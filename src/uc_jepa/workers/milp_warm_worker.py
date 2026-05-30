"""Minimal wrapper around LPWorkerTwoStage for latent candidate evaluation."""
from __future__ import annotations

import math
import pickle
import subprocess
import sys
import tempfile
import time
from collections import Counter
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import numpy as np
import torch


_LP_STAGE_ORDER = ["hard_fix", "repair_20", "repair_100", "full_soft", "round_refix"]


def _normalize_stage_name(stage: Any) -> str:
    return stage.value if hasattr(stage, "value") else str(stage or "")


def _display_stage_name(lp_result: Any, stage: str) -> str:
    if stage == "repair_20":
        return str(getattr(lp_result, "stage_name_repair_20", "") or stage)
    if stage == "repair_100":
        return str(getattr(lp_result, "stage_name_repair_100", "") or stage)
    return str(stage or "")


def _extract_stage_flags(lp_result: Any) -> Dict[str, bool]:
    final_stage = _normalize_stage_name(getattr(lp_result, "stage_used", ""))
    flags: Dict[str, bool] = {}
    for stage in _LP_STAGE_ORDER:
        time_val = float(getattr(lp_result, f"time_{stage}", 0.0) or 0.0)
        slack_val = getattr(lp_result, f"slack_{stage}", 0.0)
        flags[stage] = bool(
            time_val > 0.0
            or final_stage == stage
            or (slack_val is not None and float(slack_val) > 0.0)
        )
    return flags


def _deepest_stage_reached(lp_result: Any) -> str:
    flags = _extract_stage_flags(lp_result)
    for stage in reversed(_LP_STAGE_ORDER):
        if flags[stage]:
            return _display_stage_name(lp_result, stage)
    return _display_stage_name(
        lp_result,
        _normalize_stage_name(getattr(lp_result, "stage_used", "")),
    )


def _serialize_lp_result(lp_result: Any) -> Dict[str, Any]:
    stage_used = _normalize_stage_name(getattr(lp_result, "stage_used", ""))
    return {
        "status": str(getattr(lp_result, "status", "")),
        "stage_used": stage_used,
        "stage_label": _display_stage_name(lp_result, stage_used),
        "stage_reached": _deepest_stage_reached(lp_result),
        "lp_objective": float(getattr(lp_result, "objective_value", float("inf"))),
        "slack_mwh": float(getattr(lp_result, "slack_used", 0.0) or 0.0),
        "n_flips": int(getattr(lp_result, "n_flips", 0) or 0),
        "solve_time": float(getattr(lp_result, "solve_time", 0.0) or 0.0),
        "hard_fix_objective": float(getattr(lp_result, "hard_fix_objective", float("inf")) or float("inf")),
        "repair_objective": float(getattr(lp_result, "repair_objective", float("inf")) or float("inf")),
    }


def _build_timeout_payload(status: str, elapsed_s: float, message: str) -> Dict[str, Any]:
    return {
        "status": str(status),
        "stage_used": "failed",
        "stage_label": "failed",
        "stage_reached": "failed",
        "lp_objective": float("inf"),
        "slack_mwh": float("nan"),
        "n_flips": 0,
        "solve_time": max(0.0, float(elapsed_s)),
        "hard_fix_objective": float("inf"),
        "repair_objective": float("inf"),
        "message": str(message),
    }


@dataclass
class CandidateSolveRecord:
    candidate_idx: int
    scenario_id: str
    family: str
    energy: float
    lp_objective: float
    gap_pct: float
    feasible: bool
    status: str
    stage_used: str
    stage_label: str
    stage_reached: str
    slack_mwh: float
    n_flips: int
    solve_time: float
    hard_fix_objective: float = float("inf")
    repair_objective: float = float("inf")

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class MILPWarmWorker:
    """Cache LP workers by scenario root and score decoded latent candidates."""

    def __init__(
        self,
        scenario_roots: Sequence[str],
        solver_name: str = "appsi_highs",
        max_stages: int = 2,
        worker_kwargs: Optional[Dict[str, Any]] = None,
        enforce_hard_timeout: bool = False,
        hard_timeout_grace_s: float = 2.0,
        verbose: bool = False,
    ):
        self.scenario_roots = [str(root) for root in scenario_roots if Path(root).is_dir()]
        self.solver_name = str(solver_name)
        self.max_stages = int(max_stages)
        self.worker_kwargs = dict(worker_kwargs or {})
        self.enforce_hard_timeout = bool(enforce_hard_timeout)
        self.hard_timeout_grace_s = max(0.1, float(hard_timeout_grace_s))
        self.verbose = bool(verbose)
        self._workers: Dict[str, Any] = {}

    def _resolve_scenario_root(self, family: str, scenario_id: str) -> Optional[str]:
        if len(self.scenario_roots) == 1:
            return self.scenario_roots[0]

        for root in self.scenario_roots:
            if Path(root).name == family:
                return root
        for root in self.scenario_roots:
            path = Path(root)
            if (path / f"{scenario_id}.json").exists() or (path / "dispatch_batch" / f"{scenario_id}.json").exists():
                return root
        return None

    def _get_worker(self, scenario_root: str):
        if scenario_root not in self._workers:
            from src.milp.lp_worker_two_stage import LPWorkerTwoStage

            self._workers[scenario_root] = LPWorkerTwoStage(
                scenarios_dir=scenario_root,
                solver_name=self.solver_name,
                verbose=self.verbose,
                **self.worker_kwargs,
            )
        return self._workers[scenario_root]

    def _solve_candidate_with_optional_hard_timeout(
        self,
        scenario_root: str,
        scenario_id: str,
        decoder_output: torch.Tensor,
        force_repair_20: bool,
        candidate_time_budget_s: Optional[float],
    ) -> Dict[str, Any]:
        budget_s = (
            float(candidate_time_budget_s)
            if candidate_time_budget_s is not None and math.isfinite(float(candidate_time_budget_s))
            else None
        )
        if self.enforce_hard_timeout and budget_s is not None and budget_s > 0.0:
            payload = {
                "scenario_root": scenario_root,
                "solver_name": self.solver_name,
                "worker_kwargs": dict(self.worker_kwargs),
                "verbose": self.verbose,
                "scenario_id": str(scenario_id),
                "decoder_output": decoder_output.detach().cpu().numpy(),
                "max_stages": self.max_stages,
                "force_repair_20": bool(force_repair_20),
                "candidate_time_budget_s": budget_s,
            }
            with tempfile.TemporaryDirectory(prefix="latent_lp_timeout_") as tmp_dir:
                input_path = Path(tmp_dir) / "input.pkl"
                output_path = Path(tmp_dir) / "output.pkl"
                with input_path.open("wb") as handle:
                    pickle.dump(payload, handle)
                cmd = [
                    sys.executable,
                    "-m",
                    "src.uc_jepa.workers.lp_subprocess_worker",
                    "--input",
                    str(input_path),
                    "--output",
                    str(output_path),
                ]
                started = time.perf_counter()
                process = subprocess.Popen(
                    cmd,
                    stdout=None if self.verbose else subprocess.DEVNULL,
                    stderr=subprocess.PIPE,
                    text=True,
                )
                try:
                    _, stderr_text = process.communicate(timeout=budget_s)
                    elapsed_s = time.perf_counter() - started
                except subprocess.TimeoutExpired:
                    process.kill()
                    try:
                        _, stderr_text = process.communicate(timeout=self.hard_timeout_grace_s)
                    except subprocess.TimeoutExpired:
                        stderr_text = ""
                    elapsed_s = time.perf_counter() - started
                    return _build_timeout_payload(
                        status="hard_timeout",
                        elapsed_s=elapsed_s,
                        message=f"Candidate LP solve exceeded hard timeout of {budget_s:.2f}s.",
                    )

                if not output_path.exists():
                    stderr_suffix = f" stderr={stderr_text.strip()}" if stderr_text else ""
                    return _build_timeout_payload(
                        status="process_no_result",
                        elapsed_s=elapsed_s,
                        message=f"Candidate LP subprocess exited without output.{stderr_suffix}",
                    )

                try:
                    with output_path.open("rb") as handle:
                        message = pickle.load(handle)
                except Exception as exc:
                    return _build_timeout_payload(
                        status="process_output_error",
                        elapsed_s=elapsed_s,
                        message=f"Could not read LP subprocess output: {type(exc).__name__}: {exc}",
                    )

                if not bool(message.get("ok", False)):
                    return _build_timeout_payload(
                        status="process_error",
                        elapsed_s=elapsed_s,
                        message=str(message.get("error", "Unknown subprocess error")),
                    )

                payload_result = dict(message.get("payload", {}) or {})
                payload_result["solve_time"] = min(
                    float(payload_result.get("solve_time", elapsed_s) or elapsed_s),
                    max(elapsed_s, 0.0),
                )
                return payload_result
        worker = self._get_worker(scenario_root)
        result = worker.solve(
            scenario_id=scenario_id,
            decoder_output=decoder_output,
            max_stages=self.max_stages,
            force_repair_20=force_repair_20,
            candidate_time_budget_s=candidate_time_budget_s,
        )
        return _serialize_lp_result(result)

    @staticmethod
    def _record_from_payload(
        *,
        candidate_idx: int,
        sample,
        energy: float,
        payload: Dict[str, Any],
    ) -> Dict[str, Any]:
        ref = max(abs(float(sample.objective)), 1.0)
        lp_objective = float(payload.get("lp_objective", float("inf")))
        feasible = math.isfinite(lp_objective)
        record = CandidateSolveRecord(
            candidate_idx=int(candidate_idx),
            scenario_id=sample.scenario_id,
            family=sample.family,
            energy=float(energy),
            lp_objective=lp_objective,
            gap_pct=(
                100.0 * (lp_objective - float(sample.objective)) / ref
                if feasible else float("nan")
            ),
            feasible=feasible,
            status=str(payload.get("status", "")),
            stage_used=str(payload.get("stage_used", "")),
            stage_label=str(payload.get("stage_label", "") or payload.get("stage_used", "")),
            stage_reached=str(payload.get("stage_reached", "") or payload.get("stage_label", "") or payload.get("stage_used", "")),
            slack_mwh=float(payload.get("slack_mwh", 0.0) or 0.0),
            n_flips=int(payload.get("n_flips", 0) or 0),
            solve_time=float(payload.get("solve_time", 0.0) or 0.0),
            hard_fix_objective=float(payload.get("hard_fix_objective", float("inf")) or float("inf")),
            repair_objective=float(payload.get("repair_objective", float("inf")) or float("inf")),
        )
        return record.to_dict()

    def evaluate_candidates(
        self,
        sample,
        candidates: torch.Tensor,
        energies: Optional[Sequence[float]] = None,
        force_repair_20: bool = False,
        candidate_time_budget_s: Optional[float] = None,
    ) -> List[Dict[str, Any]]:
        """Run the LP cascade on a candidate tensor [K, Z, T, F]."""
        if candidates.dim() != 4:
            raise ValueError(f"Expected candidates [K, Z, T, F], got {tuple(candidates.shape)}")
        scenario_root = self._resolve_scenario_root(sample.family, sample.scenario_id)
        if scenario_root is None:
            raise FileNotFoundError(
                f"Could not resolve scenario root for {sample.family}/{sample.scenario_id}",
            )

        records: List[Dict[str, Any]] = []
        energy_list = list(energies) if energies is not None else [float("nan")] * candidates.shape[0]

        for idx in range(candidates.shape[0]):
            decoder_output = candidates[idx, :, :, :7].detach().cpu()
            payload = self._solve_candidate_with_optional_hard_timeout(
                scenario_root=scenario_root,
                scenario_id=sample.scenario_id,
                decoder_output=decoder_output,
                force_repair_20=force_repair_20,
                candidate_time_budget_s=candidate_time_budget_s,
            )
            record = self._record_from_payload(
                candidate_idx=idx,
                sample=sample,
                energy=float(energy_list[idx]) if idx < len(energy_list) else float("nan"),
                payload=payload,
            )
            records.append(record)

        return records

    @staticmethod
    def select_best(records: Sequence[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
        if not records:
            return None
        feasible = [row for row in records if math.isfinite(float(row.get("lp_objective", float("inf"))))]
        if feasible:
            return min(feasible, key=lambda row: float(row["lp_objective"]))
        return min(records, key=lambda row: float(row.get("energy", float("inf"))))

    @staticmethod
    def summarize(records: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
        rows = list(records)
        if not rows:
            return {
                "n": 0,
                "success_rate": 0.0,
                "mean_gap_pct": float("nan"),
                "median_gap_pct": float("nan"),
                "mean_solve_time": float("nan"),
                "stage_distribution": {},
            }

        feasible = [row for row in rows if math.isfinite(float(row.get("lp_objective", float("inf"))))]
        gaps = [float(row["gap_pct"]) for row in feasible if math.isfinite(float(row.get("gap_pct", float("nan"))))]
        solve_times = [float(row["solve_time"]) for row in rows if math.isfinite(float(row.get("solve_time", float("nan"))))]
        stage_counts = Counter((row.get("stage_used") or "unknown") for row in rows)

        return {
            "n": len(rows),
            "success_rate": len(feasible) / len(rows),
            "mean_gap_pct": float(np.mean(gaps)) if gaps else float("nan"),
            "median_gap_pct": float(np.median(gaps)) if gaps else float("nan"),
            "mean_solve_time": float(np.mean(solve_times)) if solve_times else float("nan"),
            "stage_distribution": {
                stage: {"count": count, "rate": count / len(rows)}
                for stage, count in sorted(stage_counts.items())
            },
        }
