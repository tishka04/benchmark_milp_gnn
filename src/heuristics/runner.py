"""
Runner for the RH-MO+LP heuristic baseline.

Evaluates the rolling-horizon merit-order heuristic on one scenario, or
on a whole criticality family directory under ``outputs/``. The same
LP worker as the MILP-GNN-EBM pipeline is used as Stage B so that
differences against the full pipeline are attributable to the binary
schedule generation rather than to the continuous-dispatch
reconstruction.
"""

from __future__ import annotations

import argparse
import json
import pickle
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Dict, List, Optional

import torch

try:
    from tqdm.auto import tqdm
except ImportError:  # pragma: no cover - tqdm optional
    def tqdm(x, **_kw):
        return x

from src.ebm.feasibility import (
    HierarchicalFeasibilityDecoder,
    load_physics_from_scenario,
)
from src.heuristics.rolling_horizon import (
    HeuristicConfig,
    rolling_horizon_heuristic,
)


@dataclass
class HeuristicResult:
    """Per-scenario result of the RH-MO+LP baseline."""

    scenario_id: str
    family: str = ""

    # Timing (seconds)
    time_heuristic: float = 0.0
    time_decoder: float = 0.0
    time_lp_solve: float = 0.0
    time_total: float = 0.0

    # LP outputs
    lp_status: str = ""
    lp_stage_used: str = ""
    lp_objective: float = float("nan")
    lp_slack: float = 0.0
    lp_n_flips: int = 0

    # Scenario metadata
    n_zones: int = 0
    n_timesteps: int = 0
    criticality_index: float = 0.0

    # Bookkeeping
    success: bool = True
    error_message: str = ""

    # Optional binary statistics
    binary_active_fractions: Dict[str, float] = field(default_factory=dict)


def _binary_stats(u_bin: torch.Tensor) -> Dict[str, float]:
    names = [
        "battery_charge",
        "battery_discharge",
        "pumped_charge",
        "pumped_discharge",
        "dr_active",
        "thermal_startup",
        "thermal_on",
    ]
    out = {}
    for i, name in enumerate(names):
        out[name] = float(u_bin[..., i].mean().item())
    return out


class HeuristicRunner:
    """Run the RH-MO+LP baseline on individual scenarios or families."""

    def __init__(
        self,
        heuristic_config: Optional[HeuristicConfig] = None,
        solver_name: str = "appsi_highs",
        slack_tol_mwh: float = 1.0,
        deviation_penalty: float = 10000.0,
        warm_start: bool = True,
        max_stages: int = 5,
        verbose: bool = False,
    ):
        self.heuristic_config = heuristic_config or HeuristicConfig()
        self.solver_name = solver_name
        self.slack_tol_mwh = slack_tol_mwh
        self.deviation_penalty = deviation_penalty
        self.warm_start = warm_start
        self.max_stages = max_stages
        self.verbose = verbose

    # ------------------------------------------------------------------
    def evaluate_scenario(
        self,
        scenario_path: Path,
        family: str = "",
    ) -> HeuristicResult:
        """Run RH-MO heuristic + LP worker on a single scenario JSON."""
        from src.milp.lp_worker_two_stage import LPWorkerTwoStage

        scenario_path = Path(scenario_path)
        sc_id = scenario_path.stem
        result = HeuristicResult(scenario_id=sc_id, family=family)

        try:
            with scenario_path.open("r", encoding="utf-8") as f:
                meta = json.load(f)
            result.criticality_index = float(meta.get("criticality_index", 0.0))
        except Exception:
            pass

        t_total = time.perf_counter()

        try:
            # ── Stage A: rolling-horizon merit-order heuristic ────────
            t0 = time.perf_counter()
            physics = load_physics_from_scenario(
                sc_id, str(scenario_path.parent)
            )
            u_heur = rolling_horizon_heuristic(physics, self.heuristic_config)
            result.time_heuristic = time.perf_counter() - t0
            result.n_zones = int(physics.n_zones)
            result.n_timesteps = int(physics.n_timesteps)
            result.binary_active_fractions = _binary_stats(u_heur)

            # ── Optional warm-start via pass-through decoder ──────────
            feasible_plan = None
            if self.warm_start:
                t0 = time.perf_counter()
                decoder = HierarchicalFeasibilityDecoder(physics)
                feasible_plan = decoder.decode_passthrough(u_heur)
                result.time_decoder = time.perf_counter() - t0

            # ── Stage B: LP worker cascade ────────────────────────────
            worker = LPWorkerTwoStage(
                scenarios_dir=str(scenario_path.parent),
                solver_name=self.solver_name,
                slack_tol_mwh=self.slack_tol_mwh,
                deviation_penalty=self.deviation_penalty,
                verbose=self.verbose,
            )
            t0 = time.perf_counter()
            lp_res = worker.solve(
                sc_id,
                u_heur,
                feasible_plan=feasible_plan,
                max_stages=self.max_stages,
            )
            result.time_lp_solve = time.perf_counter() - t0

            stage_used = lp_res.stage_used
            result.lp_status = str(lp_res.status)
            result.lp_stage_used = (
                stage_used.value if hasattr(stage_used, "value") else str(stage_used)
            )
            result.lp_objective = float(lp_res.objective_value)
            result.lp_slack = float(getattr(lp_res, "slack_used", 0.0))
            result.lp_n_flips = int(getattr(lp_res, "n_flips", 0) or 0)

        except Exception as exc:
            result.success = False
            result.error_message = f"{type(exc).__name__}: {exc}"

        result.time_total = time.perf_counter() - t_total
        return result

    # ------------------------------------------------------------------
    def evaluate_family(
        self,
        scenarios_dir: Path,
        family_name: str = "",
        max_scenarios: Optional[int] = None,
    ) -> List[HeuristicResult]:
        scenarios_dir = Path(scenarios_dir)
        files = sorted(scenarios_dir.glob("scenario_*.json"))
        if max_scenarios is not None:
            files = files[:max_scenarios]

        if self.verbose:
            print(
                f"\nRH-MO+LP evaluation: {len(files)} scenarios "
                f"in {family_name or scenarios_dir.name}"
            )

        results: List[HeuristicResult] = []
        iterator = tqdm(files, desc=f"RH-MO+LP [{family_name or scenarios_dir.name}]")
        for sc in iterator:
            r = self.evaluate_scenario(sc, family=family_name)
            results.append(r)
            if not r.success and self.verbose:
                print(f"  FAILED {r.scenario_id}: {r.error_message}")
        return results

    # ------------------------------------------------------------------
    @staticmethod
    def save_results(results: List[HeuristicResult], output_path: Path) -> None:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        records = [asdict(r) for r in results]
        if output_path.suffix.lower() == ".json":
            with output_path.open("w", encoding="utf-8") as f:
                json.dump(records, f, indent=2)
        else:
            with output_path.open("wb") as f:
                pickle.dump(records, f)


# ==========================================================================
# CLI
# ==========================================================================
def _build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=(
            "RH-MO+LP rolling-horizon merit-order heuristic baseline. "
            "Stage A produces a binary schedule; Stage B uses the same LP "
            "worker as the MILP-GNN-EBM pipeline."
        )
    )
    p.add_argument(
        "input",
        type=Path,
        help=(
            "Either a single scenario JSON file or a family directory "
            "containing scenario_*.json (e.g. outputs/high_criticality_scenarios)."
        ),
    )
    p.add_argument(
        "--output",
        "-o",
        type=Path,
        default=None,
        help="Path to save the result(s) as JSON (default: alongside input).",
    )
    p.add_argument("--window", "-W", type=int, default=6, help="Look-ahead window W.")
    p.add_argument("--max-scenarios", type=int, default=None)
    p.add_argument("--family", type=str, default="")
    p.add_argument("--solver", type=str, default="appsi_highs")
    p.add_argument("--slack-tol", type=float, default=1.0)
    p.add_argument("--deviation-penalty", type=float, default=10000.0)
    p.add_argument(
        "--no-warm-start",
        action="store_true",
        help="Skip the pass-through decoder warm start (faster but may be slower LP).",
    )
    p.add_argument("--max-stages", type=int, default=5)
    p.add_argument("--verbose", "-v", action="store_true")
    return p


def main(argv: Optional[List[str]] = None) -> None:
    args = _build_argparser().parse_args(argv)

    cfg = HeuristicConfig(window=args.window)
    runner = HeuristicRunner(
        heuristic_config=cfg,
        solver_name=args.solver,
        slack_tol_mwh=args.slack_tol,
        deviation_penalty=args.deviation_penalty,
        warm_start=not args.no_warm_start,
        max_stages=args.max_stages,
        verbose=args.verbose,
    )

    inp: Path = args.input
    if inp.is_file():
        result = runner.evaluate_scenario(inp, family=args.family)
        out = args.output or inp.with_name(f"{inp.stem}_rhmo.json")
        runner.save_results([result], out)
        print(json.dumps(asdict(result), indent=2))
        print(f"\nSaved result -> {out}")
        return

    if inp.is_dir():
        family = args.family or inp.name
        results = runner.evaluate_family(
            inp, family_name=family, max_scenarios=args.max_scenarios
        )
        out = args.output or inp.parent / f"{family}_rhmo_results.json"
        runner.save_results(results, out)

        n_ok = sum(1 for r in results if r.success and r.lp_objective < float("inf"))
        print(
            f"\nDone: {n_ok}/{len(results)} scenarios solved successfully -> {out}"
        )
        if results:
            avg_time = sum(r.time_total for r in results) / len(results)
            print(f"Average wall-time per scenario: {avg_time:.2f}s")
        return

    raise SystemExit(f"Input does not exist: {inp}")


if __name__ == "__main__":
    main()
