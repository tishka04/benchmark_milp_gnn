"""Partial-grid recovery under contingencies -- experiment launcher.

Implements ``docs/partial_grid_recovery_protocol.md``. For each scenario it
produces a unit-commitment vector under up to four arms, freezes the commitment,
and runs a single-line N-1 post-fault DC power-flow redispatch (plus an optional
cascade-robustness pass), then writes per-contingency rows and per-arm
aggregates.

Arms
----
* ``A`` Full MILP             (S*-independent)
* ``B`` Partial MILP + fill    (depends on S*)
* ``C`` Pipeline (full)        (S*-independent; requires --run-pipeline)
* ``D`` Pipeline-on-S* + fill  (depends on S*; requires --run-pipeline)

Example (pilot, arms A/B only, 10 scenarios/family, single-line N-1 + cascade):

    python -m scripts.run_partial_grid_recovery \
        --families low medium high \
        --scenarios-per-family 10 \
        --arms A B \
        --selection-rule top_k \
        --keep-fraction 0.4 \
        --contingency-set critical \
        --cascade \
        --out outputs/uc_jepa/partial_recovery

Add ``--run-pipeline`` to also compute arms C and D (needs the trained encoder /
EBM checkpoints and torch).
"""

from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import queue as queue_mod
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.milp.scenario_loader import load_scenario_data  # noqa: E402
from src.eval.contingency_redispatch import (  # noqa: E402
    Commitment,
    RedispatchResult,
    contingency_line_set,
    run_cascade,
    run_n1_sweep,
)
from src.eval import partial_grid as pg  # noqa: E402

FAMILY_DIRS = {
    "low": "low_criticality_scenarios",
    "medium": "medium_criticality_scenarios",
    "high": "high_criticality_scenarios",
}


# --------------------------------------------------------------------------- #
# Aggregation helpers
# --------------------------------------------------------------------------- #
def _aggregate_arm(
    n1: Dict[str, RedispatchResult], tau_mwh: float
) -> Dict[str, float]:
    """Reduce an N-1 sweep ``{outage: result}`` to per-arm summary metrics."""
    base = n1.get("__base__")
    cont = {k: v for k, v in n1.items() if k != "__base__"}
    feas = [r for r in cont.values() if r.feasible]
    lols = [r.lol_mwh for r in feas]
    risks = [r.cascade_risk_score for r in feas]
    return {
        "base_lol_mwh": float(base.lol_mwh) if base and base.feasible else float("nan"),
        "base_feasible": bool(base.feasible) if base else False,
        "n_contingencies": len(cont),
        "n_feasible": len(feas),
        "feasible_fraction": (len(feas) / len(cont)) if cont else float("nan"),
        "worst_case_lol_mwh": float(max(lols)) if lols else float("nan"),
        "mean_lol_mwh": float(np.mean(lols)) if lols else float("nan"),
        "median_lol_mwh": float(np.median(lols)) if lols else float("nan"),
        "n_critical_contingencies": int(sum(1 for x in lols if x > tau_mwh)),
        "max_cascade_risk": float(max(risks)) if risks else float("nan"),
        "mean_cascade_risk": float(np.mean(risks)) if risks else float("nan"),
        "max_overloaded_lines": int(max((r.n_overloaded_lines for r in feas), default=0)),
    }


def _contingency_rows(
    scenario_id: str,
    family: str,
    arm: str,
    rule: str,
    keep_fraction: float,
    n_zones_total: int,
    n_zones_critical: int,
    n1: Dict[str, RedispatchResult],
) -> List[Dict]:
    rows = []
    for outage, r in n1.items():
        rows.append(
            {
                "scenario_id": scenario_id,
                "family": family,
                "arm": arm,
                "rule": rule,
                "keep_fraction": keep_fraction,
                "n_zones_total": n_zones_total,
                "n_zones_critical": n_zones_critical,
                "outage": outage,
                "feasible": r.feasible,
                "termination": r.termination,
                "lol_mwh": r.lol_mwh,
                "dr_mwh": r.dr_mwh,
                "overload_mwh": r.overload_mwh,
                "n_overloaded_lines": r.n_overloaded_lines,
                "cascade_risk_score": r.cascade_risk_score,
                "redispatch_cost": r.redispatch_cost,
                "solve_seconds": r.solve_seconds,
            }
        )
    return rows


def _worst_seed_line(n1: Dict[str, RedispatchResult]) -> Optional[str]:
    """Pick the single outage with the highest post-fault load shed (cascade seed)."""
    cont = {k: v for k, v in n1.items() if k != "__base__" and v.feasible}
    if not cont:
        return None
    return max(cont.items(), key=lambda kv: kv[1].lol_mwh)[0]


# --------------------------------------------------------------------------- #
# Pipeline (arms C/D) commitment extraction
# --------------------------------------------------------------------------- #
def _build_pipeline_runner(args):
    """Construct a PipelineRunner if --run-pipeline was requested, else None."""
    if not args.run_pipeline:
        return None
    from src.eval.pipeline_runner import PipelineConfig, PipelineRunner

    cfg = PipelineConfig(
        repo_path=str(PROJECT_ROOT),
        encoder_path=args.encoder_path,
        ebm_path=args.ebm_path,
        device=args.device,
        n_samples=args.pipeline_samples,
    )
    runner = PipelineRunner(cfg)
    runner.load_models()
    return runner


# --------------------------------------------------------------------------- #
# Main per-scenario driver
# --------------------------------------------------------------------------- #
def extract_pipeline_u(scenario_path: Path, args, runner, graphs_dir: Path):
    """Run the (torch/GPU) pipeline binary extraction in the parent process.

    Returns ``(pipeline_u, seconds)`` or ``(None, nan)`` on failure / when arms
    C and D are not requested.
    """
    if runner is None or not ("C" in args.arms or "D" in args.arms):
        return None, float("nan")
    try:
        t0 = time.perf_counter()
        u = runner.extract_binary_candidate(
            scenario_path, graphs_dir, select=args.pipeline_select
        )
        return u, time.perf_counter() - t0
    except Exception as exc:  # pragma: no cover
        print(f"  [pipeline] {scenario_path.stem}: extraction failed: "
              f"{type(exc).__name__}: {exc}")
        return None, float("nan")


def process_scenario(
    scenario_path: Path,
    family: str,
    args,
    runner,
    graphs_dir: Path,
) -> Dict[str, List[Dict]]:
    """Non-isolated path: extract pipeline binaries then run the solve work."""
    pipeline_u, pl_secs = extract_pipeline_u(scenario_path, args, runner, graphs_dir)
    return process_scenario_solves(scenario_path, family, args, pipeline_u, pl_secs)


def process_scenario_solves(
    scenario_path: Path,
    family: str,
    args,
    pipeline_u=None,
    pl_secs: float = float("nan"),
) -> Dict[str, List[Dict]]:
    data = load_scenario_data(scenario_path)
    sc_id = scenario_path.stem
    n_zones = len(data.zones)
    contingency_lines = contingency_line_set(
        data, mode=args.contingency_set, top_k=args.contingency_top_k
    )

    t_scen0 = time.perf_counter()
    deadline = t_scen0 + args.scenario_time_limit if args.scenario_time_limit and args.scenario_time_limit > 0 else None

    def past_deadline() -> bool:
        return deadline is not None and time.perf_counter() >= deadline

    def commit_tl() -> float:
        if deadline is None:
            return args.milp_time_limit
        return max(0.0, min(args.milp_time_limit, deadline - time.perf_counter()))

    rd_kwargs = dict(
        contingency_lines=contingency_lines,
        overload_penalty=args.overload_penalty,
        fix_modes=args.fix_modes,
        solver_name=args.solver,
        time_limit=args.redispatch_time_limit,
        deadline=deadline,
    )

    contingency_rows: List[Dict] = []
    scenario_rows: List[Dict] = []
    cascade_rows: List[Dict] = []

    # --- S*-independent commitments (A, C) computed once -------------------- #
    base_commitments: Dict[str, Commitment] = {}
    commit_meta: Dict[str, Dict] = {}

    if "A" in args.arms:
        t0 = time.perf_counter()
        c, ok, term = pg.commitment_full_milp(
            data, solver_name=args.solver, time_limit=commit_tl()
        )
        commit_meta["A"] = {"ok": ok, "term": term, "seconds": time.perf_counter() - t0}
        if ok:
            base_commitments["A"] = c

    if pipeline_u is not None and "C" in args.arms:
        c = pg.commitment_from_binary_tensor(pipeline_u, data.zones, len(data.periods))
        base_commitments["C"] = c
        commit_meta["C"] = {"ok": True, "term": "pipeline", "seconds": pl_secs}

    # --- Run N-1 for S*-independent arms ----------------------------------- #
    for arm in ("A", "C"):
        if arm not in base_commitments:
            continue
        n1 = run_n1_sweep(data, base_commitments[arm], **rd_kwargs)
        contingency_rows += _contingency_rows(
            sc_id, family, arm, "NA", float("nan"), n_zones, n_zones, n1
        )
        agg = _aggregate_arm(n1, args.tau_mwh)
        agg.update(
            scenario_id=sc_id, family=family, arm=arm, rule="NA",
            keep_fraction=float("nan"), n_zones_total=n_zones,
            n_zones_critical=n_zones, commit_seconds=commit_meta[arm]["seconds"],
        )
        scenario_rows.append(agg)
        if args.cascade and not past_deadline():
            cascade_rows += _run_cascade_for(
                data, base_commitments[arm], n1, sc_id, family, arm, "NA",
                float("nan"), args, deadline,
            )

    # --- S*-dependent arms (B, D) over each (rule, fraction) --------------- #
    for rule in args.selection_rule:
        for frac in args.keep_fraction:
            critical = pg.select_critical_zones(data, rule=rule, keep_fraction=frac, seed=args.seed)
            n_crit = len(critical)

            if "B" in args.arms and not past_deadline():
                t0 = time.perf_counter()
                cB, okB, termB = pg.commitment_partial_milp(
                    data, critical, solver_name=args.solver, time_limit=commit_tl()
                )
                secB = time.perf_counter() - t0
                if okB:
                    n1 = run_n1_sweep(data, cB, **rd_kwargs)
                    contingency_rows += _contingency_rows(
                        sc_id, family, "B", rule, frac, n_zones, n_crit, n1
                    )
                    agg = _aggregate_arm(n1, args.tau_mwh)
                    agg.update(
                        scenario_id=sc_id, family=family, arm="B", rule=rule,
                        keep_fraction=frac, n_zones_total=n_zones,
                        n_zones_critical=n_crit, commit_seconds=secB,
                    )
                    scenario_rows.append(agg)
                    if args.cascade and not past_deadline():
                        cascade_rows += _run_cascade_for(
                            data, cB, n1, sc_id, family, "B", rule, frac, args, deadline
                        )

            if "D" in args.arms and pipeline_u is not None and not past_deadline():
                cPipe = pg.commitment_from_binary_tensor(pipeline_u, data.zones, len(data.periods))
                cD = pg.commitment_pipeline_on_critical(data, cPipe, critical)
                n1 = run_n1_sweep(data, cD, **rd_kwargs)
                contingency_rows += _contingency_rows(
                    sc_id, family, "D", rule, frac, n_zones, n_crit, n1
                )
                agg = _aggregate_arm(n1, args.tau_mwh)
                agg.update(
                    scenario_id=sc_id, family=family, arm="D", rule=rule,
                    keep_fraction=frac, n_zones_total=n_zones,
                    n_zones_critical=n_crit, commit_seconds=float("nan"),
                )
                scenario_rows.append(agg)
                if args.cascade and not past_deadline():
                    cascade_rows += _run_cascade_for(
                        data, cD, n1, sc_id, family, "D", rule, frac, args, deadline
                    )

    return {
        "contingency": contingency_rows,
        "scenario": scenario_rows,
        "cascade": cascade_rows,
    }


def _solve_worker(q, scenario_path: str, family: str, args, pipeline_u, pl_secs):
    """Subprocess entry point: run the solve work and return rows via the queue.

    Lives in a separate process so a single hung solver call (e.g. HiGHS not
    honouring its time limit during presolve) can be hard-killed by the parent.
    """
    try:
        res = process_scenario_solves(Path(scenario_path), family, args, pipeline_u, pl_secs)
        q.put(("ok", res))
    except Exception as exc:  # pragma: no cover
        q.put(("err", f"{type(exc).__name__}: {exc}"))


def _run_scenario_isolated(sp: Path, family: str, args, pipeline_u, pl_secs, hard_timeout: float):
    """Run one scenario in a child process; hard-kill it if it exceeds the budget.

    Returns the rows dict, or ``None`` on timeout / failure.
    """
    q: mp.Queue = mp.Queue()
    p = mp.Process(target=_solve_worker, args=(q, str(sp), family, args, pipeline_u, pl_secs))
    p.start()
    try:
        status, payload = q.get(timeout=hard_timeout)
    except queue_mod.Empty:
        p.terminate()
        p.join()
        print(f"      TIMEOUT (> {hard_timeout:.0f}s) - killed scenario process")
        return None
    finally:
        if p.is_alive():
            p.join(timeout=5)
            if p.is_alive():
                p.terminate()
    if status == "err":
        print(f"      FAILED in worker: {payload}")
        return None
    return payload


def _run_cascade_for(
    data, commitment, n1, sc_id, family, arm, rule, frac, args, deadline=None
) -> List[Dict]:
    seed = _worst_seed_line(n1)
    if seed is None:
        return []
    history = run_cascade(
        data, commitment, seed,
        overload_threshold=args.cascade_threshold,
        max_iterations=args.cascade_max_iter,
        overload_penalty=args.overload_penalty,
        fix_modes=args.fix_modes,
        solver_name=args.solver,
        time_limit=args.redispatch_time_limit,
        deadline=deadline,
    )
    rows = []
    for rnd, r in enumerate(history):
        rows.append(
            {
                "scenario_id": sc_id, "family": family, "arm": arm,
                "rule": rule, "keep_fraction": frac, "seed_line": seed,
                "round": rnd, "n_tripped": len(r.tripped_lines),
                "tripped_lines": ";".join(r.tripped_lines),
                "feasible": r.feasible, "lol_mwh": r.lol_mwh,
                "n_overloaded_lines": r.n_overloaded_lines,
                "cascade_risk_score": r.cascade_risk_score,
            }
        )
    return rows


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def parse_args():
    p = argparse.ArgumentParser(description="Partial-grid recovery under contingencies.")
    p.add_argument("--families", nargs="+", default=["low", "medium", "high"],
                   choices=list(FAMILY_DIRS))
    p.add_argument("--scenarios-per-family", type=int, default=10)
    p.add_argument("--scenarios-dir-root", default=str(PROJECT_ROOT / "outputs"),
                   help="Root containing <family>_criticality_scenarios directories.")
    p.add_argument("--arms", nargs="+", default=["A", "B", "C", "D"],
                   choices=["A", "B", "C", "D"])
    p.add_argument("--selection-rule", nargs="+", default=["top_k"],
                   choices=["top_k", "graph_centrality", "random"])
    p.add_argument("--keep-fraction", nargs="+", type=float, default=[0.4])
    p.add_argument("--contingency-set", default="critical", choices=["critical", "n1"])
    p.add_argument("--contingency-top-k", type=int, default=12)
    p.add_argument("--tau-mwh", type=float, default=1.0,
                   help="Load-shed threshold above which a contingency is 'critical'.")
    # cascade
    p.add_argument("--cascade", action="store_true")
    p.add_argument("--cascade-threshold", type=float, default=0.0,
                   help="Fractional overload above capacity that trips a line next round.")
    p.add_argument("--cascade-max-iter", type=int, default=5)
    # solver / budgets
    p.add_argument("--solver", default="appsi_highs")
    p.add_argument("--milp-time-limit", type=float, default=120.0)
    p.add_argument("--redispatch-time-limit", type=float, default=120.0)
    p.add_argument("--scenario-time-limit", type=float, default=1200.0,
                   help="Per-scenario wall-clock budget (s); remaining work is skipped "
                        "once exceeded. Set <=0 to disable.")
    p.add_argument("--scenario-grace", type=float, default=180.0,
                   help="Extra seconds beyond --scenario-time-limit before the scenario "
                        "subprocess is hard-killed (covers a single in-flight solve).")
    p.add_argument("--isolate", action=argparse.BooleanOptionalAction, default=True,
                   help="Run each scenario in a killable subprocess so a hung solver "
                        "cannot stall the run (default: on). Use --no-isolate to disable.")
    p.add_argument("--overload-penalty", type=float, default=5000.0)
    p.add_argument("--fix-modes", action="store_true",
                   help="Also freeze storage/DR mode binaries (default: thermal-only SCED).")
    # pipeline (arms C/D)
    p.add_argument("--run-pipeline", action="store_true")
    p.add_argument("--encoder-path", default="outputs/encoders/hierarchical_temporal_v3/best_encoder.pt")
    p.add_argument("--ebm-path", default="outputs/ebm_models/ebm_v3/ebm_v3_final.pt")
    p.add_argument("--device", default="cpu")
    p.add_argument("--pipeline-samples", type=int, default=5)
    p.add_argument("--pipeline-select", default="energy", choices=["energy", "first"])
    # output
    p.add_argument("--out", default=str(PROJECT_ROOT / "outputs" / "uc_jepa" / "partial_recovery"))
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True,
                   help="Skip scenarios already present in the output CSVs and keep their "
                        "rows (default: on). Use --no-resume for a fresh run.")
    return p.parse_args()


def main():
    args = parse_args()
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    graphs_dir = out_dir / "graphs"

    runner = _build_pipeline_runner(args)
    if ("C" in args.arms or "D" in args.arms) and runner is None:
        print("[info] arms C/D requested but --run-pipeline not set; skipping them.")
        args.arms = [a for a in args.arms if a in ("A", "B")]

    all_cont: List[Dict] = []
    all_scen: List[Dict] = []
    all_casc: List[Dict] = []

    # ---- resume: reload prior rows and skip already-processed scenarios ---- #
    done: set = set()
    if args.resume:
        cpath, spath, kpath = (out_dir / "per_contingency.csv",
                               out_dir / "per_scenario_arm.csv",
                               out_dir / "cascade.csv")
        if cpath.exists():
            all_cont = pd.read_csv(cpath).to_dict("records")
        if spath.exists():
            prev = pd.read_csv(spath)
            all_scen = prev.to_dict("records")
            done = set(zip(prev["family"].astype(str), prev["scenario_id"].astype(str)))
        if kpath.exists():
            all_casc = pd.read_csv(kpath).to_dict("records")
        if done:
            print(f"[resume] {len(done)} scenario(s) already processed; they will be skipped.")

    for family in args.families:
        fam_dir = Path(args.scenarios_dir_root) / FAMILY_DIRS[family]
        scenario_files = sorted(fam_dir.glob("scenario_*.json"))[: args.scenarios_per_family]
        print(f"\n=== Family {family}: {len(scenario_files)} scenarios ===")
        for idx, sp in enumerate(scenario_files, 1):
            if (family, sp.stem) in done:
                print(f"  [{idx}/{len(scenario_files)}] {sp.stem}: SKIPPED (resume)")
                continue
            t0 = time.perf_counter()
            use_iso = args.isolate and args.scenario_time_limit and args.scenario_time_limit > 0
            if use_iso:
                # Pipeline extraction (torch/GPU) stays in the parent; only the
                # pyomo/HiGHS solve work runs in the killable child process.
                pipeline_u, pl_secs = extract_pipeline_u(sp, args, runner, graphs_dir)
                hard = float(args.scenario_time_limit) + float(args.scenario_grace)
                res = _run_scenario_isolated(sp, family, args, pipeline_u, pl_secs, hard)
                if res is None:
                    print(f"  [{idx}/{len(scenario_files)}] {sp.stem}: skipped after "
                          f"{time.perf_counter() - t0:.1f}s")
                    continue
            else:
                try:
                    res = process_scenario(sp, family, args, runner, graphs_dir)
                except Exception as exc:
                    print(f"  [{idx}/{len(scenario_files)}] {sp.stem}: FAILED {type(exc).__name__}: {exc}")
                    continue
            all_cont += res["contingency"]
            all_scen += res["scenario"]
            all_casc += res["cascade"]
            print(f"  [{idx}/{len(scenario_files)}] {sp.stem}: "
                  f"{len(res['scenario'])} arm-rows in {time.perf_counter() - t0:.1f}s")

            # Incremental flush so long runs are crash-safe.
            pd.DataFrame(all_cont).to_csv(out_dir / "per_contingency.csv", index=False)
            pd.DataFrame(all_scen).to_csv(out_dir / "per_scenario_arm.csv", index=False)
            if all_casc:
                pd.DataFrame(all_casc).to_csv(out_dir / "cascade.csv", index=False)

    # ---- final aggregates per (arm, family, rule, keep_fraction) ---------- #
    if all_scen:
        df = pd.DataFrame(all_scen)
        group_cols = ["arm", "family", "rule", "keep_fraction"]
        agg = (
            df.groupby(group_cols, dropna=False)
            .agg(
                n_scenarios=("scenario_id", "nunique"),
                median_worst_case_lol=("worst_case_lol_mwh", "median"),
                mean_worst_case_lol=("worst_case_lol_mwh", "mean"),
                pct_fully_recoverable=("worst_case_lol_mwh", lambda s: float((s <= args.tau_mwh).mean() * 100.0)),
                mean_feasible_fraction=("feasible_fraction", "mean"),
                mean_critical_contingencies=("n_critical_contingencies", "mean"),
                median_max_cascade_risk=("max_cascade_risk", "median"),
                median_base_lol=("base_lol_mwh", "median"),
            )
            .reset_index()
        )
        agg.to_csv(out_dir / "headline_by_arm.csv", index=False)
        print("\n=== Headline (median worst-case LOL MWh, % fully recoverable) ===")
        print(agg.to_string(index=False))

    meta = {
        "args": vars(args),
        "n_contingency_rows": len(all_cont),
        "n_scenario_rows": len(all_scen),
        "n_cascade_rows": len(all_casc),
    }
    (out_dir / "run_meta.json").write_text(json.dumps(meta, indent=2, default=str), encoding="utf-8")
    print(f"\nWrote outputs to {out_dir}")


if __name__ == "__main__":
    main()
