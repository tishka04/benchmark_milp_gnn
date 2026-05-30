from __future__ import annotations

import argparse
import json
import math
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence

import numpy as np
import pandas as pd


_FAMILY_ORDER = ["low", "medium", "high"]
_STAGE_ORDER = ["hard_fix", "repair_20", "repair_100", "full_soft", "round_refix", "failed"]


def _safe_float(value: Any) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return float("nan")
    return out if math.isfinite(out) else float("nan")


def _is_feasible(row: Dict[str, Any]) -> bool:
    return bool(row.get("success", False)) and math.isfinite(_safe_float(row.get("lp_objective", float("nan"))))


def _row_abs_gap_pct(row: Dict[str, Any]) -> float:
    direct = _safe_float(row.get("milp_abs_gap_pct", float("nan")))
    if math.isfinite(direct):
        return direct
    lp_objective = _safe_float(row.get("lp_objective", float("nan")))
    milp_objective = _safe_float(row.get("milp_objective", float("nan")))
    if not math.isfinite(lp_objective) or not math.isfinite(milp_objective):
        return float("nan")
    denom = max(abs(milp_objective), 1.0)
    return abs(100.0 * (lp_objective - milp_objective) / denom)


def _load_rows(results_dir: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    scenario_dir = results_dir / "scenario_results"
    if not scenario_dir.is_dir():
        return rows

    for path in sorted(scenario_dir.glob("*/*.json")):
        try:
            with path.open("r", encoding="utf-8") as handle:
                row = json.load(handle)
        except (OSError, json.JSONDecodeError):
            continue
        if isinstance(row, dict):
            row["_path"] = str(path)
            rows.append(row)
    return rows


def _ordered_families(rows: Sequence[Dict[str, Any]]) -> List[str]:
    present = sorted({str(row.get("family", "")) for row in rows if row.get("family", "")})
    ordered = [family for family in _FAMILY_ORDER if family in present]
    ordered.extend([family for family in present if family not in ordered])
    return ordered


def _family_summary(rows: Sequence[Dict[str, Any]]) -> pd.DataFrame:
    if not rows:
        return pd.DataFrame()

    out_rows: List[Dict[str, Any]] = []
    families = _ordered_families(rows)
    for family in families + ["ALL"]:
        fam_rows = list(rows) if family == "ALL" else [row for row in rows if str(row.get("family", "")) == family]
        if not fam_rows:
            continue

        feasible = [row for row in fam_rows if _is_feasible(row)]
        feasible_rate = 100.0 * len(feasible) / max(len(fam_rows), 1)
        gap_values = [
            _row_abs_gap_pct(row)
            for row in feasible
        ]
        gap_values = [value for value in gap_values if math.isfinite(value)]

        out_rows.append(
            {
                "family": family,
                "n": int(len(fam_rows)),
                "feasible_pct": float(feasible_rate),
                "milp_gap_median_pct": float(np.median(gap_values)) if gap_values else float("nan"),
                "hard_fix_pct": float(100.0 * np.mean([str(row.get("lp_stage_used", "")) == "hard_fix" for row in fam_rows])),
                "reach_repair20_pct": float(100.0 * np.mean([bool(row.get("lp_reached_repair_20", False)) for row in fam_rows])),
                "reach_repair100_pct": float(100.0 * np.mean([bool(row.get("lp_reached_repair_100", False)) for row in fam_rows])),
                "reach_full_soft_pct": float(100.0 * np.mean([bool(row.get("lp_reached_full_soft", False)) for row in fam_rows])),
                "reach_round_refix_pct": float(100.0 * np.mean([bool(row.get("lp_reached_round_refix", False)) for row in fam_rows])),
                "time_median_s": float(np.median([_safe_float(row.get("time_total", float("nan"))) for row in fam_rows if math.isfinite(_safe_float(row.get("time_total", float("nan"))))])) if fam_rows else float("nan"),
            }
        )

    return pd.DataFrame(out_rows)


def _stage_summary(rows: Sequence[Dict[str, Any]]) -> pd.DataFrame:
    if not rows:
        return pd.DataFrame()

    counts: Dict[str, int] = {}
    for row in rows:
        stage = str(row.get("lp_stage_reached") or row.get("lp_stage_used") or "failed")
        counts[stage] = counts.get(stage, 0) + 1

    ordered = [stage for stage in _STAGE_ORDER if stage in counts]
    ordered.extend([stage for stage in sorted(counts) if stage not in ordered])
    total = max(len(rows), 1)
    return pd.DataFrame(
        [
            {
                "stage_reached": stage,
                "count": int(counts[stage]),
                "rate_pct": 100.0 * counts[stage] / total,
            }
            for stage in ordered
        ]
    )


def _print_snapshot(rows: Sequence[Dict[str, Any]], results_dir: Path) -> None:
    stamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print("\n" + "=" * 96)
    print(f"[{stamp}] Live latent summary: {results_dir}")
    print(f"Scenarios loaded: {len(rows)}")

    if not rows:
        print("No readable scenario results yet.")
        return

    feasible_count = sum(1 for row in rows if _is_feasible(row))
    feasible_pct = 100.0 * feasible_count / max(len(rows), 1)
    print(f"Feasibility: {feasible_count}/{len(rows)} = {feasible_pct:.2f}%")

    family_bits: List[str] = []
    for family in _ordered_families(rows):
        fam_rows = [row for row in rows if str(row.get("family", "")) == family]
        fam_feasible = sum(1 for row in fam_rows if _is_feasible(row))
        fam_pct = 100.0 * fam_feasible / max(len(fam_rows), 1)
        family_bits.append(f"{family}={fam_pct:.1f}%")
    if family_bits:
        print("By family: " + " | ".join(family_bits))

    family_df = _family_summary(rows)
    stage_df = _stage_summary(rows)

    if len(family_df):
        print("\nPer-family summary")
        print(
            family_df.to_string(
                index=False,
                justify="left",
                float_format=lambda value: f"{value:7.2f}",
            )
        )

    if len(stage_df):
        print("\nOverall stage-reached distribution")
        print(
            stage_df.to_string(
                index=False,
                justify="left",
                float_format=lambda value: f"{value:7.2f}",
            )
        )


def watch(results_dir: str, interval_s: float = 5.0, once: bool = False) -> None:
    root = Path(results_dir)
    seen_signature: Optional[tuple[int, float]] = None

    while True:
        rows = _load_rows(root)
        signature = (
            len(rows),
            max((Path(row["_path"]).stat().st_mtime for row in rows), default=0.0),
        )
        if once or signature != seen_signature:
            _print_snapshot(rows, root)
            seen_signature = signature
        if once:
            return
        time.sleep(max(0.5, float(interval_s)))


def _parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Watch latent scenario_results and print rolling feasibility/stage/gap summaries.")
    parser.add_argument("--results-dir", required=True, help="Evaluation output dir containing scenario_results/.")
    parser.add_argument("--interval-s", type=float, default=5.0, help="Polling interval in seconds.")
    parser.add_argument("--once", action="store_true", help="Print one snapshot and exit.")
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = _parse_args(argv)
    watch(
        results_dir=args.results_dir,
        interval_s=args.interval_s,
        once=args.once,
    )


if __name__ == "__main__":
    main()
