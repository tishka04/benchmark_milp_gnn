"""Build a per-scenario criticality-components table.

For every (family, scenario_id) in an input CSV (typically
``matched_budget_per_scenario.csv``), load the source scenario JSON from
``outputs/<family>_criticality_scenarios/<scenario_id>.json`` and compute the
22 normalized A/B sub-components from
``src.analysis.criticality_index.compute_criticality``.

The resulting table is the join key for downstream driver analysis (which
criticality axes correlate with pipeline advantage at constrained time).

Example
-------
python scripts/build_criticality_components_table.py \
    --input  outputs/uc_jepa/latent_vs_grid_milp_120_caps_k1/matched_budget_per_scenario.csv \
    --output outputs/uc_jepa/latent_vs_grid_milp_120_caps_k1/criticality_components_per_scenario.csv
"""
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd

# Ensure repo root is importable.
_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from src.analysis.criticality_index import (  # noqa: E402
    compute_criticality,
)


FAMILY_ALIAS_TO_DIR = {
    "low": "low_criticality_scenarios",
    "medium": "medium_criticality_scenarios",
    "high": "high_criticality_scenarios",
}


def _resolve_scenario_path(repo_root: Path, family: str, scenario_id: str) -> Optional[Path]:
    """Locate the scenario JSON for a (family, scenario_id) pair.

    Tries the alias mapping first, then falls back to a literal family name.
    """
    candidates: List[Path] = []
    alias = FAMILY_ALIAS_TO_DIR.get(family, family)
    candidates.append(repo_root / "outputs" / alias / f"{scenario_id}.json")
    if alias != family:
        candidates.append(repo_root / "outputs" / family / f"{scenario_id}.json")
    for cand in candidates:
        if cand.exists():
            return cand
    return None


def build_table(
    input_csv: Path,
    output_csv: Path,
    repo_root: Path,
    alpha: float = 0.5,
) -> pd.DataFrame:
    df = pd.read_csv(input_csv)
    required = {"family", "scenario_id"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Input CSV is missing columns: {missing}")

    pairs = (
        df[["family", "scenario_id"]]
        .drop_duplicates()
        .reset_index(drop=True)
    )

    rows: List[Dict[str, float]] = []
    misses: List[str] = []
    for _, row in pairs.iterrows():
        family = str(row["family"])
        scenario_id = str(row["scenario_id"])
        scenario_path = _resolve_scenario_path(repo_root, family, scenario_id)
        if scenario_path is None:
            misses.append(f"{family}/{scenario_id}")
            continue
        with scenario_path.open("r", encoding="utf-8") as fh:
            scenario = json.load(fh)
        result = compute_criticality(scenario, alpha=alpha)

        out: Dict[str, float] = {
            "family": family,
            "scenario_id": scenario_id,
            "stress_index": result.stress_index,
            "hardness_index": result.hardness_index,
            "criticality_index_recomputed": result.criticality_index,
        }
        # Normalized components in [0, 1].
        for k, v in result.stress_normalized.items():
            out[f"{k}_norm"] = float(v)
        for k, v in result.hardness_normalized.items():
            out[f"{k}_norm"] = float(v)
        # Raw component values (pre-normalization) for interpretability.
        for k, v in asdict(result.stress_metrics).items():
            out[f"{k}_raw"] = float(v)
        for k, v in asdict(result.hardness_metrics).items():
            out[f"{k}_raw"] = float(v)
        rows.append(out)

    if misses:
        print(f"[warn] {len(misses)} scenarios could not be located, e.g. {misses[:3]}")

    out_df = pd.DataFrame(rows)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(output_csv, index=False)
    print(
        f"Wrote {len(out_df)} rows ({out_df['family'].value_counts().to_dict()}) "
        f"-> {output_csv}"
    )
    return out_df


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True, help="Per-scenario input CSV.")
    parser.add_argument("--output", type=Path, required=True, help="Output components CSV path.")
    parser.add_argument("--alpha", type=float, default=0.5)
    parser.add_argument(
        "--repo-root",
        type=Path,
        default=_REPO_ROOT,
        help="Path to benchmark_milp_gnn repo root (default: inferred).",
    )
    args = parser.parse_args()
    build_table(args.input, args.output, args.repo_root, alpha=args.alpha)


if __name__ == "__main__":
    main()
