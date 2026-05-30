"""Build a criticality-radar comparing personas (or any group of scenario dirs).

For each input directory, computes the mean of every normalized A/B sub-component
(A1-A12 stress, B1-B10 hardness) across all ``scenario_*.json`` files, then
overlays them on a single radar chart.

Two figures are produced:
- ``fig_persona_radar_stress.png``    (12 stress axes A1-A12)
- ``fig_persona_radar_hardness.png``  (10 hardness axes B1-B10)
- ``fig_persona_radar_all.png``       (combined 22-axis radar)

The labels are taken from each directory's ``manifest.json`` if present
(``persona_label`` key), else from the directory name.

Example
-------
python scripts/build_persona_criticality_radar.py \
    --dirs outputs/persona_vre_battery_scenarios \
           outputs/persona_network_operator_scenarios \
           outputs/persona_mathematician_scenarios \
    --output outputs/personas_criticality_radar
"""
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from src.analysis.criticality_index import compute_criticality  # noqa: E402

STRESS_KEYS = [f"A{i}" for i in range(1, 13)]
HARDNESS_KEYS = [f"B{i}" for i in range(1, 11)]


def _short(name: str) -> str:
    """A1_vre_penetration -> A1."""
    return name.split("_", 1)[0]


def compute_dir_means(scenario_dir: Path) -> Tuple[Dict[str, float], Dict[str, float], int]:
    """Return (mean_stress_norm, mean_hardness_norm, n_scenarios)."""
    files = sorted(scenario_dir.glob("scenario_*.json"))
    if not files:
        raise FileNotFoundError(f"No scenarios in {scenario_dir}")
    s_acc: Dict[str, List[float]] = {}
    h_acc: Dict[str, List[float]] = {}
    for fp in files:
        with fp.open("r", encoding="utf-8") as fh:
            scenario = json.load(fh)
        result = compute_criticality(scenario)
        for k, v in result.stress_normalized.items():
            s_acc.setdefault(k, []).append(float(v))
        for k, v in result.hardness_normalized.items():
            h_acc.setdefault(k, []).append(float(v))
    s_mean = {k: float(np.mean(v)) for k, v in s_acc.items()}
    h_mean = {k: float(np.mean(v)) for k, v in h_acc.items()}
    return s_mean, h_mean, len(files)


_PERSONA_LABEL_MAP = {
    "persona_vre_battery_scenarios": "VRE / Battery developer",
    "persona_network_operator_scenarios": "Network operator",
    "persona_mathematician_scenarios": "Mathematician (combinatorial)",
}


def _label_for(scenario_dir: Path) -> str:
    manifest = scenario_dir / "manifest.json"
    if manifest.exists():
        try:
            with manifest.open("r", encoding="utf-8") as fh:
                payload = json.load(fh)
            for key in ("persona_label", "label", "family", "name"):
                if key in payload and payload[key]:
                    return str(payload[key])
        except Exception:  # noqa: BLE001
            pass
    if scenario_dir.name in _PERSONA_LABEL_MAP:
        return _PERSONA_LABEL_MAP[scenario_dir.name]
    return (
        scenario_dir.name.replace("persona_", "")
        .replace("_scenarios", "")
        .replace("_", " ")
        .title()
    )


def _radar_axes(n: int) -> np.ndarray:
    return np.linspace(0.0, 2.0 * np.pi, n, endpoint=False)


def _close(arr: np.ndarray) -> np.ndarray:
    return np.concatenate([arr, arr[:1]])


def plot_radar(
    title: str,
    axis_keys: List[str],
    series: Dict[str, Dict[str, float]],
    output_path: Path,
    palette: List[str] | None = None,
) -> None:
    palette = palette or ["#3498db", "#e67e22", "#2ecc71", "#9b59b6", "#e74c3c", "#1abc9c"]
    angles = _radar_axes(len(axis_keys))
    fig = plt.figure(figsize=(9, 9))
    ax = fig.add_subplot(111, projection="polar")
    ax.set_theta_offset(np.pi / 2)
    ax.set_theta_direction(-1)
    ax.set_xticks(angles)
    ax.set_xticklabels([_short(k) for k in axis_keys], fontsize=11)
    ax.set_ylim(0.0, 1.0)
    ax.set_yticks([0.25, 0.5, 0.75, 1.0])
    ax.set_yticklabels(["0.25", "0.5", "0.75", "1.0"], fontsize=8, color="gray")
    ax.set_rlabel_position(180.0 / max(len(axis_keys), 1))

    for i, (label, vals) in enumerate(series.items()):
        values = np.array([vals[k] for k in axis_keys], dtype=float)
        color = palette[i % len(palette)]
        ax.plot(_close(angles), _close(values), color=color, linewidth=2.5, label=label)
        ax.fill(_close(angles), _close(values), color=color, alpha=0.12)

    ax.set_title(title, fontweight="bold", pad=18)
    ax.legend(loc="upper right", bbox_to_anchor=(1.32, 1.05), fontsize=10)
    plt.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dirs", nargs="+", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True, help="Output dir for figures and CSV.")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)

    series_stress: Dict[str, Dict[str, float]] = {}
    series_hardness: Dict[str, Dict[str, float]] = {}
    counts: Dict[str, int] = {}

    for scenario_dir in args.dirs:
        label = _label_for(scenario_dir)
        s, h, n = compute_dir_means(scenario_dir)
        series_stress[label] = s
        series_hardness[label] = h
        counts[label] = n
        print(f"  {label}: n={n}, mean stress sum={sum(s.values()):.3f}, hardness sum={sum(h.values()):.3f}")

    # CSV table.
    import pandas as pd  # local import keeps script light if pandas missing

    rows = []
    for label in series_stress:
        row = {"label": label, "n_scenarios": counts[label]}
        row.update({f"{k}_norm": v for k, v in series_stress[label].items()})
        row.update({f"{k}_norm": v for k, v in series_hardness[label].items()})
        rows.append(row)
    df = pd.DataFrame(rows)
    df.to_csv(args.output / "persona_criticality_means.csv", index=False)

    stress_keys_sorted = sorted(next(iter(series_stress.values())).keys(), key=lambda x: int(x.split("_")[0][1:]))
    hardness_keys_sorted = sorted(next(iter(series_hardness.values())).keys(), key=lambda x: int(x.split("_")[0][1:]))

    plot_radar(
        "Stress sources of criticity (A1\u2013A12, normalized mean)",
        stress_keys_sorted,
        series_stress,
        args.output / "fig_persona_radar_stress.png",
    )
    plot_radar(
        "Combinatorial hardness sources (B1\u2013B10, normalized mean)",
        hardness_keys_sorted,
        series_hardness,
        args.output / "fig_persona_radar_hardness.png",
    )

    # Combined 22-axis radar.
    combined: Dict[str, Dict[str, float]] = {
        label: {**series_stress[label], **series_hardness[label]} for label in series_stress
    }
    combined_keys = stress_keys_sorted + hardness_keys_sorted
    plot_radar(
        "Full criticality profile (A1\u2013A12 stress + B1\u2013B10 hardness)",
        combined_keys,
        combined,
        args.output / "fig_persona_radar_all.png",
    )

    print(f"\nOutputs written under: {args.output}")
    for p in sorted(args.output.iterdir()):
        print(f"  {p.name}")


if __name__ == "__main__":
    main()
