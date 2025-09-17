from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Dict, List, Tuple

from src.gnn.graph_dataset import build_graph_record, save_graph_record
from src.milp.scenario_loader import load_scenario_data


def _ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def _load_json(path: Path) -> Dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _collect_scenarios(scenario_dir: Path) -> List[Path]:
    scenarios: List[Path] = []
    for path in sorted(scenario_dir.glob("scenario_*.json")):
        stem = path.stem
        suffix = stem[len("scenario_"):]
        if suffix.isdigit() and len(suffix) == 5:
            scenarios.append(path)
    return scenarios


def _family_id(scenario: Dict) -> str:
    regions = scenario["graph"]["regions"]
    weather = scenario["exogenous"].get("weather_profile", "unknown")
    return f"R{regions}_{weather}"


def _rebalance(train: List[str], val: List[str], test: List[str]) -> Tuple[List[str], List[str], List[str]]:
    if not train:
        if val:
            train.append(val.pop())
        elif test:
            train.append(test.pop())
    if not val:
        if test:
            val.append(test.pop())
        elif train:
            val.append(train[-1])
    if not test:
        if val:
            test.append(val[-1])
        elif train:
            test.append(train[-1])
    return train, val, test


def _split_families(families: List[str], ratios: Tuple[float, float, float], seed: int) -> Dict[str, List[str]]:
    random.seed(seed)
    families = families[:]
    random.shuffle(families)
    total = len(families)
    train_end = int(total * ratios[0])
    val_end = train_end + int(total * ratios[1])
    train = families[:train_end]
    val = families[train_end:val_end]
    test = families[val_end:]
    train, val, test = _rebalance(train, val, test)
    return {"train": train, "val": val, "test": test}


def build_dataset(
    scenario_dir: Path,
    report_dir: Path,
    output_dir: Path,
    split_ratios: Tuple[float, float, float],
    seed: int,
) -> Dict:
    _ensure_dir(output_dir)
    scenarios = _collect_scenarios(scenario_dir)
    if not scenarios:
        raise RuntimeError(f"No scenarios found in {scenario_dir}")

    index_entries = []
    family_set = set()

    for scenario_path in scenarios:
        scenario = _load_json(scenario_path)
        report_path = report_dir / f"{scenario_path.stem}.json"
        if not report_path.exists():
            raise RuntimeError(f"Missing report for {scenario_path.stem} at {report_path}")
        report = _load_json(report_path)
        scenario_data = load_scenario_data(scenario_path)
        record = build_graph_record(scenario_data, report)
        out_npz = output_dir / f"{scenario_path.stem}.npz"
        save_graph_record(record, out_npz)

        family = _family_id(scenario)
        family_set.add(family)
        entry = {
            "scenario_file": str(scenario_path),
            "report_file": str(report_path),
            "graph_file": str(out_npz),
            "scenario_id": scenario.get("id"),
            "family": family,
            "regions": scenario["graph"]["regions"],
            "weather_profile": scenario["exogenous"].get("weather_profile"),
            "zones": sum(scenario["graph"]["zones_per_region"]),
            "objective": report["mip"]["objective"],
        }
        index_entries.append(entry)

    splits = _split_families(sorted(family_set), split_ratios, seed)

    index = {
        "entries": index_entries,
        "split_families": splits,
        "splits": {
            split: [entry for entry in index_entries if entry["family"] in families]
            for split, families in splits.items()
        },
        "ratios": {"train": split_ratios[0], "val": split_ratios[1], "test": split_ratios[2]},
        "seed": seed,
    }
    return index


def main() -> None:
    parser = argparse.ArgumentParser(description="Build graph datasets from scenario and report directories.")
    parser.add_argument("scenario_dir", type=Path, help="Directory containing scenario_*.json files")
    parser.add_argument("report_dir", type=Path, help="Directory containing matching report JSON files")
    parser.add_argument("output_dir", type=Path, help="Directory to store NPZ graph files")
    parser.add_argument("--split-ratios", nargs=3, type=float, default=(0.7, 0.15, 0.15), help="Train/val/test ratios")
    parser.add_argument("--seed", type=int, default=1234, help="Random seed for splits")
    parser.add_argument("--index-path", type=Path, default=None, help="Optional path to write dataset index JSON")
    args = parser.parse_args()

    total_ratio = sum(args.split_ratios)
    if total_ratio <= 0:
        raise RuntimeError("Split ratios must be positive")
    ratios = tuple(val / total_ratio for val in args.split_ratios)

    index = build_dataset(args.scenario_dir, args.report_dir, args.output_dir, ratios, args.seed)

    index_path = args.index_path or (args.output_dir / "dataset_index.json")
    _ensure_dir(index_path.parent)
    index_path.write_text(json.dumps(index, indent=2), encoding="utf-8")
    print(f"Dataset index written to {index_path}")


if __name__ == "__main__":
    main()
