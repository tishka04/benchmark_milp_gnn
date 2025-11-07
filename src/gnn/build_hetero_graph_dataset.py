"""
Batch builder for heterogeneous multi-level graph datasets.
Mirrors the functionality of build_graph_dataset.py but for hetero graphs.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List

from tqdm import tqdm

from src.gnn.hetero_graph_dataset import build_hetero_graph_record, save_graph_record
from src.milp.scenario_loader import load_scenario_data


def _load_json(path: Path) -> Dict:
    return json.loads(path.read_text(encoding="utf-8"))


def find_matching_pairs(
    scenario_dir: Path,
    reports_dir: Path,
) -> List[tuple[Path, Path]]:
    """Find scenario/report pairs with matching stems."""
    scenario_files = {p.stem: p for p in scenario_dir.glob("scenario_*.json")}
    report_files = {p.stem: p for p in reports_dir.glob("scenario_*.json")}
    
    matching_stems = set(scenario_files.keys()) & set(report_files.keys())
    pairs = [
        (scenario_files[stem], report_files[stem])
        for stem in sorted(matching_stems)
    ]
    
    return pairs


def convert_batch(
    scenario_dir: Path,
    reports_dir: Path,
    output_dir: Path,
    *,
    resume: bool = False,
) -> Dict:
    """Convert all scenario/report pairs to heterogeneous graphs."""
    output_dir.mkdir(parents=True, exist_ok=True)
    
    pairs = find_matching_pairs(scenario_dir, reports_dir)
    if not pairs:
        print(f"No matching scenario/report pairs found in:")
        print(f"  Scenarios: {scenario_dir}")
        print(f"  Reports: {reports_dir}")
        sys.exit(1)
    
    print(f"Found {len(pairs)} scenario/report pairs")
    
    index_entries = []
    stats = {
        "success": 0,
        "failed": 0,
        "skipped": 0,
        "total_nodes": 0,
        "total_edges": 0,
    }
    
    for scenario_path, report_path in tqdm(pairs, desc="Converting"):
        stem = scenario_path.stem
        output_path = output_dir / f"{stem}.npz"
        
        if resume and output_path.exists():
            stats["skipped"] += 1
            # Still add to index
            try:
                report = _load_json(report_path)
                scenario_data = load_scenario_data(scenario_path)
                index_entries.append({
                    "graph_file": str(output_path),
                    "scenario_file": str(scenario_path),
                    "report_file": str(report_path),
                    "scenario_id": report.get("scenario_id", scenario_data.scenario_id),
                    "objective": report.get("mip", {}).get("objective", float("nan")),
                })
            except Exception:
                pass
            continue
        
        try:
            scenario_data = load_scenario_data(scenario_path)
            report = _load_json(report_path)
            
            record = build_hetero_graph_record(scenario_data, report)
            save_graph_record(record, output_path)
            
            stats["success"] += 1
            stats["total_nodes"] += len(record["node_types"])
            stats["total_edges"] += record["edge_index"].shape[1]
            
            index_entries.append({
                "graph_file": str(output_path),  # Use absolute path for compatibility
                "scenario_file": str(scenario_path),
                "report_file": str(report_path),
                "scenario_id": report.get("scenario_id", scenario_data.scenario_id),
                "objective": report.get("mip", {}).get("objective", float("nan")),  # MIP objective for cost_gap metric
                "num_nodes": int(len(record["node_types"])),
                "num_edges": int(record["edge_index"].shape[1]),
            })
            
        except Exception as e:
            stats["failed"] += 1
            print(f"\nFailed to convert {stem}: {e}")
    
    # Write dataset index
    index_path = output_dir / "dataset_index.json"
    index_data = {
        "entries": index_entries,
        "metadata": {
            "total_scenarios": len(index_entries),
            "graph_type": "heterogeneous_multi_level",
        }
    }
    index_path.write_text(json.dumps(index_data, indent=2), encoding="utf-8")
    
    print(f"\n{'='*60}")
    print(f"Conversion complete:")
    print(f"  Success: {stats['success']}")
    print(f"  Failed: {stats['failed']}")
    print(f"  Skipped: {stats['skipped']}")
    if stats['success'] > 0:
        print(f"  Avg nodes/graph: {stats['total_nodes'] / stats['success']:.1f}")
        print(f"  Avg edges/graph: {stats['total_edges'] / stats['success']:.1f}")
    print(f"\nDataset index: {index_path}")
    print(f"{'='*60}")
    
    return stats


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Convert MILP scenarios to heterogeneous multi-level graphs (batch)."
    )
    parser.add_argument(
        "scenario_dir",
        type=Path,
        help="Directory containing scenario JSON files",
    )
    parser.add_argument(
        "reports_dir",
        type=Path,
        help="Directory containing report JSON files (with detail)",
    )
    parser.add_argument(
        "output_dir",
        type=Path,
        help="Output directory for heterogeneous graph NPZ files",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Skip graphs that already exist",
    )
    
    args = parser.parse_args()
    
    convert_batch(
        args.scenario_dir,
        args.reports_dir,
        args.output_dir,
        resume=args.resume,
    )


if __name__ == "__main__":
    main()
