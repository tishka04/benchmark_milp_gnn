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

from src.gnn.hetero_graph_dataset import (
    build_hetero_graph_record,
    build_hetero_temporal_record,
    save_graph_record,
)
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
    temporal: bool = False,
    temporal_mode: str = "supra",
    window: int = None,
    stride: int = 1,
    temporal_edges: str = "soc,ramp,dr",
    time_enc: str = "sinusoidal",
    target_horizon: int = 0,
    use_solution_features: bool = True,
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
            
            # Build temporal or static graph
            if temporal:
                record = build_hetero_temporal_record(
                    scenario_data,
                    report,
                    mode=temporal_mode,
                    time_window=window,
                    stride=stride,
                    temporal_edges=tuple(temporal_edges.split(",")),
                    time_encoding=time_enc,
                    target_horizon=target_horizon,
                    use_solution_features=use_solution_features,
                )
            else:
                record = build_hetero_graph_record(scenario_data, report, use_solution_features)
            
            # Handle sequence mode (list of graphs)
            if temporal and temporal_mode == "sequence":
                # Save each snapshot separately
                for i, graph in enumerate(record):
                    snapshot_path = output_dir / f"{stem}_t{i:04d}.npz"
                    save_graph_record(graph, snapshot_path)
                    
                    index_entries.append({
                        "graph_file": str(snapshot_path),
                        "scenario_file": str(scenario_path),
                        "report_file": str(report_path),
                        "scenario_id": report.get("scenario_id", scenario_data.scenario_id),
                        "objective": report.get("mip", {}).get("objective", float("nan")),
                        "num_nodes": int(len(graph["node_types"])),
                        "num_edges": int(graph["edge_index"].shape[1]),
                        "mode": "sequence",
                        "time_step": i,
                    })
                
                stats["success"] += 1
                stats["total_nodes"] += sum(len(g["node_types"]) for g in record)
                stats["total_edges"] += sum(g["edge_index"].shape[1] for g in record)
            else:
                # Save single graph (static or supra)
                save_graph_record(record, output_path)
                
                stats["success"] += 1
                stats["total_nodes"] += len(record["node_types"])
                stats["total_edges"] += record["edge_index"].shape[1]
                
                index_entries.append({
                    "graph_file": str(output_path),
                    "scenario_file": str(scenario_path),
                    "report_file": str(report_path),
                    "scenario_id": report.get("scenario_id", scenario_data.scenario_id),
                    "objective": report.get("mip", {}).get("objective", float("nan")),
                    "num_nodes": int(len(record["node_types"])),
                    "num_edges": int(record["edge_index"].shape[1]),
                    "mode": temporal_mode if temporal else "static",
                    "T": record.get("meta", {}).get("T", 1) if temporal else 1,
                    "N_base": record.get("meta", {}).get("N_base") if temporal else None,
                    "temporal_edges": record.get("meta", {}).get("temporal_edges", []) if temporal else [],
                })
            
        except Exception as e:
            stats["failed"] += 1
            print(f"\nFailed to convert {stem}: {e}")
    
    # Write dataset index
    index_path = output_dir / "dataset_index.json"
    graph_type = "heterogeneous_multi_level"
    if temporal:
        graph_type = f"heterogeneous_multi_level_temporal_{temporal_mode}"
    
    index_data = {
        "entries": index_entries,
        "metadata": {
            "total_scenarios": len(index_entries),
            "graph_type": graph_type,
            "temporal": temporal,
            "temporal_mode": temporal_mode if temporal else None,
            "temporal_edges": temporal_edges if temporal else None,
            "time_encoding": time_enc if temporal else None,
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
    
    # Temporal graph options
    parser.add_argument(
        "--temporal",
        action="store_true",
        help="Build temporal graphs instead of static snapshots",
    )
    parser.add_argument(
        "--temporal-mode",
        choices=["sequence", "supra"],
        default="supra",
        help="Temporal graph mode: 'sequence' (list of snapshots) or 'supra' (time-expanded graph)",
    )
    parser.add_argument(
        "--window",
        type=int,
        default=None,
        help="Sequence window size in time steps (for sequence mode)",
    )
    parser.add_argument(
        "--stride",
        type=int,
        default=1,
        help="Window stride for sliding window (for sequence mode)",
    )
    parser.add_argument(
        "--temporal-edges",
        type=str,
        default="soc,ramp,dr",
        help="Comma-separated list of temporal edge types: soc, ramp, dr",
    )
    parser.add_argument(
        "--time-enc",
        choices=["sinusoidal", "cyclic-hod"],
        default="sinusoidal",
        help="Time encoding method: 'sinusoidal' (positional) or 'cyclic-hod' (hour-of-day)",
    )
    parser.add_argument(
        "--target-horizon",
        type=int,
        default=0,
        help="Prediction horizon (0 = same-step labels, >0 = predict future steps)",
    )
    parser.add_argument(
        "--input-only",
        action="store_true",
        help="Use ONLY input features (demand, renewable forecasts). Excludes MILP solution data to prevent leakage. Use this for clean generalization experiments.",
    )
    
    args = parser.parse_args()
    
    convert_batch(
        args.scenario_dir,
        args.reports_dir,
        args.output_dir,
        resume=args.resume,
        temporal=args.temporal,
        temporal_mode=args.temporal_mode,
        window=args.window,
        stride=args.stride,
        temporal_edges=args.temporal_edges,
        time_enc=args.time_enc,
        target_horizon=args.target_horizon,
        use_solution_features=not args.input_only,  # Invert: --input-only means no solution features
    )


if __name__ == "__main__":
    main()
