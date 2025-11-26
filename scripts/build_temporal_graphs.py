#!/usr/bin/env python3
"""
Build Temporal Heterogeneous Graphs
Helper script to generate temporal graphs with common configurations
"""

import argparse
import subprocess
import sys
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(
        description="Build temporal heterogeneous graphs with common configurations"
    )
    parser.add_argument(
        "--scenario-dir",
        type=Path,
        default=Path("outputs/scenarios_v1"),
        help="Directory containing scenario JSON files",
    )
    parser.add_argument(
        "--reports-dir",
        type=Path,
        default=Path("outputs/scenarios_v1"),
        help="Directory containing report JSON files",
    )
    parser.add_argument(
        "--mode",
        choices=["supra", "sequence", "both"],
        default="supra",
        help="Graph mode to build",
    )
    parser.add_argument(
        "--temporal-edges",
        type=str,
        default="soc,ramp,dr",
        help="Comma-separated temporal edge types",
    )
    parser.add_argument(
        "--time-enc",
        choices=["sinusoidal", "cyclic-hod"],
        default="sinusoidal",
        help="Time encoding method",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Skip existing graphs",
    )
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("Temporal Heterogeneous Graph Builder")
    print("=" * 60)
    print()
    
    modes = ["supra", "sequence"] if args.mode == "both" else [args.mode]
    
    for mode in modes:
        output_dir = Path(f"outputs/temporal_graphs/{mode}")
        output_dir.mkdir(parents=True, exist_ok=True)
        
        print(f"\n{'=' * 60}")
        print(f"Building {mode.upper()} mode graphs")
        print(f"{'=' * 60}")
        print(f"  Scenario Dir:    {args.scenario_dir}")
        print(f"  Reports Dir:     {args.reports_dir}")
        print(f"  Output Dir:      {output_dir}")
        print(f"  Temporal Edges:  {args.temporal_edges}")
        print(f"  Time Encoding:   {args.time_enc}")
        print()
        
        # Build command
        cmd = [
            sys.executable,
            "-m",
            "src.gnn.build_hetero_graph_dataset",
            str(args.scenario_dir),
            str(args.reports_dir),
            str(output_dir),
            "--temporal",
            "--temporal-mode",
            mode,
            "--temporal-edges",
            args.temporal_edges,
            "--time-enc",
            args.time_enc,
        ]
        
        if args.resume:
            cmd.append("--resume")
        
        print("Command:", " ".join(cmd))
        print()
        
        # Execute
        result = subprocess.run(cmd)
        
        if result.returncode != 0:
            print(f"\nError: Failed to build {mode} mode graphs")
            return 1
        
        print(f"\n✓ {mode.upper()} mode graphs complete!")
        print(f"  Output: {output_dir}")
        print(f"  Index:  {output_dir}/dataset_index.json")
    
    print("\n" + "=" * 60)
    print("All temporal graphs generated successfully!")
    print("=" * 60)
    return 0


if __name__ == "__main__":
    sys.exit(main())
