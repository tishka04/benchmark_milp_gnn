#!/usr/bin/env python3
"""
Test script to verify temporal heterogeneous graph building
"""

import sys
from pathlib import Path
import numpy as np

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.gnn.hetero_graph_dataset import (
    build_hetero_graph_record,
    build_hetero_temporal_record,
)
from src.milp.scenario_loader import load_scenario_data


def load_json(path):
    import json
    return json.loads(Path(path).read_text(encoding="utf-8"))


def test_temporal_build():
    """Test temporal graph building with a single scenario."""
    
    # Find first scenario and report
    scenario_dir = Path("outputs/scenarios_v1")
    
    scenario_files = list(scenario_dir.glob("scenario_*.json"))
    if not scenario_files:
        print("Error: No scenario files found in outputs/scenarios_v1")
        print("Please run MILP solver first to generate scenarios")
        return False
    
    # Use first scenario
    scenario_path = scenario_files[0]
    report_path = scenario_path  # Reports are in same directory
    
    print("=" * 60)
    print("Testing Temporal Heterogeneous Graph Builder")
    print("=" * 60)
    print(f"\nUsing scenario: {scenario_path.name}")
    print()
    
    # Load data
    try:
        scenario_data = load_scenario_data(scenario_path)
        report = load_json(report_path)
    except Exception as e:
        print(f"Error loading scenario/report: {e}")
        return False
    
    print("Loaded scenario successfully")
    print(f"  Zones: {len(scenario_data.zones)}")
    print(f"  Regions: {len(set(scenario_data.region_of_zone.values()))}")
    
    # Check if report has detail
    detail = report.get("detail")
    if not detail:
        print("\nError: Report missing 'detail' section")
        print("Scenarios must be solved with --save-json to include time-series data")
        return False
    
    time_steps = detail.get("time_steps", [])
    T = len(time_steps)
    print(f"  Time steps: {T}")
    print()
    
    # Test 1: Build static hetero graph
    print("-" * 60)
    print("Test 1: Building static heterogeneous graph")
    print("-" * 60)
    try:
        static_record = build_hetero_graph_record(scenario_data, report)
        N_static = len(static_record["node_types"])
        E_static = static_record["edge_index"].shape[1]
        print(f"✓ Static graph built successfully")
        print(f"  Nodes: {N_static}")
        print(f"  Edges: {E_static}")
        print(f"  Node feature dim: {static_record['node_features'].shape[1]}")
    except Exception as e:
        print(f"✗ Failed to build static graph: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    print()
    
    # Test 2: Build temporal supra-graph
    print("-" * 60)
    print("Test 2: Building temporal supra-graph")
    print("-" * 60)
    try:
        supra_record = build_hetero_temporal_record(
            scenario_data,
            report,
            mode="supra",
            temporal_edges=("soc", "ramp", "dr"),
            time_encoding="sinusoidal",
        )
        
        N_supra = len(supra_record["node_types"])
        E_supra = supra_record["edge_index"].shape[1]
        meta = supra_record["meta"]
        
        print(f"✓ Supra-graph built successfully")
        print(f"  Base nodes (N): {meta['N_base']}")
        print(f"  Time steps (T): {meta['T']}")
        print(f"  Total nodes (N×T): {N_supra}")
        print(f"  Total edges: {E_supra}")
        print(f"  Node feature dim: {supra_record['node_features'].shape[1]}")
        print(f"  Temporal edges: {meta['temporal_edges']}")
        print(f"  Time encoding: {meta['time_encoding']}")
        
        # Verify structure
        expected_nodes = meta['N_base'] * meta['T']
        if N_supra != expected_nodes:
            print(f"✗ Warning: Expected {expected_nodes} nodes, got {N_supra}")
        else:
            print(f"✓ Node count verified (N_base × T = {N_supra})")
        
        # Check edge type distribution
        edge_types = supra_record["edge_types"]
        unique_types, counts = np.unique(edge_types, return_counts=True)
        print(f"\n  Edge type distribution:")
        edge_type_names = {
            0: "Nation→Region",
            1: "Region→Zone",
            2: "Zone→Asset",
            3: "Weather→Zone",
            4: "Weather→Asset",
            5: "Transmission",
            6: "Temporal Storage",
            7: "Temporal SOC",
            8: "Temporal Ramp",
            9: "Temporal DR",
        }
        for etype, count in zip(unique_types, counts):
            name = edge_type_names.get(int(etype), f"Type {etype}")
            print(f"    {name}: {count}")
        
    except Exception as e:
        print(f"✗ Failed to build supra-graph: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    print()
    
    # Test 3: Build temporal sequence
    print("-" * 60)
    print("Test 3: Building temporal sequence (first 5 snapshots)")
    print("-" * 60)
    try:
        sequence_record = build_hetero_temporal_record(
            scenario_data,
            report,
            mode="sequence",
            time_window=5,
            time_encoding="cyclic-hod",
        )
        
        print(f"✓ Sequence built successfully")
        print(f"  Number of snapshots: {len(sequence_record)}")
        
        if sequence_record:
            first_snap = sequence_record[0]
            print(f"  Snapshot structure:")
            print(f"    Nodes: {len(first_snap['node_types'])}")
            print(f"    Edges: {first_snap['edge_index'].shape[1]}")
            print(f"    Features dim: {first_snap['node_features'].shape[1]}")
            print(f"    Time step: {first_snap.get('time_step', 'N/A')}")
            print(f"    Time index: {first_snap.get('time_index', 'N/A')}")
        
    except Exception as e:
        print(f"✗ Failed to build sequence: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    print()
    print("=" * 60)
    print("All tests passed! ✓")
    print("=" * 60)
    print()
    print("Next steps:")
    print("  1. Run full batch conversion:")
    print("     python -m src.gnn.build_hetero_graph_dataset \\")
    print("         outputs/scenarios_v1 \\")
    print("         outputs/scenarios_v1 \\")
    print("         outputs/temporal_graphs/supra \\")
    print("         --temporal --temporal-mode supra")
    print()
    print("  2. Or use the helper script:")
    print("     python scripts/build_temporal_graphs.py --mode supra")
    print()
    
    return True


if __name__ == "__main__":
    success = test_temporal_build()
    sys.exit(0 if success else 1)
