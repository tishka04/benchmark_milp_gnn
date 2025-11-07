"""
Inspection utility for heterogeneous multi-level graphs.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict

import numpy as np


NODE_TYPE_NAMES = ["Nation", "Region", "Zone", "Asset", "Weather"]
EDGE_TYPE_NAMES = [
    "Nation→Region",
    "Region→Zone", 
    "Zone→Asset",
    "Weather→Zone",
    "Weather→Asset",
    "Transmission",
    "Temporal",
]


def load_hetero_graph(path: Path) -> Dict[str, np.ndarray]:
    """Load heterogeneous graph from NPZ."""
    data = np.load(path, allow_pickle=True)
    return {key: data[key] for key in data.files}


def print_graph_summary(record: Dict[str, np.ndarray], verbose: bool = False):
    """Print summary statistics of heterogeneous graph."""
    num_nodes = len(record["node_types"])
    num_edges = record["edge_index"].shape[1]
    
    print(f"\n{'='*70}")
    print(f"HETEROGENEOUS GRAPH SUMMARY")
    print(f"{'='*70}\n")
    
    # Node statistics
    print("NODE STATISTICS:")
    print(f"  Total nodes: {num_nodes}")
    print(f"\n  Node type distribution:")
    node_types, node_counts = np.unique(record["node_types"], return_counts=True)
    for node_type, count in zip(node_types, node_counts):
        type_name = NODE_TYPE_NAMES[node_type] if node_type < len(NODE_TYPE_NAMES) else f"Type{node_type}"
        pct = 100 * count / num_nodes
        print(f"    {type_name:12s}: {count:4d} ({pct:5.1f}%)")
    
    # Edge statistics
    print(f"\n  Edge statistics:")
    print(f"    Total edges: {num_edges}")
    print(f"\n    Edge type distribution:")
    edge_types, edge_counts = np.unique(record["edge_types"], return_counts=True)
    for edge_type, count in zip(edge_types, edge_counts):
        type_name = EDGE_TYPE_NAMES[edge_type] if edge_type < len(EDGE_TYPE_NAMES) else f"Type{edge_type}"
        pct = 100 * count / num_edges
        print(f"      {type_name:20s}: {count:4d} ({pct:5.1f}%)")
    
    # Feature dimensions
    if "node_features" in record:
        node_feat_dim = record["node_features"].shape[1]
        print(f"\n  Node feature dim: {node_feat_dim}")
    
    if "edge_features" in record:
        edge_feat_dim = record["edge_features"].shape[1]
        print(f"  Edge feature dim: {edge_feat_dim}")
    
    # Labels
    if "zone_labels" in record:
        zone_labels = record["zone_labels"]
        print(f"\n  Zone labels shape: {zone_labels.shape} (T={zone_labels.shape[0]}, Z={zone_labels.shape[1]}, D={zone_labels.shape[2]})")
    
    # Connectivity analysis
    print(f"\n{'='*70}")
    print("CONNECTIVITY ANALYSIS:")
    print(f"{'='*70}\n")
    
    edge_index = record["edge_index"]
    edge_types_arr = record["edge_types"]
    node_types_arr = record["node_types"]
    
    # Degree distribution per node type
    print("  Out-degree by node type:")
    for node_type in sorted(set(node_types_arr)):
        mask = node_types_arr == node_type
        node_indices = np.where(mask)[0]
        out_degrees = [np.sum(edge_index[0] == idx) for idx in node_indices]
        if out_degrees:
            type_name = NODE_TYPE_NAMES[node_type] if node_type < len(NODE_TYPE_NAMES) else f"Type{node_type}"
            avg_deg = np.mean(out_degrees)
            max_deg = np.max(out_degrees)
            print(f"    {type_name:12s}: avg={avg_deg:6.2f}, max={max_deg:4d}")
    
    print("\n  In-degree by node type:")
    for node_type in sorted(set(node_types_arr)):
        mask = node_types_arr == node_type
        node_indices = np.where(mask)[0]
        in_degrees = [np.sum(edge_index[1] == idx) for idx in node_indices]
        if in_degrees:
            type_name = NODE_TYPE_NAMES[node_type] if node_type < len(NODE_TYPE_NAMES) else f"Type{node_type}"
            avg_deg = np.mean(in_degrees)
            max_deg = np.max(in_degrees)
            print(f"    {type_name:12s}: avg={avg_deg:6.2f}, max={max_deg:4d}")
    
    # Edge type connectivity matrix
    if verbose:
        print("\n  Edge connectivity (source_type → target_type by edge_type):")
        for edge_type in sorted(set(edge_types_arr)):
            edge_mask = edge_types_arr == edge_type
            sources = edge_index[0, edge_mask]
            targets = edge_index[1, edge_mask]
            src_types = node_types_arr[sources]
            tgt_types = node_types_arr[targets]
            
            type_name = EDGE_TYPE_NAMES[edge_type] if edge_type < len(EDGE_TYPE_NAMES) else f"Type{edge_type}"
            print(f"\n    {type_name}:")
            
            type_pairs = {}
            for src_t, tgt_t in zip(src_types, tgt_types):
                key = (int(src_t), int(tgt_t))
                type_pairs[key] = type_pairs.get(key, 0) + 1
            
            for (src_t, tgt_t), count in sorted(type_pairs.items()):
                src_name = NODE_TYPE_NAMES[src_t] if src_t < len(NODE_TYPE_NAMES) else f"T{src_t}"
                tgt_name = NODE_TYPE_NAMES[tgt_t] if tgt_t < len(NODE_TYPE_NAMES) else f"T{tgt_t}"
                print(f"      {src_name:12s} → {tgt_name:12s}: {count:4d}")
    
    print(f"\n{'='*70}\n")


def main():
    parser = argparse.ArgumentParser(description="Inspect heterogeneous graph structure")
    parser.add_argument("graph_path", type=Path, help="Path to heterogeneous graph NPZ")
    parser.add_argument("-v", "--verbose", action="store_true", help="Verbose output")
    
    args = parser.parse_args()
    
    if not args.graph_path.exists():
        print(f"Error: File not found: {args.graph_path}")
        return
    
    record = load_hetero_graph(args.graph_path)
    print_graph_summary(record, verbose=args.verbose)


if __name__ == "__main__":
    main()
