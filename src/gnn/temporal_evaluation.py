"""
Evaluation utilities for temporal heterogeneous GNN models.

Provides MILP comparison metrics similar to the static graph evaluation.
"""

import json
import time
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

from src.gnn.models.temporal_hetero_gnn import TemporalRGCN, SeparatedTemporalGNN


# Cost weights for dispatch variables ($/MWh)
# Supports both full names and simplified names
COST_WEIGHTS = {
    # Full names
    'thermal': 50.0,
    'nuclear': 10.0,
    'solar': 0.0,
    'wind': 0.0,
    'hydro_release': 5.0,
    'hydro_ror': 0.0,
    'hydro': 5.0,  # Alias for hydro_release
    'dr': 100.0,
    'battery_charge': -2.0,
    'battery_discharge': 2.0,
    'battery': 0.0,  # Average of charge/discharge
    'pumped_charge': -2.0,
    'pumped_discharge': 2.0,
    'pumped': 0.0,  # Average of charge/discharge
    'net_import': 60.0,
    'imports': 60.0,  # Alias
    'exports': -60.0,  # Negative of import
    'unserved': 10000.0,
    'overgen': 1000.0,  # Penalty for overgeneration
}

# Standard variable name order (from graph labels)
VAR_NAMES = [
    'thermal', 'nuclear', 'solar', 'wind',
    'hydro_release', 'hydro_ror', 'dr',
    'battery_charge', 'battery_discharge',
    'pumped_charge', 'pumped_discharge',
    'net_import', 'unserved'
]

# Alternative/simplified names that may be used in training
VAR_NAME_ALIASES = {
    'hydro': 'hydro_release',
    'battery': 'battery_discharge',  # Approximate as discharge
    'pumped': 'pumped_discharge',
    'imports': 'net_import',
    'exports': 'net_import',  # Exports are negative imports
}


def _resolve_path(path: Path, base_dir: Path) -> Path:
    """Resolve a potentially relative path against a base directory."""
    if path.is_absolute():
        return path
    return (base_dir / path).resolve()


def load_model_and_config(run_dir: Path, device: str = 'cpu'):
    """Load trained temporal GNN model and configuration."""
    checkpoint_path = run_dir / 'best_model.pt'
    
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"No checkpoint at {checkpoint_path}")
    
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    args = checkpoint['args']
    
    # Load a sample graph to get dimensions
    data_dir = Path(args['data_dir'])
    
    # Resolve relative paths relative to run_dir parent (project root)
    if not data_dir.is_absolute():
        # Assume run_dir is outputs/gnn_runs/temporal_hetero, so go up 3 levels to project root
        # outputs/gnn_runs/temporal_hetero -> outputs/gnn_runs -> outputs -> benchmark
        project_root = run_dir.parent.parent.parent
        data_dir = (project_root / data_dir).resolve()
    
    index_path = data_dir / 'dataset_index.json'
    
    if not index_path.exists():
        raise FileNotFoundError(
            f"Dataset index not found at {index_path}\n"
            f"Expected data_dir: {data_dir}\n"
            f"Run directory: {run_dir}\n"
            f"Make sure temporal graphs were generated with:\n"
            f"  python -m src.gnn.build_hetero_graph_dataset ... --temporal-mode supra"
        )
    
    with open(index_path) as f:
        entries = json.load(f)['entries']
    
    # Resolve sample graph path
    sample_path = _resolve_path(Path(entries[0]['graph_file']), project_root)
    sample = np.load(sample_path, allow_pickle=True)
    node_feature_dim = sample['node_features'].shape[1]
    
    target_vars = args['target_vars'].split(',')
    
    # Get output_dim from checkpoint to match saved model
    # (Don't recompute from target_vars as they may have changed)
    if 'output_dim' in args:
        output_dim = args['output_dim']
    else:
        # Infer from model state dict
        for key in checkpoint['model_state_dict'].keys():
            if 'output_head' in key and 'weight' in key:
                output_dim = checkpoint['model_state_dict'][key].shape[0]
                break
        else:
            # Fallback to computing from target_vars
            output_dim = len(target_vars)
    
    # Adjust target_vars to match actual model output dimension
    if len(target_vars) != output_dim:
        print(f"⚠️  Warning: target_vars has {len(target_vars)} variables but model outputs {output_dim}")
        print(f"   Using first {output_dim} variables from checkpoint")
        target_vars = target_vars[:output_dim]
    
    # Create model with same architecture as checkpoint
    if args['model_type'] == 'rgcn':
        model = TemporalRGCN(
            node_feature_dim=node_feature_dim,
            hidden_dim=args['hidden_dim'],
            output_dim=output_dim,
            num_edge_types=10,
            num_layers=args['num_layers'],
            dropout=args.get('dropout', 0.1),
        )
    elif args['model_type'] == 'separated':
        model = SeparatedTemporalGNN(
            node_feature_dim=node_feature_dim,
            hidden_dim=args['hidden_dim'],
            output_dim=output_dim,
            num_spatial_edge_types=7,
            num_temporal_edge_types=3,
            num_layers=args['num_layers'],
            dropout=args.get('dropout', 0.1),
        )
    else:
        raise ValueError(f"Unknown model type: {args['model_type']}")
    
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()
    
    return model, args, target_vars


def compute_dispatch_cost(dispatch: np.ndarray, var_names: List[str], timestep_hours: float = 1.0) -> float:
    """Compute total dispatch cost using predefined weights."""
    T, N, n_vars = dispatch.shape
    cost = 0.0
    
    for i, var in enumerate(var_names):
        if i < n_vars:
            weight = COST_WEIGHTS.get(var, 0.0)
            cost += np.sum(dispatch[:, :, i] * weight * timestep_hours)
    
    return cost


def evaluate_scenario(
    model: torch.nn.Module,
    graph_path: Path,
    report_path: Path,
    target_vars: List[str],
    device: str = 'cpu'
) -> Dict:
    """Evaluate model on a single scenario."""
    # Load graph
    data = np.load(graph_path, allow_pickle=True)
    
    x = torch.from_numpy(data['node_features']).float().to(device)
    edge_index = torch.from_numpy(data['edge_index']).long().to(device)
    edge_type = torch.from_numpy(data['edge_types']).long().to(device)
    node_types = torch.from_numpy(data['node_types']).long().to(device)
    
    # Ground truth
    target_indices = [VAR_NAMES.index(v) for v in target_vars if v in VAR_NAMES]
    y_true_full = data['node_labels']  # [T, N_zones, all_vars]
    y_true = y_true_full[:, :, target_indices]
    
    # Metadata
    meta = data['meta'].item()
    N_base = meta['N_base']
    T = meta['T']
    
    # Inference
    start_time = time.time()
    with torch.no_grad():
        pred = model(x, edge_index, edge_type)
    inference_time = time.time() - start_time
    
    # Extract zone predictions
    base_node_types = node_types[:N_base]
    zone_mask = base_node_types == 2
    
    y_pred = []
    for t in range(T):
        zone_indices = torch.where(zone_mask)[0] + t * N_base
        y_pred.append(pred[zone_indices].cpu().numpy())
    
    y_pred = np.stack(y_pred, axis=0)  # [T, N_zones, output_dim]
    
    # Compute GNN cost
    gnn_cost = compute_dispatch_cost(y_pred, target_vars)
    
    # Load MILP report
    with open(report_path) as f:
        report = json.load(f)
    
    milp_objective = report['mip']['objective']
    milp_runtime = report['mip']['solve_seconds']
    
    # Compute metrics
    cost_gap = (gnn_cost - milp_objective) / milp_objective if milp_objective > 0 else 0.0
    dispatch_mae = np.mean(np.abs(y_pred - y_true))
    
    # Violation rate (check unserved energy if available)
    if 'unserved' in target_vars:
        unserved_idx = target_vars.index('unserved')
        violation_rate = np.mean(y_pred[:, :, unserved_idx] > 1e-3)
    else:
        violation_rate = 0.0
    
    speedup = milp_runtime / inference_time if inference_time > 0 else 0.0
    
    return {
        'milp_objective': milp_objective,
        'gnn_cost': gnn_cost,
        'cost_gap': cost_gap,
        'dispatch_mae': dispatch_mae,
        'violation_rate': violation_rate,
        'milp_runtime': milp_runtime,
        'gnn_inference_time': inference_time,
        'speedup': speedup,
    }


def evaluate_test_set(
    run_dir: Path,
    test_split: float = 0.8,
    device: str = 'cpu',
    verbose: bool = True
) -> Tuple[pd.DataFrame, Dict]:
    """Evaluate model on test set and return results dataframe."""
    # Load model
    model, args, target_vars = load_model_and_config(run_dir, device)
    
    # Get project root for resolving relative paths
    # Go up 3 levels: temporal_hetero -> gnn_runs -> outputs -> benchmark
    project_root = run_dir.parent.parent.parent
    
    # Load dataset index (resolve relative paths)
    data_dir = Path(args['data_dir'])
    if not data_dir.is_absolute():
        data_dir = (project_root / data_dir).resolve()
    
    index_path = data_dir / 'dataset_index.json'
    
    with open(index_path) as f:
        entries = json.load(f)['entries']
    
    # Split into test set
    split_idx = int(len(entries) * test_split)
    test_entries = entries[split_idx:]
    
    if verbose:
        print(f"Evaluating {len(test_entries)} test scenarios...")
    
    # Evaluate each scenario
    results = []
    iterator = tqdm(test_entries, desc="Evaluating") if verbose else test_entries
    
    for entry in iterator:
        try:
            # Resolve paths (may be relative to project root)
            graph_path = _resolve_path(Path(entry['graph_file']), project_root)
            report_path = _resolve_path(Path(entry['report_file']), project_root)
            
            metrics = evaluate_scenario(
                model,
                graph_path,
                report_path,
                target_vars,
                device
            )
            metrics['scenario_id'] = entry['scenario_id']
            results.append(metrics)
        except Exception as e:
            if verbose:
                print(f"\nFailed {entry['scenario_id'][:16]}: {e}")
            continue
    
    # Convert to DataFrame
    results_df = pd.DataFrame(results)
    
    # Compute summary
    summary = {
        'num_scenarios': len(results_df),
        'mean_cost_gap': results_df['cost_gap'].mean(),
        'median_cost_gap': results_df['cost_gap'].median(),
        'mean_dispatch_mae': results_df['dispatch_mae'].mean(),
        'mean_violation_rate': results_df['violation_rate'].mean(),
        'mean_speedup': results_df['speedup'].mean(),
        'total_inference_time': results_df['gnn_inference_time'].sum(),
        'mean_milp_runtime': results_df['milp_runtime'].mean(),
    }
    
    return results_df, summary


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description="Evaluate temporal GNN")
    parser.add_argument('--run-dir', type=Path, default=Path('outputs/gnn_runs/temporal_hetero'))
    parser.add_argument('--test-split', type=float, default=0.8)
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--output', type=Path, default=None, help="Save results to CSV")
    
    args = parser.parse_args()
    
    results_df, summary = evaluate_test_set(args.run_dir, args.test_split, args.device)
    
    print("\n" + "="*60)
    print("Evaluation Summary")
    print("="*60)
    for key, value in summary.items():
        if 'rate' in key or 'gap' in key:
            print(f"{key:30s}: {value:8.2%}")
        elif 'time' in key:
            print(f"{key:30s}: {value:8.2f}s")
        elif 'speedup' in key:
            print(f"{key:30s}: {value:8.1f}x")
        else:
            print(f"{key:30s}: {value}")
    
    if args.output:
        results_df.to_csv(args.output, index=False)
        print(f"\n✓ Results saved to {args.output}")
