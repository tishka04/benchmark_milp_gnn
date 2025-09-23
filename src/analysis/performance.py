from __future__ import annotations

import json
import math
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import pandas as pd
import torch
import yaml
from torch.utils.data import DataLoader

from src.gnn.data import GraphTemporalDataset, collate_graph_samples
from src.gnn.models import build_model
from src.gnn.training import TrainingConfig, build_decoder

COMPONENT_ORDER = [
    'thermal',
    'nuclear',
    'solar',
    'wind',
    'renewable',
    'hydro_release',
    'demand_response',
    'unserved',
]


def _component_indices(components: Optional[Iterable[str]]) -> List[int]:
    if components is None:
        return list(range(len(COMPONENT_ORDER)))
    mapping = {name: idx for idx, name in enumerate(COMPONENT_ORDER)}
    indices: List[int] = []
    for name in components:
        if name not in mapping:
            raise ValueError(f"Unknown component '{name}'. Known: {COMPONENT_ORDER}")
        indices.append(mapping[name])
    return indices


def _build_weight_vector(weights: Optional[Dict[str, float]]) -> torch.Tensor:
    if weights is None:
        weights = {}
    values = [float(weights.get(name, 0.0)) for name in COMPONENT_ORDER]
    return torch.tensor(values, dtype=torch.float32)


def _load_training_config(resolved_path: Path) -> TrainingConfig:
    raw = yaml.safe_load(resolved_path.read_text(encoding='utf-8'))
    if not isinstance(raw, dict):
        raise RuntimeError(f'Training config at {resolved_path} is not a mapping')
    return TrainingConfig.from_dict(raw, resolved_path.parent)


def _resolve_checkpoint(run_dir: Path, checkpoint: str) -> Path:
    preferred = run_dir / checkpoint
    if preferred.exists():
        return preferred
    fallback = run_dir / 'final_model.pt'
    if fallback.exists():
        return fallback
    raise FileNotFoundError(f'No checkpoint found at {preferred} or {fallback}')


def _ensure_path(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else (Path.cwd() / path).resolve()


def _load_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding='utf-8'))


def build_milp_summary(index_path: Path | str, split: str = 'test') -> pd.DataFrame:
    index_path = _ensure_path(index_path)
    index = _load_json(index_path)
    entries = index.get('splits', {}).get(split)
    if not entries:
        raise KeyError(f"Split '{split}' not found in dataset index {index_path}")

    records: List[Dict[str, Any]] = []
    for entry in entries:
        scenario_path = _ensure_path(entry['scenario_file'])
        report_path = _ensure_path(entry['report_file'])
        scenario = _load_json(scenario_path)
        report = _load_json(report_path)

        estimates = scenario.get('estimates', {}) or {}
        meta = scenario.get('meta', {}) or {}
        graph = scenario.get('graph', {}) or {}

        scenario_id = scenario.get('id') or entry.get('scenario_id') or Path(entry['graph_file']).stem
        est_cpu_hours = estimates.get('est_cpu_hours')
        horizon_hours = scenario.get('horizon_hours')
        dt_minutes = scenario.get('dt_minutes')
        regions = meta.get('regions') or graph.get('regions')
        zones = meta.get('zones') or sum(graph.get('zones_per_region', []) or []) or None
        sites = meta.get('sites') or sum(graph.get('sites_per_zone', []) or []) or None

        mip_info = report.get('mip', {}) or {}
        lp_info = report.get('lp', {}) or {}
        mip_solve_seconds = mip_info.get('solve_seconds')
        lp_solve_seconds = lp_info.get('solve_seconds')

        records.append({
            'scenario_id': scenario_id,
            'scenario_file': str(scenario_path),
            'report_file': str(report_path),
            'milp_objective': float(mip_info.get('objective', float('nan'))),
            'milp_status': mip_info.get('status'),
            'milp_termination': mip_info.get('termination'),
            'lp_objective': float(lp_info.get('objective', float('nan'))),
            'est_cpu_hours': float(est_cpu_hours) if est_cpu_hours is not None else math.nan,
            'horizon_hours': float(horizon_hours) if horizon_hours is not None else math.nan,
            'dt_minutes': float(dt_minutes) if dt_minutes is not None else math.nan,
            'regions': regions,
            'zones': zones,
            'sites': sites,
            'milp_solve_seconds': float(mip_solve_seconds) if mip_solve_seconds is not None else math.nan,
            'lp_solve_seconds': float(lp_solve_seconds) if lp_solve_seconds is not None else math.nan,
        })

    df = pd.DataFrame.from_records(records)
    if not df.empty:
        if 'est_cpu_hours' in df.columns:
            df['milp_est_cpu_seconds'] = df['est_cpu_hours'] * 3600.0
        if 'milp_solve_seconds' in df.columns:
            df['milp_runtime_seconds'] = df['milp_solve_seconds']
        else:
            df['milp_runtime_seconds'] = math.nan
        if 'milp_runtime_seconds' in df.columns and 'milp_est_cpu_seconds' in df.columns:
            df['milp_runtime_seconds'] = df['milp_runtime_seconds'].where(~df['milp_runtime_seconds'].isna(), df['milp_est_cpu_seconds'])
    return df


def _initial_stat_record(metadata: Dict[str, Any]) -> Dict[str, Any]:
    objective = metadata.get('objective')
    try:
        objective_value = float(objective) if objective is not None else math.nan
    except (TypeError, ValueError):
        objective_value = math.nan
    scenario_id = metadata.get('scenario_id')
    scenario_id_str = str(scenario_id) if scenario_id is not None else 'unknown'
    return {
        'scenario_id': scenario_id_str,
        'split': metadata.get('split'),
        'objective': objective_value,
        'graph_file': metadata.get('graph_file'),
        'cost_pred': 0.0,
        'cost_true': 0.0,
        'dispatch_abs_sum': 0.0,
        'dispatch_sq_sum': 0.0,
        'dispatch_count': 0,
        'demand_abs_sum': 0.0,
        'demand_count': 0,
        'violation_count': 0,
        'node_count': 0,
        'inference_seconds': 0.0,
        'time_steps': set(),
    }



def evaluate_gnn_performance(
    run_dir: Path | str,
    *,
    split: str = 'test',
    checkpoint: str = 'best_model.pt',
    device: str = 'cpu',
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    run_path = Path(run_dir)
    resolved_cfg_path = run_path / 'training_config_resolved.yaml'
    if not resolved_cfg_path.exists():
        raise FileNotFoundError(f'Cannot find resolved training config at {resolved_cfg_path}')
    cfg = _load_training_config(resolved_cfg_path)

    dataset = GraphTemporalDataset(
        cfg.data.index_path,
        split=split,
        include_duals=cfg.data.include_duals,
        preload=cfg.data.preload,
    )
    loader = DataLoader(
        dataset,
        batch_size=cfg.data.batch_size,
        shuffle=False,
        num_workers=cfg.data.num_workers,
        pin_memory=cfg.data.pin_memory,
        collate_fn=collate_graph_samples,
    )

    model = build_model(cfg.model, dataset)
    device_obj = torch.device(device)
    model.to(device_obj)
    checkpoint_path = _resolve_checkpoint(run_path, checkpoint)
    state_dict = torch.load(checkpoint_path, map_location=device_obj)
    model.load_state_dict(state_dict)
    model.eval()

    decoder = build_decoder(cfg.decoder)
    component_indices = _component_indices(cfg.metrics.dispatch_error.get('components'))
    weight_vec = _build_weight_vector(cfg.metrics.cost_gap.get('weights'))
    tolerance = float(cfg.metrics.constraint_violation_rate.get('tolerance', 1e-3))

    scenario_stats: Dict[str, Dict[str, Any]] = {}
    total_forward_seconds = 0.0

    with torch.no_grad():
        for batch in loader:
            batch_gpu = batch.to(device_obj)
            if device_obj.type == 'cuda':
                torch.cuda.synchronize(device_obj)
            start = time.perf_counter()
            predictions = model(batch_gpu)
            decoded = decoder(batch_gpu, predictions)
            if device_obj.type == 'cuda':
                torch.cuda.synchronize(device_obj)
            elapsed = time.perf_counter() - start
            total_forward_seconds += elapsed
            per_sample_time = elapsed / max(1, batch_gpu.batch_size)

            batch_cpu = batch_gpu.to(torch.device('cpu'))
            decoded_cpu = decoded.detach().cpu()
            target_cpu = batch_cpu.target
            demand_cpu = batch_cpu.node_time[:, 0]
            node_batch = batch_cpu.node_batch

            for local_idx, metadata in enumerate(batch_cpu.metadata):
                scenario_id = str(metadata.get('scenario_id'))
                stats = scenario_stats.setdefault(scenario_id, _initial_stat_record(metadata))
                stats['inference_seconds'] += per_sample_time
                if 'time_index' in metadata:
                    stats['time_steps'].add(int(metadata['time_index']))

                mask = node_batch == local_idx
                if not mask.any():
                    continue

                pred_graph = decoded_cpu[mask]
                target_graph = target_cpu[mask]
                demand_graph = demand_cpu[mask]

                diff = torch.abs(pred_graph[:, component_indices] - target_graph[:, component_indices])
                stats['dispatch_abs_sum'] += float(diff.sum().item())
                stats['dispatch_sq_sum'] += float((diff ** 2).sum().item())
                stats['dispatch_count'] += int(diff.numel())
                stats['demand_abs_sum'] += float(demand_graph.abs().sum().item())
                stats['demand_count'] += int(demand_graph.numel())

                node_cost_pred = (pred_graph * weight_vec).sum(dim=1)
                node_cost_true = (target_graph * weight_vec).sum(dim=1)
                stats['cost_pred'] += float(node_cost_pred.sum().item())
                stats['cost_true'] += float(node_cost_true.sum().item())

                node_time_graph = batch_cpu.node_time[mask]
                battery_charge = node_time_graph[:, 7]
                battery_discharge = node_time_graph[:, 8]
                pumped_charge = node_time_graph[:, 9]
                pumped_discharge = node_time_graph[:, 10]
                net_import = node_time_graph[:, 15] if node_time_graph.size(1) > 15 else torch.zeros_like(demand_graph)
                net_export = node_time_graph[:, 16] if node_time_graph.size(1) > 16 else torch.zeros_like(demand_graph)

                net_exchange = net_import - net_export

                flows = batch_cpu.edge_attr[:, 1]
                net_flow = torch.zeros_like(batch_cpu.node_time[:, 0])
                if flows.numel():
                    net_flow.index_add_(0, batch_cpu.edge_index[1], flows)
                    net_flow.index_add_(0, batch_cpu.edge_index[0], -flows)
                net_flow_graph = net_flow[mask]

                supply = (
                    pred_graph[:, 0]
                    + pred_graph[:, 1]
                    + pred_graph[:, 4]
                    + pred_graph[:, 5]
                    + pred_graph[:, 6]
                    + battery_discharge
                    + pumped_discharge
                    + net_exchange
                    + net_flow_graph
                )
                demand_side = demand_graph + battery_charge + pumped_charge
                served = supply + pred_graph[:, 7]
                shortage = demand_side - served
                violation_mask = (shortage.abs() > tolerance)
                stats['violation_count'] += int(violation_mask.sum().item())
                stats['node_count'] += int(violation_mask.numel())

    records: List[Dict[str, Any]] = []
    for stats in scenario_stats.values():
        dispatch_count = max(1, stats['dispatch_count'])
        mae = stats['dispatch_abs_sum'] / dispatch_count
        rmse = math.sqrt(stats['dispatch_sq_sum'] / dispatch_count) if stats['dispatch_sq_sum'] else 0.0
        demand_mean = (stats['demand_abs_sum'] / max(1, stats['demand_count'])) if stats['demand_count'] else 0.0
        normalized_mae = mae / (demand_mean + 1e-6)
        objective = stats['objective']
        if math.isnan(objective):
            cost_gap = math.nan
        else:
            denom = max(1.0, abs(objective))
            cost_gap = (stats['cost_pred'] - objective) / denom
        violation_rate = stats['violation_count'] / max(1, stats['node_count'])

        records.append({
            'scenario_id': stats['scenario_id'],
            'split': stats['split'],
            'objective': objective,
            'gnn_pred_cost': stats['cost_pred'],
            'gnn_true_cost': stats['cost_true'],
            'gnn_cost_gap': cost_gap,
            'gnn_dispatch_mae': mae,
            'gnn_dispatch_rmse': rmse,
            'gnn_dispatch_normalized_mae': normalized_mae,
            'gnn_violation_rate': violation_rate,
            'gnn_time_steps': len(stats['time_steps']) if stats['time_steps'] else math.nan,
            'gnn_inference_seconds': stats['inference_seconds'],
            'gnn_node_evaluations': stats['node_count'],
        })

    gnn_df = pd.DataFrame.from_records(records)
    summary: Dict[str, Any] = {
        'run_dir': str(run_path),
        'split': split,
        'checkpoint': str(checkpoint_path),
        'scenarios': int(len(gnn_df)),
        'total_inference_seconds': float(gnn_df['gnn_inference_seconds'].sum()) if not gnn_df.empty else 0.0,
        'mean_dispatch_mae': float(gnn_df['gnn_dispatch_mae'].mean()) if not gnn_df.empty else math.nan,
        'mean_cost_gap': float(gnn_df['gnn_cost_gap'].mean()) if not gnn_df.empty else math.nan,
        'mean_violation_rate': float(gnn_df['gnn_violation_rate'].mean()) if not gnn_df.empty else math.nan,
        'forward_pass_seconds': total_forward_seconds,
    }
    return gnn_df, summary


def build_combined_performance(
    run_dir: Path | str,
    *,
    split: str = 'test',
    checkpoint: str = 'best_model.pt',
    device: str = 'cpu',
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    run_path = Path(run_dir)
    resolved_cfg_path = run_path / 'training_config_resolved.yaml'
    cfg = _load_training_config(resolved_cfg_path)
    milp_df = build_milp_summary(cfg.data.index_path, split=split)
    gnn_df, summary = evaluate_gnn_performance(run_dir, split=split, checkpoint=checkpoint, device=device)

    combined = milp_df.merge(gnn_df, on='scenario_id', how='inner', suffixes=('_milp', '_gnn'))
    if 'milp_runtime_seconds' not in combined.columns:
        combined['milp_runtime_seconds'] = math.nan
    if 'milp_solve_seconds' in combined.columns:
        combined['milp_runtime_seconds'] = combined['milp_runtime_seconds'].where(~combined['milp_runtime_seconds'].isna(), combined['milp_solve_seconds'])
    if 'milp_est_cpu_seconds' not in combined.columns and 'est_cpu_hours' in combined.columns:
        combined['milp_est_cpu_seconds'] = combined['est_cpu_hours'] * 3600.0
    if 'milp_est_cpu_seconds' in combined.columns:
        combined['milp_runtime_seconds'] = combined['milp_runtime_seconds'].where(~combined['milp_runtime_seconds'].isna(), combined['milp_est_cpu_seconds'])
    combined['gnn_speedup_est'] = combined['milp_runtime_seconds'] / combined['gnn_inference_seconds'].clip(lower=1e-6)
    summary.update({
        'scenarios_with_milp': int(len(combined)),
        'mean_milp_runtime_seconds': float(combined['milp_runtime_seconds'].mean()) if not combined.empty else math.nan,
        'mean_speedup_est': float(combined['gnn_speedup_est'].mean()) if not combined.empty else math.nan,
    })
    return combined, summary


def load_training_history(run_dir: Path | str) -> pd.DataFrame:
    run_path = Path(run_dir)
    history_path = run_path / 'training_history.json'
    if not history_path.exists():
        raise FileNotFoundError(f'No training history found at {history_path}')
    history = json.loads(history_path.read_text(encoding='utf-8'))
    return pd.DataFrame(history)


def load_step_losses(run_dir: Path | str) -> pd.DataFrame:
    run_path = Path(run_dir)
    step_path = run_path / 'training_step_losses.json'
    if not step_path.exists():
        raise FileNotFoundError(f'No step loss log found at {step_path}')
    content = json.loads(step_path.read_text(encoding='utf-8'))
    return pd.DataFrame(content)
