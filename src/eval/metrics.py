"""
Evaluation metrics for pipeline vs MILP comparison.

Computes:
- p90-p99 percentile metrics
- Cost gap (absolute and relative)
- Speedup ratios
- Stage distribution
- Per-family statistics
"""
from __future__ import annotations

import json
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple


def load_milp_reports(reports_dir: Path) -> Dict[str, Dict[str, Any]]:
    """Load MILP reports from a directory."""
    milp_data = {}
    for report_file in sorted(Path(reports_dir).glob('scenario_*.json')):
        sc_id = report_file.stem
        with open(report_file, 'r') as f:
            report = json.load(f)
        milp_data[sc_id] = {
            'milp_objective': report.get('mip', {}).get('objective', float('nan')),
            'milp_solve_time': report.get('mip', {}).get('solve_seconds', float('nan')),
            'milp_status': report.get('mip', {}).get('status', 'unknown'),
            'milp_termination': report.get('mip', {}).get('termination', 'unknown'),
            'cost_components': report.get('cost_components', {}),
        }
    return milp_data


def build_comparison_dataframe(
    pipeline_results: List[Dict[str, Any]],
    milp_data: Dict[str, Dict[str, Any]],
) -> pd.DataFrame:
    """
    Build a comparison DataFrame merging pipeline results with MILP data.
    
    MILP data keys should be 'family/scenario_id' to avoid collisions
    when multiple families share the same scenario IDs.
    Falls back to plain scenario_id lookup for backward compatibility.
    
    Returns DataFrame with columns for both pipeline and MILP metrics.
    """
    rows = []
    for pr in pipeline_results:
        sc_id = pr['scenario_id']
        family = pr.get('family', '')
        # Try family-prefixed key first, fall back to plain sc_id
        milp = milp_data.get(f"{family}/{sc_id}", milp_data.get(sc_id, {}))
        
        row = {
            # Pipeline
            'scenario_id': sc_id,
            'family': pr.get('family', ''),
            'pipeline_objective': pr.get('lp_objective', float('nan')),
            'pipeline_solve_time': pr.get('time_total', float('nan')),
            'pipeline_status': pr.get('lp_status', ''),
            'pipeline_stage': pr.get('lp_stage_used', ''),
            'pipeline_slack': pr.get('lp_slack', 0.0),
            'pipeline_n_flips': pr.get('lp_n_flips', 0),
            'n_zones': pr.get('n_zones', 0),
            'n_timesteps': pr.get('n_timesteps', 24),
            'criticality_index': pr.get('criticality_index', 0.0),
            'n_samples': pr.get('n_samples', 0),
            'best_sample_idx': pr.get('best_sample_idx', 0),
            'success': pr.get('success', True),
            # Timing breakdown
            'time_graph_build': pr.get('time_graph_build', 0.0),
            'time_embedding': pr.get('time_embedding', 0.0),
            'time_ebm_sampling': pr.get('time_ebm_sampling', 0.0),
            'time_decoder': pr.get('time_decoder', 0.0),
            'time_lp_solve': pr.get('time_lp_solve', 0.0),
            # MILP
            'milp_objective': milp.get('milp_objective', float('nan')),
            'milp_solve_time': milp.get('milp_solve_time', float('nan')),
            'milp_status': milp.get('milp_status', 'unknown'),
            'milp_termination': milp.get('milp_termination', 'unknown'),
        }
        rows.append(row)
    
    df = pd.DataFrame(rows)
    
    # Derived metrics
    df['speedup'] = df['milp_solve_time'] / df['pipeline_solve_time']
    df['cost_gap_pct'] = (
        (df['pipeline_objective'] - df['milp_objective']) / df['milp_objective'].abs() * 100
    )
    df['cost_gap_abs'] = (df['pipeline_objective'] - df['milp_objective'])
    
    return df


def compute_percentile_metrics(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Compute p90-p99 percentile metrics as required by eval spec.
    
    Returns dict with percentile stats for key metrics.
    """
    metrics = {}
    
    for col, label in [
        ('cost_gap_pct', 'cost_gap_pct'),
        ('speedup', 'speedup'),
        ('pipeline_solve_time', 'pipeline_time_s'),
        ('pipeline_slack', 'slack_mwh'),
    ]:
        if col not in df.columns:
            continue
        valid = df[col].dropna()
        if len(valid) == 0:
            continue
        
        metrics[label] = {
            'mean': float(valid.mean()),
            'std': float(valid.std()),
            'median': float(valid.median()),
            'min': float(valid.min()),
            'max': float(valid.max()),
            'p10': float(np.percentile(valid, 10)),
            'p25': float(np.percentile(valid, 25)),
            'p75': float(np.percentile(valid, 75)),
            'p90': float(np.percentile(valid, 90)),
            'p95': float(np.percentile(valid, 95)),
            'p99': float(np.percentile(valid, 99)),
        }
    
    return metrics


def compute_stage_distribution(df: pd.DataFrame) -> Dict[str, Any]:
    """Compute stage distribution statistics."""
    stage_counts = df['pipeline_stage'].value_counts().to_dict()
    total = len(df)
    stage_pct = {k: round(v / total * 100, 1) for k, v in stage_counts.items()}
    return {
        'counts': stage_counts,
        'percentages': stage_pct,
        'total': total,
    }


def compute_eval_metrics(
    pipeline_results: List[Dict[str, Any]],
    milp_reports_dirs: Dict[str, Path],
    families: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """
    Compute comprehensive evaluation metrics across all families.
    
    Args:
        pipeline_results: List of pipeline result dicts
        milp_reports_dirs: Dict mapping family name → reports directory path
        families: Optional list of family names to include
        
    Returns:
        Dict with global and per-family metrics
    """
    # Load all MILP data with family-prefixed keys to avoid collisions
    # (multiple families can share the same scenario IDs)
    all_milp = {}
    for family_name, reports_dir in milp_reports_dirs.items():
        milp = load_milp_reports(reports_dir)
        for sc_id, data in milp.items():
            all_milp[f"{family_name}/{sc_id}"] = data
    
    # Build comparison DataFrame
    df = build_comparison_dataframe(pipeline_results, all_milp)
    
    # Global metrics
    output = {
        'global': {
            'n_scenarios': len(df),
            'n_success': int(df['success'].sum()),
            'percentiles': compute_percentile_metrics(df),
            'stage_distribution': compute_stage_distribution(df),
        },
    }
    
    # Per-family metrics
    family_names = families or df['family'].unique().tolist()
    per_family = {}
    for fam in family_names:
        if not fam:
            continue
        df_fam = df[df['family'] == fam]
        if len(df_fam) == 0:
            continue
        per_family[fam] = {
            'n_scenarios': len(df_fam),
            'n_success': int(df_fam['success'].sum()),
            'criticality_stats': {
                'mean': float(df_fam['criticality_index'].mean()),
                'min': float(df_fam['criticality_index'].min()),
                'max': float(df_fam['criticality_index'].max()),
            },
            'percentiles': compute_percentile_metrics(df_fam),
            'stage_distribution': compute_stage_distribution(df_fam),
        }
    
    output['per_family'] = per_family
    output['dataframe'] = df
    
    return output


def format_metrics_table(metrics: Dict[str, Any], family_name: str = 'Global') -> str:
    """Format metrics as a readable table string."""
    lines = [f"\n{'='*70}", f"  {family_name} Evaluation Metrics", f"{'='*70}"]
    
    pct = metrics.get('percentiles', {})
    for metric_name, stats in pct.items():
        lines.append(f"\n  {metric_name}:")
        lines.append(f"    Mean: {stats['mean']:.2f} (std: {stats['std']:.2f})")
        lines.append(f"    Median: {stats['median']:.2f}")
        lines.append(f"    [P10, P90]: [{stats['p10']:.2f}, {stats['p90']:.2f}]")
        lines.append(f"    [P95, P99]: [{stats['p95']:.2f}, {stats['p99']:.2f}]")
    
    stage = metrics.get('stage_distribution', {})
    if stage:
        lines.append(f"\n  Stage Distribution:")
        for s, pct_val in stage.get('percentages', {}).items():
            lines.append(f"    {s}: {pct_val}%")
    
    lines.append(f"{'='*70}")
    return '\n'.join(lines)
