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


def _coerce_float_list(values: Any) -> List[float]:
    """Convert serialized timing/objective lists to floats with NaN fallback."""
    if values is None:
        return []
    if isinstance(values, float) and np.isnan(values):
        return []
    out: List[float] = []
    for value in values:
        try:
            out.append(float(value))
        except (TypeError, ValueError):
            out.append(float('nan'))
    return out


def _safe_ratio(num: Any, den: Any) -> float:
    """Return num/den with NaN for non-finite or zero denominators."""
    try:
        num_f = float(num)
        den_f = float(den)
    except (TypeError, ValueError):
        return float('nan')
    if not np.isfinite(num_f) or not np.isfinite(den_f) or den_f <= 0.0:
        return float('nan')
    return num_f / den_f


def _pipeline_timing_views(pr: Dict[str, Any]) -> Dict[str, float]:
    """
    Derive multiple pipeline timing views for fair comparisons.

    - `actual_wall_clock`: observed end-to-end latency after evaluating all K candidates.
    - `first_candidate`: latency if we stopped at K=1.
    - `best_candidate_oracle`: optimistic latency if an oracle revealed the best
      candidate in advance.
    - `parallel_ideal`: ideal latency if all candidates were post-processed in
      parallel and we waited for the slowest one.

    The headline benchmark should use `actual_wall_clock`, because it matches the
    real computation spent to return the selected best-of-K solution.
    """
    fixed_time = float(pr.get('time_graph_build', 0.0) or 0.0) + float(
        pr.get('time_embedding', 0.0) or 0.0
    )
    actual_wall_clock = float(pr.get('time_total', float('nan')))

    sample_total_times = _coerce_float_list(pr.get('all_sample_total_times'))
    sample_lp_times = _coerce_float_list(pr.get('all_sample_lp_solve_times'))
    n_samples = len(sample_total_times)

    if n_samples == 0:
        declared_samples = int(pr.get('n_samples', 0) or 0)
        if declared_samples > 0 and np.isfinite(actual_wall_clock):
            variable_total = max(actual_wall_clock - fixed_time, 0.0)
            sample_total_times = [variable_total / declared_samples] * declared_samples
            n_samples = declared_samples
            raw_lp_total = float(pr.get('time_lp_solve', float('nan')))
            if np.isfinite(raw_lp_total):
                sample_lp_times = [raw_lp_total / declared_samples] * declared_samples

    cumulative_candidates = (
        fixed_time + float(np.nansum(sample_total_times))
        if sample_total_times
        else float('nan')
    )
    if not np.isfinite(actual_wall_clock) and np.isfinite(cumulative_candidates):
        actual_wall_clock = cumulative_candidates

    best_idx = pr.get('best_sample_idx', 0)
    try:
        best_idx = int(best_idx)
    except (TypeError, ValueError):
        best_idx = 0

    first_candidate = (
        fixed_time + float(sample_total_times[0])
        if sample_total_times
        else float('nan')
    )
    best_candidate_oracle = (
        fixed_time + float(sample_total_times[best_idx])
        if 0 <= best_idx < len(sample_total_times)
        else float('nan')
    )
    parallel_ideal = (
        fixed_time + float(np.nanmax(sample_total_times))
        if sample_total_times
        else float('nan')
    )
    mean_candidate = (
        fixed_time + float(np.nanmean(sample_total_times))
        if sample_total_times
        else float('nan')
    )
    best_lp_time = (
        float(sample_lp_times[best_idx])
        if 0 <= best_idx < len(sample_lp_times)
        else float('nan')
    )

    return {
        'fixed_time': fixed_time,
        'actual_wall_clock': actual_wall_clock,
        'cumulative_candidates': cumulative_candidates,
        'first_candidate': first_candidate,
        'best_candidate_oracle': best_candidate_oracle,
        'parallel_ideal': parallel_ideal,
        'mean_candidate': mean_candidate,
        'best_candidate_lp_time': best_lp_time,
        'n_candidates_evaluated': int(len(sample_total_times)),
    }


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
        timing_views = _pipeline_timing_views(pr)
        
        row = {
            # Pipeline
            'scenario_id': sc_id,
            'family': pr.get('family', ''),
            'pipeline_objective': pr.get('lp_objective', float('nan')),
            'pipeline_solve_time': timing_views['actual_wall_clock'],
            'pipeline_wall_clock_time': timing_views['actual_wall_clock'],
            'pipeline_candidate_cumulative_time': timing_views['cumulative_candidates'],
            'pipeline_first_candidate_time': timing_views['first_candidate'],
            'pipeline_best_candidate_oracle_time': timing_views['best_candidate_oracle'],
            'pipeline_parallel_ideal_time': timing_views['parallel_ideal'],
            'pipeline_mean_candidate_time': timing_views['mean_candidate'],
            'pipeline_best_candidate_lp_time': timing_views['best_candidate_lp_time'],
            'pipeline_fixed_time': timing_views['fixed_time'],
            'pipeline_n_candidates_evaluated': timing_views['n_candidates_evaluated'],
            'pipeline_status': pr.get('lp_status', ''),
            'pipeline_stage': pr.get('lp_stage_used', ''),
            'pipeline_stage_reached': pr.get('lp_stage_reached', pr.get('lp_stage_used', '')),
            'pipeline_slack': pr.get('lp_slack', 0.0),
            'pipeline_n_flips': pr.get('lp_n_flips', 0),
            'pipeline_best_status': pr.get('best_status', ''),
            'pipeline_direct_feasible_count': pr.get('direct_feasible_count', 0),
            'pipeline_repaired_feasible_count': pr.get('repaired_feasible_count', 0),
            'pipeline_direct_feasible_rate': pr.get('direct_feasible_rate', float('nan')),
            'pipeline_repair_success_rate': pr.get('repair_success_rate', float('nan')),
            'pipeline_fallback_rate': pr.get('fallback_rate', float('nan')),
            'pipeline_fallback_used': pr.get('fallback_used', False),
            'pipeline_time_direct_lp': pr.get('time_direct_lp', 0.0),
            'pipeline_time_repair': pr.get('time_repair', 0.0),
            'pipeline_time_fallback': pr.get('time_fallback', 0.0),
            'pipeline_fallback_warm_start_vars': pr.get('fallback_warm_start_vars', 0),
            'pipeline_all_repair_flips': pr.get('all_repair_flips', []),
            'pipeline_all_repair_radii': pr.get('all_repair_radii', []),
            'pipeline_all_repair_times': pr.get('all_repair_times', []),
            'pipeline_mean_repair_flips': pr.get('mean_repair_flips', float('nan')),
            'pipeline_median_repair_flips': pr.get('median_repair_flips', float('nan')),
            'pipeline_min_repair_flips': pr.get('min_repair_flips', float('nan')),
            'pipeline_max_repair_flips': pr.get('max_repair_flips', float('nan')),
            'pipeline_median_repair_radius_used': pr.get('median_repair_radius_used', float('nan')),
            'pipeline_max_repair_radius_used': pr.get('max_repair_radius_used', float('nan')),
            'pipeline_mean_repair_time': pr.get('mean_repair_time', float('nan')),
            'pipeline_median_repair_time': pr.get('median_repair_time', float('nan')),
            'pipeline_selected_repair_flips': pr.get('selected_repair_flips', float('nan')),
            'pipeline_selected_repair_radius_used': pr.get('selected_repair_radius_used', float('nan')),
            'pipeline_selected_repair_time': pr.get('selected_repair_time', float('nan')),
            'pipeline_top_m_enabled': pr.get('top_m_enabled', False),
            'pipeline_top_m_score_mode': pr.get('top_m_score_mode', ''),
            'pipeline_top_m_projected': pr.get('top_m_projected', 0),
            'pipeline_top_m_selected_indices': pr.get('top_m_selected_indices', []),
            'pipeline_top_m_skipped_indices': pr.get('top_m_skipped_indices', []),
            'pipeline_top_m_candidate_scores': pr.get('top_m_candidate_scores', []),
            'pipeline_n_candidates_generated': pr.get('n_candidates_generated', pr.get('n_samples', 0)),
            'pipeline_n_candidates_projected': pr.get('n_candidates_projected', timing_views['n_candidates_evaluated']),
            # Per-stage slack progression (MWh). Non-zero simultaneously for
            # multiple stages when the cascade reached Stage 4/5.
            'pipeline_slack_hard_fix':    pr.get('lp_slack_hard_fix', 0.0),
            'pipeline_slack_repair_20':   pr.get('lp_slack_repair_20', 0.0),
            'pipeline_slack_repair_100':  pr.get('lp_slack_repair_100', 0.0),
            'pipeline_slack_full_soft':   pr.get('lp_slack_full_soft', 0.0),
            'pipeline_slack_round_refix': pr.get('lp_slack_round_refix', 0.0),
            'pipeline_reached_hard_fix': pr.get('lp_reached_hard_fix', False),
            'pipeline_reached_repair_20': pr.get('lp_reached_repair_20', False),
            'pipeline_reached_repair_100': pr.get('lp_reached_repair_100', False),
            'pipeline_reached_full_soft': pr.get('lp_reached_full_soft', False),
            'pipeline_reached_round_refix': pr.get('lp_reached_round_refix', False),
            'n_zones': pr.get('n_zones', 0),
            'n_timesteps': pr.get('n_timesteps', 24),
            'criticality_index': pr.get('criticality_index', 0.0),
            'n_samples': pr.get('n_samples', 0),
            'best_sample_idx': pr.get('best_sample_idx', 0),
            'candidate_mean_hamming': pr.get('sample_mean_pairwise_hamming', float('nan')),
            'candidate_max_hamming': pr.get('sample_max_pairwise_hamming', float('nan')),
            'candidate_unique_ratio': pr.get('sample_unique_ratio', float('nan')),
            'all_stages_reached': pr.get('all_stages_reached', []),
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
    df['speedup'] = df.apply(
        lambda row: _safe_ratio(row['milp_solve_time'], row['pipeline_solve_time']),
        axis=1,
    )
    df['speedup_actual'] = df['speedup']
    df['speedup_first_candidate'] = df.apply(
        lambda row: _safe_ratio(row['milp_solve_time'], row['pipeline_first_candidate_time']),
        axis=1,
    )
    df['speedup_best_candidate_oracle'] = df.apply(
        lambda row: _safe_ratio(row['milp_solve_time'], row['pipeline_best_candidate_oracle_time']),
        axis=1,
    )
    df['speedup_parallel_ideal'] = df.apply(
        lambda row: _safe_ratio(row['milp_solve_time'], row['pipeline_parallel_ideal_time']),
        axis=1,
    )
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
    stage_col = 'pipeline_stage_reached' if 'pipeline_stage_reached' in df.columns else 'pipeline_stage'
    stage_counts = df[stage_col].value_counts().to_dict()
    total = len(df)
    stage_pct = {k: round(v / total * 100, 1) for k, v in stage_counts.items()}
    return {
        'counts': stage_counts,
        'percentages': stage_pct,
        'total': total,
        'stage_col': stage_col,
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
