"""
Advanced evaluation helpers for the local pipeline notebook.

This module focuses on three paper-facing upgrades:

1. Best-of-K / solution diversity analysis for EBM candidate pools
2. Scaling-law analysis versus scenario criticality or physical complexity
3. Breakeven-lambda framing for decision-theoretic economic value
"""
from __future__ import annotations

import json
import pickle
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.preprocessing import StandardScaler

from src.milp.scenario_loader import load_scenario_data


PHYSICAL_FEATURE_COLUMNS = [
    'n_zones',
    'n_binary_variables',
    'peak_to_valley_ratio',
    'storage_adequacy_hours',
    'vre_volatility_index',
]


def _result_key(family: str, scenario_id: str) -> Tuple[str, str]:
    return (str(family), str(scenario_id))


def _coerce_float_list(values: Any) -> List[float]:
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


def _coerce_str_list(values: Any) -> List[str]:
    if values is None:
        return []
    return [str(value) for value in values]


def _sample_activation_rate(sample_stats: Any) -> float:
    """Collapse per-feature binary activity into one overall activation rate."""
    if not isinstance(sample_stats, dict) or not sample_stats:
        return float('nan')
    values = []
    for value in sample_stats.values():
        try:
            values.append(float(value))
        except (TypeError, ValueError):
            continue
    return float(np.mean(values)) if values else float('nan')


def _prefix_hamming_mean(matrix_like: Any, k: int) -> float:
    """Mean pairwise Hamming distance among the first K samples."""
    if matrix_like is None:
        return float('nan')
    matrix = np.asarray(matrix_like, dtype=float)
    if matrix.ndim != 2 or matrix.shape[0] < k or matrix.shape[1] < k or k <= 1:
        return 0.0 if k == 1 else float('nan')
    tri_upper = matrix[:k, :k][np.triu_indices(k, k=1)]
    finite = tri_upper[np.isfinite(tri_upper)]
    return float(np.mean(finite)) if finite.size else float('nan')


def _candidate_time_components(result: Dict[str, Any]) -> Tuple[float, List[float]]:
    """
    Return `(fixed_time, variable_times)` for best-of-K accounting.

    `fixed_time` corresponds to graph building + embedding.
    `variable_times` correspond to sampling + decoder + LP for each candidate.
    """
    fixed_time = float(result.get('time_graph_build', 0.0) or 0.0) + float(
        result.get('time_embedding', 0.0) or 0.0
    )
    objectives = _coerce_float_list(result.get('all_objectives'))
    n_samples = len(objectives)
    if n_samples == 0:
        n_samples = int(result.get('n_samples', 0) or 0)

    raw = _coerce_float_list(result.get('all_sample_total_times'))
    if raw and len(raw) >= n_samples:
        return fixed_time, raw[:n_samples]

    if n_samples <= 0:
        return fixed_time, []

    time_total = float(result.get('time_total', float('nan')))
    if np.isfinite(time_total):
        variable_total = max(time_total - fixed_time, 0.0)
        return fixed_time, [variable_total / n_samples] * n_samples

    return fixed_time, []


def build_candidate_pool_frame(
    pipeline_results: Sequence[Dict[str, Any]],
    comparison_df: Optional[pd.DataFrame] = None,
) -> pd.DataFrame:
    """Expand pipeline results into one row per candidate sample."""
    milp_lookup: Dict[Tuple[str, str], float] = {}
    if comparison_df is not None:
        milp_lookup = {
            _result_key(row.family, row.scenario_id): float(row.milp_objective)
            for row in comparison_df.itertuples()
            if np.isfinite(row.milp_objective)
        }

    rows: List[Dict[str, Any]] = []
    for result in pipeline_results:
        family = result.get('family', '')
        scenario_id = result.get('scenario_id')
        milp_objective = milp_lookup.get(_result_key(family, scenario_id), np.nan)
        objectives = _coerce_float_list(result.get('all_objectives'))
        stages = _coerce_str_list(result.get('all_stages'))
        fixed_time, sample_times = _candidate_time_components(result)
        sampling_times = _coerce_float_list(result.get('all_sample_sampling_times'))
        decoder_times = _coerce_float_list(result.get('all_sample_decoder_times'))
        lp_times = _coerce_float_list(result.get('all_sample_lp_solve_times'))
        active_fractions = result.get('all_binary_active_fractions') or []
        ebm_energies = _coerce_float_list(result.get('all_sample_ebm_energies'))
        sample_slacks = _coerce_float_list(result.get('all_sample_slacks'))
        stages_reached = _coerce_str_list(result.get('all_stages_reached'))
        statuses = _coerce_str_list(result.get('all_sample_statuses'))

        for sample_idx, objective in enumerate(objectives):
            cost_gap_pct = np.nan
            if np.isfinite(milp_objective) and np.isfinite(objective):
                milp_abs = max(abs(milp_objective), 1e-9)
                cost_gap_pct = 100.0 * (float(objective) - milp_objective) / milp_abs

            row = {
                'scenario_id': scenario_id,
                'family': family,
                'sample_idx': sample_idx,
                'objective': objective,
                'stage': stages[sample_idx] if sample_idx < len(stages) else '',
                'sample_total_time_s': sample_times[sample_idx] if sample_idx < len(sample_times) else np.nan,
                'sample_sampling_time_s': sampling_times[sample_idx] if sample_idx < len(sampling_times) else np.nan,
                'sample_decoder_time_s': decoder_times[sample_idx] if sample_idx < len(decoder_times) else np.nan,
                'sample_lp_time_s': lp_times[sample_idx] if sample_idx < len(lp_times) else np.nan,
                'fixed_time_s': fixed_time,
                'sample_ebm_energy': ebm_energies[sample_idx] if sample_idx < len(ebm_energies) else np.nan,
                'sample_slack_mwh': sample_slacks[sample_idx] if sample_idx < len(sample_slacks) else np.nan,
                'sample_stage_reached': stages_reached[sample_idx] if sample_idx < len(stages_reached) else '',
                'sample_status': statuses[sample_idx] if sample_idx < len(statuses) else '',
                'criticality_index': result.get('criticality_index', np.nan),
                'n_zones': result.get('n_zones', np.nan),
                'n_timesteps': result.get('n_timesteps', np.nan),
                'milp_objective': milp_objective,
                'cost_gap_pct': cost_gap_pct,
                # Paper-friendly aliases for candidate-level diagnostics.
                'energy': ebm_energies[sample_idx] if sample_idx < len(ebm_energies) else np.nan,
                'lp_objective': objective,
                'slack_mwh': sample_slacks[sample_idx] if sample_idx < len(sample_slacks) else np.nan,
                'stage_reached': stages_reached[sample_idx] if sample_idx < len(stages_reached) else '',
            }
            if sample_idx < len(active_fractions) and isinstance(active_fractions[sample_idx], dict):
                for key, value in active_fractions[sample_idx].items():
                    row[f'active_{key}'] = float(value)
                row['sample_activation_rate'] = _sample_activation_rate(active_fractions[sample_idx])
            rows.append(row)

    return pd.DataFrame(rows)


def compute_solution_diversity_frame(
    pipeline_results: Sequence[Dict[str, Any]],
    comparison_df: Optional[pd.DataFrame] = None,
) -> pd.DataFrame:
    """Compute per-scenario diversity metrics from candidate pools."""
    milp_lookup: Dict[Tuple[str, str], float] = {}
    if comparison_df is not None:
        milp_lookup = {
            _result_key(row.family, row.scenario_id): float(row.milp_objective)
            for row in comparison_df.itertuples()
            if np.isfinite(row.milp_objective)
        }

    rows: List[Dict[str, Any]] = []
    for result in pipeline_results:
        objectives = [value for value in _coerce_float_list(result.get('all_objectives')) if np.isfinite(value)]
        family = result.get('family', '')
        scenario_id = result.get('scenario_id')
        milp_objective = milp_lookup.get(_result_key(family, scenario_id), np.nan)

        row = {
            'scenario_id': scenario_id,
            'family': family,
            'criticality_index': float(result.get('criticality_index', np.nan)),
            'candidate_pool_size': int(result.get('n_samples', len(objectives)) or len(objectives)),
            'valid_candidates': int(len(objectives)),
            'binary_diversity_mean': float(result.get('sample_mean_pairwise_hamming', np.nan)),
            'binary_diversity_max': float(result.get('sample_max_pairwise_hamming', np.nan)),
            'binary_unique_ratio': float(result.get('sample_unique_ratio', np.nan)),
        }

        if objectives:
            pool_min = float(np.min(objectives))
            pool_max = float(np.max(objectives))
            row['pool_cost_min'] = pool_min
            row['pool_cost_max'] = pool_max
            row['pool_cost_spread_abs'] = pool_max - pool_min
            row['pool_first_objective'] = float(objectives[0])
            row['pool_best_minus_first'] = pool_min - float(objectives[0])
        else:
            row['pool_cost_min'] = np.nan
            row['pool_cost_max'] = np.nan
            row['pool_cost_spread_abs'] = np.nan
            row['pool_first_objective'] = np.nan
            row['pool_best_minus_first'] = np.nan

        if np.isfinite(milp_objective) and objectives:
            milp_abs = max(abs(milp_objective), 1e-9)
            row['pool_cost_spread_vs_milp_pct'] = 100.0 * row['pool_cost_spread_abs'] / milp_abs
            row['best_cost_gap_pct'] = 100.0 * (row['pool_cost_min'] - milp_objective) / milp_abs
        else:
            row['pool_cost_spread_vs_milp_pct'] = np.nan
            row['best_cost_gap_pct'] = np.nan

        rows.append(row)

    return pd.DataFrame(rows)


def compute_best_of_k_curve(
    pipeline_results: Sequence[Dict[str, Any]],
    comparison_df: pd.DataFrame,
    k_values: Sequence[int] = (1, 2, 3, 5, 10),
    heuristic_objective_col: str = 'heur_objective',
    objective_field: str = 'all_objectives',
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Aggregate quality/time trade-offs for the first K sampled candidates.

    Returns:
        summary_df: one row per K
        scenario_df: one row per (scenario, K)
    """
    comparison_lookup = {
        _result_key(row.family, row.scenario_id): row
        for row in comparison_df.itertuples()
    }
    scenario_rows: List[Dict[str, Any]] = []

    for result in pipeline_results:
        family = result.get('family', '')
        scenario_id = result.get('scenario_id')
        key = _result_key(family, scenario_id)
        comp_row = comparison_lookup.get(key)
        if comp_row is None:
            continue

        objectives = _coerce_float_list(result.get(objective_field))
        if not objectives and np.isfinite(result.get('lp_objective', np.nan)):
            objectives = [float(result['lp_objective'])]

        fixed_time, sample_times = _candidate_time_components(result)
        available_k = min(len(objectives), len(sample_times) if sample_times else len(objectives))
        if available_k == 0:
            continue

        milp_objective = float(comp_row.milp_objective)
        milp_time = float(comp_row.milp_solve_time)
        heuristic_objective = getattr(comp_row, heuristic_objective_col, np.nan)
        base_first = objectives[0]

        for k in k_values:
            if k > available_k:
                continue

            prefix_objectives = objectives[:k]
            valid_prefix = [value for value in prefix_objectives if np.isfinite(value)]
            best_objective = float(np.min(valid_prefix)) if valid_prefix else float('inf')
            total_time_s = fixed_time + float(np.sum(sample_times[:k])) if sample_times else np.nan
            milp_abs = max(abs(milp_objective), 1e-9)

            scenario_rows.append({
                'scenario_id': scenario_id,
                'family': family,
                'k': int(k),
                'criticality_index': float(result.get('criticality_index', np.nan)),
                'best_objective': best_objective,
                'first_objective': float(base_first),
                'objective_gain_vs_k1': float(base_first - best_objective) if np.isfinite(best_objective) else np.nan,
                'objective_gain_vs_k1_pct_milp': 100.0 * (base_first - best_objective) / milp_abs
                if np.isfinite(best_objective)
                else np.nan,
                'cost_gap_pct': 100.0 * (best_objective - milp_objective) / milp_abs
                if np.isfinite(best_objective)
                else np.nan,
                'total_time_s': total_time_s,
                'speedup_vs_milp': milp_time / total_time_s
                if np.isfinite(total_time_s) and total_time_s > 0
                else np.nan,
                'pipeline_faster_than_milp': bool(total_time_s < milp_time)
                if np.isfinite(total_time_s) and np.isfinite(milp_time)
                else False,
                'beats_heuristic': bool(best_objective < heuristic_objective)
                if np.isfinite(best_objective) and np.isfinite(heuristic_objective)
                else np.nan,
                'valid_candidates_in_prefix': int(len(valid_prefix)),
            })

    scenario_df = pd.DataFrame(scenario_rows)
    if scenario_df.empty:
        return pd.DataFrame(), scenario_df

    summary_rows: List[Dict[str, Any]] = []
    for k, sub_df in scenario_df.groupby('k', sort=True):
        beats_heuristic = sub_df['beats_heuristic'].dropna()
        abs_gap = sub_df['cost_gap_pct'].abs()
        summary_rows.append({
            'k': int(k),
            'n_scenarios': int(len(sub_df)),
            'mean_cost_gap_pct': float(sub_df['cost_gap_pct'].mean()),
            'median_cost_gap_pct': float(sub_df['cost_gap_pct'].median()),
            'mean_abs_cost_gap_pct': float(abs_gap.mean()),
            'median_abs_cost_gap_pct': float(abs_gap.median()),
            'mean_gain_vs_k1_pct_milp': float(sub_df['objective_gain_vs_k1_pct_milp'].mean()),
            'pct_pipeline_faster_than_milp': float(sub_df['pipeline_faster_than_milp'].mean() * 100),
            'pct_beats_heuristic': float(beats_heuristic.mean() * 100) if len(beats_heuristic) else np.nan,
            'mean_total_time_s': float(sub_df['total_time_s'].mean()),
            'median_total_time_s': float(sub_df['total_time_s'].median()),
            'mean_speedup_vs_milp': float(sub_df['speedup_vs_milp'].mean()),
            'median_speedup_vs_milp': float(sub_df['speedup_vs_milp'].median()),
        })

    summary_df = pd.DataFrame(summary_rows).sort_values('k').reset_index(drop=True)
    return summary_df, scenario_df


def compute_k_sampling_diagnostics(
    pipeline_results: Sequence[Dict[str, Any]],
    comparison_df: pd.DataFrame,
    k_values: Sequence[int] = (1, 3, 5, 10),
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Summarize sampler usefulness as K grows.

    Metrics are computed on the first K sampled candidates per scenario:
    - mean pairwise Hamming distance
    - mean activation rate
    - EBM energy std
    - best-of-K cost gap vs MILP
    - slack and LP stage reached of the selected best-of-K candidate
    """
    comparison_lookup = {
        _result_key(row.family, row.scenario_id): row
        for row in comparison_df.itertuples()
    }
    scenario_rows: List[Dict[str, Any]] = []

    for result in pipeline_results:
        family = result.get('family', '')
        scenario_id = result.get('scenario_id')
        comp_row = comparison_lookup.get(_result_key(family, scenario_id))
        if comp_row is None:
            continue

        objectives = _coerce_float_list(result.get('all_objectives'))
        if not objectives and np.isfinite(result.get('lp_objective', np.nan)):
            objectives = [float(result['lp_objective'])]
        if not objectives:
            continue

        energies = _coerce_float_list(result.get('all_sample_ebm_energies'))
        active_fractions = result.get('all_binary_active_fractions') or []
        activation_rates = [_sample_activation_rate(item) for item in active_fractions]
        slacks = _coerce_float_list(result.get('all_sample_slacks'))
        stages_reached = _coerce_str_list(result.get('all_stages_reached'))
        statuses = _coerce_str_list(result.get('all_sample_statuses'))
        hamming_matrix = result.get('pairwise_hamming_matrix')

        available_k = len(objectives)
        milp_objective = float(comp_row.milp_objective)
        milp_abs = max(abs(milp_objective), 1e-9) if np.isfinite(milp_objective) else np.nan

        for k in k_values:
            if k > available_k:
                continue

            prefix_objectives = objectives[:k]
            valid_indices = [
                idx for idx, value in enumerate(prefix_objectives)
                if np.isfinite(value)
            ]
            best_idx = min(valid_indices, key=lambda idx: prefix_objectives[idx]) if valid_indices else None
            prefix_energies = np.asarray(energies[:k], dtype=float) if energies else np.asarray([], dtype=float)
            prefix_energies = prefix_energies[np.isfinite(prefix_energies)]
            prefix_activations = np.asarray(activation_rates[:k], dtype=float) if activation_rates else np.asarray([], dtype=float)
            prefix_activations = prefix_activations[np.isfinite(prefix_activations)]

            best_objective = float(prefix_objectives[best_idx]) if best_idx is not None else float('inf')
            cost_gap_pct = (
                100.0 * (best_objective - milp_objective) / milp_abs
                if best_idx is not None and np.isfinite(milp_abs)
                else np.nan
            )

            scenario_rows.append({
                'scenario_id': scenario_id,
                'family': family,
                'criticality_index': float(result.get('criticality_index', np.nan)),
                'k': int(k),
                'available_samples': int(available_k),
                'valid_candidates': int(len(valid_indices)),
                'prefix_mean_hamming': _prefix_hamming_mean(hamming_matrix, k),
                'prefix_activation_rate_mean': float(np.mean(prefix_activations)) if prefix_activations.size else np.nan,
                'prefix_activation_rate_std': float(np.std(prefix_activations)) if prefix_activations.size else np.nan,
                'prefix_energy_mean': float(np.mean(prefix_energies)) if prefix_energies.size else np.nan,
                'prefix_energy_std': float(np.std(prefix_energies)) if prefix_energies.size else np.nan,
                'best_of_k_objective': best_objective,
                'best_of_k_cost_gap_pct': cost_gap_pct,
                'best_of_k_abs_cost_gap_pct': abs(cost_gap_pct) if np.isfinite(cost_gap_pct) else np.nan,
                'best_of_k_slack_mwh': slacks[best_idx] if best_idx is not None and best_idx < len(slacks) else np.nan,
                'best_of_k_stage_reached': stages_reached[best_idx] if best_idx is not None and best_idx < len(stages_reached) else '',
                'best_of_k_status': statuses[best_idx] if best_idx is not None and best_idx < len(statuses) else '',
            })

    scenario_df = pd.DataFrame(scenario_rows)
    if scenario_df.empty:
        return pd.DataFrame(), scenario_df

    summary_rows: List[Dict[str, Any]] = []
    for k, sub_df in scenario_df.groupby('k', sort=True):
        stage_mode = sub_df['best_of_k_stage_reached'].mode(dropna=True)
        summary_rows.append({
            'k': int(k),
            'n_scenarios': int(len(sub_df)),
            'mean_hamming': float(sub_df['prefix_mean_hamming'].mean()),
            'mean_activation_rate': float(sub_df['prefix_activation_rate_mean'].mean()),
            'mean_energy_std': float(sub_df['prefix_energy_std'].mean()),
            'mean_best_cost_gap_pct': float(sub_df['best_of_k_cost_gap_pct'].mean()),
            'median_best_cost_gap_pct': float(sub_df['best_of_k_cost_gap_pct'].median()),
            'mean_best_abs_cost_gap_pct': float(sub_df['best_of_k_abs_cost_gap_pct'].mean()),
            'median_best_abs_cost_gap_pct': float(sub_df['best_of_k_abs_cost_gap_pct'].median()),
            'mean_best_slack_mwh': float(sub_df['best_of_k_slack_mwh'].mean()),
            'median_best_slack_mwh': float(sub_df['best_of_k_slack_mwh'].median()),
            'mode_best_stage_reached': stage_mode.iloc[0] if len(stage_mode) else '',
        })

    summary_df = pd.DataFrame(summary_rows).sort_values('k').reset_index(drop=True)
    return summary_df, scenario_df


def summarize_speedup_operating_modes(
    comparison_df: pd.DataFrame,
    group_col: str = 'family',
) -> pd.DataFrame:
    modes = [
        ('actual_sequential', 'speedup_actual', 'pipeline_solve_time'),
        ('first_candidate', 'speedup_first_candidate', 'pipeline_first_candidate_time'),
        ('parallel_ideal', 'speedup_parallel_ideal', 'pipeline_parallel_ideal_time'),
        ('best_candidate_oracle', 'speedup_best_candidate_oracle', 'pipeline_best_candidate_oracle_time'),
    ]
    groups: List[Tuple[str, pd.DataFrame]] = [('ALL', comparison_df)]
    if group_col in comparison_df.columns:
        groups.extend(
            (str(name), sub_df)
            for name, sub_df in comparison_df.groupby(group_col, sort=True)
        )

    rows: List[Dict[str, Any]] = []
    for group_name, sub_df in groups:
        clean = sub_df.replace([np.inf, -np.inf], np.nan)
        for mode_name, speed_col, time_col in modes:
            if speed_col not in clean.columns or time_col not in clean.columns:
                continue
            valid_speed = clean[speed_col].dropna()
            valid_time = clean[time_col].dropna()
            abs_gap = clean['cost_gap_pct'].abs().dropna() if 'cost_gap_pct' in clean.columns else pd.Series(dtype=float)
            rows.append({
                'group': group_name,
                'mode': mode_name,
                'n_scenarios': int(len(clean)),
                'median_speedup': float(valid_speed.median()) if len(valid_speed) else np.nan,
                'mean_speedup': float(valid_speed.mean()) if len(valid_speed) else np.nan,
                'pct_faster_than_milp': float((valid_speed > 1.0).mean() * 100.0) if len(valid_speed) else np.nan,
                'median_pipeline_time_s': float(valid_time.median()) if len(valid_time) else np.nan,
                'mean_pipeline_time_s': float(valid_time.mean()) if len(valid_time) else np.nan,
                'median_cost_gap_pct': float(clean['cost_gap_pct'].median()) if 'cost_gap_pct' in clean.columns and clean['cost_gap_pct'].notna().any() else np.nan,
                'median_abs_cost_gap_pct': float(abs_gap.median()) if len(abs_gap) else np.nan,
                'p90_abs_cost_gap_pct': float(np.percentile(abs_gap, 90)) if len(abs_gap) else np.nan,
            })
    return pd.DataFrame(rows)


def _candidate_selector_feature_columns(
    candidate_df: pd.DataFrame,
    feature_cols: Optional[Sequence[str]] = None,
) -> List[str]:
    if feature_cols is not None:
        return [col for col in feature_cols if col in candidate_df.columns]
    base_cols = [
        'sample_ebm_energy',
        'sample_activation_rate',
        'criticality_index',
        'n_zones',
        'n_timesteps',
    ]
    active_cols = sorted(col for col in candidate_df.columns if col.startswith('active_'))
    return [col for col in base_cols + active_cols if col in candidate_df.columns]


def _selector_raw_features(
    candidate_df: pd.DataFrame,
    feature_cols: Sequence[str],
) -> pd.DataFrame:
    parts = []
    if feature_cols:
        numeric = candidate_df[list(feature_cols)].apply(pd.to_numeric, errors='coerce')
        parts.append(numeric)
    if 'family' in candidate_df.columns:
        parts.append(pd.get_dummies(candidate_df['family'].astype(str), prefix='family', dtype=float))
    if not parts:
        return pd.DataFrame(index=candidate_df.index)
    return pd.concat(parts, axis=1).replace([np.inf, -np.inf], np.nan)


def fit_candidate_selector(
    candidate_df: pd.DataFrame,
    feature_cols: Optional[Sequence[str]] = None,
    failure_penalty: Optional[float] = None,
    time_penalty: float = 0.0,
    test_fraction: float = 0.25,
) -> Dict[str, Any]:
    work = candidate_df.copy().replace([np.inf, -np.inf], np.nan)
    selected_features = _candidate_selector_feature_columns(work, feature_cols)
    X_raw = _selector_raw_features(work, selected_features)
    if X_raw.empty:
        raise ValueError('No usable pre-LP candidate selector features were found.')

    scenario_key = (
        work.get('family', '').astype(str) + '/' + work.get('scenario_id', '').astype(str)
        if 'family' in work.columns and 'scenario_id' in work.columns
        else pd.Series(np.arange(len(work)).astype(str), index=work.index)
    )
    split_score = scenario_key.map(
        lambda value: (sum((idx + 1) * ord(ch) for idx, ch in enumerate(str(value))) % 1000) / 1000.0
    )
    train_mask = split_score >= float(test_fraction)
    if int(train_mask.sum()) < 10:
        train_mask = pd.Series(True, index=work.index)

    medians = X_raw.loc[train_mask].median(numeric_only=True).fillna(0.0)
    X = X_raw.fillna(medians).fillna(0.0)
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X.loc[train_mask])

    finite_objective = pd.to_numeric(work.get('objective', np.nan), errors='coerce').notna()
    status = work.get('sample_status', pd.Series('', index=work.index)).astype(str)
    feasible_y = (finite_objective | status.isin(['direct_feasible', 'repaired_feasible'])).astype(int)
    feasibility_model = None
    feasibility_constant = float(feasible_y.loc[train_mask].mean()) if int(train_mask.sum()) else float(feasible_y.mean())
    if feasible_y.loc[train_mask].nunique(dropna=True) >= 2:
        feasibility_model = LogisticRegression(max_iter=2000, class_weight='balanced')
        feasibility_model.fit(X_train, feasible_y.loc[train_mask].to_numpy(dtype=int))

    gap_y = pd.to_numeric(work.get('cost_gap_pct', np.nan), errors='coerce').abs()
    gap_train_mask = train_mask & gap_y.notna()
    gap_model = None
    gap_constant = float(gap_y.loc[gap_train_mask].median()) if int(gap_train_mask.sum()) else float(gap_y.dropna().median()) if gap_y.notna().any() else 100.0
    if int(gap_train_mask.sum()) >= 10:
        gap_model = LinearRegression()
        gap_model.fit(scaler.transform(X.loc[gap_train_mask]), gap_y.loc[gap_train_mask].to_numpy(dtype=float))

    lp_y = pd.to_numeric(work.get('sample_lp_time_s', np.nan), errors='coerce')
    lp_train_mask = train_mask & lp_y.notna()
    lp_time_model = None
    lp_time_constant = float(lp_y.loc[lp_train_mask].median()) if int(lp_train_mask.sum()) else float(lp_y.dropna().median()) if lp_y.notna().any() else 0.0
    if int(lp_train_mask.sum()) >= 10:
        lp_time_model = LinearRegression()
        lp_time_model.fit(scaler.transform(X.loc[lp_train_mask]), lp_y.loc[lp_train_mask].to_numpy(dtype=float))

    if failure_penalty is None:
        finite_gap = gap_y.dropna()
        failure_penalty = float(np.percentile(finite_gap, 75)) if len(finite_gap) else 100.0

    diagnostics: Dict[str, Any] = {
        'n_rows': int(len(work)),
        'n_train_rows': int(train_mask.sum()),
        'n_test_rows': int((~train_mask).sum()),
        'feature_cols': list(selected_features),
        'model_feature_count': int(X.shape[1]),
        'failure_penalty': float(failure_penalty),
        'time_penalty': float(time_penalty),
        'train_feasible_rate': float(feasible_y.loc[train_mask].mean()) if int(train_mask.sum()) else np.nan,
    }
    if int((~train_mask).sum()) and feasibility_model is not None:
        X_test = scaler.transform(X.loc[~train_mask])
        pred = feasibility_model.predict(X_test)
        diagnostics['test_feasibility_accuracy'] = float(np.mean(pred == feasible_y.loc[~train_mask].to_numpy(dtype=int)))
    if int((~train_mask & gap_y.notna()).sum()) and gap_model is not None:
        pred_gap = np.maximum(gap_model.predict(scaler.transform(X.loc[~train_mask & gap_y.notna()])), 0.0)
        true_gap = gap_y.loc[~train_mask & gap_y.notna()].to_numpy(dtype=float)
        diagnostics['test_abs_gap_mae_pct'] = float(np.mean(np.abs(pred_gap - true_gap)))

    return {
        'feature_cols': list(selected_features),
        'model_columns': list(X.columns),
        'feature_medians': medians.to_dict(),
        'scaler': scaler,
        'feasibility_model': feasibility_model,
        'feasibility_constant': feasibility_constant,
        'gap_model': gap_model,
        'gap_constant': gap_constant,
        'lp_time_model': lp_time_model,
        'lp_time_constant': lp_time_constant,
        'failure_penalty': float(failure_penalty),
        'time_penalty': float(time_penalty),
        'diagnostics': diagnostics,
    }


def apply_candidate_selector(
    candidate_df: pd.DataFrame,
    selector: Dict[str, Any],
    score_col: str = 'selector_score',
) -> pd.DataFrame:
    scored = candidate_df.copy()
    X_raw = _selector_raw_features(scored, selector.get('feature_cols', []))
    X = X_raw.reindex(columns=selector.get('model_columns', []), fill_value=0.0)
    medians = pd.Series(selector.get('feature_medians', {}), dtype=float)
    X = X.replace([np.inf, -np.inf], np.nan).fillna(medians).fillna(0.0)
    Xz = selector['scaler'].transform(X)

    feasibility_model = selector.get('feasibility_model')
    if feasibility_model is not None:
        p_feasible = feasibility_model.predict_proba(Xz)[:, 1]
    else:
        p_feasible = np.full(len(scored), float(selector.get('feasibility_constant', 0.5)))

    gap_model = selector.get('gap_model')
    if gap_model is not None:
        pred_abs_gap = np.maximum(gap_model.predict(Xz), 0.0)
    else:
        pred_abs_gap = np.full(len(scored), float(selector.get('gap_constant', 100.0)))

    lp_time_model = selector.get('lp_time_model')
    if lp_time_model is not None:
        pred_lp_time = np.maximum(lp_time_model.predict(Xz), 0.0)
    else:
        pred_lp_time = np.full(len(scored), float(selector.get('lp_time_constant', 0.0)))

    scored['selector_p_feasible'] = p_feasible
    scored['selector_pred_abs_gap_pct'] = pred_abs_gap
    scored['selector_pred_lp_time_s'] = pred_lp_time
    scored[score_col] = (
        pred_abs_gap
        + float(selector.get('failure_penalty', 100.0)) * (1.0 - p_feasible)
        + float(selector.get('time_penalty', 0.0)) * pred_lp_time
    )
    return scored


def save_candidate_selector(selector: Dict[str, Any], path: Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('wb') as f:
        pickle.dump(selector, f)
    return path


def load_candidate_selector(path: Path) -> Dict[str, Any]:
    with Path(path).open('rb') as f:
        selector = pickle.load(f)
    if not isinstance(selector, dict):
        raise TypeError(f'Candidate selector artifact must contain a dict, got {type(selector).__name__}.')
    required = {'feature_cols', 'model_columns', 'feature_medians', 'scaler'}
    missing = sorted(required - set(selector))
    if missing:
        raise ValueError(f'Candidate selector artifact is missing required keys: {missing}')
    return selector


def compute_top_m_policy_curve(
    candidate_df: pd.DataFrame,
    comparison_df: pd.DataFrame,
    score_col: Optional[str] = 'selector_score',
    k_values: Sequence[int] = (10,),
    top_m_values: Sequence[int] = (1, 2, 3),
    policy_name: str = 'learned_selector',
    fallback: bool = True,
    generate_full_k: bool = True,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    comparison_lookup = {
        _result_key(row.family, row.scenario_id): row
        for row in comparison_df.itertuples()
    }
    rows: List[Dict[str, Any]] = []
    work = candidate_df.copy().replace([np.inf, -np.inf], np.nan)
    if score_col is not None and score_col not in work.columns:
        raise ValueError(f'Score column not found: {score_col}')

    for (family, scenario_id), group in work.groupby(['family', 'scenario_id'], sort=True):
        comp = comparison_lookup.get(_result_key(family, scenario_id))
        if comp is None:
            continue
        group = group.sort_values('sample_idx').copy()
        milp_objective = float(comp.milp_objective)
        milp_time = float(comp.milp_solve_time)
        milp_abs = max(abs(milp_objective), 1e-9)
        fixed_time = float(group['fixed_time_s'].dropna().iloc[0]) if 'fixed_time_s' in group.columns and group['fixed_time_s'].notna().any() else 0.0

        for k in k_values:
            pool = group[group['sample_idx'].astype(int) < int(k)].copy()
            if pool.empty:
                continue
            if score_col is None:
                ranked = pool.sort_values('sample_idx')
            else:
                ranked = pool.sort_values([score_col, 'sample_idx'], na_position='last')
            sampling_time = float(pool['sample_sampling_time_s'].sum()) if generate_full_k and 'sample_sampling_time_s' in pool.columns else 0.0

            for top_m in top_m_values:
                selected = ranked.head(int(top_m)).copy()
                if selected.empty:
                    continue
                if not generate_full_k and 'sample_sampling_time_s' in selected.columns:
                    sampling_time_selected = float(selected['sample_sampling_time_s'].sum())
                else:
                    sampling_time_selected = sampling_time
                post_time = 0.0
                for col in ['sample_decoder_time_s', 'sample_lp_time_s']:
                    if col in selected.columns:
                        post_time += float(selected[col].sum())
                total_time = fixed_time + sampling_time_selected + post_time

                objectives = pd.to_numeric(selected['objective'], errors='coerce')
                valid_mask = objectives.notna()
                fallback_used = False
                success = bool(valid_mask.any())
                best_objective = np.nan
                best_status = ''
                best_stage = ''
                if success:
                    best_pos = objectives[valid_mask].idxmin()
                    best_objective = float(objectives.loc[best_pos])
                    best_status = str(selected.loc[best_pos].get('sample_status', ''))
                    best_stage = str(selected.loc[best_pos].get('sample_stage_reached', ''))
                elif fallback:
                    fallback_time = float(getattr(comp, 'pipeline_time_fallback', np.nan))
                    if not np.isfinite(fallback_time) or fallback_time <= 0.0:
                        fallback_time = milp_time
                    total_time += fallback_time
                    best_objective = milp_objective
                    best_status = 'fallback_milp'
                    best_stage = 'fallback_milp'
                    fallback_used = True
                    success = True

                cost_gap_pct = (
                    100.0 * (best_objective - milp_objective) / milp_abs
                    if success and np.isfinite(best_objective)
                    else np.nan
                )
                rows.append({
                    'policy': policy_name,
                    'scenario_id': scenario_id,
                    'family': family,
                    'criticality_index': float(getattr(comp, 'criticality_index', np.nan)),
                    'k_generated': int(k),
                    'top_m_projected': int(top_m),
                    'success': bool(success),
                    'fallback_used': bool(fallback_used),
                    'best_objective': best_objective,
                    'cost_gap_pct': cost_gap_pct,
                    'abs_cost_gap_pct': abs(cost_gap_pct) if np.isfinite(cost_gap_pct) else np.nan,
                    'total_time_s': total_time,
                    'speedup_vs_milp': milp_time / total_time if np.isfinite(total_time) and total_time > 0 else np.nan,
                    'pipeline_faster_than_milp': bool(total_time < milp_time) if np.isfinite(total_time) and np.isfinite(milp_time) else False,
                    'best_status': best_status,
                    'best_stage_reached': best_stage,
                    'direct_feasible_selected': bool(best_status == 'direct_feasible'),
                    'repaired_feasible_selected': bool(best_status == 'repaired_feasible'),
                    'n_valid_selected': int(valid_mask.sum()),
                })

    scenario_df = pd.DataFrame(rows)
    if scenario_df.empty:
        return pd.DataFrame(), scenario_df

    summary_rows: List[Dict[str, Any]] = []
    for keys, sub_df in scenario_df.groupby(['policy', 'k_generated', 'top_m_projected'], sort=True):
        policy, k_generated, top_m = keys
        summary_rows.append({
            'policy': policy,
            'k_generated': int(k_generated),
            'top_m_projected': int(top_m),
            'n_scenarios': int(len(sub_df)),
            'success_rate_pct': float(sub_df['success'].mean() * 100.0),
            'fallback_rate_pct': float(sub_df['fallback_used'].mean() * 100.0),
            'direct_selected_rate_pct': float(sub_df['direct_feasible_selected'].mean() * 100.0),
            'repaired_selected_rate_pct': float(sub_df['repaired_feasible_selected'].mean() * 100.0),
            'median_cost_gap_pct': float(sub_df['cost_gap_pct'].median()),
            'median_abs_cost_gap_pct': float(sub_df['abs_cost_gap_pct'].median()),
            'p90_abs_cost_gap_pct': float(np.percentile(sub_df['abs_cost_gap_pct'].dropna(), 90)) if sub_df['abs_cost_gap_pct'].notna().any() else np.nan,
            'median_total_time_s': float(sub_df['total_time_s'].median()),
            'mean_total_time_s': float(sub_df['total_time_s'].mean()),
            'median_speedup_vs_milp': float(sub_df['speedup_vs_milp'].median()),
            'mean_speedup_vs_milp': float(sub_df['speedup_vs_milp'].mean()),
            'pct_faster_than_milp': float(sub_df['pipeline_faster_than_milp'].mean() * 100.0),
        })
    return pd.DataFrame(summary_rows), scenario_df


def build_sampler_sweep_configs() -> List[Dict[str, Any]]:
    return [
        {
            'name': 'conservative_threshold',
            'noise_scale': 0.6,
            'diversity_temperature_scale_min': 1.0,
            'diversity_temperature_scale_max': 1.3,
            'diversity_noise_scale_min': 0.8,
            'diversity_noise_scale_max': 1.2,
            'infer_binarize': 'threshold',
            'infer_threshold': 0.5,
        },
        {
            'name': 'low_noise_bernoulli',
            'noise_scale': 0.6,
            'diversity_temperature_scale_min': 1.0,
            'diversity_temperature_scale_max': 1.5,
            'diversity_noise_scale_min': 0.8,
            'diversity_noise_scale_max': 1.25,
            'infer_binarize': 'bernoulli',
            'infer_threshold': 0.5,
        },
        {
            'name': 'medium_noise_threshold',
            'noise_scale': 0.9,
            'diversity_temperature_scale_min': 1.0,
            'diversity_temperature_scale_max': 1.5,
            'diversity_noise_scale_min': 1.0,
            'diversity_noise_scale_max': 1.5,
            'infer_binarize': 'threshold',
            'infer_threshold': 0.5,
        },
        {
            'name': 'higher_on_threshold',
            'noise_scale': 0.6,
            'init_p': 0.15,
            'prior_p': 0.05,
            'diversity_temperature_scale_min': 1.0,
            'diversity_temperature_scale_max': 1.3,
            'diversity_noise_scale_min': 0.8,
            'diversity_noise_scale_max': 1.2,
            'infer_binarize': 'threshold',
            'infer_threshold': 0.45,
        },
    ]


def extract_physical_complexity_features(
    scenario_path: Path,
    family: str = '',
) -> Dict[str, Any]:
    """Build physically interpretable complexity covariates from one scenario."""
    scenario_path = Path(scenario_path)
    with scenario_path.open('r', encoding='utf-8') as f:
        scenario_json = json.load(f)

    scenario_data = load_scenario_data(scenario_path)
    zones = list(scenario_data.zones)
    periods = list(scenario_data.periods)
    dt_hours = float(scenario_data.dt_hours)

    demand = np.array([
        [scenario_data.demand[(zone, t)] for t in periods]
        for zone in zones
    ], dtype=float)
    solar = np.array([
        [scenario_data.solar_available[(zone, t)] for t in periods]
        for zone in zones
    ], dtype=float)
    wind = np.array([
        [scenario_data.wind_available[(zone, t)] for t in periods]
        for zone in zones
    ], dtype=float)
    hydro_ror = np.array([
        [scenario_data.hydro_ror_generation[(zone, t)] for t in periods]
        for zone in zones
    ], dtype=float)

    total_demand_t = demand.sum(axis=0)
    total_vre_t = solar.sum(axis=0) + wind.sum(axis=0) + hydro_ror.sum(axis=0)
    mean_demand_mw = float(np.mean(total_demand_t))
    peak_demand_mw = float(np.max(total_demand_t))
    valley_demand_mw = float(np.min(total_demand_t))
    total_demand_mwh = float(np.sum(total_demand_t) * dt_hours)

    storage_capacity_mwh = float(sum(scenario_data.battery_energy.values()) + sum(scenario_data.pumped_energy.values()))
    storage_power_mw = float(sum(scenario_data.battery_power.values()) + sum(scenario_data.pumped_power.values()))
    dr_peak_mw = float(max(sum(scenario_data.dr_limit[(zone, t)] for zone in zones) for t in periods))

    vre_mean_mw = float(np.mean(total_vre_t))
    vre_std_mw = float(np.std(total_vre_t))

    difficulty = scenario_json.get('difficulty_indicators', {})
    criticality_index = float(scenario_json.get('criticality_index', np.nan))
    n_zones = int(difficulty.get('n_zones', len(zones)))
    n_binary_variables = int(difficulty.get('n_binary_variables', 0))
    peak_to_valley_ratio = float(
        difficulty.get(
            'peak_to_valley_ratio',
            peak_demand_mw / max(valley_demand_mw, 1e-9),
        )
    )

    return {
        'scenario_id': scenario_path.stem,
        'family': family,
        'criticality_index': criticality_index,
        'n_zones': n_zones,
        'n_binary_variables': n_binary_variables,
        'peak_to_valley_ratio': peak_to_valley_ratio,
        'storage_adequacy_hours': storage_capacity_mwh / max(peak_demand_mw, 1e-9),
        'storage_adequacy_energy_ratio': storage_capacity_mwh / max(total_demand_mwh, 1e-9),
        'storage_power_ratio': storage_power_mw / max(peak_demand_mw, 1e-9),
        'dr_capacity_ratio': dr_peak_mw / max(peak_demand_mw, 1e-9),
        'vre_volatility_index': vre_std_mw / max(mean_demand_mw, 1e-9),
        'vre_cv': vre_std_mw / max(vre_mean_mw, 1e-9),
        'vre_share_mean': vre_mean_mw / max(mean_demand_mw, 1e-9),
        'mean_demand_mw': mean_demand_mw,
        'peak_demand_mw': peak_demand_mw,
        'valley_demand_mw': valley_demand_mw,
        'total_demand_mwh': total_demand_mwh,
    }


def build_physical_complexity_frame(
    family_dirs: Dict[str, Path],
    cache_path: Optional[Path] = None,
    refresh: bool = False,
) -> pd.DataFrame:
    """Compute or load physical complexity covariates for all scenarios."""
    if cache_path is not None:
        cache_path = Path(cache_path)
        if cache_path.exists() and not refresh:
            return pd.read_csv(cache_path)

    rows: List[Dict[str, Any]] = []
    for family, scenarios_dir in family_dirs.items():
        for scenario_path in sorted(Path(scenarios_dir).glob('scenario_*.json')):
            rows.append(extract_physical_complexity_features(scenario_path, family=family))

    feature_df = pd.DataFrame(rows)
    if cache_path is not None:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        feature_df.to_csv(cache_path, index=False)
    return feature_df


def merge_physical_complexity_features(
    comparison_df: pd.DataFrame,
    feature_df: pd.DataFrame,
) -> pd.DataFrame:
    """Attach physical complexity features to the main comparison frame."""
    key_cols = ['scenario_id', 'family']
    keep_cols = key_cols + [
        col for col in feature_df.columns
        if col not in key_cols and col not in comparison_df.columns
    ]
    merge_df = feature_df[keep_cols].copy()
    return comparison_df.merge(merge_df, on=key_cols, how='left')


def fit_scaling_law_models(
    df: pd.DataFrame,
    feature_col: str = 'criticality_index',
    cost_gap_col: str = 'cost_gap_pct',
    use_abs_gap: bool = True,
) -> Dict[str, Any]:
    """Fit the three core scaling-law relationships used in the notebook."""
    valid = df.copy()
    valid = valid.replace([np.inf, -np.inf], np.nan)
    valid = valid.dropna(subset=[feature_col, 'speedup', cost_gap_col])
    valid = valid[valid['speedup'] > 0]
    if valid.empty:
        return {}

    x = valid[feature_col].to_numpy(dtype=float)
    log_speedup = np.log(valid['speedup'].to_numpy(dtype=float))
    gap_target = np.abs(valid[cost_gap_col].to_numpy(dtype=float)) if use_abs_gap else valid[cost_gap_col].to_numpy(dtype=float)

    speed_fit = stats.linregress(x, log_speedup)
    gap_fit = stats.linregress(x, gap_target)
    spearman_speed = stats.spearmanr(x, valid['speedup'].to_numpy(dtype=float), nan_policy='omit')
    spearman_gap = stats.spearmanr(x, gap_target, nan_policy='omit')

    faster = (valid['speedup'].to_numpy(dtype=float) > 1.0).astype(int)
    logistic_beta = float('nan')
    logistic_intercept = float('nan')
    logistic_accuracy = float('nan')
    prob_grid = pd.DataFrame(columns=[feature_col, 'prob_pipeline_faster'])
    if np.unique(faster).size >= 2:
        logit = LogisticRegression(max_iter=2000)
        logit.fit(x.reshape(-1, 1), faster)
        logistic_beta = float(logit.coef_[0, 0])
        logistic_intercept = float(logit.intercept_[0])
        logistic_accuracy = float(logit.score(x.reshape(-1, 1), faster))
        grid = np.linspace(float(x.min()), float(x.max()), 200)
        probs = logit.predict_proba(grid.reshape(-1, 1))[:, 1]
        prob_grid = pd.DataFrame({
            feature_col: grid,
            'prob_pipeline_faster': probs,
        })

    grid = np.linspace(float(x.min()), float(x.max()), 200)
    return {
        'panel_a_points': pd.DataFrame({
            feature_col: x,
            'log_speedup': log_speedup,
            'speedup': valid['speedup'].to_numpy(dtype=float),
        }),
        'panel_a_fit': pd.DataFrame({
            feature_col: grid,
            'pred_log_speedup': speed_fit.intercept + speed_fit.slope * grid,
        }),
        'panel_b_points': pd.DataFrame({
            feature_col: x,
            'pipeline_faster': faster,
        }),
        'panel_b_fit': prob_grid,
        'panel_c_points': pd.DataFrame({
            feature_col: x,
            'cost_gap_target': gap_target,
        }),
        'panel_c_fit': pd.DataFrame({
            feature_col: grid,
            'pred_cost_gap_target': gap_fit.intercept + gap_fit.slope * grid,
        }),
        'summary': {
            'feature_col': feature_col,
            'panel_a_beta': float(speed_fit.slope),
            'panel_a_intercept': float(speed_fit.intercept),
            'panel_a_rvalue': float(speed_fit.rvalue),
            'panel_a_pvalue': float(speed_fit.pvalue),
            'panel_a_spearman_rho': float(spearman_speed.statistic),
            'panel_a_spearman_pvalue': float(spearman_speed.pvalue),
            'panel_b_beta': logistic_beta,
            'panel_b_intercept': logistic_intercept,
            'panel_b_accuracy': logistic_accuracy,
            'panel_c_beta': float(gap_fit.slope),
            'panel_c_intercept': float(gap_fit.intercept),
            'panel_c_rvalue': float(gap_fit.rvalue),
            'panel_c_pvalue': float(gap_fit.pvalue),
            'panel_c_spearman_rho': float(spearman_gap.statistic),
            'panel_c_spearman_pvalue': float(spearman_gap.pvalue),
            'use_abs_gap': bool(use_abs_gap),
        },
    }


def fit_physical_feature_robustness(
    df: pd.DataFrame,
    feature_cols: Sequence[str] = PHYSICAL_FEATURE_COLUMNS,
    cost_gap_col: str = 'cost_gap_pct',
    use_abs_gap: bool = True,
) -> pd.DataFrame:
    """
    Fit standardized coefficient models using physical complexity covariates.

    This is intended as a reviewer-facing robustness check against the
    composite criticality index.
    """
    valid = df.copy().replace([np.inf, -np.inf], np.nan)
    use_cols = list(feature_cols) + ['speedup', cost_gap_col]
    valid = valid.dropna(subset=use_cols)
    valid = valid[valid['speedup'] > 0]
    if valid.empty:
        return pd.DataFrame()

    X = valid[list(feature_cols)].to_numpy(dtype=float)
    Xz = StandardScaler().fit_transform(X)
    log_speedup = np.log(valid['speedup'].to_numpy(dtype=float))
    faster = (valid['speedup'].to_numpy(dtype=float) > 1.0).astype(int)
    gap_target = np.abs(valid[cost_gap_col].to_numpy(dtype=float)) if use_abs_gap else valid[cost_gap_col].to_numpy(dtype=float)

    speed_model = LinearRegression().fit(Xz, log_speedup)
    gap_model = LinearRegression().fit(Xz, gap_target)

    logit_coef = np.full(len(feature_cols), np.nan)
    if np.unique(faster).size >= 2:
        logit_model = LogisticRegression(max_iter=2000)
        logit_model.fit(Xz, faster)
        logit_coef = logit_model.coef_[0]

    rows: List[Dict[str, Any]] = []
    for idx, feature in enumerate(feature_cols):
        speed_corr = stats.spearmanr(valid[feature], valid['speedup'], nan_policy='omit')
        gap_corr = stats.spearmanr(valid[feature], gap_target, nan_policy='omit')
        rows.append({
            'feature': feature,
            'beta_log_speedup': float(speed_model.coef_[idx]),
            'beta_pipeline_faster_logit': float(logit_coef[idx]),
            'beta_cost_gap_target': float(gap_model.coef_[idx]),
            'spearman_speedup_rho': float(speed_corr.statistic),
            'spearman_speedup_pvalue': float(speed_corr.pvalue),
            'spearman_cost_gap_rho': float(gap_corr.statistic),
            'spearman_cost_gap_pvalue': float(gap_corr.pvalue),
        })

    return pd.DataFrame(rows).sort_values(
        by='beta_pipeline_faster_logit',
        key=lambda s: s.abs(),
        ascending=False,
    ).reset_index(drop=True)
