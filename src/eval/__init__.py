"""
Evaluation helpers for the MILP-GNN-EBM local benchmark.

Exports the core runner, summary metrics, lambda-based economics, and
paper-facing advanced analyses such as best-of-K and scaling-law utilities.
"""

from .advanced_analysis import (
    PHYSICAL_FEATURE_COLUMNS,
    build_candidate_pool_frame,
    build_physical_complexity_frame,
    compute_best_of_k_curve,
    compute_k_sampling_diagnostics,
    compute_solution_diversity_frame,
    fit_physical_feature_robustness,
    fit_scaling_law_models,
    merge_physical_complexity_features,
)
from .economic_advantage import EconomicAdvantageAnalyzer
from .metrics import compute_eval_metrics, compute_percentile_metrics
from .pipeline_runner import PipelineConfig, PipelineRunner
