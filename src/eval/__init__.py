"""
Evaluation module for the GNN+EBM pipeline.

Provides:
- pipeline_runner: Full pipeline evaluation (graph → encoder → EBM → decoder → LP)
- metrics: p90-p99 metrics, cost gap, speedup, stage distribution
- economic_advantage: Lambda-based economic indicator and sensitivity analysis
"""
from .pipeline_runner import PipelineRunner, PipelineConfig
from .metrics import compute_eval_metrics, compute_percentile_metrics
from .economic_advantage import EconomicAdvantageAnalyzer
