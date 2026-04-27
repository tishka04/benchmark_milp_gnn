"""
RH-MO+LP heuristic baseline.

Rolling-Horizon Merit-Order heuristic (Stage A) with LP-Worker
reconstruction (Stage B). Used as a deterministic, learning-free
baseline against the MILP oracle and the MILP-GNN-EBM pipeline.

Public API:
    HeuristicConfig, rolling_horizon_heuristic
    HeuristicRunner, HeuristicResult
"""

from src.heuristics.rolling_horizon import (
    HeuristicConfig,
    rolling_horizon_heuristic,
)
from src.heuristics.runner import (
    HeuristicResult,
    HeuristicRunner,
)

__all__ = [
    "HeuristicConfig",
    "rolling_horizon_heuristic",
    "HeuristicRunner",
    "HeuristicResult",
]
