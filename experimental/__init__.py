"""
Experimental Framework for Hybrid vs MILP Benchmarking

This package implements a rigorous experimental protocol for comparing:
- Monolithic MILP solvers (HiGHS/Gurobi)
- Hybrid thermodynamic + classical dispatch solver

Key principles:
1. Single canonical problem formulation (UC+Dispatch)
2. Perfect model alignment between methods
3. Reproducible instance generation
4. Fair time budgets and resource constraints
5. Comprehensive metrics and analysis
"""

__version__ = "1.0.0"
