"""
Dispatch solver - single source of truth for dispatch given commitment.

This module provides the LP/dispatch solver that both MILP and Hybrid use
to ensure perfect cost alignment.
"""
import numpy as np
from typing import Tuple, Optional
from scipy.optimize import linprog

try:
    from .data_models import UCInstance
except ImportError:
    from data_models import UCInstance


def solve_dispatch_given_commitment(
    instance: UCInstance,
    commitment: np.ndarray,
    period: Optional[int] = None
) -> Tuple[Optional[np.ndarray], float, bool]:
    """
    Solve economic dispatch for a given commitment schedule.
    
    This is the SINGLE SOURCE OF TRUTH for dispatch cost calculation.
    Both MILP and Hybrid must use this (or equivalent) to ensure alignment.
    
    Args:
        instance: Problem instance
        commitment: Binary commitment matrix [N] or [N, T]
        period: If provided, solve only for this period (commitment is [N])
    
    Returns:
        (dispatch, cost, feasible): Dispatch solution, total cost, feasibility flag
    """
    if period is not None:
        # Single period dispatch
        return _solve_single_period_dispatch(
            instance.p_min,
            instance.p_max,
            instance.marginal_cost,
            instance.demand[period],
            commitment
        )
    else:
        # Multi-period dispatch
        if commitment.ndim == 1:
            # Same commitment for all periods
            commitment = np.tile(commitment, (instance.n_periods, 1)).T
        
        return _solve_multi_period_dispatch(
            instance.p_min,
            instance.p_max,
            instance.marginal_cost,
            instance.demand,
            commitment
        )


def _solve_single_period_dispatch(
    p_min: np.ndarray,
    p_max: np.ndarray,
    marginal_cost: np.ndarray,
    demand: float,
    commitment: np.ndarray
) -> Tuple[Optional[np.ndarray], float, bool]:
    """
    Solve dispatch for a single period using greedy economic dispatch.
    
    This is deterministic and fast for single-period problems.
    """
    n_units = len(p_min)
    commitment = np.asarray(commitment, dtype=int)
    
    # Find committed units
    committed_units = np.where(commitment == 1)[0]
    
    if len(committed_units) == 0:
        return None, np.inf, False
    
    # Check feasibility
    max_gen = p_max[committed_units].sum()
    min_gen = p_min[committed_units].sum()
    
    if demand > max_gen or demand < min_gen:
        return None, np.inf, False
    
    # Greedy economic dispatch: sort by cost
    dispatch = np.zeros(n_units)
    sorted_idx = committed_units[np.argsort(marginal_cost[committed_units])]
    
    # First, set all to minimum
    for idx in sorted_idx:
        dispatch[idx] = p_min[idx]
    
    remaining_demand = demand - dispatch.sum()
    
    # Then, load up cheapest units first
    for idx in sorted_idx:
        if remaining_demand <= 0:
            break
        
        available = p_max[idx] - dispatch[idx]
        take = min(remaining_demand, available)
        dispatch[idx] += take
        remaining_demand -= take
    
    # Verify feasibility
    if abs(dispatch.sum() - demand) > 1e-3:
        return None, np.inf, False
    
    # Calculate cost
    cost = np.sum(dispatch * marginal_cost)
    
    return dispatch, cost, True


def _solve_multi_period_dispatch(
    p_min: np.ndarray,
    p_max: np.ndarray,
    marginal_cost: np.ndarray,
    demand: np.ndarray,
    commitment: np.ndarray
) -> Tuple[Optional[np.ndarray], float, bool]:
    """
    Solve dispatch for multiple periods.
    
    Args:
        commitment: [N, T] matrix
    
    Returns:
        (dispatch [N, T], total_cost, feasible)
    """
    n_units = len(p_min)
    n_periods = len(demand)
    
    dispatch_full = np.zeros((n_units, n_periods))
    total_cost = 0.0
    
    # Solve period by period
    for t in range(n_periods):
        u_t = commitment[:, t]
        dispatch_t, cost_t, feasible = _solve_single_period_dispatch(
            p_min, p_max, marginal_cost, demand[t], u_t
        )
        
        if not feasible:
            return None, np.inf, False
        
        dispatch_full[:, t] = dispatch_t
        total_cost += cost_t
    
    return dispatch_full, total_cost, True


def calculate_total_cost(
    instance: UCInstance,
    commitment: np.ndarray,
    power: np.ndarray
) -> Tuple[float, float, float]:
    """
    Calculate total cost from commitment and dispatch.
    
    This is the CANONICAL cost function used by both methods.
    
    Args:
        instance: Problem instance
        commitment: Binary commitment [N, T]
        power: Dispatch [N, T]
    
    Returns:
        (total_cost, fuel_cost, startup_cost)
    """
    # Fuel cost
    fuel_cost = np.sum(power * instance.marginal_cost[:, np.newaxis])
    
    # Startup cost: detect startups (0 -> 1 transitions)
    startup_cost = 0.0
    if commitment.ndim == 2:
        for i in range(instance.n_units):
            for t in range(instance.n_periods):
                if t == 0:
                    # Compare with initial status
                    if commitment[i, t] == 1 and instance.initial_status[i] == 0:
                        startup_cost += instance.startup_cost[i]
                else:
                    # Check transition
                    if commitment[i, t] == 1 and commitment[i, t-1] == 0:
                        startup_cost += instance.startup_cost[i]
    
    total_cost = fuel_cost + startup_cost
    
    return total_cost, fuel_cost, startup_cost


def verify_solution_feasibility(
    instance: UCInstance,
    commitment: np.ndarray,
    power: np.ndarray,
    tolerance: float = 1e-3
) -> Tuple[bool, list]:
    """
    Verify that a solution satisfies all constraints.
    
    Args:
        instance: Problem instance
        commitment: Binary commitment [N, T]
        power: Dispatch [N, T]
        tolerance: Numerical tolerance
    
    Returns:
        (is_feasible, violations): Feasibility flag and list of violation messages
    """
    violations = []
    
    n_units = instance.n_units
    n_periods = instance.n_periods
    
    # Ensure 2D
    if commitment.ndim == 1:
        commitment = commitment[:, np.newaxis]
    if power.ndim == 1:
        power = power[:, np.newaxis]
    
    # 1. Demand balance
    for t in range(n_periods):
        total_gen = power[:, t].sum()
        if abs(total_gen - instance.demand[t]) > tolerance:
            violations.append(f"Period {t}: Demand mismatch ({total_gen:.2f} vs {instance.demand[t]:.2f})")
    
    # 2. Min/max generation limits
    for i in range(n_units):
        for t in range(n_periods):
            if commitment[i, t] == 1:
                if power[i, t] < instance.p_min[i] - tolerance:
                    violations.append(f"Unit {i}, Period {t}: Below min ({power[i, t]:.2f} < {instance.p_min[i]:.2f})")
                if power[i, t] > instance.p_max[i] + tolerance:
                    violations.append(f"Unit {i}, Period {t}: Above max ({power[i, t]:.2f} > {instance.p_max[i]:.2f})")
            else:
                if power[i, t] > tolerance:
                    violations.append(f"Unit {i}, Period {t}: Generation when off ({power[i, t]:.2f})")
    
    # 3. Binary commitment
    if not np.all(np.isin(commitment, [0, 1])):
        violations.append("Commitment contains non-binary values")
    
    is_feasible = len(violations) == 0
    return is_feasible, violations
