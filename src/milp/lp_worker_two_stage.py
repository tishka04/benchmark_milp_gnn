# ==============================================================================
# LP WORKER TWO-STAGE: HARD-FIX LP + SOFT REPAIR FALLBACK
# ==============================================================================
# This module implements a two-stage LP solving strategy:
#   Stage 1 (hard_fix): Fix all binaries from decoder, solve pure LP
#   Stage 2+ (repair): If slack too high, unfix critical variables and repair
#
# Strategy C: Flip budget constraint limits total deviations from decoder
# ==============================================================================

import os
import sys
import time
import torch
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any, Set as TypingSet
from dataclasses import dataclass, field
from enum import Enum

from pyomo.environ import (
    value, SolverFactory, Var, Binary, Constraint, Set,
    NonNegativeReals, Reals, Objective, minimize
)
from pyomo.opt import TerminationCondition
from pyomo.core.base import TransformationFactory
from src.milp.model import build_uc_model
from src.milp.scenario_loader import load_scenario_data


class SolveStage(Enum):
    """Enum for tracking which stage produced the solution."""
    HARD_FIX = "hard_fix"
    REPAIR_20 = "repair_20"
    REPAIR_100 = "repair_100"
    FULL_SOFT = "full_soft"
    FAILED = "failed"


@dataclass
class TwoStageResult:
    """Result from two-stage LP optimization with detailed logging."""
    scenario_id: str
    status: str
    stage_used: SolveStage
    objective_value: float  # True UC objective (without penalty)
    solve_time: float
    continuous_vars: Dict[str, np.ndarray]
    
    # Slack and deviation metrics (in MWh, accounting for dt)
    slack_used: float = 0.0
    decoder_deviation: float = 0.0
    n_flips: int = 0
    n_unfixed: int = 0
    n_unfixed_binaries_stage1: int = 0  # Track if Stage 1 was pure LP
    
    # Penalty tracking (for analysis)
    deviation_penalty_value: float = 0.0
    
    # Per-stage timing
    time_hard_fix: float = 0.0
    time_repair_20: float = 0.0
    time_repair_100: float = 0.0
    time_full_soft: float = 0.0
    
    # Slack progression (in MWh)
    slack_hard_fix: float = 0.0
    slack_repair_20: float = 0.0
    slack_repair_100: float = 0.0
    slack_full_soft: float = 0.0
    
    # For analysis
    critical_indices: List[Tuple[str, int]] = field(default_factory=list)
    message: str = ""


class LPWorkerTwoStage:
    """
    Two-Stage LP Worker with hard-fix fast path and minimal soft repair fallback.
    
    Stage 1 (hard_fix): Fix all decoder binaries → pure LP (fast, stable)
    Stage 2 (repair_20): Unfix K=20 critical vars with flip budget
    Stage 3 (repair_100): Unfix K=100 critical vars with flip budget  
    Stage 4 (full_soft): Full soft constraints as last resort
    
    Parameters:
        slack_tol_mwh: Max allowed slack before triggering fallback (default: 1.0)
        deviation_penalty: λ for soft penalty on deviation from decoder
        flip_budget: Max number of allowed flips (Strategy C)
        time_limits: Dict with TL1, TL2, TL3, TL4 in seconds
    """
    
    def __init__(
        self,
        scenarios_dir: str,
        solver_name: str = 'appsi_highs',
        slack_tol_mwh: float = 1.0,
        deviation_penalty: float = 10000.0,
        flip_budget_20: int = 100,
        flip_budget_100: int = 1000,
        flip_budget_full_soft: int = None,  # None = no limit on full_soft
        time_limit_hard_fix: float = 20.0,
        time_limit_repair_20: float = 15.0,
        time_limit_repair_100: float = 120.0,
        time_limit_full_soft: float = 900.0,  # 5 minutes
        temporal_window: int = 1,
        dt_hours: float = 1.0,  # Timestep duration for MWh conversion
        verbose: bool = True,
    ):
        self.scenarios_dir = scenarios_dir
        self.solver_name = solver_name
        self.slack_tol_mwh = slack_tol_mwh
        self.deviation_penalty = deviation_penalty
        self.flip_budget_20 = flip_budget_20
        self.flip_budget_100 = flip_budget_100
        self.flip_budget_full_soft = flip_budget_full_soft
        self.time_limits = {
            'hard_fix': time_limit_hard_fix,
            'repair_20': time_limit_repair_20,
            'repair_100': time_limit_repair_100,
            'full_soft': time_limit_full_soft,
        }
        self.temporal_window = temporal_window
        self.dt_hours = dt_hours
        self.verbose = verbose
        
        # Initialize solver
        self.solver = SolverFactory(solver_name)
        if self.solver is None or not self.solver.available():
            for fallback in ['highs', 'glpk', 'cbc']:
                self.solver = SolverFactory(fallback)
                if self.solver is not None and self.solver.available():
                    self.solver_name = fallback
                    break
        
        if self.solver is None or not self.solver.available():
            raise RuntimeError("No LP/MILP solver available")
        
        print(f"✓ LPWorkerTwoStage initialized")
        print(f"  Solver: {self.solver_name}")
        print(f"  Slack tolerance: {slack_tol_mwh} MWh (dt={dt_hours}h)")
        print(f"  Deviation penalty (λ): {deviation_penalty}")
        print(f"  Flip budgets: K=20→{flip_budget_20}, K=100→{flip_budget_100}, full_soft→{flip_budget_full_soft}")
        print(f"  Time limits: TL1={time_limit_hard_fix}s, TL2={time_limit_repair_20}s, "
              f"TL3={time_limit_repair_100}s, TL4={time_limit_full_soft}s")

    def _find_scenario_path(self, scenario_id: str) -> Optional[str]:
        """Find the scenario JSON file path."""
        for path in [
            os.path.join(self.scenarios_dir, f"{scenario_id}.json"),
            os.path.join(self.scenarios_dir, 'dispatch_batch', f"{scenario_id}.json"),
        ]:
            if os.path.exists(path):
                return path
        return None

    def _get_decoder_targets(
        self,
        decoder_output: torch.Tensor,
        zone_names: List[str],
        n_timesteps: int = 24
    ) -> Dict[str, Dict[Tuple[str, int], float]]:
        """Extract target binary values from decoder output tensor [Z, T, F]."""
        Z = len(zone_names)
        decoder_np = decoder_output.cpu().numpy()
        
        # Handle size mismatch
        if decoder_np.shape[0] > Z:
            decoder_np = decoder_np[:Z]
        elif decoder_np.shape[0] < Z:
            pad_size = Z - decoder_np.shape[0]
            decoder_np = np.pad(decoder_np, ((0, pad_size), (0, 0), (0, 0)))
        
        targets = {
            'u_thermal': {},
            'b_charge_mode': {},
            'pumped_charge_mode': {},
            'dr_active': {},
        }
        
        for z_idx, zone in enumerate(zone_names):
            for t in range(min(n_timesteps, decoder_np.shape[1])):
                # Feature indices from decoder:
                # 0: battery_charge, 1: battery_discharge
                # 2: pumped_charge, 3: pumped_discharge
                # 4: demand_response, 5: nuclear, 6: thermal
                
                targets['u_thermal'][(zone, t)] = float(decoder_np[z_idx, t, 6] > 0.5)
                
                # Battery mode: 1 if charging > discharging
                batt_ch = decoder_np[z_idx, t, 0]
                batt_dis = decoder_np[z_idx, t, 1]
                targets['b_charge_mode'][(zone, t)] = 1.0 if batt_ch > batt_dis else 0.0
                
                # Pumped mode
                pump_ch = decoder_np[z_idx, t, 2]
                pump_dis = decoder_np[z_idx, t, 3]
                targets['pumped_charge_mode'][(zone, t)] = 1.0 if pump_ch > pump_dis else 0.0
                
                # DR active
                targets['dr_active'][(zone, t)] = float(decoder_np[z_idx, t, 4] > 0.5)
        
        return targets

    def _fix_all_binaries(self, model, targets: Dict, eps: float = 1e-6, only_u_thermal: bool = False):
        """
        Fix binary variables to decoder targets (capacity-aware).
        
        Args:
            model: Pyomo model
            targets: Dict of target values
            eps: Capacity threshold
            only_u_thermal: If True, only fix u_thermal (for Stage 1 robustness)
        
        NOTE: Never fixes v_thermal_startup - let constraints handle it.
        """
        # Always fix u_thermal
        for (z, t), val in targets['u_thermal'].items():
            if hasattr(model, 'u_thermal') and (z, t) in model.u_thermal:
                has_cap = hasattr(model, 'thermal_capacity') and value(model.thermal_capacity[z]) > eps
                if has_cap or val == 0.0:
                    if not model.u_thermal[z, t].is_fixed():
                        model.u_thermal[z, t].fix(val)
        
        # Skip other binaries in Stage 1 for robustness
        if only_u_thermal:
            return
        
        # Fix other binaries only if requested
        for (z, t), val in targets['b_charge_mode'].items():
            if hasattr(model, 'b_charge_mode') and (z, t) in model.b_charge_mode:
                has_cap = hasattr(model, 'battery_power') and value(model.battery_power[z]) > eps
                if has_cap or val == 0.0:
                    if not model.b_charge_mode[z, t].is_fixed():
                        model.b_charge_mode[z, t].fix(val)
        
        for (z, t), val in targets['pumped_charge_mode'].items():
            if hasattr(model, 'pumped_charge_mode') and (z, t) in model.pumped_charge_mode:
                has_cap = hasattr(model, 'pumped_power') and value(model.pumped_power[z]) > eps
                if has_cap or val == 0.0:
                    if not model.pumped_charge_mode[z, t].is_fixed():
                        model.pumped_charge_mode[z, t].fix(val)
        
        for (z, t), val in targets['dr_active'].items():
            if hasattr(model, 'dr_active') and (z, t) in model.dr_active:
                has_cap = hasattr(model, 'dr_max') and value(model.dr_max[z]) > eps
                if has_cap or val == 0.0:
                    if not model.dr_active[z, t].is_fixed():
                        model.dr_active[z, t].fix(val)
        
        # Fix import mode if exists
        if hasattr(model, 'import_mode'):
            for t in model.T:
                if not model.import_mode[t].is_fixed():
                    model.import_mode[t].fix(1.0)

    def _count_unfixed_binaries(self, model) -> int:
        """
        Count binary variables that are NOT fixed.
        If > 0 after _fix_all_binaries, Stage 1 is NOT a pure LP.
        """
        count = 0
        for v in model.component_data_objects(Var, active=True):
            if v.domain is Binary and not v.is_fixed():
                count += 1
        return count

    def _list_unfixed_binaries(self, model) -> List[str]:
        """
        List names of unfixed binary variables (for diagnostics).
        Returns first 10 names max.
        """
        names = []
        for v in model.component_data_objects(Var, active=True):
            if v.domain is Binary and not v.is_fixed():
                names.append(v.name)
                if len(names) >= 10:
                    break
        return names

    def _count_total_binaries(self, model) -> int:
        """
        Count total binary variables in the model for dynamic flip budget calculation.
        Counts: u_thermal, b_charge_mode, pumped_charge_mode, dr_active
        """
        count = 0
        if hasattr(model, 'u_thermal'):
            count += len(list(model.u_thermal))
        if hasattr(model, 'b_charge_mode'):
            count += len(list(model.b_charge_mode))
        if hasattr(model, 'pumped_charge_mode'):
            count += len(list(model.pumped_charge_mode))
        if hasattr(model, 'dr_active'):
            count += len(list(model.dr_active))
        return count

    def _relax_slack_bounds(self, model):
        """Allow unbounded slack for feasibility (will be penalized in objective)."""
        if hasattr(model, 'unserved'):
            for idx in model.unserved:
                model.unserved[idx].setub(None)
        if hasattr(model, 'overgen_spill'):
            for idx in model.overgen_spill:
                model.overgen_spill[idx].setub(None)

    def _extract_slack(self, model) -> float:
        """
        Calculate total slack used (unserved + overgen) in MWh.
        Multiplies MW values by dt_hours to get energy.
        """
        slack_mw = 0.0
        zones = list(model.Z)
        for z in zones:
            for t in model.T:
                if hasattr(model, 'unserved') and (z, t) in model.unserved:
                    try:
                        val = value(model.unserved[z, t])
                        slack_mw += val if val is not None else 0.0
                    except:
                        pass
                if hasattr(model, 'overgen_spill') and (z, t) in model.overgen_spill:
                    try:
                        val = value(model.overgen_spill[z, t])
                        slack_mw += val if val is not None else 0.0
                    except:
                        pass
        # Convert MW*timesteps to MWh
        return slack_mw * self.dt_hours

    def _extract_solution(self, model) -> Dict[str, np.ndarray]:
        """Extract continuous solution from solved model."""
        solution = {}
        zones = list(model.Z)
        n_zones = len(zones)
        n_timesteps = len(list(model.T))
        
        def extract_2d(var_obj):
            arr = np.zeros((n_zones, n_timesteps))
            for z_idx, z in enumerate(zones):
                for t in model.T:
                    try:
                        val = value(var_obj[z, t])
                        arr[z_idx, t] = val if val is not None else 0.0
                    except:
                        arr[z_idx, t] = 0.0
            return arr
        
        var_names = [
            'p_thermal', 'p_nuclear', 'p_solar', 'p_wind',
            'b_charge', 'b_discharge', 'b_soc',
            'pumped_charge', 'pumped_discharge', 'pumped_level',
            'h_release', 'h_level', 'dr_shed',
            'unserved', 'spill_solar', 'spill_wind', 'overgen_spill',
            'u_thermal',
        ]
        
        for name in var_names:
            if hasattr(model, name):
                solution[name] = extract_2d(getattr(model, name))
        
        return solution

    def _select_critical_indices(
        self,
        model,
        solution: Dict[str, np.ndarray],
        targets: Dict,
        K: int,
        window: int = 1
    ) -> TypingSet[Tuple[str, int]]:
        """
        Select critical (zone, time) pairs for repair.
        
        Scoring heuristics:
        1. Slack amount (unserved + overgen_spill)
        2. Thermal tension: p_thermal near capacity but u_thermal=0 (deficit signal)
        3. Thermal tension: p_thermal=0 but u_thermal=1 with overgen (surplus signal)
        
        Returns set of indices to unfix, including temporal neighbors.
        """
        zones = list(model.Z)
        n_timesteps = len(list(model.T))
        
        # Get thermal capacity for tension scoring
        thermal_caps = {}
        if hasattr(model, 'thermal_capacity'):
            for z in zones:
                try:
                    thermal_caps[z] = value(model.thermal_capacity[z])
                except:
                    thermal_caps[z] = 0.0
        
        # Score each (z, t)
        scores = {}
        for z_idx, z in enumerate(zones):
            cap = thermal_caps.get(z, 0.0)
            
            for t in range(n_timesteps):
                score = 0.0
                
                # 1. Slack-based scoring (primary)
                if 'unserved' in solution:
                    score += solution['unserved'][z_idx, t] * 2.0  # Weight slack high
                if 'overgen_spill' in solution:
                    score += solution['overgen_spill'][z_idx, t] * 2.0
                
                # 2. Thermal tension scoring (Fix #5)
                if 'p_thermal' in solution and 'u_thermal' in solution and cap > 0:
                    p_th = solution['p_thermal'][z_idx, t]
                    u_th = solution['u_thermal'][z_idx, t]
                    target_u = targets['u_thermal'].get((z, t), 0.0)
                    
                    # Deficit tension: high demand, thermal off but decoder said on
                    if 'unserved' in solution and solution['unserved'][z_idx, t] > 0:
                        if u_th < 0.5 and target_u > 0.5:
                            score += cap * 0.5  # Could gain capacity by flipping
                    
                    # Surplus tension: thermal on but not needed, decoder said off
                    if 'overgen_spill' in solution and solution['overgen_spill'][z_idx, t] > 0:
                        if u_th > 0.5 and target_u < 0.5:
                            score += p_th * 0.5  # Could reduce by flipping
                    
                    # Capacity tension: p_thermal near max but u=0 somewhere
                    if p_th > 0.8 * cap and u_th < 0.5:
                        score += (cap - p_th) * 0.3
                
                if score > 0:
                    scores[(z, t)] = score
        
        # Sort by score descending
        sorted_pairs = sorted(scores.items(), key=lambda x: -x[1])
        
        # Take top K and expand with temporal window
        critical = set()
        for (z, t), _ in sorted_pairs[:K]:
            for dt in range(-window, window + 1):
                t_new = t + dt
                if 0 <= t_new < n_timesteps:
                    critical.add((z, t_new))
        
        return critical

    def _unfix_subset(self, model, subset: TypingSet[Tuple[str, int]]):
        """
        Unfix u_thermal variables in the given subset.
        NOTE: Never unfix v_thermal_startup - let constraints handle it.
        """
        for (z, t) in subset:
            if hasattr(model, 'u_thermal') and (z, t) in model.u_thermal:
                if model.u_thermal[z, t].is_fixed():
                    model.u_thermal[z, t].unfix()

    def _unfix_subset_all(self, model, subset: TypingSet[Tuple[str, int]]):
        """
        Unfix ALL binary families on the given subset (Option A multi-binary repair).
        Unfixes: u_thermal, b_charge_mode, pumped_charge_mode, dr_active
        NOTE: Never unfix v_thermal_startup - let constraints handle it.
        """
        for (z, t) in subset:
            if hasattr(model, 'u_thermal') and (z, t) in model.u_thermal:
                if model.u_thermal[z, t].is_fixed():
                    model.u_thermal[z, t].unfix()
            if hasattr(model, 'b_charge_mode') and (z, t) in model.b_charge_mode:
                if model.b_charge_mode[z, t].is_fixed():
                    model.b_charge_mode[z, t].unfix()
            if hasattr(model, 'pumped_charge_mode') and (z, t) in model.pumped_charge_mode:
                if model.pumped_charge_mode[z, t].is_fixed():
                    model.pumped_charge_mode[z, t].unfix()
            if hasattr(model, 'dr_active') and (z, t) in model.dr_active:
                if model.dr_active[z, t].is_fixed():
                    model.dr_active[z, t].unfix()

    def _add_deviation_penalty_on_subset(
        self,
        model,
        targets: Dict,
        subset: TypingSet[Tuple[str, int]],
        penalty: float,
        penalize_all_binaries: bool = False
    ):
        """
        Add soft penalty for deviation from decoder on subset.
        
        Args:
            model: Pyomo model
            targets: Dict of target values from decoder
            subset: Set of (zone, time) pairs to penalize
            penalty: Penalty weight (lambda)
            penalize_all_binaries: If True, also penalize battery/pumped/DR modes
        
        Stores model.base_obj_expr for later retrieval of true UC objective.
        
        CRITICAL: Uses Set(dimen=2) for proper Pyomo indexing of (zone, time) tuples.
        """
        # CRITICAL: Store base objective BEFORE modification
        model.base_obj_expr = model.obj.expr
        
        # FIX #1 (BIG): Use Set(dimen=2) for proper Pyomo indexing
        # This ensures Constraint rules receive (z, t) as separate args, not as single tuple
        from pyomo.environ import Set as PyomoSet
        
        # CRITICAL FIX: Filter subset to only include (z,t) that EXIST in u_thermal
        # Otherwise dev_pos/dev_neg are unconstrained and flip_budget is meaningless!
        valid_subset = []
        if hasattr(model, 'u_thermal'):
            u_thermal_keys = set(model.u_thermal.keys())
            for (z, t) in subset:
                if (z, t) in u_thermal_keys:
                    valid_subset.append((z, t))
        
        if not valid_subset:
            # No valid indices - nothing to penalize
            model.penalty_term_expr = 0
            return
        
        model.dev_set = PyomoSet(dimen=2, initialize=valid_subset)
        
        # Create deviation variables indexed by the 2D set
        model.dev_pos = Var(model.dev_set, within=NonNegativeReals)
        model.dev_neg = Var(model.dev_set, within=NonNegativeReals)
        
        # Constraint: u - target = dev_pos - dev_neg
        # All (z,t) in dev_set are guaranteed to exist in u_thermal
        def dev_rule(m, z, t):
            target = targets['u_thermal'].get((z, t), 0.0)
            return m.u_thermal[z, t] - target == m.dev_pos[z, t] - m.dev_neg[z, t]
        
        model.deviation_constr = Constraint(model.dev_set, rule=dev_rule)
        
        # Build penalty term for u_thermal
        penalty_term = penalty * sum(
            model.dev_pos[z, t] + model.dev_neg[z, t]
            for (z, t) in model.dev_set
        )
        
        # Optionally penalize other binary variables too
        if penalize_all_binaries:
            # Battery charge mode - filter to valid keys only
            if hasattr(model, 'b_charge_mode'):
                batt_keys = set(model.b_charge_mode.keys())
                valid_batt = [(z, t) for (z, t) in subset if (z, t) in batt_keys]
                if valid_batt:
                    model.dev_batt_set = PyomoSet(dimen=2, initialize=valid_batt)
                    model.dev_batt_pos = Var(model.dev_batt_set, within=NonNegativeReals)
                    model.dev_batt_neg = Var(model.dev_batt_set, within=NonNegativeReals)
                    
                    def dev_batt_rule(m, z, t):
                        target = targets['b_charge_mode'].get((z, t), 0.0)
                        return m.b_charge_mode[z, t] - target == m.dev_batt_pos[z, t] - m.dev_batt_neg[z, t]
                    
                    model.dev_batt_constr = Constraint(model.dev_batt_set, rule=dev_batt_rule)
                    penalty_term += penalty * 0.5 * sum(
                        model.dev_batt_pos[z, t] + model.dev_batt_neg[z, t]
                        for (z, t) in model.dev_batt_set
                    )
            
            # Pumped charge mode - filter to valid keys only
            if hasattr(model, 'pumped_charge_mode'):
                pump_keys = set(model.pumped_charge_mode.keys())
                valid_pump = [(z, t) for (z, t) in subset if (z, t) in pump_keys]
                if valid_pump:
                    model.dev_pump_set = PyomoSet(dimen=2, initialize=valid_pump)
                    model.dev_pump_pos = Var(model.dev_pump_set, within=NonNegativeReals)
                    model.dev_pump_neg = Var(model.dev_pump_set, within=NonNegativeReals)
                    
                    def dev_pump_rule(m, z, t):
                        target = targets['pumped_charge_mode'].get((z, t), 0.0)
                        return m.pumped_charge_mode[z, t] - target == m.dev_pump_pos[z, t] - m.dev_pump_neg[z, t]
                    
                    model.dev_pump_constr = Constraint(model.dev_pump_set, rule=dev_pump_rule)
                    penalty_term += penalty * 0.5 * sum(
                        model.dev_pump_pos[z, t] + model.dev_pump_neg[z, t]
                        for (z, t) in model.dev_pump_set
                    )
            
            # DR active - filter to valid keys only
            if hasattr(model, 'dr_active'):
                dr_keys = set(model.dr_active.keys())
                valid_dr = [(z, t) for (z, t) in subset if (z, t) in dr_keys]
                if valid_dr:
                    model.dev_dr_set = PyomoSet(dimen=2, initialize=valid_dr)
                    model.dev_dr_pos = Var(model.dev_dr_set, within=NonNegativeReals)
                    model.dev_dr_neg = Var(model.dev_dr_set, within=NonNegativeReals)
                    
                    def dev_dr_rule(m, z, t):
                        target = targets['dr_active'].get((z, t), 0.0)
                        return m.dr_active[z, t] - target == m.dev_dr_pos[z, t] - m.dev_dr_neg[z, t]
                    
                    model.dev_dr_constr = Constraint(model.dev_dr_set, rule=dev_dr_rule)
                    penalty_term += penalty * 0.5 * sum(
                        model.dev_dr_pos[z, t] + model.dev_dr_neg[z, t]
                        for (z, t) in model.dev_dr_set
                    )
        
        # Store penalty term expression for later retrieval
        model.penalty_term_expr = penalty_term
        
        # Deactivate original and create penalized objective
        model.obj.deactivate()
        model.obj_with_penalty = Objective(expr=model.base_obj_expr + penalty_term, sense=minimize)

    def _add_flip_budget_constraint(
        self,
        model,
        targets: Dict,
        subset: TypingSet[Tuple[str, int]],
        budget: int,
        include_all_binaries: bool = False
    ) -> Tuple[bool, int]:
        """
        Add constraint limiting total flips from decoder (Strategy C).
        Uses model.dev_set for proper indexing.
        
        CRITICAL: This constrains sum(|binary - target|) <= budget.
        Since binaries are {0,1} and target in {0,1}, each flip costs exactly 1.
        
        Args:
            include_all_binaries: If True, include batt/pump/dr deviation vars in budget
        
        Returns: (success, n_constrained_vars)
        """
        if not (hasattr(model, 'dev_set') and hasattr(model, 'dev_pos') and hasattr(model, 'dev_neg')):
            return (False, 0)
        
        n_dev_vars = len(list(model.dev_set))
        if n_dev_vars == 0:
            return (False, 0)
        
        # Start with u_thermal deviations
        budget_expr = sum(
            model.dev_pos[z, t] + model.dev_neg[z, t]
            for (z, t) in model.dev_set
        )
        
        # Include other binary families if requested
        if include_all_binaries:
            if hasattr(model, 'dev_batt_set') and hasattr(model, 'dev_batt_pos'):
                budget_expr += sum(
                    model.dev_batt_pos[z, t] + model.dev_batt_neg[z, t]
                    for (z, t) in model.dev_batt_set
                )
                n_dev_vars += len(list(model.dev_batt_set))
            
            if hasattr(model, 'dev_pump_set') and hasattr(model, 'dev_pump_pos'):
                budget_expr += sum(
                    model.dev_pump_pos[z, t] + model.dev_pump_neg[z, t]
                    for (z, t) in model.dev_pump_set
                )
                n_dev_vars += len(list(model.dev_pump_set))
            
            if hasattr(model, 'dev_dr_set') and hasattr(model, 'dev_dr_pos'):
                budget_expr += sum(
                    model.dev_dr_pos[z, t] + model.dev_dr_neg[z, t]
                    for (z, t) in model.dev_dr_set
                )
                n_dev_vars += len(list(model.dev_dr_set))
        
        model.flip_budget = Constraint(expr=budget_expr <= budget)
        
        # Verify constraint was added and is active
        if hasattr(model, 'flip_budget') and model.flip_budget.active:
            return (True, n_dev_vars)
        return (False, 0)

    def _solve_with_timeout(self, model, time_limit: float) -> Tuple[Any, float, bool]:
        start = time.time()

        if self.solver_name == 'appsi_highs':
            self.solver.config.time_limit = float(time_limit)
            # ⚠️ important : éviter l’exception côté APPSI
            self.solver.config.load_solution = False

        elif 'highs' in self.solver_name.lower():
            self.solver.options['time_limit'] = float(time_limit)
        elif 'glpk' in self.solver_name.lower():
            self.solver.options['tmlim'] = int(time_limit)
        elif 'cbc' in self.solver_name.lower():
            self.solver.options['seconds'] = float(time_limit)

        results = self.solver.solve(model, tee=self.verbose, load_solutions=False)
        elapsed = time.time() - start

        tc = results.solver.termination_condition
        ok = tc in (TerminationCondition.optimal,
                    TerminationCondition.feasible,
                TerminationCondition.maxTimeLimit)

        has_solution = False
        if ok and len(results.solution) > 0:
            # ✅ charge seulement si une solution existe
            model.solutions.load_from(results)
            has_solution = True

        return results, elapsed, has_solution

    def _get_true_objective(self, model) -> float:
        """
        Get the TRUE UC objective value (without penalty terms).
        Always use this instead of value(model.obj) after adding penalties.
        """
        if hasattr(model, 'base_obj_expr'):
            return value(model.base_obj_expr)
        else:
            return value(model.obj)

    def _get_penalty_value(self, model) -> float:
        """Get the penalty term value for analysis."""
        if hasattr(model, 'penalty_term_expr'):
            try:
                return value(model.penalty_term_expr)
            except:
                return 0.0
        return 0.0

    def _calculate_deviation(self, model, targets: Dict) -> Tuple[float, int]:
        """Calculate total deviation from decoder and count flips (u_thermal only)."""
        deviation = 0.0
        n_flips = 0
        
        for (z, t), target in targets['u_thermal'].items():
            if hasattr(model, 'u_thermal') and (z, t) in model.u_thermal:
                try:
                    actual = value(model.u_thermal[z, t])
                    if actual is not None:
                        diff = abs(actual - target)
                        deviation += diff
                        if diff > 0.5:
                            n_flips += 1
                except:
                    pass
        
        return deviation, n_flips

    def _calculate_deviation_relaxed(self, model, targets: Dict) -> Dict[str, float]:
        """
        Calculate metrics for relaxed (continuous) solutions.
        Returns: {
            'l1_deviation': sum(|u - target|) for all binaries,
            'rounded_flips': count where round(u) != target
        }
        For Stage 4 where binaries are relaxed to continuous.
        """
        l1_dev = 0.0
        rounded_flips = 0
        
        # u_thermal
        for (z, t), target in targets['u_thermal'].items():
            if hasattr(model, 'u_thermal') and (z, t) in model.u_thermal:
                try:
                    actual = value(model.u_thermal[z, t])
                    if actual is not None:
                        l1_dev += abs(actual - target)
                        rounded = 1.0 if actual >= 0.5 else 0.0
                        if rounded != target:
                            rounded_flips += 1
                except:
                    pass
        
        # b_charge_mode
        for (z, t), target in targets['b_charge_mode'].items():
            if hasattr(model, 'b_charge_mode') and (z, t) in model.b_charge_mode:
                try:
                    actual = value(model.b_charge_mode[z, t])
                    if actual is not None:
                        l1_dev += abs(actual - target)
                        rounded = 1.0 if actual >= 0.5 else 0.0
                        if rounded != target:
                            rounded_flips += 1
                except:
                    pass
        
        # pumped_charge_mode
        for (z, t), target in targets['pumped_charge_mode'].items():
            if hasattr(model, 'pumped_charge_mode') and (z, t) in model.pumped_charge_mode:
                try:
                    actual = value(model.pumped_charge_mode[z, t])
                    if actual is not None:
                        l1_dev += abs(actual - target)
                        rounded = 1.0 if actual >= 0.5 else 0.0
                        if rounded != target:
                            rounded_flips += 1
                except:
                    pass
        
        # dr_active
        for (z, t), target in targets['dr_active'].items():
            if hasattr(model, 'dr_active') and (z, t) in model.dr_active:
                try:
                    actual = value(model.dr_active[z, t])
                    if actual is not None:
                        l1_dev += abs(actual - target)
                        rounded = 1.0 if actual >= 0.5 else 0.0
                        if rounded != target:
                            rounded_flips += 1
                except:
                    pass
        
        return {'l1_deviation': l1_dev, 'rounded_flips': rounded_flips}

    def _round_solution_to_targets(self, model, targets: Dict) -> Dict:
        """
        Round relaxed binary solution (u>=0.5 → 1, else 0) to create new targets.
        Used for Stage 5 round-and-refix approach.
        """
        from copy import deepcopy
        rounded = deepcopy(targets)
        
        # u_thermal
        for (z, t) in rounded['u_thermal']:
            if hasattr(model, 'u_thermal') and (z, t) in model.u_thermal:
                try:
                    u = value(model.u_thermal[z, t])
                    rounded['u_thermal'][(z, t)] = 1.0 if (u is not None and u >= 0.5) else 0.0
                except:
                    pass
        
        # b_charge_mode
        for (z, t) in rounded['b_charge_mode']:
            if hasattr(model, 'b_charge_mode') and (z, t) in model.b_charge_mode:
                try:
                    u = value(model.b_charge_mode[z, t])
                    rounded['b_charge_mode'][(z, t)] = 1.0 if (u is not None and u >= 0.5) else 0.0
                except:
                    pass
        
        # pumped_charge_mode
        for (z, t) in rounded['pumped_charge_mode']:
            if hasattr(model, 'pumped_charge_mode') and (z, t) in model.pumped_charge_mode:
                try:
                    u = value(model.pumped_charge_mode[z, t])
                    rounded['pumped_charge_mode'][(z, t)] = 1.0 if (u is not None and u >= 0.5) else 0.0
                except:
                    pass
        
        # dr_active
        for (z, t) in rounded['dr_active']:
            if hasattr(model, 'dr_active') and (z, t) in model.dr_active:
                try:
                    u = value(model.dr_active[z, t])
                    rounded['dr_active'][(z, t)] = 1.0 if (u is not None and u >= 0.5) else 0.0
                except:
                    pass
        
        return rounded

    def _calculate_deviation_all(self, model, targets: Dict) -> Dict[str, int]:
        """
        Calculate flips per binary family for detailed diagnostics.
        Returns dict with keys: u_thermal, b_charge_mode, pumped_charge_mode, dr_active
        """
        flips = {'u_thermal': 0, 'b_charge_mode': 0, 'pumped_charge_mode': 0, 'dr_active': 0}
        
        # u_thermal
        for (z, t), target in targets['u_thermal'].items():
            if hasattr(model, 'u_thermal') and (z, t) in model.u_thermal:
                try:
                    actual = value(model.u_thermal[z, t])
                    if actual is not None and abs(actual - target) > 0.5:
                        flips['u_thermal'] += 1
                except:
                    pass
        
        # b_charge_mode
        for (z, t), target in targets['b_charge_mode'].items():
            if hasattr(model, 'b_charge_mode') and (z, t) in model.b_charge_mode:
                try:
                    actual = value(model.b_charge_mode[z, t])
                    if actual is not None and abs(actual - target) > 0.5:
                        flips['b_charge_mode'] += 1
                except:
                    pass
        
        # pumped_charge_mode
        for (z, t), target in targets['pumped_charge_mode'].items():
            if hasattr(model, 'pumped_charge_mode') and (z, t) in model.pumped_charge_mode:
                try:
                    actual = value(model.pumped_charge_mode[z, t])
                    if actual is not None and abs(actual - target) > 0.5:
                        flips['pumped_charge_mode'] += 1
                except:
                    pass
        
        # dr_active
        for (z, t), target in targets['dr_active'].items():
            if hasattr(model, 'dr_active') and (z, t) in model.dr_active:
                try:
                    actual = value(model.dr_active[z, t])
                    if actual is not None and abs(actual - target) > 0.5:
                        flips['dr_active'] += 1
                except:
                    pass
        
        return flips

    def solve(self, scenario_id: str, decoder_output: torch.Tensor) -> TwoStageResult:
        """
        Solve with two-stage approach: hard-fix LP → soft repair fallback.
        
        Returns TwoStageResult with comprehensive logging.
        """
        total_start = time.time()
        
        result = TwoStageResult(
            scenario_id=scenario_id,
            status='pending',
            stage_used=SolveStage.FAILED,
            objective_value=float('inf'),
            solve_time=0.0,
            continuous_vars={},
        )
        
        # Find scenario file
        scenario_path = self._find_scenario_path(scenario_id)
        if scenario_path is None:
            result.status = 'error'
            result.message = f"File not found: {scenario_id}"
            return result
        
        try:
            # Load scenario data
            scenario_data = load_scenario_data(Path(scenario_path))
            
            # ================================================================
            # STAGE 1: HARD-FIX LP
            # ================================================================
            if self.verbose:
                print(f"  Stage 1: Hard-fix LP...")
            
            model1 = build_uc_model(scenario_data, enable_duals=False)
            zone_names = list(model1.Z)
            n_timesteps = len(list(model1.T))
            
            targets = self._get_decoder_targets(decoder_output, zone_names, n_timesteps)
            
            # Fix #2: Only fix u_thermal in Stage 1 for robustness
            # Don't fix startup, battery modes, etc. - let MILP handle them
            self._fix_all_binaries(model1, targets, only_u_thermal=False)
            
            # Relax integrality
            TransformationFactory('core.relax_integer_vars').apply_to(model1)
            # Count remaining unfixed binaries (diagnostic only - don't force fix)
            unfixed_count = self._count_unfixed_binaries(model1)
            result.n_unfixed_binaries_stage1 = unfixed_count
            if self.verbose and unfixed_count > 0:
                unfixed_names = self._list_unfixed_binaries(model1)
                print(f"    Stage 1 has {unfixed_count} unfixed binaries (MILP): {unfixed_names[:5]}...")
            
            self._relax_slack_bounds(model1)
            
            res1, time1, has1 = self._solve_with_timeout(model1, self.time_limits['hard_fix'])
            result.time_hard_fix = time1
            
            tc1 = res1.solver.termination_condition
            # Only proceed if we have a loaded solution
            if has1:
                slack1 = self._extract_slack(model1)
                result.slack_hard_fix = slack1
                
                if slack1 <= self.slack_tol_mwh:
                    # SUCCESS - hard fix is good enough
                    result.status = 'optimal'
                    result.stage_used = SolveStage.HARD_FIX
                    result.objective_value = self._get_true_objective(model1)  # Fix #1
                    result.solve_time = time.time() - total_start
                    result.continuous_vars = self._extract_solution(model1)
                    result.slack_used = slack1
                    result.decoder_deviation, result.n_flips = 0.0, 0
                    result.n_unfixed = 0
                    result.message = f"Hard-fix OK: slack={slack1:.1f} MWh"
                    return result
                
                # Need fallback - extract solution for critical index selection
                sol1 = self._extract_solution(model1)
            else:
                # Hard-fix failed entirely (no solution loaded)
                sol1 = None
                result.slack_hard_fix = float('inf')
            
            # ================================================================
            # STAGE 2: REPAIR-20 (K=20 critical variables)
            # ================================================================
            if self.verbose:
                print(f"  Stage 2: Repair-20...")
            
            model2 = build_uc_model(scenario_data, enable_duals=False)
            targets2 = self._get_decoder_targets(decoder_output, zone_names, n_timesteps)
            self._fix_all_binaries(model2, targets2)
            self._relax_slack_bounds(model2)
            
            # Dynamic flip budget: 10% of total binaries
            n_binaries = self._count_total_binaries(model2)
            flip_budget_2 = max(1, int(n_binaries * 0.10))
            if self.verbose:
                print(f"    n_binaries={n_binaries}, flip_budget_2={flip_budget_2} (10%)")
            
            # Select critical indices from hard-fix solution (Fix #5: pass targets)
            if sol1 is not None:
                critical_20 = self._select_critical_indices(model2, sol1, targets2, K=20, window=self.temporal_window)
            else:
                # Fallback: just use first few timesteps for each zone
                critical_20 = set()
                for z in zone_names[:5]:
                    for t in range(min(6, n_timesteps)):
                        critical_20.add((z, t))
            
            # Option A: Unfix ALL binary families on critical subset
            self._unfix_subset_all(model2, critical_20)
            self._add_deviation_penalty_on_subset(model2, targets2, critical_20, self.deviation_penalty, penalize_all_binaries=True)
            self._add_flip_budget_constraint(model2, targets2, critical_20, flip_budget_2, include_all_binaries=True)
            
            res2, time2, has2 = self._solve_with_timeout(model2, self.time_limits['repair_20'])
            result.time_repair_20 = time2
            
            tc2 = res2.solver.termination_condition
            if has2:
                slack2 = self._extract_slack(model2)
                result.slack_repair_20 = slack2
                
                if slack2 <= self.slack_tol_mwh:
                    # SUCCESS with repair-20
                    result.status = 'optimal'
                    result.stage_used = SolveStage.REPAIR_20
                    result.objective_value = self._get_true_objective(model2)  # Fix #1: true UC obj
                    result.deviation_penalty_value = self._get_penalty_value(model2)
                    result.solve_time = time.time() - total_start
                    result.continuous_vars = self._extract_solution(model2)
                    result.slack_used = slack2
                    result.decoder_deviation, result.n_flips = self._calculate_deviation(model2, targets2)
                    result.n_unfixed = len(critical_20)
                    result.critical_indices = list(critical_20)
                    result.message = f"Repair-20 OK: slack={slack2:.1f}, flips={result.n_flips}"
                    return result
                
                sol2 = self._extract_solution(model2)
            else:
                sol2 = sol1  # Use previous solution for next stage
                result.slack_repair_20 = float('inf')
            
            # ================================================================
            # STAGE 3: REPAIR-100 (K=100 critical variables)
            # ================================================================
            if self.verbose:
                print(f"  Stage 3: Repair-100...")
            
            model3 = build_uc_model(scenario_data, enable_duals=False)
            targets3 = self._get_decoder_targets(decoder_output, zone_names, n_timesteps)
            self._fix_all_binaries(model3, targets3)
            self._relax_slack_bounds(model3)
            
            # Dynamic flip budget: 50% of total binaries
            n_binaries_3 = self._count_total_binaries(model3)
            flip_budget_3 = max(1, int(n_binaries_3 * 0.50))
            if self.verbose:
                print(f"    n_binaries={n_binaries_3}, flip_budget_3={flip_budget_3} (50%)")
            
            ref_sol = sol2 if sol2 is not None else sol1
            if ref_sol is not None:
                critical_100 = self._select_critical_indices(model3, ref_sol, targets3, K=100, window=self.temporal_window)
            else:
                critical_100 = set()
                for z in zone_names[:20]:
                    for t in range(n_timesteps):
                        critical_100.add((z, t))
            
            # Option A: Unfix ALL binary families on critical subset
            self._unfix_subset_all(model3, critical_100)
            self._add_deviation_penalty_on_subset(model3, targets3, critical_100, self.deviation_penalty, penalize_all_binaries=True)
            self._add_flip_budget_constraint(model3, targets3, critical_100, flip_budget_3, include_all_binaries=True)
            
            res3, time3, has3 = self._solve_with_timeout(model3, self.time_limits['repair_100'])
            result.time_repair_100 = time3
            
            tc3 = res3.solver.termination_condition
            if has3:
                slack3 = self._extract_slack(model3)
                result.slack_repair_100 = slack3
                
                # Relaxed acceptance: slack <= 50 MWh OR significant improvement
                if slack3 <= 50.0 or slack3 < result.slack_hard_fix * 0.5:
                    # SUCCESS or significant improvement
                    result.status = 'optimal'
                    result.stage_used = SolveStage.REPAIR_100
                    result.objective_value = self._get_true_objective(model3)  # Fix #1: true UC obj
                    result.deviation_penalty_value = self._get_penalty_value(model3)
                    result.solve_time = time.time() - total_start
                    result.continuous_vars = self._extract_solution(model3)
                    result.slack_used = slack3
                    result.decoder_deviation, result.n_flips = self._calculate_deviation(model3, targets3)
                    result.n_unfixed = len(critical_100)
                    result.critical_indices = list(critical_100)[:50]  # Limit for logging
                    result.message = f"Repair-100 OK: slack={slack3:.1f}, flips={result.n_flips}"
                    return result
            else:
                result.slack_repair_100 = float('inf')
            
            # ================================================================
            # STAGE 4: FULL SOFT (all variables unfixed with penalty)
            # ================================================================
            if self.verbose:
                print(f"  Stage 4: Full soft...")
            
            model4 = build_uc_model(scenario_data, enable_duals=False)
            targets4 = self._get_decoder_targets(decoder_output, zone_names, n_timesteps)
            self._relax_slack_bounds(model4)
            
            # Add soft penalty on ALL (zone, time) pairs
            all_zt = set()
            for z in zone_names:
                for t in range(n_timesteps):
                    all_zt.add((z, t))
            
            # Fix #6: Penalize ALL binary types (not just u_thermal) for controlled full_soft
            self._add_deviation_penalty_on_subset(
                model4, targets4, all_zt, 
                self.deviation_penalty * 0.1,
                penalize_all_binaries=True  # Fix #6: control battery/pumped/DR too
            )

            TransformationFactory('core.relax_integer_vars').apply_to(model4)

            # Stage 4: NO flip budget - let solver find best feasible solution
            if self.verbose:
                print(f"    No flip budget (full freedom)")
            
            res4, time4, has4 = self._solve_with_timeout(model4, self.time_limits['full_soft'])
            result.time_full_soft = time4
            
            tc4 = res4.solver.termination_condition
            if has4:
                slack4 = self._extract_slack(model4)
                result.slack_full_soft = slack4
                
                # Stage 4 uses relaxed binaries - use proper metrics
                relaxed_metrics = self._calculate_deviation_relaxed(model4, targets4)
                if self.verbose:
                    print(f"    Stage 4 (relaxed): slack={slack4:.1f}, L1_dev={relaxed_metrics['l1_deviation']:.2f}, "
                          f"rounded_flips={relaxed_metrics['rounded_flips']}")
                
                # ================================================================
                # STAGE 5: ROUND & REFIX (warm-start from Stage 4)
                # ================================================================
                if self.verbose:
                    print(f"  Stage 5: Round & Refix...")
                
                # Round Stage 4 solution to discrete targets
                rounded_targets = self._round_solution_to_targets(model4, targets4)
                
                # Build new model with rounded binaries fixed
                model5 = build_uc_model(scenario_data, enable_duals=False)
                self._fix_all_binaries(model5, rounded_targets, only_u_thermal=False)
                TransformationFactory('core.relax_integer_vars').apply_to(model5)
                self._relax_slack_bounds(model5)
                
                res5, time5, has5 = self._solve_with_timeout(model5, self.time_limits['hard_fix'])
                
                if has5:
                    slack5 = self._extract_slack(model5)
                    if self.verbose:
                        print(f"    Stage 5: slack={slack5:.1f} (vs Stage 4: {slack4:.1f})")
                    
                    # Use Stage 5 if it's better
                    if slack5 < slack4:
                        if self.verbose:
                            print(f"    → Using Stage 5 (better slack)")
                        result.status = 'feasible_full_soft'
                        result.stage_used = SolveStage.FULL_SOFT
                        result.objective_value = self._get_true_objective(model5)
                        result.solve_time = time.time() - total_start
                        result.continuous_vars = self._extract_solution(model5)
                        result.slack_used = slack5
                        result.decoder_deviation, result.n_flips = self._calculate_deviation(model5, rounded_targets)
                        result.n_unfixed = 0  # All fixed in Stage 5
                        result.message = f"Round&Refix: slack={slack5:.1f} (was {slack4:.1f}), flips={result.n_flips}"
                        return result
                    else:
                        if self.verbose:
                            print(f"    → Keeping Stage 4 (slack5={slack5:.1f} >= slack4={slack4:.1f})")
                
                # Fall back to Stage 4 result
                result.status = 'feasible_full_soft'
                result.stage_used = SolveStage.FULL_SOFT
                result.objective_value = self._get_true_objective(model4)
                result.deviation_penalty_value = self._get_penalty_value(model4)
                result.solve_time = time.time() - total_start
                result.continuous_vars = self._extract_solution(model4)
                result.slack_used = slack4
                result.decoder_deviation = relaxed_metrics['l1_deviation']
                result.n_flips = relaxed_metrics['rounded_flips']
                result.n_unfixed = len(all_zt)
                result.message = f"Full soft (relaxed): slack={slack4:.1f}, L1={relaxed_metrics['l1_deviation']:.1f}, rounded_flips={relaxed_metrics['rounded_flips']}"
                return result
            
            # All stages failed
            result.status = 'infeasible'
            result.stage_used = SolveStage.FAILED
            result.solve_time = time.time() - total_start
            result.message = "All stages failed"
            return result
            
        except Exception as e:
            import traceback
            result.status = 'error'
            result.stage_used = SolveStage.FAILED
            result.solve_time = time.time() - total_start
            result.message = f"Error: {str(e)[:200]}"
            if self.verbose:
                traceback.print_exc()
            return result


def run_lp_worker_batch_two_stage(
    lp_worker: LPWorkerTwoStage,
    scenario_ids: List[str],
    decoder_outputs: List[torch.Tensor],
    max_scenarios: int = None
) -> List[TwoStageResult]:
    """
    Run two-stage LP worker on a batch of scenarios.
    
    Returns list of TwoStageResult with detailed per-stage logging.
    """
    from tqdm.auto import tqdm
    
    results = []
    n = len(scenario_ids) if max_scenarios is None else min(len(scenario_ids), max_scenarios)
    
    print(f"\n{'='*80}")
    print(f"LP WORKER TWO-STAGE - Processing {n} scenarios")
    print(f"{'='*80}\n")
    
    stage_counts = {s: 0 for s in SolveStage}
    
    for i in tqdm(range(n), desc="LP Worker Two-Stage"):
        sc_id = scenario_ids[i]
        decoder_out = decoder_outputs[i]
        
        if decoder_out is None:
            results.append(TwoStageResult(
                scenario_id=sc_id,
                status='skipped',
                stage_used=SolveStage.FAILED,
                objective_value=float('inf'),
                solve_time=0.0,
                continuous_vars={},
                message="Decoder output is None"
            ))
            continue
        
        result = lp_worker.solve(sc_id, decoder_out)
        results.append(result)
        stage_counts[result.stage_used] += 1
        
        if i <= 50:
            emoji = '✓' if result.status in ['optimal', 'feasible_full_soft'] else '✗'
            print(f"  {emoji} {sc_id}: {result.stage_used.value} | "
                  f"obj={result.objective_value:.0f} | slack={result.slack_used:.1f} | "
                  f"flips={result.n_flips} | {result.solve_time:.2f}s")
    
    # Summary
    print(f"\n{'='*80}")
    print(f"LP WORKER TWO-STAGE SUMMARY")
    print(f"{'='*80}")
    print(f"  Total scenarios:  {len(results)}")
    print(f"\n  Stage distribution:")
    for stage in SolveStage:
        pct = 100 * stage_counts[stage] / len(results) if results else 0
        print(f"    {stage.value:15s}: {stage_counts[stage]:4d} ({pct:5.1f}%)")
    
    success_results = [r for r in results if r.status in ['optimal', 'feasible_full_soft']]
    if success_results:
        obj_vals = [r.objective_value for r in success_results]
        slacks = [r.slack_used for r in success_results]
        flips = [r.n_flips for r in success_results]
        times = [r.solve_time for r in success_results]
        
        print(f"\n  Performance (n={len(success_results)} successful):")
        print(f"    Objective:  mean={np.mean(obj_vals):.0f}, min={np.min(obj_vals):.0f}, max={np.max(obj_vals):.0f}")
        print(f"    Slack:      mean={np.mean(slacks):.1f}, max={np.max(slacks):.1f} MWh")
        print(f"    Flips:      mean={np.mean(flips):.1f}, max={np.max(flips)} ")
        print(f"    Time:       mean={np.mean(times):.2f}s, max={np.max(times):.2f}s")
    
    return results
