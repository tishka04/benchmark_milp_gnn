"""
Monolithic MILP solver for UC+Dispatch.

Uses Pyomo with HiGHS solver for clean formulation and flexibility.
"""
import numpy as np
import time
from typing import Optional
import pyomo.environ as pyo

try:
    from .data_models import UCInstance, UCSolution, ExperimentConfig
    from .dispatch_solver import calculate_total_cost, verify_solution_feasibility
except ImportError:
    from data_models import UCInstance, UCSolution, ExperimentConfig
    from dispatch_solver import calculate_total_cost, verify_solution_feasibility


class MILPSolver:
    """
    Monolithic MILP solver for Unit Commitment + Dispatch.
    
    Formulation:
        Variables: u[i,t] ∈ {0,1}, P[i,t] ∈ ℝ+
        
        Minimize: Σ_i,t (marginal_cost[i] * P[i,t]) + Σ_i,t (startup_cost[i] * s[i,t])
        
        Subject to:
            Σ_i P[i,t] = demand[t]  ∀t
            P[i,t] >= p_min[i] * u[i,t]  ∀i,t
            P[i,t] <= p_max[i] * u[i,t]  ∀i,t
            s[i,t] >= u[i,t] - u[i,t-1]  ∀i,t  (startup detection)
    """
    
    def __init__(self, config: ExperimentConfig):
        self.config = config
        self.solver = config.milp_solver
        self.time_limit = None
        self.mip_gap = config.mip_gap_tolerance
    
    def solve(
        self,
        instance: UCInstance,
        time_limit: Optional[float] = None,
        checkpoint_times: Optional[list] = None
    ) -> UCSolution:
        """
        Solve the UC+Dispatch problem using monolithic MILP with Pyomo.
        
        Args:
            instance: Problem instance
            time_limit: Maximum solve time (seconds)
            checkpoint_times: Times to log intermediate solutions
        
        Returns:
            UCSolution: Solution object
        """
        start_time = time.time()
        
        N = instance.n_units
        T = instance.n_periods
        
        print(f"\n{'='*80}")
        print(f"MILP SOLVER (Pyomo + HiGHS): {instance.scale_name}")
        print(f"{'='*80}")
        print(f"Units: {N}, Periods: {T}, Variables: {3*N*T} ({2*N*T} binary, {N*T} continuous)")
        
        # Build Pyomo model
        print("Building Pyomo model...")
        model = pyo.ConcreteModel()
        
        # Sets
        model.Units = pyo.RangeSet(0, N-1)
        model.Periods = pyo.RangeSet(0, T-1)
        
        # Variables
        model.u = pyo.Var(model.Units, model.Periods, domain=pyo.Binary)  # Commitment
        model.P = pyo.Var(model.Units, model.Periods, domain=pyo.NonNegativeReals)  # Power
        model.s = pyo.Var(model.Units, model.Periods, domain=pyo.Binary)  # Startup
        
        # Objective: minimize fuel cost + startup cost
        def obj_rule(m):
            fuel_cost = sum(
                instance.marginal_cost[i] * m.P[i, t]
                for i in m.Units for t in m.Periods
            )
            startup_cost = sum(
                instance.startup_cost[i] * m.s[i, t]
                for i in m.Units for t in m.Periods
            )
            return fuel_cost + startup_cost
        
        model.objective = pyo.Objective(rule=obj_rule, sense=pyo.minimize)
        
        # Constraints
        print("Adding constraints...")
        
        # 1. Demand balance: Σ_i P[i,t] = demand[t]
        def demand_balance_rule(m, t):
            return sum(m.P[i, t] for i in m.Units) == instance.demand[t]
        
        model.demand_balance = pyo.Constraint(model.Periods, rule=demand_balance_rule)
        
        # 2. Minimum generation: P[i,t] >= p_min[i] * u[i,t]
        def min_generation_rule(m, i, t):
            return m.P[i, t] >= instance.p_min[i] * m.u[i, t]
        
        model.min_generation = pyo.Constraint(
            model.Units, model.Periods, rule=min_generation_rule
        )
        
        # 3. Maximum generation: P[i,t] <= p_max[i] * u[i,t]
        def max_generation_rule(m, i, t):
            return m.P[i, t] <= instance.p_max[i] * m.u[i, t]
        
        model.max_generation = pyo.Constraint(
            model.Units, model.Periods, rule=max_generation_rule
        )
        
        # 4. Startup detection: s[i,t] >= u[i,t] - u[i,t-1]
        def startup_detection_rule(m, i, t):
            if t == 0:
                # Compare with initial status
                return m.s[i, t] >= m.u[i, t] - instance.initial_status[i]
            else:
                return m.s[i, t] >= m.u[i, t] - m.u[i, t-1]
        
        model.startup_detection = pyo.Constraint(
            model.Units, model.Periods, rule=startup_detection_rule
        )
        
        n_constraints = (
            len(list(model.demand_balance)) +
            len(list(model.min_generation)) +
            len(list(model.max_generation)) +
            len(list(model.startup_detection))
        )
        print(f"Constraints: {n_constraints}")
        
        # Solve with HiGHS
        print(f"Solving with HiGHS (time limit: {time_limit if time_limit else 3600}s)...")
        solve_start = time.time()
        
        # Create solver
        solver = pyo.SolverFactory('appsi_highs')
        
        # Set solver options
        solver.options['mip_rel_gap'] = self.mip_gap
        solver.options['time_limit'] = time_limit if time_limit else 3600
        solver.options['threads'] = self.config.milp_threads
        
        # Solve
        results = solver.solve(model, tee=True)
        
        solve_time = time.time() - solve_start
        
        # Extract solution
        termination = results.solver.termination_condition
        
        if termination in [pyo.TerminationCondition.optimal, 
                          pyo.TerminationCondition.feasible,
                          pyo.TerminationCondition.maxTimeLimit]:
            
            # Extract values
            commitment = np.zeros((N, T), dtype=int)
            power = np.zeros((N, T))
            
            for i in range(N):
                for t in range(T):
                    commitment[i, t] = int(round(pyo.value(model.u[i, t])))
                    power[i, t] = pyo.value(model.P[i, t])
            
            # Calculate costs using canonical function
            total_cost, fuel_cost, startup_cost = calculate_total_cost(
                instance, commitment, power
            )
            
            # Verify feasibility
            is_feasible, violations = verify_solution_feasibility(
                instance, commitment, power
            )
            
            if not is_feasible:
                print(f"WARNING: Solution has violations:")
                for v in violations[:5]:
                    print(f"  - {v}")
            
            # Determine optimality
            optimal = (termination == pyo.TerminationCondition.optimal)
            
            # Get MIP gap if available
            mip_gap = None
            best_bound = None
            
            if hasattr(results.problem, 'lower_bound') and hasattr(results.problem, 'upper_bound'):
                lb = results.problem.lower_bound
                ub = results.problem.upper_bound
                if lb is not None and ub is not None and abs(lb) > 1e-6:
                    mip_gap = abs(ub - lb) / abs(lb)
                    best_bound = lb
            
            print(f"\n{'='*80}")
            print(f"MILP SOLUTION FOUND")
            print(f"{'='*80}")
            print(f"Status: {termination}")
            print(f"Time: {solve_time:.2f}s")
            print(f"Cost: ${total_cost:,.2f}")
            print(f"  Fuel: ${fuel_cost:,.2f}")
            print(f"  Startup: ${startup_cost:,.2f}")
            print(f"Optimal: {optimal}")
            if mip_gap is not None:
                print(f"MIP Gap: {mip_gap*100:.2f}%")
            if best_bound is not None:
                print(f"Best Bound: ${best_bound:,.2f}")
            print(f"Feasible: {is_feasible}")
            
            return UCSolution(
                commitment=commitment,
                power=power,
                total_cost=total_cost,
                fuel_cost=fuel_cost,
                startup_cost=startup_cost,
                solve_time=solve_time,
                method="MILP-Pyomo",
                feasible=is_feasible,
                optimal=optimal,
                mip_gap=mip_gap,
                best_bound=best_bound
            )
        
        else:
            print(f"\n{'='*80}")
            print(f"MILP FAILED")
            print(f"{'='*80}")
            print(f"Termination condition: {termination}")
            
            return UCSolution(
                commitment=np.zeros((N, T), dtype=int),
                power=np.zeros((N, T)),
                total_cost=np.inf,
                fuel_cost=0.0,
                startup_cost=0.0,
                solve_time=solve_time,
                method="MILP-Pyomo",
                feasible=False,
                optimal=False
            )
