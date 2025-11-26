"""
Hybrid solver: Thermodynamic sampling + Classical dispatch.

Uses thrml for Ising-based commitment sampling, then solves dispatch
using the same canonical dispatch solver as MILP.
"""
import numpy as np
import jax
import jax.numpy as jnp
import time
from typing import Optional, Tuple
from thrml import SpinNode, Block, SamplingSchedule, sample_states
from thrml.models import IsingEBM, IsingSamplingProgram, hinton_init

try:
    from .data_models import UCInstance, UCSolution, ExperimentConfig
    from .dispatch_solver import (
        solve_dispatch_given_commitment,
        calculate_total_cost,
        verify_solution_feasibility
    )
except ImportError:
    from data_models import UCInstance, UCSolution, ExperimentConfig
    from dispatch_solver import (
        solve_dispatch_given_commitment,
        calculate_total_cost,
        verify_solution_feasibility
    )


class HybridSolver:
    """
    Hybrid Thermodynamic + Classical Solver.
    
    Architecture:
    1. Discrete Manager: Thermodynamic Ising sampling for commitment u[i,t]
    2. Continuous Worker: Classical dispatch solver (same as MILP uses)
    
    This ensures perfect cost alignment with MILP.
    """
    
    def __init__(self, config: ExperimentConfig):
        self.config = config
        self.n_samples_per_period = config.hybrid_n_samples_per_period
        self.n_seeds = config.hybrid_n_seeds
        self.temperature = config.hybrid_temperature
        self.warmup = config.hybrid_warmup
        self.steps_per_sample = config.hybrid_steps_per_sample
    
    def solve(
        self,
        instance: UCInstance,
        time_limit: Optional[float] = None,
        checkpoint_times: Optional[list] = None
    ) -> UCSolution:
        """
        Solve UC+Dispatch using hybrid thermodynamic approach.
        
        Strategy:
        - Sample commitment configurations period-by-period
        - For each candidate commitment, solve dispatch using canonical solver
        - Track best feasible solution found
        
        Args:
            instance: Problem instance
            time_limit: Maximum solve time (seconds)
            checkpoint_times: Times to log intermediate solutions
        
        Returns:
            UCSolution: Best solution found
        """
        start_time = time.time()
        
        N = instance.n_units
        T = instance.n_periods
        
        print(f"\n{'='*80}")
        print(f"HYBRID SOLVER: {instance.scale_name}")
        print(f"{'='*80}")
        print(f"Units: {N}, Periods: {T}")
        print(f"Sampling: {self.n_samples_per_period} candidates/period × {self.n_seeds} seeds")
        
        # Initialize best solution
        best_commitment = np.zeros((N, T), dtype=int)
        best_power = np.zeros((N, T))
        best_cost = np.inf
        time_to_first_feasible = None
        n_evaluated = 0
        
        # Checkpoint tracking
        checkpoint_times_list = []
        checkpoint_costs_list = []
        
        # Solve period by period
        commitment_full = np.zeros((N, T), dtype=int)
        power_full = np.zeros((N, T))
        total_cost_accumulated = 0.0
        
        for t in range(T):
            if time_limit and (time.time() - start_time) > time_limit:
                print(f"\nTime limit reached at period {t}/{T}")
                break
            
            # Sample commitment for this period
            u_candidates = self._sample_commitment_for_period(
                instance, t, time.time() - start_time, time_limit
            )
            
            # Evaluate each candidate
            best_u_t = None
            best_dispatch_t = None
            best_cost_t = np.inf
            
            for u_t in u_candidates:
                n_evaluated += 1
                
                dispatch_t, cost_t, feasible = solve_dispatch_given_commitment(
                    instance, u_t, period=t
                )
                
                if feasible and cost_t < best_cost_t:
                    best_cost_t = cost_t
                    best_u_t = u_t
                    best_dispatch_t = dispatch_t
            
            if best_u_t is None:
                # No feasible solution for this period - try fallback
                best_u_t, best_dispatch_t, best_cost_t = self._fallback_greedy(
                    instance, t
                )
                
                if best_u_t is None:
                    print(f"FAILED: No feasible solution for period {t}")
                    return UCSolution(
                        commitment=np.zeros((N, T), dtype=int),
                        power=np.zeros((N, T)),
                        total_cost=np.inf,
                        fuel_cost=0.0,
                        startup_cost=0.0,
                        solve_time=time.time() - start_time,
                        method="Hybrid",
                        feasible=False
                    )
            
            # Store best for this period
            commitment_full[:, t] = best_u_t
            power_full[:, t] = best_dispatch_t
            total_cost_accumulated += best_cost_t
            
            # Track first feasible
            if time_to_first_feasible is None:
                time_to_first_feasible = time.time() - start_time
            
            if t % 10 == 0 or t == T-1:
                elapsed = time.time() - start_time
                print(f"  Period {t+1}/{T} - Cost so far: ${total_cost_accumulated:,.2f} - Time: {elapsed:.1f}s")
        
        solve_time = time.time() - start_time
        
        # Use the complete solution built period-by-period
        best_commitment = commitment_full
        best_power = power_full
        
        # Calculate total cost with startups using canonical function
        total_cost, fuel_cost, startup_cost = calculate_total_cost(
            instance, best_commitment, best_power
        )
        
        # Verify feasibility
        is_feasible, violations = verify_solution_feasibility(
            instance, best_commitment, best_power
        )
        
        if not is_feasible:
            print(f"WARNING: Solution has violations:")
            for v in violations[:5]:
                print(f"  - {v}")
        
        print(f"\n{'='*80}")
        print(f"HYBRID SOLUTION FOUND")
        print(f"{'='*80}")
        print(f"Time: {solve_time:.2f}s")
        print(f"Time to first feasible: {time_to_first_feasible:.2f}s")
        print(f"Candidates evaluated: {n_evaluated}")
        print(f"Cost: ${total_cost:,.2f}")
        print(f"  Fuel: ${fuel_cost:,.2f}")
        print(f"  Startup: ${startup_cost:,.2f}")
        print(f"Feasible: {is_feasible}")
        
        return UCSolution(
            commitment=best_commitment,
            power=best_power,
            total_cost=total_cost,
            fuel_cost=fuel_cost,
            startup_cost=startup_cost,
            solve_time=solve_time,
            method="Hybrid",
            feasible=is_feasible,
            iterations=n_evaluated,
            time_to_first_feasible=time_to_first_feasible
        )
    
    def _sample_commitment_for_period(
        self,
        instance: UCInstance,
        period: int,
        elapsed_time: float,
        time_limit: Optional[float]
    ) -> list:
        """
        Sample commitment configurations for a single period using Ising model.
        
        Builds a QUBO that balances:
        - Matching demand (penalty for mismatch)
        - Minimizing cost (preference for cheaper units)
        """
        N = instance.n_units
        demand_t = instance.demand[period]
        
        # Normalize for numerical stability
        scale_power = 100.0
        scale_cost = 10.0
        
        P_max = instance.p_max / scale_power
        P_min = instance.p_min / scale_power
        C = instance.marginal_cost / scale_cost
        D = demand_t / scale_power
        
        # QUBO weights
        ALPHA = 200.0  # Demand mismatch penalty
        BETA = 1.0     # Cost penalty
        
        # Build QUBO: minimize u'Q u + L'u
        L = np.zeros(N)
        Q = np.zeros((N, N))
        
        for i in range(N):
            # Linear term: ALPHA*(P_max^2 - 2*D*P_max) + BETA*C*P_min
            L[i] = ALPHA * (P_max[i]**2 - 2*D*P_max[i]) + BETA * C[i] * P_min[i]
            
            for j in range(i + 1, N):
                # Quadratic term: 2*ALPHA*P_max[i]*P_max[j]
                Q[i, j] = 2 * ALPHA * P_max[i] * P_max[j]
        
        # Convert QUBO to Ising
        h = -L / 2.0
        J = np.zeros((N, N))
        
        for i in range(N):
            for j in range(i + 1, N):
                if Q[i, j] != 0:
                    J_val = Q[i, j] / 4.0
                    J[i, j] -= J_val
                    J[j, i] -= J_val
                    h[i] -= J_val
                    h[j] -= J_val
        
        # Sample using thrml
        nodes = [SpinNode() for _ in range(N)]
        edges = [(nodes[i], nodes[j]) for i in range(N) for j in range(i+1, N)]
        weights = [J[i, j] for i in range(N) for j in range(i+1, N)]
        
        model = IsingEBM(
            nodes, edges,
            jnp.array(list(h)),
            jnp.array(weights),
            jnp.array(self.temperature)
        )
        program = IsingSamplingProgram(model, [Block(nodes)], clamped_blocks=[])
        
        # Sample multiple times with different seeds
        candidates = []
        seen = set()
        
        for seed_offset in range(self.n_seeds):
            if time_limit and elapsed_time > time_limit * 0.8:
                break  # Leave time for dispatch
            
            key = jax.random.key(42 + seed_offset + period * 100)
            k_init, k_samp = jax.random.split(key)
            
            init_state = hinton_init(k_init, model, [Block(nodes)], ())
            schedule = SamplingSchedule(
                n_warmup=self.warmup,
                n_samples=self.n_samples_per_period,
                steps_per_sample=self.steps_per_sample
            )
            
            samples = sample_states(k_samp, program, schedule, init_state, [], [Block(nodes)])
            
            for s in samples:
                u_vec = np.array(s[0]).flatten().astype(int)
                u_tuple = tuple(u_vec)
                
                if u_tuple not in seen:
                    seen.add(u_tuple)
                    candidates.append(u_vec)
        
        # Add greedy heuristic candidates
        candidates.extend(self._get_heuristic_candidates(instance, period))
        
        return candidates
    
    def _get_heuristic_candidates(self, instance: UCInstance, period: int) -> list:
        """Get greedy heuristic candidates."""
        N = instance.n_units
        demand_t = instance.demand[period]
        
        candidates = []
        
        # 1. All on
        candidates.append(np.ones(N, dtype=int))
        
        # 2. Greedy by cost
        sorted_cost = np.argsort(instance.marginal_cost)
        u_greedy = np.zeros(N, dtype=int)
        cumulative = 0
        for idx in sorted_cost:
            if cumulative < demand_t:
                u_greedy[idx] = 1
                cumulative += instance.p_max[idx]
        candidates.append(u_greedy)
        
        # 3. Greedy by capacity (largest first)
        sorted_cap = np.argsort(-instance.p_max)
        u_cap = np.zeros(N, dtype=int)
        cumulative = 0
        for idx in sorted_cap:
            if cumulative < demand_t:
                u_cap[idx] = 1
                cumulative += instance.p_max[idx]
        candidates.append(u_cap)
        
        return candidates
    
    def _fallback_greedy(
        self,
        instance: UCInstance,
        period: int
    ) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], float]:
        """
        Fallback greedy commitment for when sampling fails.
        """
        N = instance.n_units
        demand_t = instance.demand[period]
        
        # Try greedy by cost
        sorted_cost = np.argsort(instance.marginal_cost)
        u = np.zeros(N, dtype=int)
        
        for idx in sorted_cost:
            u[idx] = 1
            dispatch, cost, feasible = solve_dispatch_given_commitment(
                instance, u, period
            )
            if feasible:
                return u, dispatch, cost
        
        return None, None, np.inf
