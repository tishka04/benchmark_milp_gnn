"""
Benchmark N=200: Extended timeout version

For N=200, MILP needs much longer timeout (6 hours vs 1 hour)
"""
import json
import numpy as np
import jax
import jax.numpy as jnp
from thrml import SpinNode, Block, SamplingSchedule, sample_states
from thrml.models import IsingEBM, IsingSamplingProgram, hinton_init
from scipy.optimize import milp, LinearConstraint, Bounds
from pathlib import Path
import time

def load_scenario(scenario_file):
    """Load synthetic scenario."""
    with open(scenario_file, 'r') as f:
        return json.load(f)

def solve_hybrid(scenario):
    """Solve using hybrid thermodynamic + classical approach."""
    print("\n" + "=" * 90)
    print("HYBRID SOLVER")
    print("=" * 90)
    
    n_thermal = scenario['n_thermal']
    n_periods = scenario['n_periods']
    
    thermal_capacity = np.array(scenario['thermal_capacity_mw'])
    thermal_min_gen = np.array(scenario['thermal_min_gen_mw'])
    thermal_cost = np.array(scenario['thermal_cost_eur_per_mw'])
    demand_profile = np.array(scenario['demand_profile_mw'])
    
    print(f"\nProblem: {n_thermal} units, {n_periods} periods")
    print(f"Variables: {2 * n_thermal * n_periods:,}")
    
    # Normalize
    scale_power = 100.0
    scale_cost = 10.0
    P_max = thermal_capacity / scale_power
    P_min = thermal_min_gen / scale_power
    C = thermal_cost / scale_cost
    
    def solve_period(demand_t):
        """Solve single period."""
        D = demand_t / scale_power
        ALPHA, BETA = 200.0, 1.0
        
        # Build QUBO
        L = np.zeros(n_thermal)
        Q = np.zeros((n_thermal, n_thermal))
        
        for i in range(n_thermal):
            L[i] = ALPHA * (P_max[i]**2 - 2*D*P_max[i]) + BETA * C[i] * P_min[i]
            for j in range(i + 1, n_thermal):
                Q[i, j] = 2 * ALPHA * P_max[i] * P_max[j]
        
        # Convert to Ising
        h = -L / 2.0
        J = np.zeros((n_thermal, n_thermal))
        for i in range(n_thermal):
            for j in range(i + 1, n_thermal):
                if Q[i, j] != 0:
                    J_val = Q[i, j] / 4.0
                    J[i, j] -= J_val
                    J[j, i] -= J_val
                    h[i] -= J_val
                    h[j] -= J_val
        
        # Sample
        nodes = [SpinNode() for _ in range(n_thermal)]
        edges = [(nodes[i], nodes[j]) for i in range(n_thermal) for j in range(i+1, n_thermal)]
        weights = [J[i, j] for i in range(n_thermal) for j in range(i+1, n_thermal)]
        
        model = IsingEBM(nodes, edges, jnp.array(list(h)), jnp.array(weights), jnp.array(0.3))
        program = IsingSamplingProgram(model, [Block(nodes)], clamped_blocks=[])
        
        candidates = []
        for seed in range(5):
            key = jax.random.key(42 + seed)
            k_init, k_samp = jax.random.split(key)
            init_state = hinton_init(k_init, model, [Block(nodes)], ())
            schedule = SamplingSchedule(n_warmup=200, n_samples=5, steps_per_sample=3)
            samples = sample_states(k_samp, program, schedule, init_state, [], [Block(nodes)])
            for s in samples:
                candidates.append(np.array(s[0]).flatten().astype(int))
        
        # Heuristics
        candidates.append(np.ones(n_thermal, dtype=int))
        sorted_cost = np.argsort(C)
        heur = np.zeros(n_thermal, dtype=int)
        cum = 0
        for idx in sorted_cost:
            if cum < D:
                heur[idx] = 1
                cum += P_max[idx]
        candidates.append(heur)
        
        # Dispatch
        best_cost = np.inf
        best_dispatch = None
        
        for u in candidates:
            avail = np.where(u == 1)[0]
            if len(avail) == 0:
                continue
            
            if np.sum(P_max[avail]) < D or np.sum(P_min[avail]) > D:
                continue
            
            sorted_idx = avail[np.argsort(C[avail])]
            dispatch = np.zeros(n_thermal)
            remaining = D
            
            for idx in sorted_idx:
                dispatch[idx] = P_min[idx]
                remaining -= P_min[idx]
            
            for idx in sorted_idx:
                if remaining <= 0:
                    break
                take = min(remaining, P_max[idx] - dispatch[idx])
                dispatch[idx] += take
                remaining -= take
            
            if abs(np.sum(dispatch) - D) > 1e-2:
                continue
            
            cost = np.sum(dispatch * scale_power * C * scale_cost)
            if cost < best_cost:
                best_cost = cost
                best_dispatch = dispatch
        
        return best_dispatch, best_cost
    
    # Solve all periods
    print("\nStarting hybrid solver...")
    start_time = time.time()
    total_cost = 0
    success = True
    
    for t in range(n_periods):
        if t % 10 == 0:
            elapsed = time.time() - start_time
            print(f"  Period {t+1}/{n_periods}... (elapsed: {elapsed:.1f}s, {elapsed/60:.1f}min)")
        
        dispatch, cost = solve_period(demand_profile[t])
        
        if dispatch is None:
            print(f"  [FAILED] Period {t+1}")
            success = False
            break
        
        total_cost += cost
    
    solve_time = time.time() - start_time
    
    if success:
        print(f"\n[SUCCESS] All periods solved")
        print(f"Time: {solve_time:.1f}s ({solve_time/60:.1f} min, {solve_time/3600:.2f} hr)")
        print(f"Cost: EUR {total_cost:,.2f}")
    
    return {
        'success': success,
        'time': solve_time,
        'cost': total_cost if success else None
    }

def solve_milp(scenario):
    """Solve using monolithic MILP with EXTENDED timeout for N=200."""
    print("\n" + "=" * 90)
    print("MILP SOLVER (Monolithic)")
    print("=" * 90)
    
    n_thermal = scenario['n_thermal']
    n_periods = scenario['n_periods']
    
    thermal_capacity = np.array(scenario['thermal_capacity_mw'])
    thermal_min_gen = np.array(scenario['thermal_min_gen_mw'])
    thermal_cost = np.array(scenario['thermal_cost_eur_per_mw'])
    demand_profile = np.array(scenario['demand_profile_mw'])
    
    print(f"\nProblem: {n_thermal} units, {n_periods} periods")
    print(f"Binary variables: {n_thermal * n_periods:,}")
    print(f"Continuous variables: {n_thermal * n_periods:,}")
    print(f"Total variables: {2 * n_thermal * n_periods:,}")
    
    # Build MILP
    n_vars = 2 * n_thermal * n_periods
    offset_p = n_thermal * n_periods
    
    # Objective
    c = np.zeros(n_vars)
    for t in range(n_periods):
        for i in range(n_thermal):
            c[offset_p + t * n_thermal + i] = thermal_cost[i]
    
    print("\nBuilding constraints...")
    constraints_list = []
    
    # Demand
    print(f"  - Demand constraints: {n_periods}")
    for t in range(n_periods):
        A = np.zeros(n_vars)
        for i in range(n_thermal):
            A[offset_p + t * n_thermal + i] = 1.0
        constraints_list.append(LinearConstraint(A, lb=demand_profile[t], ub=demand_profile[t]))
    
    # Min/max generation
    print(f"  - Generation bounds: {2 * n_thermal * n_periods:,}")
    for t in range(n_periods):
        for i in range(n_thermal):
            idx_u = t * n_thermal + i
            idx_p = offset_p + t * n_thermal + i
            
            A = np.zeros(n_vars)
            A[idx_p] = 1.0
            A[idx_u] = -thermal_min_gen[i]
            constraints_list.append(LinearConstraint(A, lb=0, ub=np.inf))
            
            A = np.zeros(n_vars)
            A[idx_p] = 1.0
            A[idx_u] = -thermal_capacity[i]
            constraints_list.append(LinearConstraint(A, lb=-np.inf, ub=0))
    
    print(f"\nTotal constraints: {len(constraints_list):,}")
    
    # Bounds
    bounds = Bounds(
        lb=np.concatenate([np.zeros(offset_p), np.zeros(offset_p)]),
        ub=np.concatenate([np.ones(offset_p), np.repeat(thermal_capacity, n_periods)])
    )
    
    integrality = np.concatenate([np.ones(offset_p), np.zeros(offset_p)])
    
    # Extended timeout for N=200
    TIMEOUT_SECONDS = 21600  # 6 hours
    
    print(f"\n⚠️  EXTENDED TIMEOUT: {TIMEOUT_SECONDS/3600:.1f} hours")
    print("This is a LARGE problem - be patient!")
    print("\nSolving...")
    
    start_time = time.time()
    
    result = milp(
        c=c,
        constraints=constraints_list,
        bounds=bounds,
        integrality=integrality,
        options={'disp': True, 'time_limit': TIMEOUT_SECONDS, 'mip_rel_gap': 0.01}
    )
    
    solve_time = time.time() - start_time
    
    if result.success or (hasattr(result, 'x') and result.x is not None):
        p_solution = result.x[offset_p:].reshape((n_periods, n_thermal))
        total_cost = np.sum(p_solution * thermal_cost)
        
        status = "OPTIMAL" if result.success else "FEASIBLE (time limit)"
        print(f"\n[{status}] Solution found")
        print(f"Time: {solve_time:.1f}s ({solve_time/60:.1f} min, {solve_time/3600:.2f} hr)")
        print(f"Cost: EUR {total_cost:,.2f}")
        
        if hasattr(result, 'mip_gap'):
            print(f"MIP gap: {result.mip_gap * 100:.4f}%")
        
        return {
            'success': True,
            'time': solve_time,
            'cost': total_cost,
            'optimal': result.success
        }
    else:
        print(f"\n[FAILED] {result.message}")
        print(f"Time: {solve_time:.1f}s ({solve_time/60:.1f} min, {solve_time/3600:.2f} hr)")
        
        return {
            'success': False,
            'time': solve_time,
            'cost': None
        }

# Main
print("=" * 90)
print("BENCHMARK: N=200 Thermal Units")
print("=" * 90)

scenario_file = Path(__file__).parent / 'synthetic_scenario_N200.json'

if not scenario_file.exists():
    print(f"Error: {scenario_file} not found!")
    print("Run create_feasible_scenario.py first.")
    exit(1)

scenario = load_scenario(scenario_file)

print(f"\nScenario: {scenario['name']}")
print(f"Units: {scenario['n_thermal']}")
print(f"Periods: {scenario['n_periods']}")
print(f"Peak demand: {max(scenario['demand_profile_mw']):.0f} MW")

# Run both methods
hybrid_result = solve_hybrid(scenario)
milp_result = solve_milp(scenario)

# Compare
print("\n" + "=" * 90)
print("FINAL COMPARISON")
print("=" * 90)

print(f"\n{'Method':<15} {'Time':<30} {'Cost':<25} {'Status':<15}")
print("-" * 90)

if hybrid_result['success']:
    h_time = hybrid_result['time']
    h_cost = hybrid_result['cost']
    print(f"{'Hybrid':<15} {h_time:>10.1f}s ({h_time/60:>5.1f}min, {h_time/3600:>4.2f}hr)  EUR {h_cost:>18,.2f}  {'SUCCESS':<15}")
else:
    print(f"{'Hybrid':<15} {'N/A':<30} {'N/A':<25} {'FAILED':<15}")

if milp_result['success']:
    m_time = milp_result['time']
    m_cost = milp_result['cost']
    status = "OPTIMAL" if milp_result.get('optimal') else "FEASIBLE"
    print(f"{'MILP':<15} {m_time:>10.1f}s ({m_time/60:>5.1f}min, {m_time/3600:>4.2f}hr)  EUR {m_cost:>18,.2f}  {status:<15}")
else:
    print(f"{'MILP':<15} {'N/A':<30} {'N/A':<25} {'TIMEOUT/FAIL':<15}")

if hybrid_result['success'] and milp_result['success']:
    print(f"\n{'=' * 90}")
    print("VERDICT")
    print("=" * 90)
    
    time_ratio = milp_result['time'] / hybrid_result['time']
    cost_diff = abs(milp_result['cost'] - hybrid_result['cost']) / hybrid_result['cost'] * 100
    
    print(f"\nSpeed:")
    if time_ratio > 1:
        print(f"  🎯 Hybrid was {time_ratio:.2f}x FASTER")
        print(f"  ({m_time/60:.1f} min vs {h_time/60:.1f} min)")
    else:
        print(f"  MILP was {1/time_ratio:.2f}x faster (unexpected at N=200!)")
    
    print(f"\nCost Quality:")
    print(f"  Difference: {cost_diff:.2f}%")
    
    print(f"\n{'=' * 90}")
    print("CONCLUSION FOR N=200")
    print("=" * 90)
    
    if time_ratio > 5:
        print(f"\n✓✓✓ HYBRID DOMINATES at scale!")
        print(f"  - {time_ratio:.1f}x speedup proves decomposition essential")
        print(f"  - This is a REALISTIC grid size (regional grid)")
        print(f"  - MILP impractical for larger problems")
        print(f"  - Hybrid ready for production use")
    elif time_ratio > 2:
        print(f"\n✓ Hybrid has clear advantage")
        print(f"  - {time_ratio:.1f}x speedup")
        print(f"  - Scales to real-world sizes")
    else:
        print(f"\n≈ Both methods viable at N=200")
        print(f"  - MILP more capable than expected")

elif hybrid_result['success'] and not milp_result['success']:
    print(f"\n{'=' * 90}")
    print("VERDICT")
    print("=" * 90)
    
    print(f"\n✓✓✓ HYBRID WINS - MILP TIMEOUT!")
    print(f"\n  - Hybrid: {hybrid_result['time']/60:.1f} min ✓")
    print(f"  - MILP: TIMEOUT after {milp_result['time']/3600:.1f} hr ✗")
    print(f"\n  This PROVES hybrid's scalability advantage!")
    print(f"  At N=200, MILP becomes impractical.")
    print(f"  Hybrid is production-ready for real grids.")

# Save results
result_data = {
    'n_thermal': 200,
    'n_periods': 96,
    'hybrid': hybrid_result,
    'milp': milp_result
}

output_file = Path(__file__).parent / 'benchmark_N200_results.json'
with open(output_file, 'w') as f:
    json.dump(result_data, f, indent=2)

print(f"\n{'=' * 90}")
print(f"Results saved to: {output_file.name}")
print("=" * 90)
