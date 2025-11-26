"""
Hybrid Thermodynamic-Classical Solver for Large-Scale Scenario

Applies hybrid methodology to scenario_00286.json - the most computationally
expensive scenario with 40 thermal units and 139,872 variables.
"""
import json
import numpy as np
import jax
import jax.numpy as jnp
from thrml import SpinNode, Block, SamplingSchedule, sample_states
from thrml.models import IsingEBM, IsingSamplingProgram, hinton_init
from pathlib import Path
import time

print("=" * 90)
print("HYBRID SOLVER: LARGE-SCALE SCENARIO")
print("Most Computationally Expensive Scenario in Dataset")
print("=" * 90)

# Load scenario
scenario_path = Path(__file__).parent.parent / "outputs" / "scenarios_v1" / "scenario_00286.json"

with open(scenario_path, 'r') as f:
    scenario = json.load(f)

print(f"\nScenario ID: {scenario['id']}")
print(f"File: scenario_00286.json")
print(f"\nComplexity Metrics:")
print(f"  Full MILP variables: {scenario['estimates']['vars_total']:,}")
print(f"  Full MILP constraints: {scenario['estimates']['cons_total']:,}")
print(f"  Estimated MILP CPU time: {scenario['estimates']['est_cpu_hours']:.2f} hours")

# Extract asset counts
assets = scenario['meta']['assets']
n_thermal = assets['thermal']

print(f"\nAssets Overview:")
print(f"  Thermal units: {n_thermal}")
print(f"  Solar: {assets['solar']}")
print(f"  Wind: {assets['wind']}")
print(f"  Battery: {assets['battery']}")
print(f"  Nuclear: {assets['nuclear']}")
print(f"  Total assets: {sum(assets.values())}")

print(f"\nGrid Structure:")
print(f"  Regions: {scenario['meta']['regions']}")
print(f"  Zones: {scenario['meta']['zones']}")
print(f"  Sites: {scenario['meta']['sites']}")

# ==========================================
# SIMPLIFICATION: Single-Period Thermal Commitment
# ==========================================
print("\n" + "=" * 90)
print("SIMPLIFICATION: Single-Period Thermal Unit Commitment")
print("=" * 90)
print("\nNote: Full multi-period optimization would solve 96 time steps.")
print("This demo focuses on thermal commitment for computational comparison.")

# Generate synthetic thermal parameters based on scenario
np.random.seed(42)

thermal_capacity = np.random.uniform(100, 800, n_thermal)  # MW (larger range for big scenario)
thermal_min_gen = thermal_capacity * 0.3

# Cost structure
base_cost = np.random.uniform(10, 30, n_thermal)
fuel_cost = scenario['operation_costs']['thermal_fuel_eur_per_mwh']
co2_intensity = np.random.uniform(0.3, 1.0, n_thermal)
co2_price = scenario['econ_policy']['co2_price']

thermal_cost = base_cost + fuel_cost * scenario['tech']['thermal_marg_cost'] / 10 + co2_intensity * co2_price

# Peak demand (scaled for larger grid)
base_demand = 5000.0  # MW for this large grid
demand_scale = scenario['exogenous']['demand_scale_factor']
target_demand = base_demand * demand_scale

# Renewable availability
renewable_factor = 0.25
renewable_gen = (assets['solar'] * 50 + assets['wind'] * 80) * renewable_factor
net_demand = target_demand - renewable_gen

print(f"\nOperational Parameters:")
print(f"  Base demand: {base_demand:.0f} MW")
print(f"  Demand scale factor: {demand_scale:.3f}")
print(f"  Target demand: {target_demand:.1f} MW")
print(f"  Renewable generation: {renewable_gen:.1f} MW")
print(f"  Net thermal demand: {net_demand:.1f} MW")
print(f"  Total thermal capacity: {thermal_capacity.sum():.1f} MW")
print(f"  Thermal cost range: €{thermal_cost.min():.1f} - €{thermal_cost.max():.1f} per MW")

# ==========================================
# HYBRID SOLVER
# ==========================================
print("\n" + "=" * 90)
print("HYBRID SOLVER: STAGE 1 - Thermodynamic Sampling")
print("=" * 90)

# Normalize
scale_power = 100.0
scale_cost = 10.0

P_max = thermal_capacity / scale_power
P_min = thermal_min_gen / scale_power
D = net_demand / scale_power
C = thermal_cost / scale_cost

# QUBO parameters
ALPHA = 200.0  # High penalty for demand
BETA = 1.0     # Low weight for cost

print(f"\nQUBO Parameters:")
print(f"  ALPHA (demand penalty): {ALPHA}")
print(f"  BETA (cost penalty): {BETA}")

# Build QUBO
Q_qubo = np.zeros((n_thermal, n_thermal))
L_qubo = np.zeros(n_thermal)

for i in range(n_thermal):
    demand_term = ALPHA * (P_max[i]**2 - 2*D*P_max[i])
    cost_term = BETA * C[i] * P_min[i]
    L_qubo[i] = demand_term + cost_term
    
    for j in range(i + 1, n_thermal):
        Q_qubo[i, j] = 2 * ALPHA * P_max[i] * P_max[j]

# Convert to Ising
h_ising = np.zeros(n_thermal)
J_ising = np.zeros((n_thermal, n_thermal))

h_ising -= L_qubo / 2.0

for i in range(n_thermal):
    for j in range(i + 1, n_thermal):
        if Q_qubo[i, j] == 0:
            continue
        J_val = Q_qubo[i, j] / 4.0
        J_ising[i, j] -= J_val
        J_ising[j, i] -= J_val
        h_ising[i] -= J_val
        h_ising[j] -= J_val

print(f"\nIsing Hamiltonian:")
print(f"  Bias range: [{h_ising.min():.1f}, {h_ising.max():.1f}]")
print(f"  Coupling range: [{J_ising[J_ising != 0].min():.1f}, {J_ising[J_ising != 0].max():.1f}]")

# Build thermodynamic model
nodes = [SpinNode() for _ in range(n_thermal)]
edges = []
weights_list = []
biases_list = []

for i in range(n_thermal):
    biases_list.append(h_ising[i])
    for j in range(i + 1, n_thermal):
        edges.append((nodes[i], nodes[j]))
        weights_list.append(J_ising[i, j])

print(f"\nThermodynamic Model:")
print(f"  Nodes (spins): {n_thermal}")
print(f"  Edges (couplings): {len(edges)}")

# VERY HIGH temperature for exploration
beta_temp = jnp.array(0.3)  # Even hotter for larger problem
model = IsingEBM(nodes, edges, jnp.array(biases_list), jnp.array(weights_list), beta_temp)
program = IsingSamplingProgram(model, [Block(nodes)], clamped_blocks=[])

# Sample with multiple seeds
print(f"\nSampling Parameters:")
print(f"  Temperature (1/beta): {1.0/float(beta_temp):.2f} (VERY HOT for exploration)")
print(f"  Number of seeds: 15 (increased for larger problem)")
print(f"  Samples per seed: 15")
print(f"  Total candidates: ~225")

print(f"\nSampling... (this may take a minute for 40 units)")

start_time = time.time()

candidates = []

for seed in range(15):
    if seed % 5 == 0:
        print(f"  Seed {seed+1}/15...")
    
    key = jax.random.key(42 + seed)
    k_init, k_samp = jax.random.split(key)
    init_state = hinton_init(k_init, model, [Block(nodes)], ())
    
    schedule = SamplingSchedule(n_warmup=500, n_samples=15, steps_per_sample=5)
    samples = sample_states(k_samp, program, schedule, init_state, [], [Block(nodes)])
    
    for s in samples:
        spins = np.array(s[0]).flatten()
        u_vec = spins.astype(int)
        candidates.append(u_vec)

# Add heuristics
sorted_by_cost = np.argsort(C)
heuristic_1 = np.zeros(n_thermal, dtype=int)
cumulative_cap = 0
for idx in sorted_by_cost:
    if cumulative_cap < D:
        heuristic_1[idx] = 1
        cumulative_cap += P_max[idx]
candidates.append(heuristic_1)

heuristic_2 = np.ones(n_thermal, dtype=int)
candidates.append(heuristic_2)

np.random.seed(42)
for _ in range(20):
    n_on = np.random.randint(max(1, n_thermal // 3), n_thermal)
    random_commitment = np.zeros(n_thermal, dtype=int)
    random_commitment[np.random.choice(n_thermal, n_on, replace=False)] = 1
    candidates.append(random_commitment)

sampling_time = time.time() - start_time

print(f"\nSampling complete in {sampling_time:.2f} seconds")
print(f"Generated {len(candidates)} candidates")

# ==========================================
# STAGE 2: Classical Dispatch
# ==========================================
print("\n" + "=" * 90)
print("HYBRID SOLVER: STAGE 2 - Classical Economic Dispatch")
print("=" * 90)

def solve_economic_dispatch(u_commitment, demand, p_min, p_max, costs):
    """Economic dispatch using merit order."""
    available = np.where(u_commitment == 1)[0]
    
    if len(available) == 0:
        return None, np.inf, False, "No units committed"
    
    max_gen = np.sum(p_max[available])
    min_gen = np.sum(p_min[available])
    
    if demand > max_gen:
        return None, np.inf, False, f"Insufficient capacity"
    
    if demand < min_gen:
        return None, np.inf, False, f"Below minimum generation"
    
    sorted_idx = available[np.argsort(costs[available])]
    
    dispatch = np.zeros(len(costs))
    remaining = demand
    
    # Minimum generation first
    for idx in sorted_idx:
        dispatch[idx] = p_min[idx]
        remaining -= p_min[idx]
    
    # Add remaining load
    for idx in sorted_idx:
        if remaining <= 0:
            break
        available_capacity = p_max[idx] - dispatch[idx]
        take = min(remaining, available_capacity)
        dispatch[idx] += take
        remaining -= take
    
    total_gen = np.sum(dispatch)
    if abs(total_gen - demand) > 1e-2:
        return None, np.inf, False, f"Dispatch error"
    
    total_cost = np.sum(dispatch * scale_power * costs * scale_cost)
    
    return dispatch, total_cost, True, "Feasible"

# Evaluate candidates
print(f"Evaluating {len(candidates)} candidates...")

start_eval = time.time()

results = []
seen_configs = set()

for i, u in enumerate(candidates):
    if i % 50 == 0 and i > 0:
        print(f"  Evaluated {i}/{len(candidates)}...")
    
    u_tuple = tuple(u)
    if u_tuple in seen_configs:
        continue
    seen_configs.add(u_tuple)
    
    dispatch, cost, feasible, msg = solve_economic_dispatch(u, D, P_min, P_max, C)
    
    results.append({
        'id': i,
        'commitment': u,
        'dispatch': dispatch,
        'cost': cost,
        'feasible': feasible,
        'message': msg,
        'n_units_on': np.sum(u)
    })

eval_time = time.time() - start_eval

print(f"\nEvaluation complete in {eval_time:.2f} seconds")
print(f"Unique configurations: {len(results)}")

# ==========================================
# STAGE 3: Selection
# ==========================================
print("\n" + "=" * 90)
print("HYBRID SOLVER: STAGE 3 - Selection")
print("=" * 90)

feasible_results = [r for r in results if r['feasible']]
print(f"\nFeasible solutions found: {len(feasible_results)} / {len(results)}")

if len(feasible_results) == 0:
    print("\n[FAILED] No feasible solution found!")
    print("\nTop infeasibility reasons:")
    reason_counts = {}
    for r in results[:20]:
        msg = r['message']
        reason_counts[msg] = reason_counts.get(msg, 0) + 1
    for reason, count in sorted(reason_counts.items(), key=lambda x: -x[1])[:5]:
        print(f"  {reason}: {count} occurrences")
else:
    # Sort by cost
    feasible_results.sort(key=lambda x: x['cost'])
    
    best = feasible_results[0]
    
    print(f"\n**BEST SOLUTION FOUND**")
    print(f"  Cost: €{best['cost']:,.2f}")
    print(f"  Units committed: {best['n_units_on']} / {n_thermal}")
    print(f"  Utilization: {best['n_units_on'] / n_thermal * 100:.1f}%")
    
    # Total time
    total_time = sampling_time + eval_time
    print(f"\n**HYBRID SOLVER PERFORMANCE**")
    print(f"  Sampling time: {sampling_time:.2f}s ({sampling_time/total_time*100:.1f}%)")
    print(f"  Evaluation time: {eval_time:.2f}s ({eval_time/total_time*100:.1f}%)")
    print(f"  Total time: {total_time:.2f}s")
    
    # ==========================================
    # DETAILED RESULTS
    # ==========================================
    print("\n" + "=" * 90)
    print("OPTIMAL DISPATCH DETAILS")
    print("=" * 90)
    
    u_best = best['commitment']
    p_best = best['dispatch']
    
    total_dispatch = 0.0
    total_cost_check = 0.0
    
    print(f"\n{'Unit':<6} {'Status':<8} {'Capacity':<12} {'Dispatch':<12} {'Cost/MW':<12} {'Total Cost':<14}")
    print("-" * 90)
    
    units_on = []
    for i in range(n_thermal):
        if u_best[i] == 1:
            status = "ON"
            capacity = thermal_capacity[i]
            dispatch_mw = p_best[i] * scale_power
            cost_per_mw = thermal_cost[i] * scale_cost
            unit_cost = dispatch_mw * cost_per_mw
            
            total_dispatch += dispatch_mw
            total_cost_check += unit_cost
            units_on.append(i+1)
            
            print(f"{i+1:<6} {status:<8} {capacity:>10.1f} MW {dispatch_mw:>10.1f} MW "
                  f"€{cost_per_mw:>9.2f} €{unit_cost:>12.2f}")
    
    print("-" * 90)
    print(f"{'TOTAL':<6} {'':<8} {thermal_capacity.sum():>10.1f} MW {total_dispatch:>10.1f} MW "
          f"{'':>11} €{total_cost_check:>12.2f}")
    
    print(f"\nDemand Satisfaction:")
    print(f"  Net thermal demand: {net_demand:.1f} MW")
    print(f"  Total dispatch: {total_dispatch:.1f} MW")
    print(f"  Error: {abs(total_dispatch - net_demand):.3f} MW ({abs(total_dispatch - net_demand)/net_demand*100:.3f}%)")
    
    print(f"\nOperating Reserve:")
    available_capacity = np.sum(thermal_capacity[u_best == 1])
    reserve = available_capacity - total_dispatch
    reserve_pct = reserve / total_dispatch * 100
    print(f"  Committed capacity: {available_capacity:.1f} MW")
    print(f"  Reserve margin: {reserve:.1f} MW ({reserve_pct:.1f}%)")
    
    # Solution diversity
    if len(feasible_results) > 1:
        print(f"\n" + "-" * 90)
        print(f"Solution Diversity (Top {min(10, len(feasible_results))} Feasible)")
        print("-" * 90)
        
        print(f"\n{'Rank':<6} {'Units ON':<12} {'Cost':<18} {'Delta vs Best':<16}")
        print("-" * 90)
        
        for rank, sol in enumerate(feasible_results[:min(10, len(feasible_results))], 1):
            delta = sol['cost'] - best['cost']
            delta_pct = delta / best['cost'] * 100
            print(f"{rank:<6} {sol['n_units_on']:<12} €{sol['cost']:>15,.2f} +€{delta:>10,.2f} (+{delta_pct:.2f}%)")
    
    # Save results
    result_data = {
        'scenario': 'scenario_00286.json',
        'n_thermal': n_thermal,
        'net_demand_mw': float(net_demand),
        'total_time_seconds': total_time,
        'sampling_time_seconds': sampling_time,
        'evaluation_time_seconds': eval_time,
        'best_cost_eur': float(best['cost']),
        'units_committed': int(best['n_units_on']),
        'units_on_ids': units_on,
        'total_dispatch_mw': float(total_dispatch),
        'feasible_solutions_found': len(feasible_results),
        'total_candidates_evaluated': len(results)
    }
    
    output_file = Path(__file__).parent / 'hybrid_result_large.json'
    with open(output_file, 'w') as f:
        json.dump(result_data, f, indent=2)
    
    print(f"\n" + "=" * 90)
    print("Results saved to: hybrid_result_large.json")
    print("=" * 90)
