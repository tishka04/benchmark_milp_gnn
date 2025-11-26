"""
Hybrid Solver for SIMPLEST Scenario (scenario_00285.json)
10 thermal units, 96 periods
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
print("HYBRID SOLVER: SIMPLEST SCENARIO (10 units)")
print("=" * 90)

# Load scenario
scenario_path = Path(__file__).parent.parent / "outputs" / "scenarios_v1" / "scenario_00285.json"
with open(scenario_path, 'r') as f:
    scenario = json.load(f)

n_periods = 96
n_thermal = scenario['meta']['assets']['thermal']

print(f"\nScenario: {scenario['id']}")
print(f"File: scenario_00285.json (SIMPLEST)")
print(f"Thermal units: {n_thermal}")
print(f"Estimated full MILP time: {scenario['estimates']['est_cpu_hours']:.2f} hours")

# Generate parameters
np.random.seed(42)
thermal_capacity = np.random.uniform(100, 800, n_thermal)
thermal_min_gen = thermal_capacity * 0.3
thermal_ramp_rate = thermal_capacity * scenario['tech']['thermal_ramp_pct']

base_cost = np.random.uniform(10, 30, n_thermal)
fuel_cost = scenario['operation_costs']['thermal_fuel_eur_per_mwh']
co2_intensity = np.random.uniform(0.3, 1.0, n_thermal)
co2_price = scenario['econ_policy']['co2_price']
thermal_cost = base_cost + fuel_cost * scenario['tech']['thermal_marg_cost'] / 10 + co2_intensity * co2_price

# Demand profile
base_demand = 5000.0
demand_scale = scenario['exogenous']['demand_scale_factor']
hours = np.linspace(0, 24, n_periods)
demand_profile = np.zeros(n_periods)

for t, h in enumerate(hours):
    morning_peak = 1.2 * np.exp(-((h - 8)**2) / 2)
    evening_peak = 1.4 * np.exp(-((h - 19)**2) / 2)
    night_valley = -0.3 * np.exp(-((h - 3)**2) / 2)
    demand_profile[t] = base_demand * demand_scale * (1.0 + morning_peak + evening_peak + night_valley)

assets = scenario['meta']['assets']
renewable_profile = np.zeros(n_periods)
for t, h in enumerate(hours):
    solar = assets['solar'] * 50 * max(0, np.sin(np.pi * (h - 6) / 12))
    wind = assets['wind'] * 80 * (0.3 + 0.2 * np.sin(2 * np.pi * h / 24 + 1.5))
    renewable_profile[t] = solar + wind

net_demand = demand_profile - renewable_profile

print(f"\nDemand: {net_demand.min():.0f} - {net_demand.max():.0f} MW")
print(f"Total capacity: {thermal_capacity.sum():.0f} MW")

# Normalize
scale_power = 100.0
scale_cost = 10.0
P_max = thermal_capacity / scale_power
P_min = thermal_min_gen / scale_power
C = thermal_cost / scale_cost

print(f"\nStarting hybrid solver...")

def solve_period(demand_t):
    D = demand_t / scale_power
    ALPHA, BETA = 200.0, 1.0
    
    # Build QUBO
    Q = np.zeros((n_thermal, n_thermal))
    L = np.zeros(n_thermal)
    
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

# Solve
total_start = time.time()
total_cost = 0
schedule = []

for t in range(n_periods):
    if t % 10 == 0:
        print(f"  Period {t+1}/96...")
    
    dispatch, cost = solve_period(net_demand[t])
    
    if dispatch is None:
        print(f"[FAILED] Period {t+1}")
        break
    
    total_cost += cost
    schedule.append({'period': t, 'cost': cost})

total_time = time.time() - total_start

if len(schedule) == n_periods:
    print(f"\n[SUCCESS] All {n_periods} periods solved")
    print(f"\nTime: {total_time:.1f}s ({total_time/60:.1f} min)")
    print(f"Cost: EUR {total_cost:,.2f}")
    
    result_data = {
        'scenario': 'scenario_00285.json',
        'method': 'hybrid',
        'n_thermal': n_thermal,
        'success': True,
        'solve_time_seconds': total_time,
        'total_cost_eur': float(total_cost)
    }
else:
    result_data = {'scenario': 'scenario_00285.json', 'method': 'hybrid', 'success': False}

with open(Path(__file__).parent / 'hybrid_result_simple.json', 'w') as f:
    json.dump(result_data, f, indent=2)

print(f"\nSaved to: hybrid_result_simple.json")
