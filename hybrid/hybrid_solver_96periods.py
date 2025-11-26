"""
Hybrid Solver: FULL 96-PERIOD Problem

Tests hybrid approach on complete 24-hour schedule (96 x 15-minute periods)
with 40 thermal units for scenario_00286.json.
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
print("HYBRID SOLVER: FULL 96-PERIOD MULTI-TEMPORAL OPTIMIZATION")
print("Scenario 00286 - Complete 24-hour Schedule")
print("=" * 90)

# Load scenario
scenario_path = Path(__file__).parent.parent / "outputs" / "scenarios_v1" / "scenario_00286.json"

with open(scenario_path, 'r') as f:
    scenario = json.load(f)

print(f"\nScenario: {scenario['id']}")
print(f"Horizon: {scenario['horizon_hours']} hours")
print(f"Resolution: {scenario['dt_minutes']} minutes")

n_periods = (scenario['horizon_hours'] * 60) // scenario['dt_minutes']
print(f"Time periods: {n_periods}")

assets = scenario['meta']['assets']
n_thermal = assets['thermal']

print(f"Thermal units: {n_thermal}")
print(f"\nFull problem size (estimated):")
print(f"  Variables: {scenario['estimates']['vars_total']:,}")
print(f"  Constraints: {scenario['estimates']['cons_total']:,}")

# Generate thermal parameters
np.random.seed(42)
thermal_capacity = np.random.uniform(100, 800, n_thermal)
thermal_min_gen = thermal_capacity * 0.3
thermal_ramp_rate = thermal_capacity * scenario['tech']['thermal_ramp_pct']

base_cost = np.random.uniform(10, 30, n_thermal)
fuel_cost = scenario['operation_costs']['thermal_fuel_eur_per_mwh']
co2_intensity = np.random.uniform(0.3, 1.0, n_thermal)
co2_price = scenario['econ_policy']['co2_price']
thermal_cost = base_cost + fuel_cost * scenario['tech']['thermal_marg_cost'] / 10 + co2_intensity * co2_price

print(f"\nUnit characteristics:")
print(f"  Capacity range: {thermal_capacity.min():.0f} - {thermal_capacity.max():.0f} MW")
print(f"  Ramp rate: {thermal_ramp_rate.min():.1f} - {thermal_ramp_rate.max():.1f} MW/period")
print(f"  Cost range: €{thermal_cost.min():.1f} - €{thermal_cost.max():.1f} per MW")

# Generate realistic demand profile (cold snap scenario)
base_demand = 5000.0
demand_scale = scenario['exogenous']['demand_scale_factor']

# Create 24-hour demand curve
hours = np.linspace(0, 24, n_periods)
demand_profile = np.zeros(n_periods)

for t, h in enumerate(hours):
    # Morning peak (7-9am)
    morning_peak = 1.2 * np.exp(-((h - 8)**2) / 2)
    # Evening peak (18-20pm)
    evening_peak = 1.4 * np.exp(-((h - 19)**2) / 2)
    # Night valley (2-5am)
    night_valley = -0.3 * np.exp(-((h - 3)**2) / 2)
    
    demand_profile[t] = base_demand * demand_scale * (1.0 + morning_peak + evening_peak + night_valley)

# Renewable generation (solar peaks midday, wind variable)
renewable_profile = np.zeros(n_periods)
for t, h in enumerate(hours):
    # Solar (peaks at noon)
    solar = assets['solar'] * 50 * max(0, np.sin(np.pi * (h - 6) / 12))
    # Wind (variable with some randomness)
    wind = assets['wind'] * 80 * (0.3 + 0.2 * np.sin(2 * np.pi * h / 24 + 1.5))
    renewable_profile[t] = solar + wind

net_demand = demand_profile - renewable_profile

print(f"\nDemand profile:")
print(f"  Min: {net_demand.min():.0f} MW (period {np.argmin(net_demand)})")
print(f"  Max: {net_demand.max():.0f} MW (period {np.argmax(net_demand)})")
print(f"  Mean: {net_demand.mean():.0f} MW")

# ==========================================
# HYBRID SOLVER
# ==========================================
print("\n" + "=" * 90)
print("MULTI-PERIOD HYBRID OPTIMIZATION")
print("=" * 90)

total_start = time.time()

# Normalize
scale_power = 100.0
scale_cost = 10.0

P_max = thermal_capacity / scale_power
P_min = thermal_min_gen / scale_power
R = thermal_ramp_rate / scale_power
C = thermal_cost / scale_cost

def solve_period_commitment(demand_t, prev_commitment=None):
    """
    Sample commitment candidates for period t using thermodynamic sampling.
    Considers previous commitment for ramping if provided.
    """
    D = demand_t / scale_power
    
    # QUBO parameters
    ALPHA = 200.0
    BETA = 1.0
    
    # Build QUBO
    Q_qubo = np.zeros((n_thermal, n_thermal))
    L_qubo = np.zeros(n_thermal)
    
    for i in range(n_thermal):
        demand_term = ALPHA * (P_max[i]**2 - 2*D*P_max[i])
        cost_term = BETA * C[i] * P_min[i]
        
        # Penalty for changing state (if prev_commitment provided)
        if prev_commitment is not None:
            # Small penalty to encourage continuity
            change_penalty = 5.0 if prev_commitment[i] == 0 else -5.0
            L_qubo[i] = demand_term + cost_term + change_penalty
        else:
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
    
    beta_temp = jnp.array(0.3)
    model = IsingEBM(nodes, edges, jnp.array(biases_list), jnp.array(weights_list), beta_temp)
    program = IsingSamplingProgram(model, [Block(nodes)], clamped_blocks=[])
    
    # Sample candidates (balance between quality and speed)
    candidates = []
    
    for seed in range(8):  # 8 seeds for more diversity
        key = jax.random.key(42 + seed + (0 if prev_commitment is None else 1000))
        k_init, k_samp = jax.random.split(key)
        init_state = hinton_init(k_init, model, [Block(nodes)], ())
        
        schedule = SamplingSchedule(n_warmup=200, n_samples=6, steps_per_sample=3)
        samples = sample_states(k_samp, program, schedule, init_state, [], [Block(nodes)])
        
        for s in samples:
            spins = np.array(s[0]).flatten()
            u_vec = spins.astype(int)
            candidates.append(u_vec)
    
    # Add heuristics (CRITICAL for feasibility)
    if prev_commitment is not None:
        # Reuse previous commitment
        candidates.append(prev_commitment.copy())
        # Add +1 unit to previous
        extended = prev_commitment.copy()
        off_units = np.where(prev_commitment == 0)[0]
        if len(off_units) > 0:
            # Turn on cheapest off unit
            extended[off_units[np.argmin(C[off_units])]] = 1
            candidates.append(extended)
    
    # Greedy cheapest
    sorted_by_cost = np.argsort(C)
    heuristic = np.zeros(n_thermal, dtype=int)
    cumulative = 0
    for idx in sorted_by_cost:
        if cumulative < D:
            heuristic[idx] = 1
            cumulative += P_max[idx]
    candidates.append(heuristic)
    
    # All-on (maximum flexibility)
    candidates.append(np.ones(n_thermal, dtype=int))
    
    # High-capacity units (for high demand)
    sorted_by_capacity = np.argsort(-P_max)  # Descending
    high_cap = np.zeros(n_thermal, dtype=int)
    cumulative = 0
    for idx in sorted_by_capacity:
        if cumulative < D * 1.5:  # 150% coverage
            high_cap[idx] = 1
            cumulative += P_max[idx]
    candidates.append(high_cap)
    
    return candidates


def solve_economic_dispatch(u_commitment, demand, p_min, p_max_orig, costs, prev_dispatch=None, ramp_limits=None):
    """
    Solve economic dispatch with optional ramping constraints.
    """
    available = np.where(u_commitment == 1)[0]
    
    if len(available) == 0:
        return None, np.inf, False, "No units"
    
    # Copy p_max to modify with ramping
    p_max = p_max_orig.copy()
    p_min_adj = p_min.copy()
    
    # Apply ramping constraints if previous dispatch provided
    if prev_dispatch is not None and ramp_limits is not None:
        for idx in available:
            if prev_dispatch[idx] > 1e-6:  # Was ON in previous period
                # Apply ramp limits
                p_max[idx] = min(p_max[idx], prev_dispatch[idx] + ramp_limits[idx])
                p_min_adj[idx] = max(p_min[idx], prev_dispatch[idx] - ramp_limits[idx])
            # If was OFF, can start at p_min (no ramping from 0)
    
    max_gen = np.sum(p_max[available])
    min_gen = np.sum(p_min_adj[available])
    
    if demand > max_gen:
        return None, np.inf, False, "Insufficient capacity"
    
    if demand < min_gen:
        return None, np.inf, False, "Below minimum"
    
    sorted_idx = available[np.argsort(costs[available])]
    
    dispatch = np.zeros(len(costs))
    remaining = demand
    
    # Minimum generation (use adjusted minimum)
    for idx in sorted_idx:
        dispatch[idx] = p_min_adj[idx]
        remaining -= p_min_adj[idx]
    
    # Add remaining
    for idx in sorted_idx:
        if remaining <= 0:
            break
        available_capacity = p_max[idx] - dispatch[idx]
        take = min(remaining, available_capacity)
        dispatch[idx] += take
        remaining -= take
    
    total_gen = np.sum(dispatch)
    if abs(total_gen - demand) > 1e-2:
        return None, np.inf, False, "Dispatch error"
    
    total_cost = np.sum(dispatch * scale_power * costs * scale_cost)
    
    return dispatch, total_cost, True, "Feasible"


# Solve period by period
print(f"\nSolving {n_periods} periods sequentially...")
print("(This will take several minutes)")

schedule = []
total_cost = 0.0
prev_commitment = None
prev_dispatch = None

sampling_times = []
evaluation_times = []

for t in range(n_periods):
    if t % 10 == 0:
        print(f"  Period {t+1}/{n_periods} (hour {t*0.25:.1f})...")
    
    # Sample candidates
    t_sample_start = time.time()
    candidates = solve_period_commitment(net_demand[t], prev_commitment)
    sampling_times.append(time.time() - t_sample_start)
    
    # Evaluate candidates
    t_eval_start = time.time()
    best_u = None
    best_dispatch = None
    best_cost = np.inf
    
    seen = set()
    for u in candidates:
        u_tuple = tuple(u)
        if u_tuple in seen:
            continue
        seen.add(u_tuple)
        
        dispatch, cost, feasible, msg = solve_economic_dispatch(
            u, net_demand[t] / scale_power, P_min, P_max, C,
            prev_dispatch, R
        )
        
        if feasible and cost < best_cost:
            best_cost = cost
            best_dispatch = dispatch
            best_u = u
    
    # If no solution found, try with relaxed ramping (1.5x)
    if best_u is None and prev_dispatch is not None:
        print(f"    Period {t+1}: Relaxing ramping constraints...")
        R_relaxed = R * 1.5
        
        for u in candidates:
            u_tuple = tuple(u)
            dispatch, cost, feasible, msg = solve_economic_dispatch(
                u, net_demand[t] / scale_power, P_min, P_max, C,
                prev_dispatch, R_relaxed
            )
            
            if feasible and cost < best_cost:
                best_cost = cost
                best_dispatch = dispatch
                best_u = u
    
    evaluation_times.append(time.time() - t_eval_start)
    
    if best_u is None:
        print(f"\n[ERROR] No feasible solution for period {t+1}")
        print(f"  Demand: {net_demand[t]:.0f} MW")
        print(f"  Previous dispatch: {prev_dispatch * scale_power if prev_dispatch is not None else 'None'}")
        break
    
    schedule.append({
        'period': t,
        'hour': t * 0.25,
        'demand': net_demand[t],
        'commitment': best_u,
        'dispatch': best_dispatch * scale_power,
        'cost': best_cost,
        'units_on': np.sum(best_u)
    })
    
    total_cost += best_cost
    prev_commitment = best_u
    prev_dispatch = best_dispatch

total_time = time.time() - total_start

print(f"\n{'=' * 90}")
print("HYBRID SOLUTION COMPLETE")
print("=" * 90)

if len(schedule) == n_periods:
    print(f"\n[SUCCESS] Solved all {n_periods} periods")
    
    avg_sampling = np.mean(sampling_times)
    avg_evaluation = np.mean(evaluation_times)
    avg_period = avg_sampling + avg_evaluation
    
    print(f"\nTiming Breakdown:")
    print(f"  Total time: {total_time:.1f} seconds ({total_time/60:.1f} minutes)")
    print(f"  Avg per period: {avg_period:.3f}s")
    print(f"    - Sampling: {avg_sampling:.3f}s ({avg_sampling/avg_period*100:.1f}%)")
    print(f"    - Evaluation: {avg_evaluation:.3f}s ({avg_evaluation/avg_period*100:.1f}%)")
    
    print(f"\nSolution Quality:")
    print(f"  Total cost: €{total_cost:,.2f}")
    
    total_generation = sum(s['dispatch'].sum() for s in schedule)
    total_demand = net_demand.sum()
    
    print(f"  Total generation: {total_generation:,.0f} MW")
    print(f"  Total demand: {total_demand:,.0f} MW")
    print(f"  Satisfaction error: {abs(total_generation - total_demand):.1f} MW")
    
    units_on_avg = np.mean([s['units_on'] for s in schedule])
    units_on_max = max(s['units_on'] for s in schedule)
    units_on_min = min(s['units_on'] for s in schedule)
    
    print(f"\nCommitment Statistics:")
    print(f"  Avg units ON: {units_on_avg:.1f} / {n_thermal}")
    print(f"  Max units ON: {units_on_max} (period {[s['period'] for s in schedule if s['units_on'] == units_on_max][0]})")
    print(f"  Min units ON: {units_on_min} (period {[s['period'] for s in schedule if s['units_on'] == units_on_min][0]})")
    
    # Save results
    result_data = {
        'scenario': 'scenario_00286.json',
        'method': 'hybrid',
        'n_periods': n_periods,
        'n_thermal': n_thermal,
        'total_time_seconds': total_time,
        'avg_time_per_period': avg_period,
        'total_cost_eur': float(total_cost),
        'total_generation_mw': float(total_generation),
        'total_demand_mw': float(total_demand),
        'units_on_avg': float(units_on_avg),
        'units_on_max': int(units_on_max),
        'units_on_min': int(units_on_min),
        'success': True
    }
    
    output_file = Path(__file__).parent / 'hybrid_result_96periods.json'
    with open(output_file, 'w') as f:
        json.dump(result_data, f, indent=2)
    
    print(f"\n{'=' * 90}")
    print(f"Results saved to: hybrid_result_96periods.json")
    print("=" * 90)
    
else:
    print(f"\n[FAILED] Only solved {len(schedule)} / {n_periods} periods")
    
    result_data = {
        'scenario': 'scenario_00286.json',
        'method': 'hybrid',
        'success': False,
        'periods_solved': len(schedule),
        'periods_total': n_periods,
        'total_time_seconds': total_time
    }
    
    output_file = Path(__file__).parent / 'hybrid_result_96periods.json'
    with open(output_file, 'w') as f:
        json.dump(result_data, f, indent=2)
