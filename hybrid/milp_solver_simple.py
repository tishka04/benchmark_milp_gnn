"""
MILP Monolithic Solver for SIMPLEST Scenario (scenario_00285.json)
10 thermal units, 96 periods - WITH SLACK VARIABLES
"""
import json
import numpy as np
from scipy.optimize import milp, LinearConstraint, Bounds
from pathlib import Path
import time

print("=" * 90)
print("MILP SOLVER: SIMPLEST SCENARIO (10 units, monolithic)")
print("=" * 90)

# Load scenario
scenario_path = Path(__file__).parent.parent / "outputs" / "scenarios_v1" / "scenario_00285.json"
with open(scenario_path, 'r') as f:
    scenario = json.load(f)

n_periods = 96
n_thermal = scenario['meta']['assets']['thermal']

print(f"\nScenario: scenario_00285.json (SIMPLEST)")
print(f"Thermal units: {n_thermal}")
print(f"Problem: {n_thermal * n_periods * 2:,} variables (monolithic)")

# Generate parameters
np.random.seed(42)
thermal_capacity = np.random.uniform(100, 800, n_thermal)
thermal_min_gen = thermal_capacity * 0.3

base_cost = np.random.uniform(10, 30, n_thermal)
fuel_cost = scenario['operation_costs']['thermal_fuel_eur_per_mwh']
co2_intensity = np.random.uniform(0.3, 1.0, n_thermal)
co2_price = scenario['econ_policy']['co2_price']
thermal_cost = base_cost + fuel_cost * scenario['tech']['thermal_marg_cost'] / 10 + co2_intensity * co2_price

# Demand
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

print(f"Demand: {net_demand.min():.0f} - {net_demand.max():.0f} MW")

# Build MILP with slack
n_vars = 2 * n_thermal * n_periods + 2 * n_periods
offset_p = n_thermal * n_periods
offset_slack_under = 2 * n_thermal * n_periods
offset_slack_over = 2 * n_thermal * n_periods + n_periods

SLACK_PENALTY = 10000.0

c = np.zeros(n_vars)
for t in range(n_periods):
    for i in range(n_thermal):
        c[offset_p + t * n_thermal + i] = thermal_cost[i]
for t in range(n_periods):
    c[offset_slack_under + t] = SLACK_PENALTY
    c[offset_slack_over + t] = SLACK_PENALTY

print(f"\nBuilding constraints...")

constraints_list = []

# Demand with slack
for t in range(n_periods):
    A = np.zeros(n_vars)
    for i in range(n_thermal):
        A[offset_p + t * n_thermal + i] = 1.0
    A[offset_slack_under + t] = 1.0
    A[offset_slack_over + t] = -1.0
    constraints_list.append(LinearConstraint(A, lb=net_demand[t], ub=net_demand[t]))

# Min/max generation
for t in range(n_periods):
    for i in range(n_thermal):
        A = np.zeros(n_vars)
        idx_u = t * n_thermal + i
        idx_p = offset_p + t * n_thermal + i
        A[idx_p] = 1.0
        A[idx_u] = -thermal_min_gen[i]
        constraints_list.append(LinearConstraint(A, lb=0, ub=np.inf))
        
        A = np.zeros(n_vars)
        A[idx_p] = 1.0
        A[idx_u] = -thermal_capacity[i]
        constraints_list.append(LinearConstraint(A, lb=-np.inf, ub=0))

print(f"Total constraints: {len(constraints_list):,}")

# Bounds
bounds = Bounds(
    lb=np.concatenate([
        np.zeros(n_thermal * n_periods),
        np.zeros(n_thermal * n_periods),
        np.zeros(n_periods),
        np.zeros(n_periods)
    ]),
    ub=np.concatenate([
        np.ones(n_thermal * n_periods),
        np.repeat(thermal_capacity, n_periods),
        np.full(n_periods, 10000.0),
        np.full(n_periods, 10000.0)
    ])
)

integrality = np.concatenate([
    np.ones(n_thermal * n_periods),
    np.zeros(n_thermal * n_periods),
    np.zeros(n_periods),
    np.zeros(n_periods)
])

print(f"\n{'=' * 90}")
print("SOLVING MILP")
print("=" * 90)
print(f"\nThis should be MUCH faster with only {n_thermal} units!\n")

start_time = time.time()

result = milp(
    c=c,
    constraints=constraints_list,
    bounds=bounds,
    integrality=integrality,
    options={'disp': True, 'time_limit': 3600, 'mip_rel_gap': 0.01}
)

solve_time = time.time() - start_time

if result.success or (hasattr(result, 'x') and result.x is not None):
    print(f"\n[SUCCESS] MILP solved!")
    print(f"Time: {solve_time:.1f}s ({solve_time/60:.1f} min)")
    
    p_solution = result.x[offset_p:offset_slack_under].reshape((n_periods, n_thermal))
    slack_under = result.x[offset_slack_under:offset_slack_over]
    slack_over = result.x[offset_slack_over:]
    
    gen_cost = np.sum(p_solution * thermal_cost)
    slack_cost = SLACK_PENALTY * (np.sum(slack_under) + np.sum(slack_over))
    total_cost = gen_cost + slack_cost
    
    periods_with_slack = np.sum((slack_under > 0.1) | (slack_over > 0.1))
    
    print(f"\nCost: EUR {gen_cost:,.2f} (generation)")
    print(f"Slack: {np.sum(slack_under):.1f} MW under, {np.sum(slack_over):.1f} MW over")
    print(f"Periods with slack: {periods_with_slack}/{n_periods}")
    
    # Compare with hybrid
    try:
        with open(Path(__file__).parent / 'hybrid_result_simple.json', 'r') as f:
            hybrid_data = json.load(f)
        
        if hybrid_data.get('success'):
            hybrid_time = hybrid_data['solve_time_seconds']
            hybrid_cost = hybrid_data['total_cost_eur']
            
            print(f"\n{'=' * 90}")
            print("COMPARISON")
            print("=" * 90)
            print(f"\nHybrid: {hybrid_time:.1f}s, EUR {hybrid_cost:,.2f}")
            print(f"MILP:   {solve_time:.1f}s, EUR {gen_cost:,.2f}")
            
            if solve_time < hybrid_time:
                print(f"\n[RESULT] MILP was {hybrid_time/solve_time:.1f}x FASTER!")
                print(f"  (Simple problem - MILP excels)")
            else:
                print(f"\n[RESULT] Hybrid was {solve_time/hybrid_time:.1f}x faster")
    except:
        pass
    
    result_data = {
        'scenario': 'scenario_00285.json',
        'method': 'milp_monolithic',
        'n_thermal': n_thermal,
        'success': result.success,
        'solve_time_seconds': solve_time,
        'generation_cost_eur': float(gen_cost),
        'slack_under_mw': float(np.sum(slack_under)),
        'periods_with_slack': int(periods_with_slack)
    }
else:
    print(f"\n[FAILED] {result.message}")
    result_data = {'scenario': 'scenario_00285.json', 'method': 'milp_monolithic', 'success': False}

with open(Path(__file__).parent / 'milp_result_simple.json', 'w') as f:
    json.dump(result_data, f, indent=2)

print(f"\n{'=' * 90}")
print("Saved to: milp_result_simple.json")
print("=" * 90)
