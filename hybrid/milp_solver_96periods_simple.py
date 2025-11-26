"""
MILP Solver: Simplified formulation matching hybrid exactly
"""
import json
import numpy as np
from scipy.optimize import milp, LinearConstraint, Bounds
from pathlib import Path
import time

print("=" * 90)
print("MILP SOLVER: SIMPLIFIED FORMULATION")
print("=" * 90)

# Load and setup (same as hybrid)
scenario_path = Path(__file__).parent.parent / "outputs" / "scenarios_v1" / "scenario_00286.json"
with open(scenario_path, 'r') as f:
    scenario = json.load(f)

n_periods = 96
n_thermal = 40

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

print(f"Periods: {n_periods}, Units: {n_thermal}")
print(f"Demand range: {net_demand.min():.0f} - {net_demand.max():.0f} MW\n")

# ==========================================
# SOLVE PERIOD BY PERIOD (like hybrid)
# ==========================================
print("Solving period-by-period (like hybrid)...")
print("=" * 90)

total_cost = 0
total_gen = 0
start_time = time.time()

for t in range(n_periods):
    if t % 10 == 0:
        print(f"  Period {t+1}/96...")
    
    demand_t = net_demand[t]
    
    # MILP for single period
    # Variables: u[i] (binary), p[i] (continuous) for each unit
    n_vars = 2 * n_thermal
    offset_p = n_thermal
    
    # Objective: minimize cost
    c = np.zeros(n_vars)
    c[offset_p:] = thermal_cost
    
    # Constraints
    constraints = []
    
    # 1. Meet demand: sum(p[i]) = demand_t
    A_demand = np.zeros(n_vars)
    A_demand[offset_p:] = 1.0
    constraints.append(LinearConstraint(A_demand, lb=demand_t, ub=demand_t))
    
    # 2. Min/max generation for each unit
    for i in range(n_thermal):
        # p[i] >= u[i] * p_min[i]
        A = np.zeros(n_vars)
        A[offset_p + i] = 1.0
        A[i] = -thermal_min_gen[i]
        constraints.append(LinearConstraint(A, lb=0, ub=np.inf))
        
        # p[i] <= u[i] * p_max[i]
        A = np.zeros(n_vars)
        A[offset_p + i] = 1.0
        A[i] = -thermal_capacity[i]
        constraints.append(LinearConstraint(A, lb=-np.inf, ub=0))
    
    # Bounds
    bounds = Bounds(
        lb=np.concatenate([np.zeros(n_thermal), np.zeros(n_thermal)]),
        ub=np.concatenate([np.ones(n_thermal), thermal_capacity])
    )
    
    # Integrality
    integrality = np.concatenate([np.ones(n_thermal), np.zeros(n_thermal)])
    
    # Solve
    result = milp(c=c, constraints=constraints, bounds=bounds, integrality=integrality,
                  options={'disp': False, 'time_limit': 60})
    
    if not result.success:
        print(f"\n[FAILED] Period {t+1} infeasible!")
        print(f"  Demand: {demand_t:.1f} MW")
        print(f"  Message: {result.message}")
        break
    
    # Extract solution
    p_solution = result.x[offset_p:]
    period_cost = np.sum(p_solution * thermal_cost)
    period_gen = np.sum(p_solution)
    
    total_cost += period_cost
    total_gen += period_gen

solve_time = time.time() - start_time

if t == n_periods - 1:
    print(f"\n{'=' * 90}")
    print("MILP SOLUTION COMPLETE")
    print("=" * 90)
    
    print(f"\nSolve time: {solve_time:.1f}s ({solve_time/60:.1f} min)")
    print(f"Total cost: EUR {total_cost:,.2f}")
    print(f"Total generation: {total_gen:,.0f} MW")
    print(f"Total demand: {net_demand.sum():,.0f} MW")
    
    # Compare with hybrid
    try:
        with open(Path(__file__).parent / 'hybrid_result_96periods.json', 'r') as f:
            hybrid_data = json.load(f)
        
        if hybrid_data.get('success'):
            hybrid_cost = hybrid_data['total_cost_eur']
            hybrid_time = hybrid_data['total_time_seconds']
            
            gap = (total_cost - hybrid_cost) / hybrid_cost * 100
            speedup = hybrid_time / solve_time  # Who is faster
            
            print(f"\n{'=' * 90}")
            print("COMPARISON")
            print("=" * 90)
            
            print(f"\nHybrid:")
            print(f"  Time: {hybrid_time:.1f}s ({hybrid_time/60:.1f} min)")
            print(f"  Cost: EUR {hybrid_cost:,.2f}")
            
            print(f"\nMILP (period-by-period):")
            print(f"  Time: {solve_time:.1f}s ({solve_time/60:.1f} min)")
            print(f"  Cost: EUR {total_cost:,.2f}")
            
            print(f"\nResults:")
            print(f"  Cost difference: {gap:+.2f}%")
            
            if speedup > 1:
                print(f"  MILP was {speedup:.1f}x FASTER than Hybrid")
            else:
                print(f"  Hybrid was {1/speedup:.1f}x FASTER than MILP")
    except:
        pass
    
    result_data = {
        'scenario': 'scenario_00286.json',
        'method': 'milp_period_by_period',
        'success': True,
        'solve_time_seconds': solve_time,
        'total_cost_eur': float(total_cost),
        'total_generation_mw': float(total_gen)
    }
else:
    result_data = {
        'scenario': 'scenario_00286.json',
        'method': 'milp_period_by_period',
        'success': False,
        'periods_solved': t + 1
    }

output_file = Path(__file__).parent / 'milp_result_96periods_simple.json'
with open(output_file, 'w') as f:
    json.dump(result_data, f, indent=2)

print(f"\n{'=' * 90}")
print("Results saved")
print("=" * 90)
