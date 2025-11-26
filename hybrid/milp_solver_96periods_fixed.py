"""
MILP Solver: FULL 96-PERIOD Problem (FIXED - Proper Startup Handling)

Fixed version that properly handles unit startups/shutdowns in ramping constraints.
"""
import json
import numpy as np
from scipy.optimize import milp, LinearConstraint, Bounds
from pathlib import Path
import time

print("=" * 90)
print("MILP SOLVER: FULL 96-PERIOD (FIXED - Proper Ramping)")
print("Scenario 00286 - Complete 24-hour Schedule")
print("=" * 90)
print("\nFIX: Ramping constraints only apply when unit remains ON")
print("     (Startups/shutdowns exempt from ramping limits)")

# Load scenario
scenario_path = Path(__file__).parent.parent / "outputs" / "scenarios_v1" / "scenario_00286.json"

with open(scenario_path, 'r') as f:
    scenario = json.load(f)

n_periods = (scenario['horizon_hours'] * 60) // scenario['dt_minutes']
assets = scenario['meta']['assets']
n_thermal = assets['thermal']

print(f"\nScenario configuration:")
print(f"  Time periods: {n_periods}")
print(f"  Thermal units: {n_thermal}")

# Generate thermal parameters (same as hybrid)
np.random.seed(42)
thermal_capacity = np.random.uniform(100, 800, n_thermal)
thermal_min_gen = thermal_capacity * 0.3
thermal_ramp_rate = thermal_capacity * scenario['tech']['thermal_ramp_pct']

base_cost = np.random.uniform(10, 30, n_thermal)
fuel_cost = scenario['operation_costs']['thermal_fuel_eur_per_mwh']
co2_intensity = np.random.uniform(0.3, 1.0, n_thermal)
co2_price = scenario['econ_policy']['co2_price']
thermal_cost = base_cost + fuel_cost * scenario['tech']['thermal_marg_cost'] / 10 + co2_intensity * co2_price

# Generate demand profile (same as hybrid)
base_demand = 5000.0
demand_scale = scenario['exogenous']['demand_scale_factor']

hours = np.linspace(0, 24, n_periods)
demand_profile = np.zeros(n_periods)

for t, h in enumerate(hours):
    morning_peak = 1.2 * np.exp(-((h - 8)**2) / 2)
    evening_peak = 1.4 * np.exp(-((h - 19)**2) / 2)
    night_valley = -0.3 * np.exp(-((h - 3)**2) / 2)
    demand_profile[t] = base_demand * demand_scale * (1.0 + morning_peak + evening_peak + night_valley)

renewable_profile = np.zeros(n_periods)
for t, h in enumerate(hours):
    solar = assets['solar'] * 50 * max(0, np.sin(np.pi * (h - 6) / 12))
    wind = assets['wind'] * 80 * (0.3 + 0.2 * np.sin(2 * np.pi * h / 24 + 1.5))
    renewable_profile[t] = solar + wind

net_demand = demand_profile - renewable_profile

print(f"\nDemand characteristics:")
print(f"  Min: {net_demand.min():.0f} MW")
print(f"  Max: {net_demand.max():.0f} MW")
print(f"  Mean: {net_demand.mean():.0f} MW")

# ==========================================
# MILP FORMULATION WITH PROPER RAMPING
# ==========================================
print("\n" + "=" * 90)
print("BUILDING MILP FORMULATION")
print("=" * 90)

# Variables: u[i,t], p[i,t]
n_vars = 2 * n_thermal * n_periods

print(f"\nVariables: {n_vars:,}")
print(f"  Binary (commitment): {n_thermal * n_periods:,}")
print(f"  Continuous (dispatch): {n_thermal * n_periods:,}")

# Objective
c = np.zeros(n_vars)
offset_p = n_thermal * n_periods

for t in range(n_periods):
    for i in range(n_thermal):
        idx_p = offset_p + t * n_thermal + i
        c[idx_p] = thermal_cost[i]

print("\nBuilding constraints...")
print("(This may take a minute...)")

constraints_list = []

# 1. Demand constraints
print("  - Demand constraints...")
for t in range(n_periods):
    A_demand = np.zeros(n_vars)
    for i in range(n_thermal):
        idx_p = offset_p + t * n_thermal + i
        A_demand[idx_p] = 1.0
    constraints_list.append(LinearConstraint(A_demand, lb=net_demand[t], ub=net_demand[t]))

# 2. Minimum generation
print("  - Minimum generation...")
for t in range(n_periods):
    for i in range(n_thermal):
        A_min = np.zeros(n_vars)
        idx_u = t * n_thermal + i
        idx_p = offset_p + t * n_thermal + i
        
        A_min[idx_p] = 1.0
        A_min[idx_u] = -thermal_min_gen[i]
        constraints_list.append(LinearConstraint(A_min, lb=0, ub=np.inf))

# 3. Maximum generation
print("  - Maximum generation...")
for t in range(n_periods):
    for i in range(n_thermal):
        A_max = np.zeros(n_vars)
        idx_u = t * n_thermal + i
        idx_p = offset_p + t * n_thermal + i
        
        A_max[idx_p] = 1.0
        A_max[idx_u] = -thermal_capacity[i]
        constraints_list.append(LinearConstraint(A_max, lb=-np.inf, ub=0))

# 4. FIXED RAMPING CONSTRAINTS
# Only apply ramping if unit is ON in BOTH periods
# Use Big-M formulation to disable ramping when unit is OFF

print("  - Ramping constraints (with startup/shutdown handling)...")

M = 10000  # Big-M constant (larger than any possible ramp)

for t in range(n_periods - 1):
    for i in range(n_thermal):
        idx_u_t = t * n_thermal + i
        idx_u_t1 = (t + 1) * n_thermal + i
        idx_p_t = offset_p + t * n_thermal + i
        idx_p_t1 = offset_p + (t + 1) * n_thermal + i
        
        # Ramp up: p[i,t+1] - p[i,t] <= ramp[i] + M*(2 - u[i,t] - u[i,t+1])
        # Active only if u[i,t]=1 AND u[i,t+1]=1 (unit stays ON)
        A_ramp_up = np.zeros(n_vars)
        A_ramp_up[idx_p_t1] = 1.0
        A_ramp_up[idx_p_t] = -1.0
        A_ramp_up[idx_u_t] = -M
        A_ramp_up[idx_u_t1] = -M
        constraints_list.append(LinearConstraint(A_ramp_up, lb=-np.inf, ub=thermal_ramp_rate[i] + 2*M))
        
        # Ramp down: p[i,t] - p[i,t+1] <= ramp[i] + M*(2 - u[i,t] - u[i,t+1])
        A_ramp_down = np.zeros(n_vars)
        A_ramp_down[idx_p_t] = 1.0
        A_ramp_down[idx_p_t1] = -1.0
        A_ramp_down[idx_u_t] = -M
        A_ramp_down[idx_u_t1] = -M
        constraints_list.append(LinearConstraint(A_ramp_down, lb=-np.inf, ub=thermal_ramp_rate[i] + 2*M))

print(f"\nTotal constraints: {len(constraints_list):,}")

# Variable bounds
bounds = Bounds(
    lb=np.concatenate([
        np.zeros(n_thermal * n_periods),
        np.zeros(n_thermal * n_periods)
    ]),
    ub=np.concatenate([
        np.ones(n_thermal * n_periods),
        np.repeat(thermal_capacity, n_periods)
    ])
)

# Integrality
integrality = np.concatenate([
    np.ones(n_thermal * n_periods),
    np.zeros(n_thermal * n_periods)
])

print(f"\nProblem summary:")
print(f"  Variables: {n_vars:,}")
print(f"  Constraints: {len(constraints_list):,}")
print(f"  Binary vars: {n_thermal * n_periods:,}")

# ==========================================
# SOLVE
# ==========================================
print("\n" + "=" * 90)
print("SOLVING MILP")
print("=" * 90)

TIMEOUT_SECONDS = 18000  # 5 hours

print(f"\nTimeout: {TIMEOUT_SECONDS/3600:.1f} hours")
print("Gap tolerance: 1%")
print("\nStarting solver...\n")

start_time = time.time()

try:
    result = milp(
        c=c,
        constraints=constraints_list,
        bounds=bounds,
        integrality=integrality,
        options={
            'disp': True,
            'time_limit': TIMEOUT_SECONDS,
            'mip_rel_gap': 0.01
        }
    )
    
    solve_time = time.time() - start_time
    
    if result.success or (hasattr(result, 'x') and result.x is not None):
        print(f"\n{'=' * 90}")
        print("MILP SOLUTION")
        print("=" * 90)
        
        status = "OPTIMAL" if result.success else "FEASIBLE"
        print(f"\nStatus: {status}")
        print(f"Solve time: {solve_time:.1f}s ({solve_time/60:.1f} min)")
        
        u_solution = result.x[:n_thermal * n_periods].reshape((n_periods, n_thermal))
        p_solution = result.x[offset_p:].reshape((n_periods, n_thermal))
        
        u_binary = (u_solution > 0.5).astype(int)
        
        total_cost = np.sum(p_solution * thermal_cost)
        total_generation = np.sum(p_solution)
        total_demand = np.sum(net_demand)
        
        units_on_per_period = np.sum(u_binary, axis=1)
        
        print(f"\nSolution Quality:")
        print(f"  Total cost: €{total_cost:,.2f}")
        print(f"  Total generation: {total_generation:,.0f} MW")
        print(f"  Total demand: {total_demand:,.0f} MW")
        print(f"  Error: {abs(total_generation - total_demand):.1f} MW")
        
        if hasattr(result, 'mip_gap'):
            print(f"  MIP gap: {result.mip_gap * 100:.2f}%")
        
        print(f"\nCommitment:")
        print(f"  Avg units ON: {units_on_per_period.mean():.1f} / {n_thermal}")
        print(f"  Max units ON: {units_on_per_period.max()}")
        print(f"  Min units ON: {units_on_per_period.min()}")
        
        result_data = {
            'scenario': 'scenario_00286.json',
            'method': 'milp_fixed',
            'n_periods': n_periods,
            'n_thermal': n_thermal,
            'success': result.success,
            'solve_time_seconds': solve_time,
            'total_cost_eur': float(total_cost),
            'total_generation_mw': float(total_generation),
            'total_demand_mw': float(total_demand),
            'units_on_avg': float(units_on_per_period.mean()),
            'mip_gap': float(result.mip_gap) if hasattr(result, 'mip_gap') else None
        }
    else:
        solve_time = time.time() - start_time
        print(f"\n[FAILED] No solution found")
        print(f"Time: {solve_time:.1f}s")
        print(f"Message: {result.message}")
        
        result_data = {
            'scenario': 'scenario_00286.json',
            'method': 'milp_fixed',
            'success': False,
            'solve_time_seconds': solve_time,
            'message': result.message
        }

except Exception as e:
    solve_time = time.time() - start_time
    print(f"\n[ERROR] {str(e)}")
    
    result_data = {
        'scenario': 'scenario_00286.json',
        'method': 'milp_fixed',
        'success': False,
        'error': str(e),
        'solve_time_seconds': solve_time
    }

output_file = Path(__file__).parent / 'milp_result_96periods_fixed.json'
with open(output_file, 'w') as f:
    json.dump(result_data, f, indent=2)

print(f"\n{'=' * 90}")
print(f"Results saved to: milp_result_96periods_fixed.json")
print("=" * 90)
