"""
MILP Solver: FULL 96-PERIOD Problem

Tests classical MILP on complete 24-hour schedule with 40 thermal units.
This is a LARGE problem that may take hours to solve optimally.
"""
import json
import numpy as np
from scipy.optimize import milp, LinearConstraint, Bounds
from pathlib import Path
import time

print("=" * 90)
print("MILP SOLVER: FULL 96-PERIOD MULTI-TEMPORAL OPTIMIZATION")
print("Scenario 00286 - Complete 24-hour Schedule")
print("=" * 90)
print("\n⚠️  WARNING: This is a LARGE MILP problem!")
print("Expected problem size:")
print("  - 7,680 variables (3,840 binary + 3,840 continuous)")
print("  - ~11,500 constraints")
print("  - Estimated time: 1-3 hours")
print("\nTimeout set to 5 hours. Solution may be suboptimal.")

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
print(f"  Total binary variables: {n_thermal * n_periods:,}")
print(f"  Total continuous variables: {n_thermal * n_periods:,}")

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
# MILP FORMULATION
# ==========================================
print("\n" + "=" * 90)
print("BUILDING MILP FORMULATION")
print("=" * 90)

n_vars = 2 * n_thermal * n_periods  # u[i,t] and p[i,t] for each unit and period

print(f"\nVariables: {n_vars:,}")
print(f"  Binary: {n_thermal * n_periods:,}")
print(f"  Continuous: {n_thermal * n_periods:,}")

# Objective: Minimize total cost over all periods
c = np.zeros(n_vars)

# Cost for dispatch variables p[i,t]
# Variables ordered as: [u[0,0], u[1,0], ..., u[N-1,T-1], p[0,0], p[1,0], ..., p[N-1,T-1]]
offset_p = n_thermal * n_periods

for t in range(n_periods):
    for i in range(n_thermal):
        idx_p = offset_p + t * n_thermal + i
        c[idx_p] = thermal_cost[i]

print("\nBuilding constraints...")

constraints_list = []
n_constraints_added = 0

# 1. Demand constraints (one per period)
print("  - Demand constraints (96)...")
for t in range(n_periods):
    A_demand = np.zeros(n_vars)
    for i in range(n_thermal):
        idx_p = offset_p + t * n_thermal + i
        A_demand[idx_p] = 1.0
    
    constraints_list.append(LinearConstraint(A_demand, lb=net_demand[t], ub=net_demand[t]))
    n_constraints_added += 1

# 2. Minimum generation constraints
print(f"  - Minimum generation ({n_thermal * n_periods:,})...")
for t in range(n_periods):
    for i in range(n_thermal):
        A_min = np.zeros(n_vars)
        idx_u = t * n_thermal + i
        idx_p = offset_p + t * n_thermal + i
        
        A_min[idx_p] = 1.0
        A_min[idx_u] = -thermal_min_gen[i]
        
        constraints_list.append(LinearConstraint(A_min, lb=0, ub=np.inf))
        n_constraints_added += 1

# 3. Maximum generation constraints
print(f"  - Maximum generation ({n_thermal * n_periods:,})...")
for t in range(n_periods):
    for i in range(n_thermal):
        A_max = np.zeros(n_vars)
        idx_u = t * n_thermal + i
        idx_p = offset_p + t * n_thermal + i
        
        A_max[idx_p] = 1.0
        A_max[idx_u] = -thermal_capacity[i]
        
        constraints_list.append(LinearConstraint(A_max, lb=-np.inf, ub=0))
        n_constraints_added += 1

# 4. Ramping constraints (between consecutive periods)
print(f"  - Ramping constraints ({n_thermal * (n_periods-1) * 2:,})...")
for t in range(n_periods - 1):
    for i in range(n_thermal):
        idx_p_t = offset_p + t * n_thermal + i
        idx_p_t1 = offset_p + (t + 1) * n_thermal + i
        
        # Ramp up constraint: p[i,t+1] - p[i,t] <= ramp_rate[i]
        A_ramp_up = np.zeros(n_vars)
        A_ramp_up[idx_p_t1] = 1.0
        A_ramp_up[idx_p_t] = -1.0
        constraints_list.append(LinearConstraint(A_ramp_up, lb=-np.inf, ub=thermal_ramp_rate[i]))
        n_constraints_added += 1
        
        # Ramp down constraint: p[i,t] - p[i,t+1] <= ramp_rate[i]
        A_ramp_down = np.zeros(n_vars)
        A_ramp_down[idx_p_t] = 1.0
        A_ramp_down[idx_p_t1] = -1.0
        constraints_list.append(LinearConstraint(A_ramp_down, lb=-np.inf, ub=thermal_ramp_rate[i]))
        n_constraints_added += 1

print(f"\nTotal constraints: {n_constraints_added:,}")

# Variable bounds
bounds = Bounds(
    lb=np.concatenate([
        np.zeros(n_thermal * n_periods),  # u >= 0
        np.zeros(n_thermal * n_periods)   # p >= 0
    ]),
    ub=np.concatenate([
        np.ones(n_thermal * n_periods),   # u <= 1
        np.repeat(thermal_capacity, n_periods)  # p <= capacity
    ])
)

# Integrality
integrality = np.concatenate([
    np.ones(n_thermal * n_periods),   # u is binary
    np.zeros(n_thermal * n_periods)   # p is continuous
])

print(f"\nProblem summary:")
print(f"  Variables: {n_vars:,}")
print(f"  Constraints: {len(constraints_list):,}")
print(f"  Binary vars: {n_thermal * n_periods:,}")
print(f"  Continuous vars: {n_thermal * n_periods:,}")

# ==========================================
# SOLVE WITH TIMEOUT
# ==========================================
print("\n" + "=" * 90)
print("SOLVING MILP")
print("=" * 90)

TIMEOUT_SECONDS = 18000  # 1 hour

print(f"\nTimeout: {TIMEOUT_SECONDS} seconds ({TIMEOUT_SECONDS/60:.0f} minutes)")
print("Gap tolerance: 1% (may terminate early if within 1% of optimal)")
print("\nStarting solver... (this will take a while)")
print("Progress will be displayed by HiGHS solver...\n")

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
            'mip_rel_gap': 0.01  # 1% gap tolerance
        }
    )
    
    solve_time = time.time() - start_time
    
    # ==========================================
    # RESULTS
    # ==========================================
    if result.success or (hasattr(result, 'x') and result.x is not None):
        print(f"\n{'=' * 90}")
        print("MILP SOLUTION")
        print("=" * 90)
        
        status = "OPTIMAL" if result.success else "FEASIBLE (within gap)"
        print(f"\nStatus: {status}")
        print(f"Solve time: {solve_time:.1f} seconds ({solve_time/60:.1f} minutes)")
        
        # Extract solution
        u_solution = result.x[:n_thermal * n_periods].reshape((n_periods, n_thermal))
        p_solution = result.x[offset_p:].reshape((n_periods, n_thermal))
        
        # Round binary variables
        u_binary = (u_solution > 0.5).astype(int)
        
        # Calculate metrics
        total_cost = np.sum(p_solution * thermal_cost)
        total_generation = np.sum(p_solution)
        total_demand = np.sum(net_demand)
        
        units_on_per_period = np.sum(u_binary, axis=1)
        
        print(f"\nSolution Quality:")
        print(f"  Total cost: €{total_cost:,.2f}")
        print(f"  Total generation: {total_generation:,.0f} MW")
        print(f"  Total demand: {total_demand:,.0f} MW")
        print(f"  Satisfaction error: {abs(total_generation - total_demand):.1f} MW")
        
        if hasattr(result, 'mip_gap'):
            print(f"  MIP gap: {result.mip_gap * 100:.2f}%")
        
        print(f"\nCommitment Statistics:")
        print(f"  Avg units ON: {units_on_per_period.mean():.1f} / {n_thermal}")
        print(f"  Max units ON: {units_on_per_period.max()} (period {np.argmax(units_on_per_period)})")
        print(f"  Min units ON: {units_on_per_period.min()} (period {np.argmin(units_on_per_period)})")
        
        # Check ramping violations
        ramp_violations = 0
        for t in range(n_periods - 1):
            for i in range(n_thermal):
                ramp_change = abs(p_solution[t+1, i] - p_solution[t, i])
                if ramp_change > thermal_ramp_rate[i] + 1e-3:
                    ramp_violations += 1
        
        if ramp_violations > 0:
            print(f"\n⚠️  Ramping violations detected: {ramp_violations}")
        else:
            print(f"\n✓ All ramping constraints satisfied")
        
        # Save results
        result_data = {
            'scenario': 'scenario_00286.json',
            'method': 'milp',
            'n_periods': n_periods,
            'n_thermal': n_thermal,
            'success': result.success,
            'solve_time_seconds': solve_time,
            'total_cost_eur': float(total_cost),
            'total_generation_mw': float(total_generation),
            'total_demand_mw': float(total_demand),
            'units_on_avg': float(units_on_per_period.mean()),
            'units_on_max': int(units_on_per_period.max()),
            'units_on_min': int(units_on_per_period.min()),
            'mip_gap': float(result.mip_gap) if hasattr(result, 'mip_gap') else None,
            'ramp_violations': int(ramp_violations)
        }
        
    else:
        solve_time = time.time() - start_time
        print(f"\n[FAILED] MILP could not find solution")
        print(f"Time elapsed: {solve_time:.1f}s ({solve_time/60:.1f} min)")
        print(f"Message: {result.message}")
        
        result_data = {
            'scenario': 'scenario_00286.json',
            'method': 'milp',
            'success': False,
            'solve_time_seconds': solve_time,
            'message': result.message
        }

except Exception as e:
    solve_time = time.time() - start_time
    print(f"\n[ERROR] {str(e)}")
    print(f"Time elapsed: {solve_time:.1f}s ({solve_time/60:.1f} min)")
    
    result_data = {
        'scenario': 'scenario_00286.json',
        'method': 'milp',
        'success': False,
        'solve_time_seconds': solve_time,
        'error': str(e)
    }

# Save results
output_file = Path(__file__).parent / 'milp_result_96periods.json'
with open(output_file, 'w') as f:
    json.dump(result_data, f, indent=2)

print(f"\n{'=' * 90}")
print(f"Results saved to: milp_result_96periods.json")
print("=" * 90)
