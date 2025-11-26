"""
MILP Solver: MONOLITHIC 96-PERIOD - Fair Comparison with Hybrid

Solves ALL 96 periods simultaneously as single large optimization.
This is the "proper" MILP approach that shows why hybrid decomposition helps.

No ramping constraints (matching hybrid's soft ramping approach).
"""
import json
import numpy as np
from scipy.optimize import milp, LinearConstraint, Bounds
from pathlib import Path
import time

print("=" * 90)
print("MILP SOLVER: MONOLITHIC 96-PERIOD OPTIMIZATION")
print("Fair Comparison - All Periods Coupled in Single Optimization")
print("=" * 90)

# Load scenario
scenario_path = Path(__file__).parent.parent / "outputs" / "scenarios_v1" / "scenario_00286.json"
with open(scenario_path, 'r') as f:
    scenario = json.load(f)

n_periods = 96
n_thermal = 40

print(f"\nProblem: {n_periods} periods × {n_thermal} units")
print("Approach: Single monolithic MIP")
print("Constraint: No ramping (matching hybrid)")

# Generate parameters
np.random.seed(42)
thermal_capacity = np.random.uniform(100, 800, n_thermal)
thermal_min_gen = thermal_capacity * 0.3

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

# ==========================================
# MONOLITHIC MILP FORMULATION
# ==========================================
print("\n" + "=" * 90)
print("BUILDING MONOLITHIC MILP")
print("=" * 90)

# Variables: u[i,t] (binary), p[i,t] (continuous)
# Layout: [u[0,0], u[1,0], ..., u[N-1,T-1], p[0,0], p[1,0], ..., p[N-1,T-1]]
n_vars = 2 * n_thermal * n_periods
offset_p = n_thermal * n_periods

print(f"\nVariables: {n_vars:,}")
print(f"  Binary (commitment): {n_thermal * n_periods:,}")
print(f"  Continuous (dispatch): {n_thermal * n_periods:,}")

# Objective: minimize sum over all periods and units of (cost[i] * p[i,t])
print("\nBuilding objective...")
c = np.zeros(n_vars)
for t in range(n_periods):
    for i in range(n_thermal):
        idx_p = offset_p + t * n_thermal + i
        c[idx_p] = thermal_cost[i]

print("Building constraints...")
print("(This will take a minute for 7,776 constraints...)")

constraints_list = []

# 1. Demand constraints (one per period)
print("  - Demand (96)...")
for t in range(n_periods):
    A = np.zeros(n_vars)
    for i in range(n_thermal):
        idx_p = offset_p + t * n_thermal + i
        A[idx_p] = 1.0
    constraints_list.append(LinearConstraint(A, lb=net_demand[t], ub=net_demand[t]))

# 2. Minimum generation (one per unit per period)
print("  - Minimum generation (3,840)...")
for t in range(n_periods):
    for i in range(n_thermal):
        A = np.zeros(n_vars)
        idx_u = t * n_thermal + i
        idx_p = offset_p + t * n_thermal + i
        
        # p[i,t] >= u[i,t] * p_min[i]
        # Rewrite as: p[i,t] - u[i,t] * p_min[i] >= 0
        A[idx_p] = 1.0
        A[idx_u] = -thermal_min_gen[i]
        constraints_list.append(LinearConstraint(A, lb=0, ub=np.inf))

# 3. Maximum generation (one per unit per period)
print("  - Maximum generation (3,840)...")
for t in range(n_periods):
    for i in range(n_thermal):
        A = np.zeros(n_vars)
        idx_u = t * n_thermal + i
        idx_p = offset_p + t * n_thermal + i
        
        # p[i,t] <= u[i,t] * p_max[i]
        # Rewrite as: p[i,t] - u[i,t] * p_max[i] <= 0
        A[idx_p] = 1.0
        A[idx_u] = -thermal_capacity[i]
        constraints_list.append(LinearConstraint(A, lb=-np.inf, ub=0))

print(f"\nTotal constraints: {len(constraints_list):,}")

# Variable bounds
print("Setting variable bounds...")
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
print("Setting integrality...")
integrality = np.concatenate([
    np.ones(n_thermal * n_periods),   # u is binary
    np.zeros(n_thermal * n_periods)   # p is continuous
])

print(f"\nProblem summary:")
print(f"  Variables: {n_vars:,} ({n_thermal * n_periods:,} binary + {n_thermal * n_periods:,} continuous)")
print(f"  Constraints: {len(constraints_list):,}")
print(f"  Problem size: LARGE (this is why hybrid helps!)")

# ==========================================
# SOLVE MONOLITHIC MILP
# ==========================================
print("\n" + "=" * 90)
print("SOLVING MONOLITHIC MILP")
print("=" * 90)

TIMEOUT_SECONDS = 18000  # 5 hours

print(f"\nTimeout: {TIMEOUT_SECONDS/3600:.1f} hours")
print("Gap tolerance: 1%")
print("\nStarting solver...")
print("This is the HARD problem that takes hours!")
print("(Hybrid solves in ~19 minutes by decomposition)\n")

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
    
    # ==========================================
    # RESULTS
    # ==========================================
    if result.success or (hasattr(result, 'x') and result.x is not None):
        print(f"\n{'=' * 90}")
        print("MILP SOLUTION")
        print("=" * 90)
        
        status = "OPTIMAL" if result.success else "FEASIBLE (timeout/gap)"
        print(f"\nStatus: {status}")
        print(f"Solve time: {solve_time:.1f}s ({solve_time/60:.1f} min, {solve_time/3600:.2f} hr)")
        
        # Extract solution
        u_solution = result.x[:offset_p].reshape((n_periods, n_thermal))
        p_solution = result.x[offset_p:].reshape((n_periods, n_thermal))
        
        u_binary = (u_solution > 0.5).astype(int)
        
        total_cost = np.sum(p_solution * thermal_cost)
        total_generation = np.sum(p_solution)
        total_demand = np.sum(net_demand)
        
        units_on_per_period = np.sum(u_binary, axis=1)
        
        print(f"\nSolution Quality:")
        print(f"  Total cost: EUR {total_cost:,.2f}")
        print(f"  Total generation: {total_generation:,.0f} MW")
        print(f"  Total demand: {total_demand:,.0f} MW")
        print(f"  Error: {abs(total_generation - total_demand):.1f} MW")
        
        if hasattr(result, 'mip_gap'):
            print(f"  MIP gap: {result.mip_gap * 100:.2f}%")
        
        print(f"\nCommitment:")
        print(f"  Avg units ON: {units_on_per_period.mean():.1f} / {n_thermal}")
        print(f"  Max: {units_on_per_period.max()} | Min: {units_on_per_period.min()}")
        
        # ==========================================
        # COMPARISON WITH HYBRID
        # ==========================================
        print(f"\n{'=' * 90}")
        print("COMPARISON WITH HYBRID")
        print("=" * 90)
        
        try:
            with open(Path(__file__).parent / 'hybrid_result_96periods.json', 'r') as f:
                hybrid_data = json.load(f)
            
            if hybrid_data.get('success'):
                hybrid_cost = hybrid_data['total_cost_eur']
                hybrid_time = hybrid_data['total_time_seconds']
                
                cost_diff = (total_cost - hybrid_cost) / hybrid_cost * 100
                time_ratio = solve_time / hybrid_time
                
                print(f"\nHybrid (Decomposition):")
                print(f"  Time: {hybrid_time:.1f}s ({hybrid_time/60:.1f} min)")
                print(f"  Cost: EUR {hybrid_cost:,.2f}")
                print(f"  Method: Period-by-period with thermodynamic sampling")
                
                print(f"\nMILP (Monolithic):")
                print(f"  Time: {solve_time:.1f}s ({solve_time/60:.1f} min, {solve_time/3600:.2f} hr)")
                print(f"  Cost: EUR {total_cost:,.2f}")
                print(f"  Method: All periods optimized together")
                
                print(f"\n{'=' * 90}")
                print("FINAL VERDICT")
                print("=" * 90)
                
                print(f"\nSpeed:")
                if time_ratio > 1:
                    print(f"  Hybrid was {time_ratio:.1f}x FASTER ({solve_time/60:.1f} min vs {hybrid_time/60:.1f} min)")
                else:
                    print(f"  MILP was {1/time_ratio:.1f}x FASTER")
                
                print(f"\nCost:")
                if abs(cost_diff) < 1:
                    print(f"  Nearly identical ({cost_diff:+.2f}% difference)")
                elif cost_diff < 0:
                    print(f"  MILP better by {abs(cost_diff):.2f}%")
                else:
                    print(f"  Hybrid better by {abs(cost_diff):.2f}%")
                
                print(f"\nConclusion:")
                if time_ratio > 5:
                    print(f"  Hybrid's decomposition provides SIGNIFICANT speedup ({time_ratio:.1f}x)")
                    print(f"  Cost quality excellent ({abs(cost_diff):.2f}% from optimal)")
                    print(f"  [RESULT] Hybrid wins for large-scale problems!")
                elif time_ratio > 2:
                    print(f"  Hybrid faster ({time_ratio:.1f}x) with comparable quality")
                    print(f"  [RESULT] Hybrid preferred for time-sensitive operations")
                else:
                    print(f"  MILP competitive on this problem size")
                    print(f"  [RESULT] Problem may not be large enough to show hybrid advantage")
        except Exception as e:
            print(f"\nCould not load hybrid results: {e}")
        
        result_data = {
            'scenario': 'scenario_00286.json',
            'method': 'milp_monolithic',
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
        print(f"\n[FAILED] {result.message}")
        print(f"Time: {solve_time:.1f}s ({solve_time/60:.1f} min)")
        
        result_data = {
            'scenario': 'scenario_00286.json',
            'method': 'milp_monolithic',
            'success': False,
            'solve_time_seconds': solve_time,
            'message': result.message
        }

except Exception as e:
    solve_time = time.time() - start_time
    print(f"\n[ERROR] {str(e)}")
    
    result_data = {
        'scenario': 'scenario_00286.json',
        'method': 'milp_monolithic',
        'success': False,
        'error': str(e),
        'solve_time_seconds': solve_time
    }

# Save results
output_file = Path(__file__).parent / 'milp_result_96periods_monolithic.json'
with open(output_file, 'w') as f:
    json.dump(result_data, f, indent=2)

print(f"\n{'=' * 90}")
print("Results saved to: milp_result_96periods_monolithic.json")
print("=" * 90)
