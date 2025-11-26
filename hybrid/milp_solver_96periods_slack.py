"""
MILP Solver: MONOLITHIC with SLACK VARIABLES

Adds slack variables to demand constraints to ensure feasibility.
Penalizes under/over generation heavily to find best feasible solution.
"""
import json
import numpy as np
from scipy.optimize import milp, LinearConstraint, Bounds
from pathlib import Path
import time

print("=" * 90)
print("MILP SOLVER: MONOLITHIC WITH SLACK VARIABLES")
print("=" * 90)
print("\nSlack variables added to demand constraints")
print("Heavily penalized to find best near-feasible solution\n")

# Load scenario
scenario_path = Path(__file__).parent.parent / "outputs" / "scenarios_v1" / "scenario_00286.json"
with open(scenario_path, 'r') as f:
    scenario = json.load(f)

n_periods = 96
n_thermal = 40

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

print(f"Problem: {n_periods} periods × {n_thermal} units")
print(f"Demand range: {net_demand.min():.0f} - {net_demand.max():.0f} MW\n")

# ==========================================
# MILP WITH SLACK VARIABLES
# ==========================================
print("=" * 90)
print("BUILDING MILP WITH SLACKS")
print("=" * 90)

# Variables:
# - u[i,t]: binary commitment (n_thermal × n_periods)
# - p[i,t]: continuous dispatch (n_thermal × n_periods)
# - slack_under[t]: undergeneration slack (n_periods)
# - slack_over[t]: overgeneration slack (n_periods)

n_vars = 2 * n_thermal * n_periods + 2 * n_periods
offset_p = n_thermal * n_periods
offset_slack_under = 2 * n_thermal * n_periods
offset_slack_over = 2 * n_thermal * n_periods + n_periods

print(f"\nVariables: {n_vars:,}")
print(f"  Binary (commitment): {n_thermal * n_periods:,}")
print(f"  Continuous (dispatch): {n_thermal * n_periods:,}")
print(f"  Slack (under): {n_periods}")
print(f"  Slack (over): {n_periods}")

# Objective: minimize generation cost + heavy penalty for slack
# Penalty: 10,000 EUR/MW (very expensive to use slack)
SLACK_PENALTY = 10000.0

c = np.zeros(n_vars)

# Generation cost
for t in range(n_periods):
    for i in range(n_thermal):
        idx_p = offset_p + t * n_thermal + i
        c[idx_p] = thermal_cost[i]

# Slack penalties
for t in range(n_periods):
    c[offset_slack_under + t] = SLACK_PENALTY  # Penalty for undergeneration
    c[offset_slack_over + t] = SLACK_PENALTY   # Penalty for overgeneration

print(f"\nObjective:")
print(f"  Generation cost: normal")
print(f"  Slack penalty: €{SLACK_PENALTY:,.0f}/MW")

print("\nBuilding constraints...")

constraints_list = []

# 1. Demand constraints WITH SLACK
# sum(p[i,t]) + slack_under[t] - slack_over[t] = demand[t]
print("  - Demand with slack (96)...")
for t in range(n_periods):
    A = np.zeros(n_vars)
    
    # Generation
    for i in range(n_thermal):
        A[offset_p + t * n_thermal + i] = 1.0
    
    # Slack variables
    A[offset_slack_under + t] = 1.0   # Allows undergeneration
    A[offset_slack_over + t] = -1.0   # Allows overgeneration
    
    constraints_list.append(LinearConstraint(A, lb=net_demand[t], ub=net_demand[t]))

# 2. Minimum generation
print("  - Minimum generation (3,840)...")
for t in range(n_periods):
    for i in range(n_thermal):
        A = np.zeros(n_vars)
        idx_u = t * n_thermal + i
        idx_p = offset_p + t * n_thermal + i
        
        A[idx_p] = 1.0
        A[idx_u] = -thermal_min_gen[i]
        constraints_list.append(LinearConstraint(A, lb=0, ub=np.inf))

# 3. Maximum generation
print("  - Maximum generation (3,840)...")
for t in range(n_periods):
    for i in range(n_thermal):
        A = np.zeros(n_vars)
        idx_u = t * n_thermal + i
        idx_p = offset_p + t * n_thermal + i
        
        A[idx_p] = 1.0
        A[idx_u] = -thermal_capacity[i]
        constraints_list.append(LinearConstraint(A, lb=-np.inf, ub=0))

print(f"\nTotal constraints: {len(constraints_list):,}")

# Variable bounds
bounds = Bounds(
    lb=np.concatenate([
        np.zeros(n_thermal * n_periods),  # u >= 0
        np.zeros(n_thermal * n_periods),  # p >= 0
        np.zeros(n_periods),              # slack_under >= 0
        np.zeros(n_periods)               # slack_over >= 0
    ]),
    ub=np.concatenate([
        np.ones(n_thermal * n_periods),   # u <= 1
        np.repeat(thermal_capacity, n_periods),  # p <= capacity
        np.full(n_periods, 10000.0),      # slack_under <= 10000 MW
        np.full(n_periods, 10000.0)       # slack_over <= 10000 MW
    ])
)

# Integrality
integrality = np.concatenate([
    np.ones(n_thermal * n_periods),    # u is binary
    np.zeros(n_thermal * n_periods),   # p is continuous
    np.zeros(n_periods),               # slack_under is continuous
    np.zeros(n_periods)                # slack_over is continuous
])

print(f"\nProblem size: {n_vars:,} variables, {len(constraints_list):,} constraints")

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
        print(f"Solve time: {solve_time:.1f}s ({solve_time/60:.1f} min, {solve_time/3600:.2f} hr)")
        
        # Extract solution
        u_solution = result.x[:offset_p].reshape((n_periods, n_thermal))
        p_solution = result.x[offset_p:offset_slack_under].reshape((n_periods, n_thermal))
        slack_under = result.x[offset_slack_under:offset_slack_over]
        slack_over = result.x[offset_slack_over:]
        
        u_binary = (u_solution > 0.5).astype(int)
        
        # Calculate costs
        generation_cost = np.sum(p_solution * thermal_cost)
        slack_cost = SLACK_PENALTY * (np.sum(slack_under) + np.sum(slack_over))
        total_cost = generation_cost + slack_cost
        
        total_generation = np.sum(p_solution)
        total_demand = np.sum(net_demand)
        total_slack_under = np.sum(slack_under)
        total_slack_over = np.sum(slack_over)
        
        units_on_per_period = np.sum(u_binary, axis=1)
        
        print(f"\nSolution Quality:")
        print(f"  Generation cost: EUR {generation_cost:,.2f}")
        print(f"  Slack penalty: EUR {slack_cost:,.2f}")
        print(f"  Total cost: EUR {total_cost:,.2f}")
        
        print(f"\nDemand Satisfaction:")
        print(f"  Total generation: {total_generation:,.0f} MW")
        print(f"  Total demand: {total_demand:,.0f} MW")
        print(f"  Undergeneration slack: {total_slack_under:.1f} MW")
        print(f"  Overgeneration slack: {total_slack_over:.1f} MW")
        print(f"  Net error: {total_generation + total_slack_under - total_slack_over - total_demand:.1f} MW")
        
        if hasattr(result, 'mip_gap'):
            print(f"\nMIP gap: {result.mip_gap * 100:.2f}%")
        
        print(f"\nCommitment:")
        print(f"  Avg units ON: {units_on_per_period.mean():.1f} / {n_thermal}")
        print(f"  Max: {units_on_per_period.max()} | Min: {units_on_per_period.min()}")
        
        # Check how much slack was used
        periods_with_slack = np.sum((slack_under > 0.1) | (slack_over > 0.1))
        print(f"\nSlack Usage:")
        print(f"  Periods with slack: {periods_with_slack} / {n_periods}")
        
        if periods_with_slack > 0:
            print(f"  Max undergen: {slack_under.max():.1f} MW")
            print(f"  Max overgen: {slack_over.max():.1f} MW")
        
        # ==========================================
        # COMPARISON
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
                
                # Compare generation cost only (fair comparison)
                cost_diff = (generation_cost - hybrid_cost) / hybrid_cost * 100
                time_ratio = solve_time / hybrid_time
                
                print(f"\nHybrid:")
                print(f"  Time: {hybrid_time:.1f}s ({hybrid_time/60:.1f} min)")
                print(f"  Cost: EUR {hybrid_cost:,.2f}")
                
                print(f"\nMILP Monolithic:")
                print(f"  Time: {solve_time:.1f}s ({solve_time/60:.1f} min, {solve_time/3600:.2f} hr)")
                print(f"  Cost (gen only): EUR {generation_cost:,.2f}")
                print(f"  Cost (with slack): EUR {total_cost:,.2f}")
                
                print(f"\n{'=' * 90}")
                print("VERDICT")
                print("=" * 90)
                
                if time_ratio > 1:
                    print(f"\n[SPEED] Hybrid was {time_ratio:.1f}x FASTER")
                    print(f"  Hybrid: {hybrid_time/60:.1f} min")
                    print(f"  MILP: {solve_time/60:.1f} min")
                else:
                    print(f"\n[SPEED] MILP was {1/time_ratio:.1f}x faster (unexpected!)")
                
                print(f"\n[QUALITY] Cost difference: {cost_diff:+.2f}%")
                
                if periods_with_slack == 0:
                    print(f"  No slack used - MILP found exact solution")
                else:
                    print(f"  Slack used in {periods_with_slack} periods")
                    print(f"  MILP had trouble meeting all constraints exactly")
                
                print(f"\n[CONCLUSION]")
                if time_ratio > 5:
                    print(f"  Hybrid decomposition provides MAJOR speedup!")
                    print(f"  This validates the hybrid approach for large problems.")
                elif time_ratio > 2:
                    print(f"  Hybrid faster, demonstrating decomposition advantage")
                else:
                    print(f"  Comparable performance on this problem size")
        except:
            pass
        
        result_data = {
            'scenario': 'scenario_00286.json',
            'method': 'milp_monolithic_slack',
            'success': result.success,
            'solve_time_seconds': solve_time,
            'generation_cost_eur': float(generation_cost),
            'slack_cost_eur': float(slack_cost),
            'total_cost_eur': float(total_cost),
            'total_generation_mw': float(total_generation),
            'slack_under_mw': float(total_slack_under),
            'slack_over_mw': float(total_slack_over),
            'periods_with_slack': int(periods_with_slack),
            'mip_gap': float(result.mip_gap) if hasattr(result, 'mip_gap') else None
        }
        
    else:
        solve_time = time.time() - start_time
        print(f"\n[FAILED] {result.message}")
        
        result_data = {
            'scenario': 'scenario_00286.json',
            'method': 'milp_monolithic_slack',
            'success': False,
            'solve_time_seconds': solve_time
        }

except Exception as e:
    solve_time = time.time() - start_time
    print(f"\n[ERROR] {str(e)}")
    
    result_data = {
        'scenario': 'scenario_00286.json',
        'method': 'milp_monolithic_slack',
        'success': False,
        'error': str(e)
    }

output_file = Path(__file__).parent / 'milp_result_96periods_slack.json'
with open(output_file, 'w') as f:
    json.dump(result_data, f, indent=2)

print(f"\n{'=' * 90}")
print("Results saved to: milp_result_96periods_slack.json")
print("=" * 90)
