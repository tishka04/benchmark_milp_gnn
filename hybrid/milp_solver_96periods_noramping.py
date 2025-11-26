"""
MILP Solver: FULL 96-PERIOD Problem (NO RAMPING - For Fair Comparison)

Removes ramping constraints to allow fair comparison with hybrid solver.
This matches the hybrid approach which uses ramping as soft preference, not hard constraint.
"""
import json
import numpy as np
from scipy.optimize import milp, LinearConstraint, Bounds
from pathlib import Path
import time

print("=" * 90)
print("MILP SOLVER: FULL 96-PERIOD (NO RAMPING CONSTRAINTS)")
print("Fair Comparison with Hybrid Approach")
print("=" * 90)
print("\nNote: Ramping constraints removed for fair comparison")
print("      (Hybrid uses soft ramping preference, not hard constraint)")

# Load scenario
scenario_path = Path(__file__).parent.parent / "outputs" / "scenarios_v1" / "scenario_00286.json"

with open(scenario_path, 'r') as f:
    scenario = json.load(f)

n_periods = (scenario['horizon_hours'] * 60) // scenario['dt_minutes']
assets = scenario['meta']['assets']
n_thermal = assets['thermal']

print(f"\nScenario: {n_periods} periods, {n_thermal} thermal units")

# Generate parameters (same as hybrid)
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

renewable_profile = np.zeros(n_periods)
for t, h in enumerate(hours):
    solar = assets['solar'] * 50 * max(0, np.sin(np.pi * (h - 6) / 12))
    wind = assets['wind'] * 80 * (0.3 + 0.2 * np.sin(2 * np.pi * h / 24 + 1.5))
    renewable_profile[t] = solar + wind

net_demand = demand_profile - renewable_profile

print(f"Demand: {net_demand.min():.0f} - {net_demand.max():.0f} MW")

# ==========================================
# SIMPLE MILP FORMULATION (NO RAMPING)
# ==========================================
print("\n" + "=" * 90)
print("BUILDING MILP")
print("=" * 90)

n_vars = 2 * n_thermal * n_periods
offset_p = n_thermal * n_periods

print(f"\nVariables: {n_vars:,} ({n_thermal * n_periods:,} binary + {n_thermal * n_periods:,} continuous)")

# Objective
c = np.zeros(n_vars)
for t in range(n_periods):
    for i in range(n_thermal):
        idx_p = offset_p + t * n_thermal + i
        c[idx_p] = thermal_cost[i]

print("Building constraints...")

constraints_list = []

# 1. Demand
print("  - Demand (96)...")
for t in range(n_periods):
    A = np.zeros(n_vars)
    for i in range(n_thermal):
        A[offset_p + t * n_thermal + i] = 1.0
    constraints_list.append(LinearConstraint(A, lb=net_demand[t], ub=net_demand[t]))

# 2. Min generation
print("  - Min generation (3,840)...")
for t in range(n_periods):
    for i in range(n_thermal):
        A = np.zeros(n_vars)
        idx_u = t * n_thermal + i
        idx_p = offset_p + t * n_thermal + i
        A[idx_p] = 1.0
        A[idx_u] = -thermal_min_gen[i]
        constraints_list.append(LinearConstraint(A, lb=0, ub=np.inf))

# 3. Max generation
print("  - Max generation (3,840)...")
for t in range(n_periods):
    for i in range(n_thermal):
        A = np.zeros(n_vars)
        idx_u = t * n_thermal + i
        idx_p = offset_p + t * n_thermal + i
        A[idx_p] = 1.0
        A[idx_u] = -thermal_capacity[i]
        constraints_list.append(LinearConstraint(A, lb=-np.inf, ub=0))

print(f"\nTotal constraints: {len(constraints_list):,}")

# Bounds
bounds = Bounds(
    lb=np.concatenate([np.zeros(n_thermal * n_periods), np.zeros(n_thermal * n_periods)]),
    ub=np.concatenate([np.ones(n_thermal * n_periods), np.repeat(thermal_capacity, n_periods)])
)

# Integrality
integrality = np.concatenate([
    np.ones(n_thermal * n_periods),
    np.zeros(n_thermal * n_periods)
])

# ==========================================
# SOLVE
# ==========================================
print("\n" + "=" * 90)
print("SOLVING MILP")
print("=" * 90)

TIMEOUT_SECONDS = 18000  # 5 hours

print(f"\nTimeout: {TIMEOUT_SECONDS/3600:.1f} hours")
print("Gap tolerance: 1%")
print("\nThis will take a while (expecting 1-3 hours)...\n")

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
        
        # Compare with hybrid
        print(f"\n{'=' * 90}")
        print("COMPARISON WITH HYBRID")
        print("=" * 90)
        
        try:
            with open(Path(__file__).parent / 'hybrid_result_96periods.json', 'r') as f:
                hybrid_data = json.load(f)
            
            if hybrid_data.get('success'):
                hybrid_cost = hybrid_data['total_cost_eur']
                hybrid_time = hybrid_data['total_time_seconds']
                
                gap = (total_cost - hybrid_cost) / hybrid_cost * 100
                speedup = solve_time / hybrid_time
                
                print(f"\nHybrid:")
                print(f"  Time: {hybrid_time:.1f}s ({hybrid_time/60:.1f} min)")
                print(f"  Cost: EUR {hybrid_cost:,.2f}")
                
                print(f"\nMILP:")
                print(f"  Time: {solve_time:.1f}s ({solve_time/60:.1f} min)")
                print(f"  Cost: EUR {total_cost:,.2f}")
                
                print(f"\nComparison:")
                print(f"  MILP time / Hybrid time: {speedup:.1f}x")
                print(f"  Cost difference: {gap:+.2f}%")
                
                if speedup > 1:
                    print(f"  [RESULT] Hybrid was {speedup:.1f}x FASTER")
                else:
                    print(f"  [RESULT] MILP was {1/speedup:.1f}x FASTER")
        except:
            pass
        
        result_data = {
            'scenario': 'scenario_00286.json',
            'method': 'milp_noramping',
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
        print(f"\n[FAILED] No solution")
        print(f"Time: {solve_time:.1f}s")
        
        result_data = {
            'scenario': 'scenario_00286.json',
            'method': 'milp_noramping',
            'success': False,
            'solve_time_seconds': solve_time
        }

except Exception as e:
    solve_time = time.time() - start_time
    print(f"\n[ERROR] {str(e)}")
    
    result_data = {
        'scenario': 'scenario_00286.json',
        'method': 'milp_noramping',
        'success': False,
        'error': str(e),
        'solve_time_seconds': solve_time
    }

output_file = Path(__file__).parent / 'milp_result_96periods_noramping.json'
with open(output_file, 'w') as f:
    json.dump(result_data, f, indent=2)

print(f"\n{'=' * 90}")
print("Results saved to: milp_result_96periods_noramping.json")
print("=" * 90)
