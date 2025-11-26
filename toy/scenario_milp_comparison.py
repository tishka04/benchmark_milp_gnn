"""
MILP Solver for comparison with hybrid approach.
Solves the same thermal commitment problem optimally.
"""
import json
import numpy as np
from scipy.optimize import milp, LinearConstraint, Bounds
from pathlib import Path

print("=" * 80)
print("CLASSICAL MILP SOLVER (Reference Solution)")
print("Scenario: scenario_00001.json")
print("=" * 80)

# Load scenario (same as hybrid solver)
scenario_path = Path(__file__).parent.parent / "outputs" / "scenarios_v1" / "scenario_00001.json"

with open(scenario_path, 'r') as f:
    scenario = json.load(f)

assets = scenario['meta']['assets']
n_thermal = assets['thermal']

# Same parameters as hybrid solver
np.random.seed(42)
thermal_capacity = np.random.uniform(50, 500, n_thermal)
thermal_min_gen = thermal_capacity * 0.3

base_cost = np.random.uniform(5, 15, n_thermal)
fuel_cost = scenario['operation_costs']['thermal_fuel_eur_per_mwh']
co2_intensity = np.random.uniform(0.4, 0.9, n_thermal)
co2_price = scenario['econ_policy']['co2_price']
thermal_cost = base_cost + fuel_cost * scenario['tech']['thermal_marg_cost'] / 10 + co2_intensity * co2_price

base_demand = 800.0
demand_scale = scenario['exogenous']['demand_scale_factor']
target_demand = base_demand * demand_scale
renewable_factor = 0.3
renewable_gen = (assets['solar'] * 20 + assets['wind'] * 30) * renewable_factor
net_demand = target_demand - renewable_gen

print(f"\nThermal Units: {n_thermal}")
print(f"Net Thermal Demand: {net_demand:.1f} MW")

# ==========================================
# MILP FORMULATION
# ==========================================
# Variables: u_i (binary commitment), p_i (continuous dispatch)
# Indices: [u_0, ..., u_{n-1}, p_0, ..., p_{n-1}]

n_vars = 2 * n_thermal  # n binary + n continuous

# Objective: Minimize sum(cost_i * p_i)
c = np.zeros(n_vars)
c[n_thermal:] = thermal_cost  # Cost for dispatch variables

# Constraints:
constraints_list = []

# 1. Demand constraint: sum(p_i) = net_demand
A_demand = np.zeros((1, n_vars))
A_demand[0, n_thermal:] = 1.0  # Sum of all p_i
constraints_list.append(LinearConstraint(A_demand, lb=net_demand, ub=net_demand))

# 2. Minimum generation when ON: p_i >= u_i * P_min_i
# Rearranged: p_i - u_i * P_min_i >= 0
for i in range(n_thermal):
    A_min = np.zeros(n_vars)
    A_min[n_thermal + i] = 1.0  # p_i
    A_min[i] = -thermal_min_gen[i]  # -u_i * P_min_i
    constraints_list.append(LinearConstraint(A_min, lb=0, ub=np.inf))

# 3. Maximum generation: p_i <= u_i * P_max_i
# Rearranged: p_i - u_i * P_max_i <= 0
for i in range(n_thermal):
    A_max = np.zeros(n_vars)
    A_max[n_thermal + i] = 1.0  # p_i
    A_max[i] = -thermal_capacity[i]  # -u_i * P_max_i
    constraints_list.append(LinearConstraint(A_max, lb=-np.inf, ub=0))

# Variable bounds
bounds = Bounds(
    lb=np.concatenate([np.zeros(n_thermal), np.zeros(n_thermal)]),  # All >= 0
    ub=np.concatenate([np.ones(n_thermal), thermal_capacity])  # u <= 1, p <= P_max
)

# Integrality: first n_thermal variables are binary
integrality = np.concatenate([np.ones(n_thermal), np.zeros(n_thermal)])

# ==========================================
# SOLVE
# ==========================================
print("\nSolving MILP...")
print(f"  Variables: {n_vars} ({n_thermal} binary + {n_thermal} continuous)")
print(f"  Constraints: {len(constraints_list)}")

result = milp(
    c=c,
    constraints=constraints_list,
    bounds=bounds,
    integrality=integrality,
    options={'disp': False}
)

# ==========================================
# RESULTS
# ==========================================
if result.success:
    print("\n[SUCCESS] Optimal solution found")
    
    u_opt = result.x[:n_thermal]
    p_opt = result.x[n_thermal:]
    
    # Round binary variables
    u_binary = (u_opt > 0.5).astype(int)
    
    total_dispatch = np.sum(p_opt)
    total_cost = np.sum(thermal_cost * p_opt)
    n_units_on = np.sum(u_binary)
    
    print(f"\n" + "=" * 80)
    print("OPTIMAL MILP SOLUTION")
    print("=" * 80)
    
    print(f"\n{'Unit':<6} {'Status':<8} {'Capacity':<12} {'Dispatch':<12} {'Cost/MW':<12} {'Total Cost':<12}")
    print("-" * 80)
    
    for i in range(n_thermal):
        if u_binary[i] == 1:
            status = "ON"
            capacity = thermal_capacity[i]
            dispatch_mw = p_opt[i]
            cost_per_mw = thermal_cost[i]
            unit_cost = dispatch_mw * cost_per_mw
            
            print(f"{i+1:<6} {status:<8} {capacity:>10.1f} MW {dispatch_mw:>10.1f} MW "
                  f"€{cost_per_mw:>9.2f} €{unit_cost:>10.2f}")
    
    print("-" * 80)
    print(f"{'TOTAL':<6} {'':<8} {thermal_capacity.sum():>10.1f} MW {total_dispatch:>10.1f} MW "
          f"{'':>11} €{total_cost:>10.2f}")
    
    print(f"\nSummary:")
    print(f"  Units committed: {n_units_on} / {n_thermal}")
    print(f"  Total dispatch: {total_dispatch:.1f} MW")
    print(f"  Target demand: {net_demand:.1f} MW")
    print(f"  Error: {abs(total_dispatch - net_demand):.4f} MW")
    print(f"  Total cost: €{total_cost:.2f}")
    print(f"  Optimality gap: {result.mip_gap * 100:.2f}%")
    
    available_capacity = np.sum(thermal_capacity[u_binary == 1])
    reserve = available_capacity - total_dispatch
    reserve_pct = reserve / total_dispatch * 100
    print(f"  Operating reserve: {reserve:.1f} MW ({reserve_pct:.1f}%)")
    
    print("\n" + "=" * 80)
    print("This is the GLOBALLY OPTIMAL solution")
    print("Compare with hybrid solver to assess solution quality")
    print("=" * 80)
    
else:
    print(f"\n[FAILED] Optimization failed: {result.message}")
