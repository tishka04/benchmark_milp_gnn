"""
MILP Solver for Large-Scale Scenario (with timeout)

Compares classical MILP approach to hybrid on scenario_00286.json
"""
import json
import numpy as np
from scipy.optimize import milp, LinearConstraint, Bounds
from pathlib import Path
import time
import signal

# Timeout handling
class TimeoutException(Exception):
    pass

def timeout_handler(signum, frame):
    raise TimeoutException()

print("=" * 90)
print("MILP SOLVER: LARGE-SCALE SCENARIO (WITH TIMEOUT)")
print("=" * 90)

# Load scenario
scenario_path = Path(__file__).parent.parent / "outputs" / "scenarios_v1" / "scenario_00286.json"

with open(scenario_path, 'r') as f:
    scenario = json.load(f)

print(f"\nScenario ID: {scenario['id']}")
print(f"File: scenario_00286.json")

assets = scenario['meta']['assets']
n_thermal = assets['thermal']

print(f"\nComplexity:")
print(f"  Thermal units: {n_thermal}")
print(f"  Estimated full MILP time: {scenario['estimates']['est_cpu_hours']:.2f} hours")

# Same parameters as hybrid solver
np.random.seed(42)

thermal_capacity = np.random.uniform(100, 800, n_thermal)
thermal_min_gen = thermal_capacity * 0.3

base_cost = np.random.uniform(10, 30, n_thermal)
fuel_cost = scenario['operation_costs']['thermal_fuel_eur_per_mwh']
co2_intensity = np.random.uniform(0.3, 1.0, n_thermal)
co2_price = scenario['econ_policy']['co2_price']

thermal_cost = base_cost + fuel_cost * scenario['tech']['thermal_marg_cost'] / 10 + co2_intensity * co2_price

base_demand = 5000.0
demand_scale = scenario['exogenous']['demand_scale_factor']
target_demand = base_demand * demand_scale

renewable_factor = 0.25
renewable_gen = (assets['solar'] * 50 + assets['wind'] * 80) * renewable_factor
net_demand = target_demand - renewable_gen

print(f"\nProblem Size:")
print(f"  Net thermal demand: {net_demand:.1f} MW")
print(f"  Total capacity: {thermal_capacity.sum():.1f} MW")
print(f"  Variables: {2 * n_thermal} ({n_thermal} binary + {n_thermal} continuous)")

# ==========================================
# MILP FORMULATION
# ==========================================
print("\n" + "=" * 90)
print("MILP FORMULATION")
print("=" * 90)

n_vars = 2 * n_thermal

# Objective: Minimize sum(cost_i * p_i)
c = np.zeros(n_vars)
c[n_thermal:] = thermal_cost

# Constraints
constraints_list = []

# 1. Demand constraint
A_demand = np.zeros((1, n_vars))
A_demand[0, n_thermal:] = 1.0
constraints_list.append(LinearConstraint(A_demand, lb=net_demand, ub=net_demand))

# 2. Minimum generation constraints
for i in range(n_thermal):
    A_min = np.zeros(n_vars)
    A_min[n_thermal + i] = 1.0
    A_min[i] = -thermal_min_gen[i]
    constraints_list.append(LinearConstraint(A_min, lb=0, ub=np.inf))

# 3. Maximum generation constraints
for i in range(n_thermal):
    A_max = np.zeros(n_vars)
    A_max[n_thermal + i] = 1.0
    A_max[i] = -thermal_capacity[i]
    constraints_list.append(LinearConstraint(A_max, lb=-np.inf, ub=0))

# Variable bounds
bounds = Bounds(
    lb=np.concatenate([np.zeros(n_thermal), np.zeros(n_thermal)]),
    ub=np.concatenate([np.ones(n_thermal), thermal_capacity])
)

# Integrality
integrality = np.concatenate([np.ones(n_thermal), np.zeros(n_thermal)])

print(f"\nMILP Problem:")
print(f"  Variables: {n_vars}")
print(f"  Constraints: {len(constraints_list)}")
print(f"  Binary variables: {n_thermal}")
print(f"  Continuous variables: {n_thermal}")

# ==========================================
# SOLVE WITH TIMEOUT
# ==========================================
print("\n" + "=" * 90)
print("SOLVING MILP")
print("=" * 90)

TIMEOUT_SECONDS = 300  # 5 minutes

print(f"\nTimeout set to: {TIMEOUT_SECONDS} seconds ({TIMEOUT_SECONDS/60:.1f} minutes)")
print(f"Note: Full optimal MILP would take ~{scenario['estimates']['est_cpu_hours']:.1f} hours")
print(f"\nSolving... (this may take several minutes)")

start_time = time.time()

try:
    # Set timeout (Unix-like systems)
    try:
        signal.signal(signal.SIGALRM, timeout_handler)
        signal.alarm(TIMEOUT_SECONDS)
    except AttributeError:
        # Windows doesn't support SIGALRM
        pass
    
    result = milp(
        c=c,
        constraints=constraints_list,
        bounds=bounds,
        integrality=integrality,
        options={'disp': True, 'time_limit': TIMEOUT_SECONDS}
    )
    
    # Cancel alarm
    try:
        signal.alarm(0)
    except AttributeError:
        pass
    
    solve_time = time.time() - start_time
    
    # ==========================================
    # RESULTS
    # ==========================================
    if result.success:
        print(f"\n[SUCCESS] Solution found in {solve_time:.2f} seconds")
        
        u_opt = result.x[:n_thermal]
        p_opt = result.x[n_thermal:]
        
        u_binary = (u_opt > 0.5).astype(int)
        
        total_dispatch = np.sum(p_opt)
        total_cost = np.sum(thermal_cost * p_opt)
        n_units_on = np.sum(u_binary)
        
        print("\n" + "=" * 90)
        print("MILP SOLUTION")
        print("=" * 90)
        
        print(f"\n{'Unit':<6} {'Status':<8} {'Capacity':<12} {'Dispatch':<12} {'Cost/MW':<12} {'Total Cost':<14}")
        print("-" * 90)
        
        units_on = []
        for i in range(n_thermal):
            if u_binary[i] == 1:
                status = "ON"
                capacity = thermal_capacity[i]
                dispatch_mw = p_opt[i]
                cost_per_mw = thermal_cost[i]
                unit_cost = dispatch_mw * cost_per_mw
                units_on.append(i+1)
                
                print(f"{i+1:<6} {status:<8} {capacity:>10.1f} MW {dispatch_mw:>10.1f} MW "
                      f"€{cost_per_mw:>9.2f} €{unit_cost:>12.2f}")
        
        print("-" * 90)
        print(f"{'TOTAL':<6} {'':<8} {thermal_capacity.sum():>10.1f} MW {total_dispatch:>10.1f} MW "
              f"{'':>11} €{total_cost:>12.2f}")
        
        print(f"\nSummary:")
        print(f"  Units committed: {n_units_on} / {n_thermal}")
        print(f"  Total dispatch: {total_dispatch:.1f} MW")
        print(f"  Target demand: {net_demand:.1f} MW")
        print(f"  Error: {abs(total_dispatch - net_demand):.4f} MW")
        print(f"  Total cost: €{total_cost:,.2f}")
        print(f"  Solution time: {solve_time:.2f}s")
        
        if hasattr(result, 'mip_gap'):
            print(f"  MIP gap: {result.mip_gap * 100:.2f}%")
        
        available_capacity = np.sum(thermal_capacity[u_binary == 1])
        reserve = available_capacity - total_dispatch
        reserve_pct = reserve / total_dispatch * 100
        print(f"  Operating reserve: {reserve:.1f} MW ({reserve_pct:.1f}%)")
        
        # Save results
        result_data = {
            'scenario': 'scenario_00286.json',
            'success': True,
            'n_thermal': n_thermal,
            'solve_time_seconds': solve_time,
            'total_cost_eur': float(total_cost),
            'units_committed': int(n_units_on),
            'units_on_ids': units_on,
            'total_dispatch_mw': float(total_dispatch),
            'mip_gap': float(result.mip_gap) if hasattr(result, 'mip_gap') else None
        }
        
    else:
        solve_time = time.time() - start_time
        print(f"\n[PARTIAL] Solution not optimal within timeout")
        print(f"Time elapsed: {solve_time:.2f}s")
        print(f"Message: {result.message}")
        
        result_data = {
            'scenario': 'scenario_00286.json',
            'success': False,
            'message': result.message,
            'solve_time_seconds': solve_time,
            'timeout': TIMEOUT_SECONDS
        }

except TimeoutException:
    solve_time = time.time() - start_time
    print(f"\n[TIMEOUT] MILP solver exceeded {TIMEOUT_SECONDS}s timeout")
    print(f"Time elapsed: {solve_time:.2f}s")
    print(f"\nThis demonstrates why hybrid approach is valuable:")
    print(f"  - MILP: Timed out after {TIMEOUT_SECONDS}s")
    print(f"  - Estimated full MILP time: {scenario['estimates']['est_cpu_hours']:.1f} hours")
    print(f"  - Hybrid: Found solution in ~15s")
    
    result_data = {
        'scenario': 'scenario_00286.json',
        'success': False,
        'timeout': True,
        'solve_time_seconds': solve_time,
        'timeout_limit': TIMEOUT_SECONDS
    }

except Exception as e:
    solve_time = time.time() - start_time
    print(f"\n[ERROR] {str(e)}")
    print(f"Time elapsed: {solve_time:.2f}s")
    
    result_data = {
        'scenario': 'scenario_00286.json',
        'success': False,
        'error': str(e),
        'solve_time_seconds': solve_time
    }

# Save results
output_file = Path(__file__).parent / 'milp_result_large.json'
with open(output_file, 'w') as f:
    json.dump(result_data, f, indent=2)

print(f"\n" + "=" * 90)
print(f"Results saved to: milp_result_large.json")
print("=" * 90)
