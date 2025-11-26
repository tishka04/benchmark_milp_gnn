"""
Reference MILP solution for the unit commitment problem.
This provides the ground truth optimal solution for comparison.
"""
import numpy as np
from scipy.optimize import milp, LinearConstraint, Bounds

# --- PROBLEM SETUP ---
print("=" * 60)
print("REFERENCE MILP SOLUTION")
print("=" * 60)

# Generator specifications
gen_capacity = np.array([100.0, 200.0, 50.0, 150.0, 300.0])  # MW
gen_cost = np.array([10.0, 15.0, 5.0, 20.0, 25.0])  # $/MW
target_demand = 450.0  # MW

num_units = len(gen_capacity)

print(f"\nGenerators:")
for i in range(num_units):
    print(f"  Gen {i+1}: {gen_capacity[i]:6.1f} MW @ ${gen_cost[i]:5.1f}/MW")
print(f"\nTarget Demand: {target_demand} MW")

# --- FORMULATION 1: Exact Demand Matching ---
print("\n" + "-" * 60)
print("FORMULATION 1: Exact Demand Constraint")
print("-" * 60)

# Objective: Minimize sum(C_i * P_i * u_i)
# where u_i is binary (0 or 1)
c = gen_cost * gen_capacity  # Total cost if generator i is ON

# Constraint: sum(P_i * u_i) = D
# In matrix form: A @ u = [b_lower, b_upper]
A = gen_capacity.reshape(1, -1)
constraints = LinearConstraint(A, lb=target_demand, ub=target_demand)

# Variable bounds: u_i ∈ {0, 1}
bounds = Bounds(lb=0, ub=1)

# Integer constraints (all variables are binary)
integrality = np.ones(num_units)

# Solve
result = milp(c=c, constraints=constraints, bounds=bounds, integrality=integrality)

if result.success:
    print("\n[OK] Optimal solution found")
    solution = result.x
    total_gen = 0.0
    total_cost = 0.0
    
    print("\nDispatch:")
    for i in range(num_units):
        is_on = solution[i] > 0.5  # Binary rounding
        status = "ON " if is_on else "OFF"
        power = gen_capacity[i] if is_on else 0.0
        cost = gen_cost[i] * power
        
        if is_on:
            total_gen += power
            total_cost += cost
        
        print(f"Gen {i+1}: {status} | {gen_capacity[i]:6.1f} MW @ ${gen_cost[i]:5.1f}/MW | Power: {power:6.1f} MW | Cost: ${cost:7.1f}")
    
    print("-" * 60)
    print(f"Total Generation: {total_gen:6.1f} MW")
    print(f"Target Demand:    {target_demand:6.1f} MW")
    print(f"Mismatch:         {total_gen - target_demand:+6.1f} MW")
    print(f"Total Cost:       ${total_cost:7.1f}")
else:
    print(f"\n[FAILED] Optimization failed: {result.message}")

# --- FORMULATION 2: Inequality with Cost Penalty ---
print("\n" + "-" * 60)
print("FORMULATION 2: Inequality Constraint (>= D)")
print("-" * 60)

# Objective: Minimize sum(C_i * P_i * u_i)
# Constraint: sum(P_i * u_i) >= D
constraints_ineq = LinearConstraint(A, lb=target_demand, ub=np.inf)

result2 = milp(c=c, constraints=constraints_ineq, bounds=bounds, integrality=integrality)

if result2.success:
    print("\n[OK] Optimal solution found")
    solution2 = result2.x
    total_gen2 = 0.0
    total_cost2 = 0.0
    
    print("\nDispatch:")
    for i in range(num_units):
        is_on = solution2[i] > 0.5
        status = "ON " if is_on else "OFF"
        power = gen_capacity[i] if is_on else 0.0
        cost = gen_cost[i] * power
        
        if is_on:
            total_gen2 += power
            total_cost2 += cost
        
        print(f"Gen {i+1}: {status} | {gen_capacity[i]:6.1f} MW @ ${gen_cost[i]:5.1f}/MW | Power: {power:6.1f} MW | Cost: ${cost:7.1f}")
    
    print("-" * 60)
    print(f"Total Generation: {total_gen2:6.1f} MW")
    print(f"Target Demand:    {target_demand:6.1f} MW")
    print(f"Mismatch:         {total_gen2 - target_demand:+6.1f} MW")
    print(f"Total Cost:       ${total_cost2:7.1f}")
else:
    print(f"\n[FAILED] Optimization failed: {result2.message}")

# --- ANALYSIS ---
print("\n" + "=" * 60)
print("ANALYSIS")
print("=" * 60)

# Check all possible combinations (brute force for this small problem)
print("\nAll feasible solutions that meet demand exactly:")

feasible_solutions = []
for mask in range(2**num_units):
    binary = np.array([(mask >> i) & 1 for i in range(num_units)])
    gen = np.sum(binary * gen_capacity)
    if abs(gen - target_demand) < 1e-6:  # Exact match
        cost = np.sum(binary * gen_capacity * gen_cost)
        feasible_solutions.append((binary, gen, cost))

if feasible_solutions:
    # Sort by cost
    feasible_solutions.sort(key=lambda x: x[2])
    
    print(f"\nFound {len(feasible_solutions)} feasible solution(s):")
    for idx, (binary, gen, cost) in enumerate(feasible_solutions):
        gens_on = [i+1 for i in range(num_units) if binary[i] == 1]
        print(f"  {idx+1}. Gens {gens_on}: {gen:.1f} MW, ${cost:.1f}")
    
    print(f"\n* Optimal solution: Gens {[i+1 for i in range(num_units) if feasible_solutions[0][0][i] == 1]}")
    print(f"   Cost: ${feasible_solutions[0][2]:.1f}")
else:
    print(f"\n[WARNING] No exact solutions exist for target demand {target_demand} MW")
    print("   Generators cannot be combined to exactly meet demand.")
    
    # Find closest solutions
    print("\nClosest solutions:")
    all_solutions = []
    for mask in range(2**num_units):
        binary = np.array([(mask >> i) & 1 for i in range(num_units)])
        gen = np.sum(binary * gen_capacity)
        cost = np.sum(binary * gen_capacity * gen_cost)
        error = abs(gen - target_demand)
        all_solutions.append((binary, gen, cost, error))
    
    # Sort by error, then by cost
    all_solutions.sort(key=lambda x: (x[3], x[2]))
    
    for idx in range(min(5, len(all_solutions))):
        binary, gen, cost, error = all_solutions[idx]
        gens_on = [i+1 for i in range(num_units) if binary[i] == 1]
        print(f"  {idx+1}. Gens {gens_on}: {gen:.1f} MW (±{error:.1f} MW), ${cost:.1f}")

print("=" * 60)
