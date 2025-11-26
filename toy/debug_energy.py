"""
Debug script to analyze the energy landscape and understand why the solver fails.
"""
import numpy as np

# Problem parameters
gen_capacity = np.array([100.0, 200.0, 50.0, 150.0, 300.0])
gen_cost = np.array([10.0, 15.0, 5.0, 20.0, 25.0])
target_demand = 450.0

scale_power = 100.0
scale_cost = 10.0

P = gen_capacity / scale_power
D = target_demand / scale_power
C = gen_cost / scale_cost

ALPHA = 50.0
BETA = 1.0

num_units = len(P)

# Build QUBO
L_qubo = np.zeros(num_units)
for i in range(num_units):
    L_qubo[i] = ALPHA * (P[i]**2 - 2*D*P[i]) + BETA * C[i]

print("=" * 60)
print("ENERGY LANDSCAPE ANALYSIS")
print("=" * 60)

print(f"\nScaled parameters:")
print(f"P = {P}")
print(f"D = {D}")
print(f"C = {C}")

print(f"\nLinear QUBO coefficients:")
for i in range(num_units):
    print(f"  L[{i}] = {L_qubo[i]:.3f}")

# Compute QUBO energy for all states
print(f"\n" + "-" * 60)
print("QUBO Energy for different states:")
print("-" * 60)

test_states = [
    [0, 0, 0, 0, 0],  # All OFF
    [1, 1, 1, 1, 1],  # All ON
    [1, 1, 0, 1, 0],  # Optimal: Gen 1, 2, 4
    [1, 0, 1, 0, 1],  # Alternative: Gen 1, 3, 5
    [0, 0, 0, 1, 1],  # Alternative: Gen 4, 5
]

for state in test_states:
    u = np.array(state)
    
    # Compute QUBO energy
    gen_total = np.sum(u * P)
    demand_penalty = ALPHA * (gen_total - D)**2
    cost_penalty = BETA * np.sum(u * C)
    energy_qubo = demand_penalty + cost_penalty
    
    # Compute linear contribution
    linear_contrib = np.sum(L_qubo * u)
    
    # Compute quadratic contribution
    quad_contrib = 0.0
    for i in range(num_units):
        for j in range(i + 1, num_units):
            Q_ij = 2 * ALPHA * P[i] * P[j]
            quad_contrib += Q_ij * u[i] * u[j]
    
    energy_qubo_check = linear_contrib + quad_contrib + ALPHA * D**2  # Constant term
    
    gens_on = [i+1 for i in range(num_units) if u[i] == 1]
    gen_mw = np.sum(u * gen_capacity)
    
    print(f"\nState {gens_on if gens_on else 'ALL OFF'}:")
    print(f"  Generation: {gen_mw:.1f} MW (target: {target_demand} MW)")
    print(f"  QUBO Energy: {energy_qubo:.3f}")
    print(f"    - Demand penalty: {demand_penalty:.3f}")
    print(f"    - Cost penalty: {cost_penalty:.3f}")
    print(f"  Linear contrib: {linear_contrib:.3f}")
    print(f"  Quad contrib: {quad_contrib:.3f}")

print("\n" + "=" * 60)
print("DIAGNOSIS:")
print("=" * 60)

print("\nThe problem is clear:")
print("- L_qubo coefficients are MOSTLY NEGATIVE (due to -2*D*P term)")
print("- This makes u=0 (OFF) attractive in the linear term")
print("- The quadratic penalty (which should enforce demand) is NOT strong enough")
print(f"- Linear term for all OFF: {np.sum(L_qubo * 0):.3f} (zero)")
print(f"- But demand penalty for all OFF: {ALPHA * (0 - D)**2:.3f} (huge!)")

print("\nThe QUBO formulation is mathematically correct, BUT:")
print("When mapping to Ising, the negative biases dominate,")
print("causing the thermodynamic solver to prefer u=0 (all OFF).")

print("\nSOLUTION: We need to reformulate the problem or adjust the mapping.")
print("The thermodynamic hardware/simulator is minimizing energy correctly,")
print("but the energy function doesn't correctly represent our optimization goal.")
