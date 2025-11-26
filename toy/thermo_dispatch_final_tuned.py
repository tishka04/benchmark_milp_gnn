import jax
import jax.numpy as jnp
import numpy as np
from thrml import SpinNode, Block, SamplingSchedule, sample_states
from thrml.models import IsingEBM, IsingSamplingProgram, hinton_init

def qubo_to_ising(Q, L):
    """Standard QUBO to Ising conversion without offset tricks."""
    n = len(L)
    h_ising = np.zeros(n)
    J_ising = np.zeros((n, n))
    
    for i in range(n):
        h_ising[i] += L[i] / 2.0
    
    for i in range(n):
        for j in range(i + 1, n):
            if Q[i, j] == 0:
                continue
            J_ising[i, j] += Q[i, j] / 4.0
            J_ising[j, i] += Q[i, j] / 4.0
            h_ising[i] += Q[i, j] / 4.0
            h_ising[j] += Q[i, j] / 4.0
    
    return h_ising, J_ising

def spin_to_binary(spin_state):
    """Convert thrml spin state to binary."""
    spin_values = np.where(spin_state, 1, -1)
    binary = (spin_values + 1) // 2
    return binary

print("=" * 70)
print("THERMODYNAMIC DISPATCH - FINAL TUNED VERSION")
print("=" * 70)

gen_capacity = np.array([100.0, 200.0, 50.0, 150.0, 300.0])
gen_cost = np.array([10.0, 15.0, 5.0, 20.0, 25.0])
target_demand = 450.0
num_units = len(gen_capacity)

print(f"\nGenerators:")
for i in range(num_units):
    print(f"  Gen {i+1}: {gen_capacity[i]:6.1f} MW @ ${gen_cost[i]:5.1f}/MW")
print(f"\nTarget: {target_demand} MW")
print("\nOptimal solution (MILP): Gens [1, 2, 4] ON, Cost $7000")

# Normalization
scale_power = 100.0
scale_cost = 10.0
P = gen_capacity / scale_power
D = target_demand / scale_power
C = gen_cost / scale_cost

# KEY: Balance ALPHA and BETA carefully
# ALPHA must be large enough that the quadratic coupling terms overcome negative linear biases
# But not so large that it makes everything turn on
ALPHA = 200.0  # Medium-high penalty
BETA = 0.1     # Low cost weight (constraint satisfaction is priority)

print(f"\nPenalty Coefficients: ALPHA={ALPHA}, BETA={BETA}")

# Build QUBO
Q_qubo = np.zeros((num_units, num_units))
L_qubo = np.zeros(num_units)

for i in range(num_units):
    L_qubo[i] = ALPHA * (P[i]**2 - 2*D*P[i]) + BETA * C[i]
    for j in range(i + 1, num_units):
        Q_qubo[i, j] = 2 * ALPHA * P[i] * P[j]

# Convert to Ising
h_ising, J_ising = qubo_to_ising(Q_qubo, L_qubo)

print(f"Ising Hamiltonian: h in [{h_ising.min():.1f}, {h_ising.max():.1f}], " + 
      f"J in [{J_ising.min():.1f}, {J_ising.max():.1f}]")

# Build model
nodes = [SpinNode() for _ in range(num_units)]
edges = []
weights_list = []
biases_list = []

for i in range(num_units):
    biases_list.append(h_ising[i])
    for j in range(i + 1, num_units):
        if J_ising[i, j] != 0:
            edges.append((nodes[i], nodes[j]))
            weights_list.append(J_ising[i, j])

# KEY: Use VERY LOW temperature (high beta) for deterministic optimization
# This makes the system more likely to settle into global minimum
beta = jnp.array(30.0)  # INCREASED from 10 to 30 (colder)

print(f"Inverse temperature: beta={float(beta)} (cold = deterministic)")

model = IsingEBM(nodes, edges, jnp.array(biases_list), jnp.array(weights_list), beta)
program = IsingSamplingProgram(model, [Block(nodes)], clamped_blocks=[])

# Try multiple random seeds to find best solution
print(f"\nRunning multiple trials with different initializations...")

best_solution = None
best_error = float('inf')
best_cost = float('inf')

for trial in range(10):  # Run 10 trials
    key = jax.random.key(42 + trial)
    k_init, k_samp = jax.random.split(key)
    init_state = hinton_init(k_init, model, [Block(nodes)], ())
    
    # Long warmup with many samples
    schedule = SamplingSchedule(n_warmup=10000, n_samples=500, steps_per_sample=20)
    samples = sample_states(k_samp, program, schedule, init_state, [], [Block(nodes)])
    
    # Check last few samples for best result
    for sample in samples[-50:]:  # Check last 50 samples
        spin_state = np.array(sample[0]).flatten()
        binary = spin_to_binary(spin_state)
        
        gen_total = np.sum(binary * gen_capacity)
        error = abs(gen_total - target_demand)
        cost = np.sum(binary * gen_capacity * gen_cost)
        
        # Prioritize constraint satisfaction, then cost
        if error < best_error or (error == best_error and cost < best_cost):
            best_error = error
            best_cost = cost
            best_solution = binary.copy()
    
    print(f"  Trial {trial+1}: Best error = {best_error:.1f} MW")

# Display best solution found
print("\n" + "=" * 70)
print("BEST SOLUTION FOUND")
print("=" * 70)

total_generation = 0.0
total_cost = 0.0

for i in range(num_units):
    is_on = best_solution[i]
    status = "ON " if is_on else "OFF"
    power = gen_capacity[i] if is_on else 0.0
    cost = gen_cost[i] * power
    
    if is_on:
        total_generation += power
        total_cost += cost
    
    print(f"Gen {i+1}: {status} | {gen_capacity[i]:6.1f} MW @ ${gen_cost[i]:5.1f}/MW | " + 
          f"Power: {power:6.1f} MW | Cost: ${cost:7.1f}")

print("-" * 70)
print(f"Total Generation: {total_generation:6.1f} MW")
print(f"Target Demand:    {target_demand:6.1f} MW")
print(f"Mismatch:         {total_generation - target_demand:+6.1f} MW")
print(f"Total Cost:       ${total_cost:7.1f}")
print("=" * 70)

if abs(total_generation - target_demand) < 1.0:
    print("[OK] Constraint satisfied!")
    if abs(total_cost - 7000.0) < 100.0:
        print("[OPTIMAL] Found the optimal MILP solution!")
    else:
        print(f"[SUBOPTIMAL] Cost is ${total_cost:.1f} vs optimal $7000")
else:
    error_pct = abs(total_generation - target_demand) / target_demand * 100
    print(f"[PARTIAL] Constraint violated by {abs(total_generation - target_demand):.1f} MW ({error_pct:.1f}%)")
    
print("\n" + "=" * 70)
print("CONCLUSION:")
print("=" * 70)
print("The thermodynamic approach struggles with this problem because:")
print("1. Tight equality constraint (sum P_i * u_i = 450) is hard for continuous methods")
print("2. Negative linear QUBO terms bias the system toward all-OFF state")
print("3. Thermal fluctuations prevent exact constraint satisfaction")
print("4. The energy landscape has many local minima")
print("\nFor discrete optimization with tight constraints, classical MILP is superior.")
print("Thermodynamic computing works better for:")
print("- Inequality constraints (e.g., >= instead of =)")
print("- Softer problems with many near-optimal solutions")
print("- Problems where approximate solutions are acceptable")
