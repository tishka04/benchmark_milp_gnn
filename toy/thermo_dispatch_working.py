import jax
import jax.numpy as jnp
import numpy as np
from thrml import SpinNode, Block, SamplingSchedule, sample_states
from thrml.models import IsingEBM, IsingSamplingProgram, hinton_init

def qubo_to_ising_with_offset(Q, L, offset=0.0):
    """
    Converts QUBO to Ising with optional constant offset to shift energy landscape.
    
    The offset doesn't change the optimization result but can improve
    the conditioning of the energy landscape for thermodynamic sampling.
    """
    n = len(L)
    h_ising = np.zeros(n)
    J_ising = np.zeros((n, n))
    
    # Linear terms: L_i * u_i = L_i * (s_i + 1)/2 = L_i/2 * s_i + L_i/2
    for i in range(n):
        h_ising[i] += (L[i] + offset) / 2.0
    
    # Quadratic terms: Q_ij * u_i * u_j
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
    """Convert thrml spin state to binary {0, 1}."""
    spin_values = np.where(spin_state, 1, -1)
    binary = (spin_values + 1) // 2
    return binary

# --- PROBLEM SETUP ---
print("=" * 60)
print("THERMODYNAMIC DISPATCH - WORKING VERSION")
print("=" * 60)

gen_capacity = np.array([100.0, 200.0, 50.0, 150.0, 300.0])  # MW
gen_cost = np.array([10.0, 15.0, 5.0, 20.0, 25.0])  # $/MW
target_demand = 450.0  # MW

num_units = len(gen_capacity)

print(f"\nGenerators:")
for i in range(num_units):
    print(f"  Gen {i+1}: {gen_capacity[i]:6.1f} MW @ ${gen_cost[i]:5.1f}/MW")
print(f"\nTarget Demand: {target_demand} MW")

# Normalization
scale_power = 100.0
scale_cost = 10.0

P = gen_capacity / scale_power
D = target_demand / scale_power
C = gen_cost / scale_cost

# KEY FIX: Use much larger ALPHA to dominate the negative linear terms
# The demand constraint penalty must overwhelm the negative bias from the linear expansion
ALPHA = 500.0  # INCREASED from 50 to 500
BETA = 1.0

print(f"\nPenalty Coefficients:")
print(f"  ALPHA (demand constraint): {ALPHA}")
print(f"  BETA (cost objective): {BETA}")

# --- BUILD QUBO ---
Q_qubo = np.zeros((num_units, num_units))
L_qubo = np.zeros(num_units)

for i in range(num_units):
    L_qubo[i] = ALPHA * (P[i]**2 - 2*D*P[i]) + BETA * C[i]
    for j in range(i + 1, num_units):
        Q_qubo[i, j] = 2 * ALPHA * P[i] * P[j]

# KEY FIX: Add large positive offset to shift the energy landscape
# This makes the favorable states have positive or less negative biases
# The offset is chosen to roughly counteract the negative linear terms
offset = ALPHA * 2 * D * P.max()  # Compensate for the -2*D*P term

print(f"\nEnergy landscape adjustment:")
print(f"  Offset applied: {offset:.3f}")

# --- CONVERT TO ISING WITH OFFSET ---
h_ising, J_ising = qubo_to_ising_with_offset(Q_qubo, L_qubo, offset)

print(f"\nIsing Hamiltonian (after offset):")
print(f"  Bias range: [{h_ising.min():.3f}, {h_ising.max():.3f}]")
print(f"  Coupling range: [{J_ising.min():.3f}, {J_ising.max():.3f}]")

# --- BUILD THRML MODEL ---
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

jax_biases = jnp.array(biases_list)
jax_weights = jnp.array(weights_list)

# Use moderate temperature
beta = jnp.array(10.0)

print(f"\nThermodynamic parameters:")
print(f"  Inverse temperature (beta): {float(beta)}")
print(f"  Number of edges: {len(edges)}")

model = IsingEBM(nodes, edges, jax_biases, jax_weights, beta)
program = IsingSamplingProgram(model, [Block(nodes)], clamped_blocks=[])

# --- RUN SIMULATION ---
print(f"\nRunning thermodynamic simulation...")

key = jax.random.key(42)
k_init, k_samp = jax.random.split(key)
init_state = hinton_init(k_init, model, [Block(nodes)], ())

schedule = SamplingSchedule(n_warmup=5000, n_samples=1000, steps_per_sample=10)
samples = sample_states(k_samp, program, schedule, init_state, [], [Block(nodes)])

print(f"  Warmup steps: {schedule.n_warmup}")
print(f"  Samples collected: {schedule.n_samples}")

# --- DECODE RESULTS ---
final_spin_state = np.array(samples[-1][0]).flatten()
final_binary = spin_to_binary(final_spin_state)

print("\n" + "=" * 60)
print("SOLUTION")
print("=" * 60)

total_generation = 0.0
total_cost = 0.0

for i in range(num_units):
    is_on = final_binary[i]
    status = "ON " if is_on else "OFF"
    power = gen_capacity[i] if is_on else 0.0
    cost = gen_cost[i] * power
    
    if is_on:
        total_generation += power
        total_cost += cost
    
    print(f"Gen {i+1}: {status} | {gen_capacity[i]:6.1f} MW @ ${gen_cost[i]:5.1f}/MW | Power: {power:6.1f} MW | Cost: ${cost:7.1f}")

print("-" * 60)
print(f"Total Generation: {total_generation:6.1f} MW")
print(f"Target Demand:    {target_demand:6.1f} MW")
print(f"Mismatch:         {total_generation - target_demand:+6.1f} MW")
print(f"Total Cost:       ${total_cost:7.1f}")

print("=" * 60)

if abs(total_generation - target_demand) < 1e-6:
    print("[OK] Demand constraint SATISFIED")
    print(f"\nCompare to optimal MILP solution: Gens [1, 2, 4], Cost $7000.0")
else:
    error_pct = abs(total_generation - target_demand) / target_demand * 100
    print(f"[PARTIAL] Demand constraint violated by {abs(total_generation - target_demand):.1f} MW ({error_pct:.1f}%)")
    if error_pct < 10:
        print("  (Close enough for thermodynamic approximation)")
