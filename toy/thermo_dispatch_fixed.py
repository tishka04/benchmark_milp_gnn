import jax
import jax.numpy as jnp
import numpy as np
from thrml import SpinNode, Block, SamplingSchedule, sample_states
from thrml.models import IsingEBM, IsingSamplingProgram, hinton_init

def qubo_to_ising(Q, L):
    """
    Converts QUBO (binary 0/1) to Ising (spin -1/+1) model.
    
    QUBO Objective: Minimize E_QUBO = sum_i L_i * u_i + sum_{i<j} Q_ij * u_i * u_j
    where u_i ∈ {0, 1}
    
    Ising Energy: E_Ising = sum_i h_i * s_i + sum_{i<j} J_ij * s_i * s_j
    where s_i ∈ {-1, +1}
    
    Transformation: u_i = (s_i + 1) / 2
    
    Returns:
        h_ising: bias terms (field)
        J_ising: coupling terms (interactions)
    """
    n = len(L)
    h_ising = np.zeros(n)
    J_ising = np.zeros((n, n))
    
    # Expand linear terms: L_i * u_i = L_i * (s_i + 1)/2
    # = L_i/2 * s_i + L_i/2
    for i in range(n):
        h_ising[i] += L[i] / 2.0
    
    # Expand quadratic terms: Q_ij * u_i * u_j 
    # = Q_ij * (s_i + 1)/2 * (s_j + 1)/2
    # = Q_ij/4 * (s_i * s_j + s_i + s_j + 1)
    for i in range(n):
        for j in range(i + 1, n):
            if Q[i, j] == 0:
                continue
            
            # Coupling term: Q_ij/4 * s_i * s_j
            J_ising[i, j] += Q[i, j] / 4.0
            J_ising[j, i] += Q[i, j] / 4.0
            
            # Linear cross-terms: Q_ij/4 * s_i + Q_ij/4 * s_j
            h_ising[i] += Q[i, j] / 4.0
            h_ising[j] += Q[i, j] / 4.0
    
    return h_ising, J_ising

def spin_to_binary(spin_state):
    """
    Convert Ising spin {-1, +1} to binary {0, 1}.
    
    In thrml, SpinNode can output True/False (boolean).
    We need to determine the mapping:
    - If True represents spin +1: u = (spin + 1) / 2 = (1 + 1) / 2 = 1 (ON)
    - If False represents spin -1: u = (spin + 1) / 2 = (-1 + 1) / 2 = 0 (OFF)
    
    For boolean: True → +1 → u=1, False → -1 → u=0
    """
    # Convert boolean to spin values first
    # In thrml convention: True typically means spin +1, False means spin -1
    spin_values = np.where(spin_state, 1, -1)
    # Convert spin to binary
    binary = (spin_values + 1) // 2
    return binary

# --- PROBLEM SETUP ---
print("=" * 60)
print("THERMODYNAMIC UNIT COMMITMENT DISPATCH")
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

# Normalization for numerical stability
scale_power = 100.0
scale_cost = 10.0

P = gen_capacity / scale_power
D = target_demand / scale_power
C = gen_cost / scale_cost

# Penalty coefficients
# ALPHA: Penalty for demand mismatch (constraint)
# BETA: Penalty for cost (objective)
# ALPHA >> BETA ensures constraint satisfaction is prioritized
ALPHA = 50.0
BETA = 1.0

print(f"\nPenalty Coefficients:")
print(f"  ALPHA (demand constraint): {ALPHA}")
print(f"  BETA (cost objective): {BETA}")

# --- BUILD QUBO FORMULATION ---
# Objective: ALPHA * (sum_i P_i * u_i - D)^2 + BETA * sum_i C_i * u_i
# 
# Expand the squared term:
# (sum_i P_i * u_i - D)^2 = (sum_i P_i * u_i)^2 - 2*D*(sum_i P_i * u_i) + D^2
#                         = sum_i sum_j P_i * P_j * u_i * u_j - 2*D*sum_i P_i * u_i + D^2
#
# For binary variables: u_i^2 = u_i (idempotent property)
# So diagonal terms: P_i * P_i * u_i * u_i = P_i^2 * u_i
#
# Final QUBO form:
# L_i = ALPHA * (P_i^2 - 2*D*P_i) + BETA * C_i  (linear terms)
# Q_ij = 2 * ALPHA * P_i * P_j  (quadratic terms, i ≠ j)

Q_qubo = np.zeros((num_units, num_units))
L_qubo = np.zeros(num_units)

for i in range(num_units):
    # Linear term from demand constraint + cost objective
    L_qubo[i] = ALPHA * (P[i]**2 - 2*D*P[i]) + BETA * C[i]
    
    for j in range(i + 1, num_units):
        # Quadratic coupling from demand constraint
        Q_qubo[i, j] = 2 * ALPHA * P[i] * P[j]

print(f"\nQUBO Matrix built: {num_units}x{num_units}")

# --- CONVERT TO ISING ---
h_ising, J_ising = qubo_to_ising(Q_qubo, L_qubo)

print(f"Ising Hamiltonian parameters:")
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
        if J_ising[i, j] != 0:  # Only add non-zero couplings
            edges.append((nodes[i], nodes[j]))
            weights_list.append(J_ising[i, j])

jax_biases = jnp.array(biases_list)
jax_weights = jnp.array(weights_list)

# Temperature parameter (inverse temperature beta)
# Higher beta → lower temperature → more deterministic
# Lower beta → higher temperature → more exploration
beta = jnp.array(15.0)

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

# Sampling schedule
# n_warmup: Steps to reach equilibrium
# n_samples: Number of samples to collect
# steps_per_sample: Steps between samples
schedule = SamplingSchedule(n_warmup=3000, n_samples=500, steps_per_sample=5)
samples = sample_states(k_samp, program, schedule, init_state, [], [Block(nodes)])

print(f"  Warmup steps: {schedule.n_warmup}")
print(f"  Samples collected: {schedule.n_samples}")

# --- DECODE RESULTS ---
# Take the final sample as the equilibrium state
final_spin_state = np.array(samples[-1][0]).flatten()

# Convert from spin to binary
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
    cost = gen_cost[i] * power  # Cost per MW times MW
    
    if is_on:
        total_generation += power
        total_cost += cost
    
    print(f"Gen {i+1}: {status} | {gen_capacity[i]:6.1f} MW @ ${gen_cost[i]:5.1f}/MW | Power: {power:6.1f} MW | Cost: ${cost:7.1f}")

print("-" * 60)
print(f"Total Generation: {total_generation:6.1f} MW")
print(f"Target Demand:    {target_demand:6.1f} MW")
print(f"Mismatch:         {total_generation - target_demand:+6.1f} MW")
print(f"Total Cost:       ${total_cost:7.1f}")

# Calculate the actual objective value
demand_error = (total_generation - target_demand) ** 2
objective_value = ALPHA * (demand_error / (scale_power**2)) + BETA * (total_cost / scale_cost)
print(f"Objective Value:  {objective_value:.3f}")

print("=" * 60)

# Verify the solution quality
if abs(total_generation - target_demand) < 1e-6:
    print("[OK] Demand constraint SATISFIED")
else:
    print(f"[FAILED] Demand constraint VIOLATED by {abs(total_generation - target_demand):.1f} MW")
