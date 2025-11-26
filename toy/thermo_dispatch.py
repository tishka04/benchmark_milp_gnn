import jax
import jax.numpy as jnp
import numpy as np
from thrml import SpinNode, Block, SamplingSchedule, sample_states
from thrml.models import IsingEBM, IsingSamplingProgram, hinton_init

# --- 1. DEFINE THE GRID SCENARIO ---
# 5 Generators with different Capacities (MW) and Costs ($)
gen_capacity = np.array([100, 200, 50, 150, 300])  # MW
gen_cost     = np.array([10,  15,  5,  20,  25])   # Cost coefficient
target_demand = 450.0  # MW to match

# Penalty Factors (The "Lagrange Multipliers")
# ALPHA: Penalty for missing demand (Must be dominant!)
# BETA:  Weight for minimizing cost
ALPHA = 1.0  
BETA  = 0.1  

print(f"Target Demand: {target_demand} MW")
print("Solving via Thermodynamic Simulation...")

# --- 2. MAP TO ISING MODEL (MATH) ---
# The Ising variable 's' is {-1, +1}. We need {0, 1}.
# Transformation: u = (s + 1) / 2  =>  s = 2u - 1
# This math converts our physical constraints into J (weights) and h (biases).

num_units = len(gen_capacity)

# Initialize couplings (J) and biases (h)
# J matrix is symmetric (interactions between units)
J = np.zeros((num_units, num_units))
h = np.zeros(num_units)

# Expand the squared term: ALPHA * (Sum(P_i * u_i) - D)^2
# This creates quadratic interactions (u_i * u_j) and linear terms (u_i)

for i in range(num_units):
    # Linear terms from the expansion + Cost term
    term1 = -2 * ALPHA * target_demand * gen_capacity[i]
    term2 = ALPHA * (gen_capacity[i] ** 2)
    term3 = BETA * gen_cost[i]
    
    # We work in {0,1} logic here for clarity, will map to Ising biases below.
    # Note: For exact Ising coefficients, one usually expands fully. 
    # For this demo, we use a simplified heuristic mapping often used in QUBO.
    h[i] = term1 + term2 + term3

    for j in range(i + 1, num_units):
        # Quadratic terms: 2 * ALPHA * P_i * P_j * u_i * u_j
        weight = 2 * ALPHA * gen_capacity[i] * gen_capacity[j]
        J[i, j] = weight
        J[j, i] = weight # Symmetric

# --- 3. BUILD THE THRML GRAPH ---
nodes = [SpinNode() for _ in range(num_units)]

# Define Edges (All-to-All connectivity because Sum(P) couples everyone)
edges = []
weights_list = []
biases_list = []

for i in range(num_units):
    biases_list.append(h[i])
    for j in range(i + 1, num_units):
        edges.append((nodes[i], nodes[j]))
        weights_list.append(J[i, j])

# Convert to JAX arrays
jax_biases = jnp.array(biases_list)
jax_weights = jnp.array(weights_list)
# Temperature (Low temp = deterministic optimization, High = exploration)
beta_temp = jnp.array(10.0) 

# Create the Energy Based Model
model = IsingEBM(nodes, edges, jax_biases, jax_weights, beta_temp)

# --- 4. RUN THERMODYNAMIC INFERENCE ---
# Define Block Sampling (Standard 2-color for fully connected graphs is tricky, 
# so we treat all nodes as one block for this small example)
free_blocks = [Block(nodes)] 

program = IsingSamplingProgram(model, free_blocks, clamped_blocks=[])

# Initialize random state
key = jax.random.key(42)
k_init, k_samp = jax.random.split(key, 2)
init_state = hinton_init(k_init, model, free_blocks, ())

# Run the Physics (Langevin / Gibbs Sampling)
# This loop simulates the TSU hardware settling into equilibrium
schedule = SamplingSchedule(n_warmup=500, n_samples=100, steps_per_sample=1)
samples = sample_states(k_samp, program, schedule, init_state, [], free_blocks)

# --- 5. DECODE RESULTS ---
# Take the last sample as the "converged" state
final_state = samples[-1] 
# Convert from Boolean (False/True) to Binary (0/1)
# Note: In thrml, SpinNode is often True(+1)/False(-1).
# We mapped our logic assuming standard 0/1, so we just treat True=ON.

print("\n--- Optimal Dispatch Results ---")
total_gen = 0
total_cost = 0

# The output format depends on thrml version, usually a list of arrays corresponding to blocks
# Flatten the result
states_flat = np.array(final_state[0]).flatten() 

for i, is_on in enumerate(states_flat):
    status = "ON " if is_on else "OFF"
    p_out = gen_capacity[i] if is_on else 0
    if is_on:
        total_gen += p_out
        total_cost += gen_cost[i]
    print(f"Gen {i+1} ({gen_capacity[i]}MW): {status}")

print(f"\nTotal Generation: {total_gen} MW")
print(f"Target Demand:    {target_demand} MW")
print(f"Difference:       {total_gen - target_demand} MW")