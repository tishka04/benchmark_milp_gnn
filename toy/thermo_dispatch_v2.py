import jax
import jax.numpy as jnp
import numpy as np
from thrml import SpinNode, Block, SamplingSchedule, sample_states
from thrml.models import IsingEBM, IsingSamplingProgram, hinton_init

def qubo_to_ising(Q, linear_biases):
    """
    Converts a 0/1 QUBO problem to a -1/+1 Ising model.
    QUBO: Minimize sum(L_i * u_i) + sum(Q_ij * u_i * u_j)
    Ising: E = - sum(h_i * s_i) - sum(J_ij * s_i * s_j)
    Returns: h (Ising biases), J (Ising weights)
    """
    n = len(linear_biases)
    h_ising = np.zeros(n)
    J_ising = np.zeros((n, n))
    
    # Mathematical transformation u = (s+1)/2
    # This generates constant, linear, and quadratic terms in s.
    
    # 1. Handle Linear parts from QUBO L vector
    # L_i * (s_i + 1)/2 -> contributes L_i/2 to h_ising
    h_ising -= linear_biases / 2.0  # Note sign flip for Ising Energy convention
    
    # 2. Handle Quadratic parts from QUBO Q matrix
    for i in range(n):
        for j in range(i + 1, n):
            weight = Q[i, j]
            if weight == 0: continue
            
            # Term: Q_ij * u_i * u_j
            # = Q_ij * (s_i + 1)/2 * (s_j + 1)/2
            # = Q_ij/4 * (s_i * s_j + s_i + s_j + 1)
            
            # The s_i * s_j part maps to Ising J
            J_val = weight / 4.0
            J_ising[i, j] -= J_val # Sign flip: Positive Cost -> Negative J (Antiferromagnetic)
            J_ising[j, i] -= J_val 
            
            # The linear parts s_i and s_j update the biases!
            h_ising[i] -= weight / 4.0
            h_ising[j] -= weight / 4.0
            
    return h_ising, J_ising

# --- 1. SETUP GRID SCENARIO ---
# Generators
gen_capacity = np.array([100.0, 200.0, 50.0, 150.0, 300.0]) # MW
gen_cost     = np.array([10.0,  15.0,  5.0,  20.0,  25.0])  # $/hr (simplified)
target_demand = 450.0

# NORMALIZATION (Critical for Physics/ML stability)
# We scale down MW values so energy isn't 1,000,000+
scale_factor = 100.0 
P = gen_capacity / scale_factor
D = target_demand / scale_factor
C = gen_cost / 10.0 # Scale costs similarly

# Penalties
ALPHA = 20.0 # Strong penalty for demand mismatch
BETA  = 0.5  # Weak penalty for economic cost

print(f"Target: {target_demand} MW")
print("Compiling Thermodynamic Hamiltonians...")

num_units = len(P)

# --- 2. BUILD QUBO MATRIX (0/1 Logic) ---
# Objective: ALPHA * (sum(P_i u_i) - D)^2 + BETA * sum(C_i u_i)
# Expand (sum P u - D)^2 = sum(P_i P_j u_i u_j) - 2D sum(P_i u_i) + D^2

Q_qubo = np.zeros((num_units, num_units))
L_qubo = np.zeros(num_units)

for i in range(num_units):
    # Linear Term: -2 * ALPHA * D * P_i  +  BETA * C_i
    # Plus self-quadratic term P_i^2 (since u_i^2 = u_i)
    L_qubo[i] = -2 * ALPHA * D * P[i] + ALPHA * (P[i]**2) + BETA * C[i]
    
    for j in range(i + 1, num_units):
        # Cross Term: 2 * ALPHA * P_i * P_j
        weight = 2 * ALPHA * P[i] * P[j]
        Q_qubo[i, j] = weight
        # Symmetric Q usually not needed if we iterate i<j, but keeping logic clean

# --- 3. CONVERT TO ISING (Physics Logic) ---
h_ising, J_ising = qubo_to_ising(Q_qubo, L_qubo)

# --- 4. THRML SIMULATION ---
# Map numpy arrays to JAX
nodes = [SpinNode() for _ in range(num_units)]
edges = []
weights_list = []
biases_list = []

for i in range(num_units):
    biases_list.append(h_ising[i])
    for j in range(i + 1, num_units):
        edges.append((nodes[i], nodes[j]))
        weights_list.append(J_ising[i, j]) # Standard Ising J

jax_biases = jnp.array(biases_list)
jax_weights = jnp.array(weights_list)

# Set Temperature (Beta). 
# Higher Beta = Colder = More deterministic (Greedy descent)
# Lower Beta = Hotter = More exploration
beta = jnp.array(5.0) 

model = IsingEBM(nodes, edges, jax_biases, jax_weights, beta)
program = IsingSamplingProgram(model, [Block(nodes)], clamped_blocks=[])

# Initialize and Sample
key = jax.random.key(123)
k_init, k_samp = jax.random.split(key)
init_state = hinton_init(k_init, model, [Block(nodes)], ())

# Sampling Schedule: Warmup helps escape bad initial states
schedule = SamplingSchedule(n_warmup=2000, n_samples=500, steps_per_sample=10)
samples = sample_states(k_samp, program, schedule, init_state, [], [Block(nodes)])

# --- 5. RESULTS ---
final_spins = np.array(samples[-1][0]).flatten() # True/False
# Map Back: True(+1) -> u=1, False(-1) -> u=0
final_u = final_spins.astype(int) 

print("\n--- Thermodynamic Solution ---")
total_gen = 0
total_cost_val = 0

for i, u in enumerate(final_u):
    status = "ON " if u else "OFF"
    mw = gen_capacity[i] if u else 0
    cost = gen_cost[i] if u else 0
    total_gen += mw
    total_cost_val += cost
    print(f"Gen {i+1} ({gen_capacity[i]:5.1f} MW): {status} | Cost: ${cost}")

print("-" * 30)
print(f"Total Generation: {total_gen} MW")
print(f"Target Demand:    {target_demand} MW")
print(f"Mismatch:         {total_gen - target_demand} MW")