import jax
import jax.numpy as jnp
import numpy as np
from thrml import SpinNode, Block, SamplingSchedule, sample_states
from thrml.models import IsingEBM, IsingSamplingProgram, hinton_init

def qubo_to_ising_negated(Q, linear_biases):
    """
    Converts Cost Function (QUBO) to Physics Energy (Ising).
    
    Optimization: Minimize Cost = x'Qx + L'x
    Physics:      Minimize Energy = - sum(h*s) - sum(J*s*s)
    
    Mapping Rules:
    1. u = (s+1)/2
    2. Cost coefficients must be NEGATED to become Physics coefficients.
       (Positive Cost = Bad = Negative Reward)
    """
    n = len(linear_biases)
    h_ising = np.zeros(n)
    J_ising = np.zeros((n, n))
    
    # --- 1. Map Linear Terms (L * u) ---
    # u = s/2 + 1/2.
    # Cost term L*u contributes L/2 * s.
    # We want Energy ~ Cost.
    # Energy term is -h*s.
    # So: -h = L/2  =>  h = -L/2
    h_ising -= linear_biases / 2.0
    
    # --- 2. Map Quadratic Terms (u_i * Q_ij * u_j) ---
    # u_i * u_j = (s_i s_j + s_i + s_j + 1) / 4
    for i in range(n):
        for j in range(i + 1, n):
            weight = Q[i, j]
            if weight == 0: continue
            
            # The 's_i * s_j' term coeff is weight/4.
            # Energy term is -J * s_i * s_j.
            # So: -J = weight/4 => J = -weight/4
            J_val = weight / 4.0
            J_ising[i, j] -= J_val
            J_ising[j, i] -= J_val 
            
            # The linear parts (s_i, s_j) from the expansion also affect bias
            # Contribution to Cost is weight/4 * s.
            # Contribution to h must be negated.
            h_ising[i] -= J_val
            h_ising[j] -= J_val
            
    return h_ising, J_ising

# --- 1. SETUP GRID SCENARIO ---
# 5 Generators
gen_capacity = np.array([100.0, 200.0, 50.0, 150.0, 300.0]) 
gen_cost     = np.array([10.0,  15.0,  5.0,  20.0,  25.0])
target_demand = 450.0

# Normalize (Crucial for numerical stability)
# We work in "1 unit = 100 MW" space
scale = 100.0
P = gen_capacity / scale
D = target_demand / scale
C = gen_cost / 10.0 

# Penalty Strengths
# ALPHA: Penalty for Missing Demand (Constraint)
# BETA:  Penalty for Spending Money (Objective)
ALPHA = 40.0 
BETA  = 1.0  

print(f"Target: {target_demand} MW")
print("Compiling Hamiltonians with Corrected Signs...")

num_units = len(P)

# --- 2. BUILD QUBO (Cost Function) ---
Q_qubo = np.zeros((num_units, num_units))
L_qubo = np.zeros(num_units)

for i in range(num_units):
    # Linear part of Cost:
    # 1. From Demand term (P*u - D)^2 -> -2*D*P*u + P^2*u^2
    # 2. From Economic term -> C*u
    L_qubo[i] = ALPHA * (P[i]**2 - 2*D*P[i]) + BETA * C[i]
    
    for j in range(i + 1, num_units):
        # Quadratic part of Cost:
        # From Demand term: 2 * P_i * P_j * u_i * u_j
        Q_qubo[i, j] = ALPHA * (2 * P[i] * P[j])

# --- 3. CONVERT TO ISING (Physics) ---
h_ising, J_ising = qubo_to_ising_negated(Q_qubo, L_qubo)

# --- 4. THRML SIMULATION ---
nodes = [SpinNode() for _ in range(num_units)]
edges = []
weights_list = []
biases_list = []

for i in range(num_units):
    biases_list.append(h_ising[i])
    for j in range(i + 1, num_units):
        edges.append((nodes[i], nodes[j]))
        weights_list.append(J_ising[i, j])

# Compile to JAX
model = IsingEBM(nodes, edges, jnp.array(biases_list), jnp.array(weights_list), jnp.array(20.0)) # Beta=20 (Cold/Deterministic)
program = IsingSamplingProgram(model, [Block(nodes)], clamped_blocks=[])

# Run Inference
key = jax.random.key(42)
k_init, k_samp = jax.random.split(key)
init_state = hinton_init(k_init, model, [Block(nodes)], ())

# Warmup allows the physics to "fall" into the well
schedule = SamplingSchedule(n_warmup=2000, n_samples=100, steps_per_sample=1)
samples = sample_states(k_samp, program, schedule, init_state, [], [Block(nodes)])

# --- 5. RESULTS ---
final_spins = np.array(samples[-1][0]).flatten()
final_u = final_spins.astype(int) 

print("\n--- Thermodynamic Solution ---")
total_gen = 0
total_cost_val = 0

for i, u in enumerate(final_u):
    status = "ON " if u else "OFF"
    mw = gen_capacity[i] if u else 0
    cost = gen_cost[i] if u else 0
    if u:
        total_gen += mw
        total_cost_val += cost
    print(f"Gen {i+1} ({gen_capacity[i]:5.1f} MW): {status} | Cost: ${cost}")

print("-" * 30)
print(f"Total Generation: {total_gen} MW")
print(f"Target Demand:    {target_demand} MW")
print(f"Mismatch:         {total_gen - target_demand} MW")