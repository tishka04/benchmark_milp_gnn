import jax
import jax.numpy as jnp
import numpy as np
from thrml import SpinNode, Block, SamplingSchedule, sample_states
from thrml.models import IsingEBM, IsingSamplingProgram, hinton_init

def qubo_to_ising_positive(Q, linear_biases):
    """
    Converts 0/1 QUBO to -1/+1 Ising for 'Minimization' convention.
    Objective: Min E = sum(h_i * s_i) + sum(J_ij * s_i * s_j)
    """
    n = len(linear_biases)
    h_ising = np.zeros(n)
    J_ising = np.zeros((n, n))
    
    # 1. Linear Mapping: u = (s+1)/2
    # Term L*u -> L/2 * s + const
    h_ising += linear_biases / 2.0
    
    # 2. Quadratic Mapping: u_i * u_j = (s_i+1)/2 * (s_j+1)/2
    # Term Q*u*u -> Q/4 * s_i*s_j + Q/4 * s_i + Q/4 * s_j + const
    for i in range(n):
        for j in range(i + 1, n):
            weight = Q[i, j]
            if weight == 0: continue
            
            # Coupling Term
            J_val = weight / 4.0
            J_ising[i, j] += J_val # Positive accumulation
            J_ising[j, i] += J_val 
            
            # Update Linear Biases from cross-terms
            h_ising[i] += weight / 4.0
            h_ising[j] += weight / 4.0
            
    return h_ising, J_ising

# --- 1. SETUP GRID SCENARIO ---
gen_capacity = np.array([100.0, 200.0, 50.0, 150.0, 300.0]) 
gen_cost     = np.array([10.0,  15.0,  5.0,  20.0,  25.0])
target_demand = 450.0

# Normalize
scale_factor = 100.0
P = gen_capacity / scale_factor
D = target_demand / scale_factor
C = gen_cost / 10.0

ALPHA = 30.0 # Penalty strength
BETA  = 0.5  # Cost strength

num_units = len(P)

# --- 2. BUILD QUBO ---
Q_qubo = np.zeros((num_units, num_units))
L_qubo = np.zeros(num_units)

for i in range(num_units):
    # Linear: -2*alpha*D*P + alpha*P^2 + beta*C
    L_qubo[i] = -2 * ALPHA * D * P[i] + ALPHA * (P[i]**2) + BETA * C[i]
    for j in range(i + 1, num_units):
        # Quad: 2*alpha*P*P
        Q_qubo[i, j] = 2 * ALPHA * P[i] * P[j]

# --- 3. CONVERT TO ISING (Positive Sign) ---
h_ising, J_ising = qubo_to_ising_positive(Q_qubo, L_qubo)

# --- 4. RUN SIMULATION ---
nodes = [SpinNode() for _ in range(num_units)]
edges = []
weights_list = []
biases_list = []

for i in range(num_units):
    biases_list.append(h_ising[i])
    for j in range(i + 1, num_units):
        edges.append((nodes[i], nodes[j]))
        weights_list.append(J_ising[i, j])

jax_biases = jnp.array(biases_list)
jax_weights = jnp.array(weights_list)

# Inverse Temp (Higher = More Deterministic)
beta = jnp.array(10.0)

model = IsingEBM(nodes, edges, jax_biases, jax_weights, beta)
program = IsingSamplingProgram(model, [Block(nodes)], clamped_blocks=[])

key = jax.random.key(999)
k_init, k_samp = jax.random.split(key)
init_state = hinton_init(k_init, model, [Block(nodes)], ())

# Longer schedule for stability
schedule = SamplingSchedule(n_warmup=2000, n_samples=1000, steps_per_sample=5)
samples = sample_states(k_samp, program, schedule, init_state, [], [Block(nodes)])

# --- 5. RESULTS ---
final_spins = np.array(samples[-1][0]).flatten() # True/False
# Spin True -> +1 -> u=1 (ON)
final_u = final_spins.astype(int) 

print(f"\nTarget: {target_demand} MW")
print("--- Thermodynamic Solution ---")
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
print(f"Mismatch:         {total_gen - target_demand} MW")