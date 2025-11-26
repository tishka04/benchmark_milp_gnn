import jax
import jax.numpy as jnp
import numpy as np
from thrml import SpinNode, Block, SamplingSchedule, sample_states
from thrml.models import IsingEBM, IsingSamplingProgram, hinton_init

# ==========================================
# PART 1: SYSTEM DEFINITION
# ==========================================
# 5 Generators
# P_min = 0 for simplicity in this toy model
gen_capacity = np.array([100.0, 200.0, 50.0, 150.0, 300.0]) 
gen_cost     = np.array([10.0,  15.0,  5.0,  20.0,  25.0])
target_demand = 450.0

# Normalize for Numerical Stability (Crucial for Ising)
scale = 100.0
P = gen_capacity / scale
D = target_demand / scale
C = gen_cost / 10.0 

num_units = len(P)

print(f"Target Demand: {target_demand} MW")
print("==========================================")

# ==========================================
# PART 2: THE THERMODYNAMIC MANAGER (Ising)
# ==========================================
# Goal: Find u (Binary) that *roughly* allows sum(P) ~ D.
# We map the QUBO Cost -> Ising Energy.

# Weights for the Thermodynamic Proxy
ALPHA = 40.0  # Penalty for Demand Mismatch
BETA  = 1.0   # Penalty for Cost

# 1. Build QUBO Matrix (Cost Function)
# Minimize Cost = u'Qu + L'u
Q_qubo = np.zeros((num_units, num_units))
L_qubo = np.zeros(num_units)

for i in range(num_units):
    # Linear Term: ALPHA*(P^2 - 2DP) + BETA*C
    # Note: P^2 comes from u^2=u. 
    L_qubo[i] = ALPHA * (P[i]**2 - 2*D*P[i]) + BETA * C[i]
    
    for j in range(i + 1, num_units):
        # Quadratic Term: 2 * ALPHA * P_i * P_j
        Q_qubo[i, j] = 2 * ALPHA * P[i] * P[j]

# 2. Convert to Ising Parameters (Physics)
# Rule: Minimize Cost <-> Minimize Energy
# Energy = - sum(h*s) - sum(J*s*s)
# Transformation u = (s+1)/2
# implies: h = - (L/2 + sum(Q_row)/4)
#          J = - Q/4

h_ising = np.zeros(num_units)
J_ising = np.zeros((num_units, num_units))

# Linear Mapping
h_ising -= L_qubo / 2.0

# Quadratic Mapping
for i in range(num_units):
    for j in range(i + 1, num_units):
        weight = Q_qubo[i, j]
        if weight == 0: continue
        
        J_val = weight / 4.0
        J_ising[i, j] -= J_val
        J_ising[j, i] -= J_val 
        
        # Cross terms affect linear bias
        h_ising[i] -= J_val
        h_ising[j] -= J_val

# 3. Setup thrml Simulation
nodes = [SpinNode() for _ in range(num_units)]
edges = []
weights_list = []
biases_list = []

for i in range(num_units):
    biases_list.append(h_ising[i])
    for j in range(i + 1, num_units):
        edges.append((nodes[i], nodes[j]))
        weights_list.append(J_ising[i, j])

# Compile Model
# Low Beta (Hot) to explore, High Beta (Cold) to exploit.
# We use a moderate temp to get diversity.
model = IsingEBM(nodes, edges, jnp.array(biases_list), jnp.array(weights_list), jnp.array(5.0))
program = IsingSamplingProgram(model, [Block(nodes)], clamped_blocks=[])

# 4. Run "Thermodynamic Sampling"
# We generate MULTIPLE samples (candidates)
n_candidates = 20
print(f"Sampling {n_candidates} candidates from Thermodynamic Engine...")

key = jax.random.key(123)
k_init, k_samp = jax.random.split(key)
init_state = hinton_init(k_init, model, [Block(nodes)], ())
schedule = SamplingSchedule(n_warmup=1000, n_samples=n_candidates, steps_per_sample=10)
samples = sample_states(k_samp, program, schedule, init_state, [], [Block(nodes)])

# Extract candidates (Binary 0/1)
# thrml outputs boolean PyTrees, flatten them
candidates = []
for s in samples:
    spins = np.array(s[0]).flatten() # True/False
    u_vec = spins.astype(int)       # 1/0
    candidates.append(u_vec)

# ==========================================
# PART 3: THE DIFFERENTIABLE WORKER (JAX)
# ==========================================
# Goal: For each candidate u, solve Exact Dispatch min Cost.
# Constraint: sum(P) = D, 0 <= P <= P_max * u

def solve_dispatch_worker(u_vec, demand):
    """
    A simple differentiable worker (Analytic Solution for Economic Dispatch).
    Since cost is linear, we just pick the cheapest available units 
    among the committed ones until demand is met.
    """
    # Filter available units
    available_indices = np.where(u_vec == 1)[0]
    if len(available_indices) == 0:
        return None, np.inf, False # Infeasible

    current_gen = 0.0
    dispatch = np.zeros_like(gen_capacity)
    
    # Sort available units by cost
    # (In a complex JAX solver, this would be a gradient descent step)
    sorted_idx = available_indices[np.argsort(gen_cost[available_indices])]
    
    for idx in sorted_idx:
        needed = demand - current_gen
        if needed <= 0:
            break
        
        # Max output for this unit
        p_max = gen_capacity[idx]
        
        # Dispatch logic
        take = min(needed, p_max)
        dispatch[idx] = take
        current_gen += take
        
    # Check Feasibility
    # Allow small float tolerance
    is_feasible = abs(current_gen - demand) < 1e-3
    
    # Calculate Real Economic Cost
    real_cost = np.sum(dispatch * gen_cost)
    
    return dispatch, real_cost, is_feasible

# ==========================================
# PART 4: HYBRID SELECTION
# ==========================================
print("\n--- Evaluating Candidates ---")

best_sol = None
best_cost = np.inf
best_u = None

seen_configs = set()

for i, u in enumerate(candidates):
    u_tuple = tuple(u)
    if u_tuple in seen_configs: continue
    seen_configs.add(u_tuple)
    
    dispatch, cost, feasible = solve_dispatch_worker(u, target_demand)
    
    status_str = "FEASIBLE" if feasible else "INFEASIBLE (Under-generation)"
    
    # Simple logging
    on_units = [j+1 for j, val in enumerate(u) if val]
    print(f"Cand {i}: Units {on_units} -> {status_str}, Cost: ${cost:.2f}")
    
    if feasible and cost < best_cost:
        best_cost = cost
        best_sol = dispatch
        best_u = u

# ==========================================
# PART 5: FINAL RESULT
# ==========================================
print("\n==========================================")
print("HYBRID SYSTEM RESULT")
print("==========================================")

if best_sol is not None:
    print(f"Optimal Configuration Found!")
    total_mw = 0
    for i, mw in enumerate(best_sol):
        status = "ON " if best_u[i] else "OFF"
        if best_u[i]:
            print(f"Gen {i+1} ({gen_capacity[i]} MW): {status} -> Dispatched: {mw:5.1f} MW")
            total_mw += mw
    print("-" * 30)
    print(f"Total Dispatch: {total_mw} MW")
    print(f"Total Cost:     ${best_cost:.2f}")
else:
    print("No feasible solution found among thermodynamic samples.")
    print("Try increasing n_candidates or adjusting Alpha/Beta.")