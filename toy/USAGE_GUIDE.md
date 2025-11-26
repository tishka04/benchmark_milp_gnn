# Hybrid Thermodynamic-Classical Solver: Usage Guide

## Quick Start

```bash
cd c:/Users/Dell/projects/multilayer_milp_gnn/benchmark/toy

# 1. Run the original toy problem (5 generators, 450 MW)
python thermal_hybrid.py

# 2. Run scenario-based problem (13 generators, 637 MW from scenario_00001.json)
python scenario_hybrid_solver.py

# 3. Compare with optimal MILP solution
python scenario_milp_comparison.py

# 4. See all analysis and comparisons
python compare_all_methods.py
```

## Files Overview

### Core Implementations

#### `thermal_hybrid.py` (Your Original)
- **Purpose:** Demonstrates hybrid concept on toy problem
- **Problem:** 5 generators, 450 MW demand
- **Status:** Working prototype
- **Key features:**
  - QUBO → Ising conversion
  - Thermodynamic sampling (n=20 candidates)
  - Classical dispatch worker
  - Simple merit-order dispatch

#### `scenario_hybrid_solver.py` (Scenario-Based)
- **Purpose:** Applies hybrid to realistic scenario
- **Problem:** 13 generators from scenario_00001.json
- **Status:** **Finds globally optimal solution!** ✓
- **Key improvements:**
  - Very high temperature (β=0.5)
  - Multiple random seeds (10)
  - Heuristic augmentation
  - 122 total candidates

### Reference Solutions

#### `scenario_milp_comparison.py`
- **Purpose:** Provides optimal baseline
- **Method:** Classical MILP (scipy.optimize.milp)
- **Result:** €84,602.59 (globally optimal)
- **Use:** Validate hybrid solution quality

#### `reference_milp_solution.py`
- **Purpose:** Original toy problem reference
- **Problem:** 5 generators, 450 MW
- **Result:** Gens [1,2,4], $7000 cost
- **Use:** Show what optimal looks like

### Analysis Files

#### `hybrid_analysis.md`
- **Content:** Deep technical analysis of architecture
- **Topics:**
  - Why separation of concerns works
  - Complexity analysis
  - Comparison to alternatives
  - Extensions and limitations

#### `HYBRID_RESULTS_SUMMARY.md`
- **Content:** Experimental results and findings
- **Highlights:**
  - Hybrid = MILP optimal (0% gap!)
  - Solution diversity analysis
  - Scalability projections
  - Lessons learned

#### `ANALYSIS_SUMMARY.md`
- **Content:** Why pure thermodynamic fails
- **Topics:**
  - Energy landscape analysis
  - QUBO-to-Ising issues
  - Bug identification
  - Root cause diagnosis

### Failed Approaches (Educational)

- `thermo_dispatch.py` - Original attempt (all OFF)
- `thermo_dispatch_corrected.py` - Negation strategy (all OFF)
- `thermo_dispatch_final.py` - Positive signs (all OFF)
- `thermo_dispatch_v2.py` - Has accumulation bug (all OFF)
- `thermo_dispatch_fixed.py` - Correct conversion (all OFF)
- `thermo_dispatch_working.py` - Offset approach (all ON)
- `thermo_dispatch_final_tuned.py` - Multi-trial (all OFF)

**Key lesson:** All pure thermodynamic approaches fail due to negative bias trap

### Utilities

- `debug_energy.py` - Energy landscape visualizer
- `compare_all_methods.py` - Side-by-side comparison
- `README.md` - Quick reference

## Using the Hybrid Solver

### Step 1: Define Your Problem

```python
# Generator parameters
gen_capacity = np.array([...])  # MW
gen_cost = np.array([...])      # €/MW
target_demand = 500.0           # MW

# Optional: minimum generation, ramp rates
gen_min = gen_capacity * 0.3
ramp_rate = gen_capacity * 0.5
```

### Step 2: Build QUBO

```python
# Penalty coefficients
ALPHA = 100.0  # Demand constraint (high)
BETA = 1.0     # Cost objective (low)

# QUBO matrices
Q_qubo = np.zeros((n, n))
L_qubo = np.zeros(n)

for i in range(n):
    L_qubo[i] = ALPHA * (P[i]**2 - 2*D*P[i]) + BETA * C[i]
    for j in range(i+1, n):
        Q_qubo[i, j] = 2 * ALPHA * P[i] * P[j]
```

### Step 3: Convert to Ising

```python
# Ising parameters
h_ising = -L_qubo / 2.0

J_ising = np.zeros((n, n))
for i in range(n):
    for j in range(i+1, n):
        J_val = Q_qubo[i, j] / 4.0
        J_ising[i, j] -= J_val
        J_ising[j, i] -= J_val
        h_ising[i] -= J_val
        h_ising[j] -= J_val
```

### Step 4: Thermodynamic Sampling

```python
from thrml import SpinNode, Block, IsingEBM, IsingSamplingProgram

# Build model
nodes = [SpinNode() for _ in range(n)]
edges = [(nodes[i], nodes[j]) for i in range(n) for j in range(i+1, n)]
weights = [J_ising[i, j] for i in range(n) for j in range(i+1, n)]
biases = list(h_ising)

# CRITICAL: Use HIGH temperature for exploration!
beta = jnp.array(0.5)  # HOT (not cold!)

model = IsingEBM(nodes, edges, jnp.array(biases), jnp.array(weights), beta)
program = IsingSamplingProgram(model, [Block(nodes)], [])

# Sample with multiple seeds
candidates = []
for seed in range(10):
    key = jax.random.key(42 + seed)
    k_init, k_samp = jax.random.split(key)
    init_state = hinton_init(k_init, model, [Block(nodes)], ())
    
    schedule = SamplingSchedule(n_warmup=500, n_samples=10, steps_per_sample=5)
    samples = sample_states(k_samp, program, schedule, init_state, [], [Block(nodes)])
    
    for s in samples:
        u = np.array(s[0]).flatten().astype(int)
        candidates.append(u)

# Add heuristics (IMPORTANT!)
# ... (see scenario_hybrid_solver.py lines 209-230)
```

### Step 5: Classical Dispatch

```python
def solve_economic_dispatch(u_commitment, demand, p_min, p_max, costs):
    """Merit order dispatch for given commitment."""
    available = np.where(u_commitment == 1)[0]
    
    # Check feasibility
    if len(available) == 0:
        return None, np.inf, False
    if demand > np.sum(p_max[available]):
        return None, np.inf, False
    if demand < np.sum(p_min[available]):
        return None, np.inf, False
    
    # Sort by cost
    sorted_idx = available[np.argsort(costs[available])]
    
    # Greedy dispatch
    dispatch = np.zeros(len(costs))
    remaining = demand
    
    # Min generation first
    for idx in sorted_idx:
        dispatch[idx] = p_min[idx]
        remaining -= p_min[idx]
    
    # Add remaining to cheapest
    for idx in sorted_idx:
        take = min(remaining, p_max[idx] - dispatch[idx])
        dispatch[idx] += take
        remaining -= take
    
    # Calculate cost
    total_cost = np.sum(dispatch * costs)
    
    return dispatch, total_cost, True

# Evaluate all candidates
results = []
for u in candidates:
    dispatch, cost, feasible = solve_economic_dispatch(u, demand, p_min, p_max, costs)
    if feasible:
        results.append({'u': u, 'dispatch': dispatch, 'cost': cost})
```

### Step 6: Selection

```python
# Pick best feasible solution
results.sort(key=lambda x: x['cost'])
best = results[0]

print(f"Optimal cost: €{best['cost']:.2f}")
print(f"Units ON: {np.sum(best['u'])}")
print(f"Dispatch: {best['dispatch']}")
```

## Parameter Tuning Guide

### Temperature (beta)

```python
# Too cold (beta > 10): Gets stuck in local minimum
beta = 20.0  # ❌ All OFF state

# Too hot (beta < 0.1): Random, no guidance
beta = 0.01  # ❌ No structure

# Just right (beta ≈ 0.5-2.0): Diverse + guided
beta = 0.5   # ✓ Explores well
```

**Rule of thumb:** Start at β = 1/√N, adjust up if too random, down if stuck.

### QUBO Penalties (ALPHA, BETA)

```python
# ALPHA: Constraint importance
ALPHA = 10    # Too weak: ignores constraint
ALPHA = 100   # Good: guides without dominating
ALPHA = 1000  # Too strong: amplifies negative bias

# BETA: Cost importance
BETA = 0.1   # Feasibility prioritized
BETA = 1.0   # Balanced
BETA = 10.0  # Cost prioritized (risk infeasibility)
```

**Rule of thumb:** ALPHA >> BETA for hard constraints.

### Sampling Budget

```python
# Number of seeds
n_seeds = max(5, N // 5)  # Scale with problem size

# Samples per seed
n_samples = 10-20  # Diminishing returns after 20

# Total candidates
total = n_seeds * n_samples + heuristics
# Aim for 50-200 total candidates
```

**Rule of thumb:** Budget 0.5-2% of search space (2^N).

### Warmup Steps

```python
warmup = 100   # Too short: not equilibrated
warmup = 500   # Good for small (N < 20)
warmup = 2000  # Good for medium (N < 100)
warmup = 5000  # Large problems (N > 100)
```

**Rule of thumb:** warmup ≈ 50 × N for full equilibration.

## Common Issues and Solutions

### Issue 1: All Candidates Infeasible

**Symptom:** `Feasible solutions: 0 / X`

**Causes:**
- Temperature too low (stuck in all-OFF)
- No heuristic candidates
- Demand exceeds capacity

**Solutions:**
```python
# Increase temperature
beta = 0.5  # or lower

# Add heuristic candidates
candidates.append(greedy_cheapest_first())
candidates.append(all_on())
candidates.append(random_selection())

# Check problem feasibility
assert demand <= np.sum(p_max)
assert demand >= np.sum(p_min * any_valid_u)
```

### Issue 2: Poor Solution Quality

**Symptom:** Cost much higher than expected

**Causes:**
- Not enough candidates
- Temperature too high (random)
- Missing good heuristics

**Solutions:**
```python
# Increase sampling
n_seeds = 20  # more diversity
n_samples = 20  # more per seed

# Adjust temperature
beta = 1.0  # cooler, more structured

# Better heuristics
candidates.append(cost_sorted_greedy())
candidates.append(capacity_sorted_greedy())
```

### Issue 3: Slow Performance

**Symptom:** Takes too long to run

**Causes:**
- Too many candidates
- Long warmup
- Large problem (N > 100)

**Solutions:**
```python
# Reduce sampling
n_seeds = 5
n_samples = 10
warmup = 500

# Parallelize (if possible)
from multiprocessing import Pool
with Pool(10) as p:
    results = p.map(evaluate_candidate, candidates)

# Use sparse Ising if possible
# Only include non-zero J_ising edges
```

## Advanced Features

### Multi-Period Scheduling

```python
# Extend to time series
T = 96  # time periods
schedule = np.zeros((T, n))

for t in range(T):
    # Sample commitment for period t
    candidates_t = thermodynamic_sample(demand[t])
    
    # Add temporal constraints
    if t > 0:
        candidates_t = filter_by_ramp(
            candidates_t, 
            schedule[t-1], 
            ramp_rate
        )
    
    # Dispatch and select
    best_t = select_best(candidates_t, demand[t])
    schedule[t] = best_t
```

### Stochastic Demand/Renewable

```python
# Sample from forecast distribution
demand_scenarios = np.random.normal(demand_forecast, std, size=100)
renewable_scenarios = sample_wind_solar_forecast(100)

# Solve for each scenario
solutions = []
for d, r in zip(demand_scenarios, renewable_scenarios):
    net_d = d - r
    candidates = thermodynamic_sample(net_d)
    best = select_best(candidates, net_d)
    solutions.append(best)

# Aggregate (e.g., most frequent commitment)
robust_solution = mode(solutions)
```

### Warm-Start from Previous Solution

```python
# Use yesterday's solution as initialization
prev_commitment = load_previous_day_solution()

# Clamp some spins in Ising model
clamped_nodes = [nodes[i] for i in range(n) if prev_commitment[i] == 1]
clamped_values = [True] * len(clamped_nodes)

program = IsingSamplingProgram(
    model, 
    free_blocks=[Block([n for n in nodes if n not in clamped_nodes])],
    clamped_blocks=[(Block(clamped_nodes), clamped_values)]
)
```

## Validation Checklist

Before deploying, verify:

- [ ] Demand satisfaction: `|dispatch - demand| < tolerance`
- [ ] Capacity limits: `p_min * u <= dispatch <= p_max * u`
- [ ] Cost calculation: `cost = sum(dispatch * unit_cost)`
- [ ] Reserve margin: `sum(p_max * u) - dispatch >= min_reserve`
- [ ] Solution diversity: Multiple feasible solutions found
- [ ] Comparison to MILP: Gap < 5% (if MILP tractable)
- [ ] Sensitivity analysis: Robust to parameter changes
- [ ] Edge cases: All OFF, all ON, infeasible demand

## Citation

If you use this hybrid approach, please cite:

```bibtex
@software{hybrid_thermodynamic_classical_2024,
  title={Hybrid Thermodynamic-Classical Solver for Unit Commitment},
  author={[Your Name]},
  year={2024},
  note={Combines Ising model sampling with classical economic dispatch}
}
```

## Support and Resources

- **Documentation:** See `hybrid_analysis.md` for technical details
- **Results:** See `HYBRID_RESULTS_SUMMARY.md` for experimental findings
- **Issues:** See `ANALYSIS_SUMMARY.md` for troubleshooting
- **Examples:** Run `python scenario_hybrid_solver.py` for working example

## License

[Add your license here]

---

**Happy optimizing! 🔌⚡**
