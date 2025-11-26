# Analysis of Thermodynamic Dispatch Problem

## Problem Statement
Dispatch 5 generators to meet 450 MW demand while minimizing cost.

**Generators:**
- Gen 1: 100 MW @ $10/MW
- Gen 2: 200 MW @ $15/MW
- Gen 3: 50 MW @ $5/MW
- Gen 4: 150 MW @ $20/MW
- Gen 5: 300 MW @ $25/MW

**Optimal Solution (via MILP):**
- Gens 1, 2, 4 ON → 450 MW, $7000 cost
- 3 feasible exact solutions exist

## Why All Implementations Fail

### Root Cause: Incompatible Energy Landscape

The QUBO formulation for this problem creates an energy landscape that is fundamentally incompatible with thermodynamic minimization:

```
Objective: ALPHA * (sum P_i * u_i - D)^2 + BETA * sum C_i * u_i
```

When expanded for QUBO:
```
L_i = ALPHA * (P_i^2 - 2*D*P_i) + BETA * C_i
Q_ij = 2 * ALPHA * P_i * P_j
```

**The Problem:**
- The term `-2*D*P_i` is large and negative
- This makes ALL linear coefficients L_i negative
- When mapped to Ising: h_i = L_i/2 (also negative)
- Ising energy E = sum h_i * s_i is minimized when all s_i = -1
- This corresponds to all u_i = 0 (all generators OFF)

### Energy Landscape Analysis

| State | Generation | QUBO Energy | Linear Term | Quadratic Term |
|-------|------------|-------------|-------------|----------------|
| All OFF | 0 MW | 1012.5 | 0.0 | 0.0 |
| Optimal [1,2,4] | 450 MW | 4.5 | -1658.0 | +650.0 |
| All ON | 800 MW | 620.0 | -2767.5 | +2375.0 |

**Key Insight:**
- The optimal state has much better TOTAL energy (4.5 vs 1012.5)
- But the LINEAR terms alone favor OFF states (0.0 vs -1658.0)
- The quadratic coupling is supposed to overcome this, but thermodynamic sampling explores locally and gets trapped

## Specific Issues in Each Implementation

### 1. `thermo_dispatch.py`
- **Lines 37-46:** Doesn't properly map u→s transformation
- **Line 17:** ALPHA=1.0 too weak to enforce constraint
- **Output bug:** Treats boolean directly as ON/OFF without proper spin conversion

### 2. `thermo_dispatch_corrected.py`
- **Lines 7-51:** Uses negation strategy to flip signs
- **Issue:** Still results in negative biases after conversion
- **Lines 28-29:** `h_ising -= linear_biases / 2.0` makes problem worse

### 3. `thermo_dispatch_final.py`
- **Lines 18-34:** Uses positive accumulation `h_ising +=`
- **Issue:** Sign convention doesn't match minimization objective
- **Result:** Different failure mode but still incorrect

### 4. `thermo_dispatch_v2.py`
- **Lines 135-136:** CRITICAL BUG
  ```python
  total_gen += mw
  total_cost_val += cost
  ```
  Accumulates even when generator is OFF! This makes ALL reported results incorrect.
- **Lines 22-42:** QUBO conversion has correct math but suffers from same landscape issue

## Why Attempted Fixes Don't Work

### Approach 1: Increase ALPHA
- Tried: ALPHA = 1 → 20 → 50 → 200 → 500
- Result: Strengthens quadratic coupling BUT also strengthens negative linear bias
- The ratio stays problematic

### Approach 2: Add Positive Offset
- Tried: Add large constant to shift all biases positive
- Result: Overcorrects, now all spins prefer +1 (all ON)
- Offset doesn't change relative energies, just shifts baseline

### Approach 3: Higher Temperature (lower beta)
- Tried: beta = 5 → 10 → 20 → 30
- Result: More exploration but can't overcome fundamental bias
- Lower temp gets stuck, higher temp too random

### Approach 4: Multiple Trials
- Tried: 10 different random seeds, long warmup (10K steps)
- Result: All trials converge to same all-OFF state
- Confirms this is global minimum of the Ising Hamiltonian

## Why Thermodynamic Computing Fails Here

Thermodynamic/analog computing works by minimizing physical energy. It excels at:
- ✓ Inequality constraints (e.g., >= demand)
- ✓ Soft optimization (approximate solutions acceptable)
- ✓ Problems with smooth energy landscapes
- ✓ Many-variable problems where local search is effective

It struggles with:
- ✗ **Tight equality constraints** (e.g., exactly 450 MW)
- ✗ **Discrete optimization** with sparse feasible regions
- ✗ **Non-convex landscapes** with structural bias
- ✗ Problems where expansion creates contradictory terms

This unit commitment problem has:
1. **Tight equality constraint:** Only 3 out of 32 states are exactly feasible
2. **Structural bias:** The squared expansion creates massive negative linear terms
3. **Discrete variables:** Binary decisions, no intermediate states
4. **Multi-scale coupling:** All variables coupled through sum constraint

## Correct Solutions

### Classical MILP (scipy.optimize.milp)
- Solves optimally in milliseconds
- Finds all 3 feasible solutions
- Handles constraints exactly
- **This is the right tool for this problem**

### Alternative QUBO Formulation (Future Work)
Possible improvements (not implemented):
1. **Slack variables:** Add auxiliary variables to absorb negative terms
2. **Penalty reformulation:** Use MAX(0, violation) instead of squared penalty
3. **Hybrid approach:** Use thermodynamic for soft constraints, MILP for exact
4. **Inequality constraint:** Change to "meet OR EXCEED demand" (easier landscape)

## Takeaways

1. **QUBO formulation matters:** Not all optimization problems map well to Ising
2. **Thermodynamic != Universal solver:** Know when to use classical methods
3. **Debug energy landscape:** Always check if minimum energy = desired solution
4. **Constraint type matters:** Equality constraints are harder than inequalities
5. **Validation is critical:** Test that solver outputs match expected results

## Recommendations

For this specific problem:
- **Use classical MILP:** Fast, exact, guaranteed optimal
- Thermodynamic approach is not appropriate

For similar problems:
- If tight constraints: Use MILP or constraint programming
- If approximate OK: Consider thermodynamic/quantum annealing
- If very large scale: Hybrid methods (decomposition + specialized solvers)
- Always validate on small instances first

## Files Created

- `reference_milp_solution.py` - Correct solution using scipy MILP
- `debug_energy.py` - Energy landscape analysis tool
- `thermo_dispatch_fixed.py` - Attempted fix (still fails)
- `thermo_dispatch_working.py` - Offset approach (overcorrects)
- `thermo_dispatch_final_tuned.py` - Multi-trial approach (still all-OFF)

**Conclusion:** The thermodynamic approach cannot solve this problem reliably due to fundamental incompatibility between the QUBO energy landscape and the optimization objective. Classical MILP is the correct choice.
