# Research Hypotheses: Hybrid vs MILP

## Overview

This document formalizes the hypotheses that guide the experimental design and interpretation of results.

## Core Claim

**"The hybrid solver acts as a scalable heuristic surrogate for large multi-period unit commitment problems. It is dominated by monolithic MILP at small scales but becomes competitive or superior when the combinatorial explosion kicks in."**

This is a defensible, publishable claim that acknowledges MILP's strengths while demonstrating the hybrid's value proposition.

---

## H1: Small/Medium Scale Instances

### Hypothesis

For small to medium instances (N ≤ 50 units, T ≤ 96 periods), monolithic MILP with a modern solver (HiGHS/Gurobi):

1. **Is faster:** Solves to optimality or near-optimality in seconds to minutes
2. **Has equal or better objective value:** Finds provably optimal or near-optimal solutions
3. **Is the preferred method:** No advantage to hybrid decomposition

### Rationale

At small scales:
- The combinatorial search space is manageable
- Modern MILP solvers excel with tight LP relaxations
- Branch-and-bound pruning is highly effective
- Overhead of hybrid sampling is not justified

### Expected Experimental Evidence

- MILP solve times: < 60 seconds
- MILP optimality rate: > 80%
- MILP MIP gaps: < 1%
- Hybrid times: comparable or slower
- Hybrid costs: equal or 1-5% worse

### Interpretation

**Verdict:** "For N ≤ 50, MILP is preferable. The problem structure at this scale does not warrant decomposition."

---

## H2: Large Scale / Stressed Instances

### Hypothesis

For large or structurally complex instances (N > 100 units, T > 96 periods, or with additional complexity like network constraints, DR, storage):

1. **MILP deteriorates:**
   - Frequently hits time limits without proving optimality
   - MIP gaps remain large (5-20%)
   - May fail to find any feasible solution within budget

2. **Hybrid remains effective:**
   - Consistently delivers feasible solutions
   - Achieves competitive costs (within 1-3% of best-known)
   - Does so in significantly less wallclock time
   - Sometimes beats MILP's best solution within same budget

### Rationale

At large scales:
- Combinatorial explosion makes exhaustive search infeasible
- MILP's LP relaxation bounds become weaker
- Branch-and-bound tree grows exponentially
- Hybrid's thermodynamic sampling:
  - Focuses search on "good" regions of commitment space
  - Avoids exhaustive enumeration
  - Leverages physics-inspired heuristics for commitment
  - Uses exact dispatch solver for cost evaluation

### Expected Experimental Evidence

**MILP:**
- Solve times: hit time limits (30-60 min)
- Optimality rate: < 30%
- MIP gaps at timeout: 5-20%
- Feasibility issues for largest instances

**Hybrid:**
- Solve times: 5-15 minutes
- Consistent feasibility
- Gaps vs best-known: 1-3%
- Time to first feasible: < 1 minute

### Quantitative Targets

For N ≥ 200:
- Hybrid finds solutions ≥ 2x faster than MILP
- Hybrid achieves smaller gaps in allotted time
- Hybrid wins on cost-quality in 50-70% of instances

### Interpretation

**Verdict:** "For N > 100, the hybrid approach becomes attractive. It trades MILP's optimality guarantees for scalability, delivering high-quality feasible solutions when MILP struggles."

---

## Additional Hypotheses

### H3: Anytime Performance

**Hypothesis:** Hybrid exhibits better anytime performance than MILP.

**Evidence:**
- Time-to-first-feasible: Hybrid < 1 min, MILP > 5 min
- Solution quality improves smoothly with time for Hybrid
- MILP shows discontinuous improvement (branch-and-bound jumps)

### H4: Scaling Behavior

**Hypothesis:** Hybrid runtime scales more gracefully with problem size.

**Evidence:**
- Hybrid runtime ~ O(N × T) (polynomial)
- MILP runtime ~ exponential or worse
- Crossover point: N ≈ 75-100 units

### H5: Robustness

**Hypothesis:** Hybrid is more robust to tight constraints and low feasibility margins.

**Evidence:**
- Success rate on "hard" instances
- Behavior when capacity margins are reduced
- Sensitivity to demand variability

---

## Null Hypotheses (to reject)

### N1: "Hybrid always beats MILP"

**False.** We expect MILP to dominate at small scales. This strengthens the claim that hybrid is a *scalable alternative*, not a universal replacement.

### N2: "MILP always beats Hybrid"

**False.** At large scales, we expect Hybrid to find better solutions within fixed time budgets, demonstrating its value proposition.

### N3: "The differences are due to model mismatch"

**False.** Perfect model alignment (same dispatch solver, same cost function) ensures observed differences reflect algorithmic choices, not formulation artifacts.

---

## Experimental Design Implications

### What we're NOT trying to show

- ❌ Hybrid is always better
- ❌ MILP is obsolete
- ❌ One method dominates everywhere

### What we ARE trying to show

- ✅ There exists a crossover point (~N=75-100)
- ✅ Hybrid scales better to large instances
- ✅ MILP is preferable for small-scale problems
- ✅ The hybrid is a principled, scalable heuristic
- ✅ Thermodynamic sampling + classical dispatch is a viable architecture

---

## Statistical Testing

### Primary Tests

1. **Paired t-test:** Compare costs on same instances
2. **Wilcoxon signed-rank:** Non-parametric alternative
3. **Win rate analysis:** % of instances where Hybrid < MILP

### Significance Levels

- α = 0.05 for primary comparisons
- Bonferroni correction for multiple scales

### Effect Sizes

Report:
- Cohen's d for cost differences
- Time ratio distributions
- Gap distributions

---

## Publication Strategy

### Title (suggested)

"Scalable Unit Commitment via Hybrid Thermodynamic-Classical Decomposition: When to Use MILP and When to Sample"

### Key Message

"We demonstrate that for UC problems:
- N ≤ 50: Use MILP
- N > 100: Use Hybrid
- Transition zone (50-100): Context-dependent"

### Figures

1. **Runtime vs N:** Show crossover point
2. **Gap vs Time:** Anytime performance curves
3. **Cost ratio boxplots:** By scale
4. **Success rate bars:** Feasibility and optimality

### Tables

1. Aggregated metrics by scale
2. Statistical test results
3. Hardware specifications and fairness controls

---

## Conclusion

These hypotheses guide a nuanced, defensible narrative:

> "We neither claim hybrid always wins, nor that MILP is obsolete. Instead, we characterize a crossover point beyond which the hybrid's scalability becomes valuable, while acknowledging MILP's superiority at small scales."

This is the kind of honest, rigorous story that reviewers appreciate.
