# Changelog

## [1.1.0] - 2024-11-24

### Changed

#### MILP Solver: Migrated from scipy to Pyomo + HiGHS

**What changed:**
- Replaced `scipy.optimize.milp` with Pyomo algebraic modeling
- Now uses `pyomo.environ` + HiGHS solver backend

**Why:**
- **Cleaner formulation:** Constraints written as mathematical rules instead of matrix manipulation
- **Better maintainability:** Named constraints, easier to debug, natural syntax
- **More flexible:** Easy to extend with ramping, reserves, storage, etc.
- **Industry standard:** Pyomo is the de facto Python optimization framework
- **Same performance:** Both use HiGHS backend, solve times identical

**What stayed the same:**
- Interface: `solve(instance, time_limit)` → `UCSolution`
- Cost calculation: Still uses canonical `calculate_total_cost()`
- Feasibility checking: Still uses `verify_solution_feasibility()`
- Results format: Same `UCSolution` object
- **Perfect model alignment maintained!**

**Dependencies updated:**
```diff
- scipy>=1.9.0
+ pyomo>=6.7.0
+ highspy>=1.5.0
```

**Installation:**
```bash
pip install -r requirements.txt --upgrade
```

**Example constraint comparison:**

*Before (scipy):*
```python
A = np.zeros(n_vars)
A[offset_p + t*N + i] = 1.0
A[offset_u + t*N + i] = -instance.p_max[i]
constraints_list.append(LinearConstraint(A, lb=-np.inf, ub=0))
```

*After (Pyomo):*
```python
def max_generation_rule(m, i, t):
    return m.P[i, t] <= instance.p_max[i] * m.u[i, t]

model.max_generation = pyo.Constraint(
    model.Units, model.Periods, rule=max_generation_rule
)
```

**Migration impact:**
- ✅ All existing code works unchanged
- ✅ Tests pass with identical results
- ✅ Fair comparison maintained
- ✅ Better foundation for future research

**For more details:** See [PYOMO_IMPLEMENTATION.md](PYOMO_IMPLEMENTATION.md)

---

## [1.0.0] - 2024-11-24

### Added

- Initial release of experimental framework
- Complete UC+Dispatch formulation
- Instance generator with reproducible seeds
- MILP solver (scipy-based)
- Hybrid thermodynamic solver
- Experiment runner with time budgets
- Metrics and analysis tools
- Comprehensive documentation

### Features

- Perfect model alignment between MILP and Hybrid
- Hypothesis-driven experimental protocol
- Fair time budgets and resource constraints
- Statistical analysis and reporting
- Publication-ready outputs
