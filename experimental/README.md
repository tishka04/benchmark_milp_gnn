# Experimental Framework: Hybrid vs MILP Benchmark

This framework implements a rigorous experimental protocol for comparing:
- **Monolithic MILP** solvers (using HiGHS via scipy)
- **Hybrid thermodynamic + classical** dispatch solver

## Research Hypotheses

### H1 (Small/Medium Scale)
For small to medium instances (N ≤ 50), monolithic MILP:
- Is faster
- Has equal or better objective value
- Is the preferred method

### H2 (Large Scale / Stressed Instances)
For large or complex instances (N > 100), monolithic MILP deteriorates (timeouts/large gaps), while the hybrid:
- Consistently delivers feasible solutions
- Achieves competitive costs in significantly less time
- Becomes the attractive alternative

**Key Claim:** "The hybrid is a scalable heuristic surrogate that becomes attractive when MILP starts to struggle."

## Framework Architecture

### 1. Single Canonical Problem Formulation

**Unit Commitment + Dispatch:**
- Variables: `u[i,t] ∈ {0,1}` (commitment), `P[i,t] ∈ ℝ+` (dispatch)
- Objective: `Σ(marginal_cost * P) + Σ(startup_cost * startups)`
- Constraints:
  - Demand balance: `Σ_i P[i,t] = demand[t]`
  - Min/Max: `p_min[i] * u[i,t] ≤ P[i,t] ≤ p_max[i] * u[i,t]`

### 2. Perfect Model Alignment

Both MILP and Hybrid use:
- **Same data model** (`UCInstance`)
- **Same dispatch solver** (`dispatch_solver.py`)
- **Same cost function** (`calculate_total_cost`)

This ensures differences reflect **algorithmic quality**, not model mismatch.

### 3. Instance Generation

**Scales:** N ∈ {20, 50, 100, 200, 400}, T ∈ {24, 96}

**Features:**
- Reproducible (fixed seeds)
- Realistic demand profiles (daily patterns)
- Guaranteed feasibility
- Varied unit sizes and costs

### 4. Experimental Protocol

```
1. Generate instances (K=10 per scale)
2. Reference MILP runs (large budget → J_ref)
3. Standard-budget MILP runs
4. Standard-budget Hybrid runs
5. Metrics computation and analysis
```

**Time Budgets:**
- N ≤ 50: 5 minutes
- 50 < N ≤ 100: 10 minutes
- 100 < N ≤ 200: 30 minutes
- N > 200: 60 minutes

## File Structure

```
experimental/
├── __init__.py                 # Package initialization
├── data_models.py              # UCInstance, UCSolution, ExperimentConfig
├── instance_generator.py       # Instance generation with seeds
├── dispatch_solver.py          # Single source of truth for dispatch
├── milp_solver.py             # Monolithic MILP solver
├── hybrid_solver.py           # Thermodynamic + dispatch hybrid
├── experiment_runner.py        # Orchestrates full protocol
├── analysis.py                # Metrics computation and reporting
├── README.md                  # This file
└── run_experiments.py         # Main entry point
```

## Installation

### Requirements

```bash
pip install numpy pyomo highspy jax jaxlib thrml
```

**Key dependencies:**
- `pyomo` ≥ 6.7 (optimization modeling language)
- `highspy` ≥ 1.5 (HiGHS solver Python interface)
- `jax` + `jaxlib` (for differentiable operations)
- `thrml` (for thermodynamic Ising sampling)

### Optional
- `matplotlib` (for plotting)
- `pandas` (for data analysis)

## Usage

### Quick Start

```python
from pathlib import Path
from experimental import ExperimentConfig, ExperimentRunner

# Configure experiment
config = ExperimentConfig(
    n_units_list=[20, 50, 100],
    n_periods_list=[24, 96],
    instances_per_scale=10,
    time_budget_small=300,  # 5 minutes
    time_budget_medium=600,  # 10 minutes
)

# Run full protocol
output_dir = Path("./results")
runner = ExperimentRunner(config, output_dir)
runner.run_full_protocol(regenerate_instances=True)
```

### Running from Command Line

```bash
cd experimental
python run_experiments.py
```

### Custom Experiments

```python
from experimental import UCInstanceGenerator, MILPSolver, HybridSolver
from experimental.data_models import ExperimentConfig

# Generate single instance
generator = UCInstanceGenerator()
instance = generator.generate(n_units=50, n_periods=24, seed=42)

# Solve with MILP
config = ExperimentConfig()
milp_solver = MILPSolver(config)
milp_solution = milp_solver.solve(instance, time_limit=300)

# Solve with Hybrid
hybrid_solver = HybridSolver(config)
hybrid_solution = hybrid_solver.solve(instance, time_limit=300)

# Compare
print(f"MILP: ${milp_solution.total_cost:,.2f} in {milp_solution.solve_time:.1f}s")
print(f"Hybrid: ${hybrid_solution.total_cost:,.2f} in {hybrid_solution.solve_time:.1f}s")
```

## Metrics and Analysis

### Per-Instance Metrics

- **Feasibility:** Constraint violations
- **Objective value:** Total cost (fuel + startups)
- **Gap vs Reference:** `(cost - J_ref) / |J_ref| × 100%`
- **Runtime:** Time to solution
- **Time to first feasible** (Hybrid only)

### Aggregated Metrics (by Scale)

- Mean/median gap, runtime
- Standard deviation
- Feasibility rate
- Optimality rate (MILP)
- Win rate (Hybrid vs MILP)

### Reports Generated

1. **per_instance_metrics.json:** Detailed metrics for every instance
2. **aggregated_metrics.json:** Statistics by scale
3. **REPORT.md:** Human-readable summary

## Fairness and Reproducibility

### Hardware Control
- Single-threaded execution
- Fixed random seeds
- Same time budgets for both methods

### Model Alignment
- Both methods use identical:
  - Problem formulation
  - Constraint definitions
  - Cost calculations
  - Data structures

### Randomness Control
- Instance generation: `seed = base_seed + instance_id`
- Thermodynamic sampling: `seed = 42 + seed_offset + period * 100`

## Expected Results

### Small Scale (N=20, N=50)
- **MILP:** Solves to optimality in seconds/minutes
- **Hybrid:** Slower, similar or slightly worse cost
- **Verdict:** MILP preferable

### Medium Scale (N=100)
- **MILP:** May struggle with time limits
- **Hybrid:** Quick feasible solutions
- **Verdict:** Competitive

### Large Scale (N=200, N=400)
- **MILP:** Often hits time limit with large gaps (5-20%)
- **Hybrid:** Systematically finds feasible solutions quickly, reaches 1-3% gap
- **Verdict:** Hybrid preferable for scalability

## Customization

### Adjusting MILP Settings

```python
config = ExperimentConfig(
    milp_solver="highs",  # or "gurobi", "cplex"
    mip_gap_tolerance=0.001,  # 0.1%
    milp_threads=1  # for fairness
)
```

### Adjusting Hybrid Settings

```python
config = ExperimentConfig(
    hybrid_n_samples_per_period=10,
    hybrid_n_seeds=5,
    hybrid_temperature=5.0,
    hybrid_warmup=200,
    hybrid_steps_per_sample=3
)
```

### Custom Instance Generator

```python
generator = UCInstanceGenerator(
    capacity_range=(50, 500),
    cost_range=(20, 80),
    demand_peak_factor=0.85,
    demand_base_factor=0.40
)
```

## Extending the Framework

### Adding a New Solver

1. Create solver class implementing `.solve(instance, time_limit)` → `UCSolution`
2. Use `dispatch_solver.py` for cost alignment
3. Add to `experiment_runner.py`

### Adding Constraints

1. Extend `UCInstance` in `data_models.py`
2. Update `dispatch_solver.py` for feasibility checks
3. Update both `milp_solver.py` and `hybrid_solver.py`

### Custom Metrics

Extend `ResultsAnalyzer` in `analysis.py`:

```python
def _compute_custom_metric(self, instance_metrics):
    # Your metric logic
    pass
```

## Citation

If you use this framework, please cite:

```
@software{hybrid_milp_benchmark,
  title={Experimental Framework for Hybrid vs MILP Benchmarking},
  author={Your Name},
  year={2024},
  url={https://github.com/...}
}
```

## License

MIT License

## Contact

For questions or issues, please open an issue on GitHub.

---

**Key Principle:** This framework is designed to produce defensible, publishable results by ensuring perfect model alignment and rigorous experimental control.
