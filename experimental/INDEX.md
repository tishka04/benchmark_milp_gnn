# Experimental Framework Index

Complete experimental framework for rigorous Hybrid vs MILP benchmarking.

## Quick Start

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Test the framework
python test_framework.py

# 3. Run small-scale experiment
python run_experiments.py --scales small

# 4. Review results
cat results/analysis/REPORT.md
```

## Documentation

### Core Documents

1. **[README.md](README.md)** - Overview and usage guide
2. **[HYPOTHESES.md](HYPOTHESES.md)** - Research hypotheses and expected outcomes
3. **[PROTOCOL.md](PROTOCOL.md)** - Detailed step-by-step experimental protocol
4. **[PYOMO_IMPLEMENTATION.md](PYOMO_IMPLEMENTATION.md)** - Why we use Pyomo + HiGHS
5. **[requirements.txt](requirements.txt)** - Python dependencies

### Source Code

#### Data Models
- **[data_models.py](data_models.py)** - `UCInstance`, `UCSolution`, `ExperimentConfig`

#### Problem Components
- **[instance_generator.py](instance_generator.py)** - Reproducible instance generation
- **[dispatch_solver.py](dispatch_solver.py)** - Single source of truth for dispatch

#### Solvers
- **[milp_solver.py](milp_solver.py)** - Monolithic MILP solver
- **[hybrid_solver.py](hybrid_solver.py)** - Thermodynamic + classical hybrid

#### Experiment Infrastructure
- **[experiment_runner.py](experiment_runner.py)** - Orchestrates full protocol
- **[analysis.py](analysis.py)** - Metrics computation and reporting

#### Entry Points
- **[run_experiments.py](run_experiments.py)** - Main CLI entry point
- **[test_framework.py](test_framework.py)** - Framework verification tests

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    Experimental Protocol                     │
└─────────────────────────────────────────────────────────────┘
                            │
        ┌───────────────────┴───────────────────┐
        ▼                                       ▼
┌───────────────────┐                  ┌───────────────────┐
│ Instance Generator│                  │  Experiment       │
│                   │                  │  Runner           │
│ - Reproducible    │                  │                   │
│ - Multiple scales │                  │ - Reference runs  │
│ - Fixed seeds     │                  │ - Standard runs   │
└─────────┬─────────┘                  │ - Time budgets    │
          │                            └─────────┬─────────┘
          │ UCInstance                           │
          ▼                                      │
┌─────────────────────────────────────────┐     │
│        Single Source of Truth           │     │
│                                         │     │
│  ┌──────────────────────────────────┐  │     │
│  │   Dispatch Solver                │  │     │
│  │   - solve_dispatch_given_commit  │  │     │
│  │   - calculate_total_cost         │  │     │
│  │   - verify_feasibility           │  │     │
│  └──────────────────────────────────┘  │     │
└────────────┬────────────────────────────┘     │
             │                                  │
    ┌────────┴────────┐                        │
    ▼                 ▼                        ▼
┌─────────┐    ┌──────────────┐    ┌──────────────────┐
│  MILP   │    │   Hybrid     │    │   Analysis       │
│ Solver  │    │   Solver     │    │                  │
│         │    │              │    │ - Per-instance   │
│ - HiGHS │    │ - Ising      │    │ - Aggregated     │
│ - Binary│    │ - Sampling   │    │ - Statistical    │
│ - LP    │    │ - Dispatch   │    │ - Plots          │
└─────────┘    └──────────────┘    └──────────────────┘
     │                │                       │
     └────────────────┴───────────────────────┘
                      │
                      ▼
              ┌──────────────┐
              │   Results    │
              │              │
              │ - Metrics    │
              │ - Reports    │
              │ - Artifacts  │
              └──────────────┘
```

## Key Features

### 1. Model Alignment ✓
Both MILP and Hybrid use **identical**:
- Problem formulation
- Dispatch solver
- Cost calculation
- Data structures

### 2. Reproducibility ✓
- Fixed random seeds
- Documented hardware
- Saved configurations
- Archived datasets

### 3. Fairness ✓
- Same time budgets
- Single-threaded execution
- Standardized metrics
- Controlled experiments

### 4. Rigor ✓
- Reference solutions (J_ref)
- Multiple instances per scale
- Statistical analysis
- Hypothesis testing

## Experimental Scales

| Scale | N (units) | T (periods) | Instances | Time Budget | Expected |
|-------|-----------|-------------|-----------|-------------|----------|
| Small | 20, 50 | 24 | 10 | 5 min | MILP wins |
| Medium | 50, 100 | 24, 96 | 10 | 10 min | Competitive |
| Large | 100, 200 | 24, 96 | 10 | 30 min | Hybrid wins |
| X-Large | 400 | 24, 96 | 10 | 60 min | Hybrid dominates |

## Output Structure

```
results/
├── experiment_config.json          # Configuration used
├── instances/                      # Generated instances
│   ├── dataset_index.json
│   ├── N20_T24/
│   │   ├── instance_000.json
│   │   ├── instance_001.json
│   │   └── ...
│   └── ...
├── results_reference/              # Reference MILP (large budget)
│   ├── N20_T24/
│   │   ├── instance_000_milp_ref.json
│   │   └── ...
│   └── ...
├── results_standard/               # Standard budget experiments
│   ├── N20_T24/
│   │   ├── instance_000_milp.json
│   │   ├── instance_000_hybrid.json
│   │   └── ...
│   └── ...
└── analysis/                       # Analysis outputs
    ├── per_instance_metrics.json
    ├── aggregated_metrics.json
    └── REPORT.md
```

## Metrics Computed

### Per-Instance
- **Feasibility:** Constraint satisfaction
- **Cost:** Total objective value
- **Gap vs Reference:** `(cost - J_ref) / |J_ref| × 100%`
- **Runtime:** Wall-clock time
- **Optimality:** Proven optimal (MILP only)

### Aggregated (by scale)
- **Mean/Median/StdDev:** For all metrics
- **Win Rates:** % instances where Hybrid < MILP
- **Time Ratios:** MILP time / Hybrid time
- **Cost Ratios:** Hybrid cost / MILP cost
- **Feasibility Rates:** % feasible solutions

## Customization Points

### Instance Generation
- Unit size distributions
- Cost structure
- Demand patterns
- Feasibility margins

### MILP Configuration
- Solver choice (HiGHS, Gurobi, CPLEX)
- MIP gap tolerance
- Thread count
- Presolve options

### Hybrid Configuration
- Temperature schedule
- Sampling iterations
- Warmup steps
- Fallback heuristics

### Time Budgets
- Per-scale budgets
- Reference budget
- Checkpoint times

## Testing

```bash
# Quick test (10 units, 24 periods)
python test_framework.py

# Small experiment (20-50 units, 5 instances)
python run_experiments.py --scales small --instances-per-scale 5

# Medium experiment
python run_experiments.py --scales medium

# Full experiment (warning: takes days)
python run_experiments.py --scales full
```

## Troubleshooting

### Common Issues

**Import errors:**
```bash
pip install -r requirements.txt --upgrade
```

**MILP too slow:**
- Reduce time budgets
- Use smaller instances
- Check scipy version (need ≥1.9 for HiGHS)

**Hybrid infeasible:**
- Increase samples per period
- Adjust temperature
- Check instance generation (ensure margin)

**Memory issues:**
- Process one scale at a time
- Reduce instances per scale
- Run on larger machine

## Citation

```bibtex
@software{hybrid_milp_experimental_framework,
  title={Experimental Framework for Hybrid vs MILP Unit Commitment Benchmarking},
  author={},
  year={2024},
  url={https://github.com/...}
}
```

## License

MIT

## Contact

For issues or questions, please refer to the documentation or create an issue.

---

**Status:** ✓ Framework Complete and Tested

**Last Updated:** November 2024
