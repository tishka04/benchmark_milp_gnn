# Setup Guide: Getting Started in 5 Minutes

Quick start guide to get your experimental framework running.

## Prerequisites

- Python 3.9+
- pip
- 8GB+ RAM recommended

## Step 1: Install Dependencies (2 minutes)

```bash
cd c:\Users\Dell\projects\multilayer_milp_gnn\benchmark\experimental
pip install -r requirements.txt
```

**Expected output:**
```
Successfully installed numpy-1.24.0 pyomo-6.7.0 highspy-1.5.0 jax-0.4.0 jaxlib-0.4.0 thrml-0.1.0
```

**Note:** The framework now uses Pyomo + HiGHS for MILP solving, providing:
- Cleaner model formulation
- Better solver control
- More flexible optimization modeling

## Step 2: Verify Installation (1 minute)

```bash
python test_framework.py
```

**Expected output:**
```
EXPERIMENTAL FRAMEWORK TEST SUITE
================================================================================

TEST 1: Instance Generation
✓ Generated instance: N10_T24
...

ALL TESTS PASSED ✓
```

If you see errors:
- **Import errors:** Run `pip install -r requirements.txt --upgrade`
- **JAX errors:** May need CPU-only version: `pip install jax[cpu]`
- **thrml errors:** Check installation: `pip install thrml --upgrade`

## Step 3: Run Quick Experiment (2 minutes)

```bash
python run_experiments.py --scales small --instances-per-scale 3
```

This will:
1. Generate 6 instances (N∈{20,50}, T=24, 3 per scale)
2. Run reference MILP
3. Run standard MILP and Hybrid
4. Analyze results

**Expected runtime:** ~5-10 minutes

## Step 4: View Results

```bash
# View summary report
cat results/analysis/REPORT.md

# Or open in text editor
notepad results/analysis/REPORT.md
```

## What You Should See

### Instance Generation
```
GENERATING INSTANCE DATASET
================================================================================

N20_T24:
  Generated 3 instances for N20_T24
N50_T24:
  Generated 3 instances for N50_T24

DATASET COMPLETE: 6 instances
```

### MILP Results (Small Scale)
```
MILP SOLVER: N20_T24
================================================================================
Units: 20, Periods: 24, Variables: 1440
...
MILP SOLUTION FOUND
Cost: $123,456.78
Time: 12.5s
Optimal: True
```

### Hybrid Results (Small Scale)
```
HYBRID SOLVER: N20_T24
================================================================================
Units: 20, Periods: 24
...
HYBRID SOLUTION FOUND
Cost: $124,567.89
Time: 8.3s
Candidates evaluated: 150
```

### Analysis Report
```markdown
# Experimental Results: Hybrid vs MILP

## Summary by Scale

### N20_T24

**Instances:** 3

#### MILP
- Gap vs Ref: 0.00% ± 0.00%
- Solve Time: 12.5s ± 3.2s
- Feasible: 3/3
- Optimal: 3/3

#### Hybrid
- Gap vs Ref: 0.85% ± 0.32%
- Solve Time: 8.1s ± 2.1s
- Feasible: 3/3

#### Comparison
- **MILP is 1.54x faster**
- Cost Ratio (Hybrid/MILP): 1.008
- Hybrid wins on time: 0/3
```

## Next Steps

### Run Larger Experiments

**Medium scale** (N∈{20,50,100}, T∈{24,96}, 10 instances):
```bash
python run_experiments.py --scales medium
```
Runtime: ~2-4 hours

**Full scale** (N∈{20,50,100,200,400}, T∈{24,96}, 10 instances):
```bash
python run_experiments.py --scales full
```
Runtime: ~1-2 days (consider running on cluster)

### Customize Configuration

Edit `run_experiments.py` or create your own:

```python
from pathlib import Path
from data_models import ExperimentConfig
from experiment_runner import ExperimentRunner

config = ExperimentConfig(
    n_units_list=[20, 50, 100, 200],
    n_periods_list=[24, 96],
    instances_per_scale=10,
    time_budget_small=300,  # 5 minutes
    time_budget_medium=600,  # 10 minutes
    time_budget_large=1800,  # 30 minutes
)

runner = ExperimentRunner(config, Path("my_results"))
runner.run_full_protocol(regenerate_instances=True)
```

### Explore Your Data

```python
# Load and analyze results
from analysis import ResultsAnalyzer
from pathlib import Path

analyzer = ResultsAnalyzer(Path("results"))
metrics, aggregated = analyzer.analyze_all()

# Print summary
import json
print(json.dumps(aggregated["N50_T24"], indent=2))
```

## Common Issues

### Issue: "ModuleNotFoundError: No module named 'thrml'"

**Solution:**
```bash
pip install thrml
# Or if that fails:
pip install git+https://github.com/tansey/thrml.git
```

### Issue: MILP solver very slow or not working

**Solution:** Ensure Pyomo and HiGHS are properly installed:
```bash
pip install pyomo highspy --upgrade
python -c "import pyomo.environ as pyo; print(pyo.__version__)"
python -c "import highspy; print('HiGHS available')"
```

**Alternative:** If `highspy` installation fails, you can also use HiGHS as an external executable:
```bash
# Download HiGHS from https://github.com/ERGO-Code/HiGHS/releases
# Add to PATH, then use:
solver = pyo.SolverFactory('highs')  # instead of 'appsi_highs'
```

### Issue: "JAX requires jaxlib..."

**Solution:** Install matching versions:
```bash
pip install jax jaxlib --upgrade
# For CPU-only:
pip install jax[cpu] --upgrade
```

### Issue: Out of memory

**Solution:** Reduce problem size:
```python
config = ExperimentConfig(
    n_units_list=[20, 50],  # Smaller sizes
    n_periods_list=[24],    # Fewer periods
    instances_per_scale=5   # Fewer instances
)
```

## Directory Structure After Setup

```
experimental/
├── results/                        # Created after first run
│   ├── experiment_config.json
│   ├── instances/
│   ├── results_reference/
│   ├── results_standard/
│   └── analysis/
│       ├── per_instance_metrics.json
│       ├── aggregated_metrics.json
│       └── REPORT.md
└── [source files...]
```

## Verification Checklist

- [ ] All dependencies installed (`pip list | grep -E "numpy|scipy|jax|thrml"`)
- [ ] Test framework passes (`python test_framework.py`)
- [ ] Quick experiment completes (`python run_experiments.py --scales small`)
- [ ] Results generated (`ls results/analysis/REPORT.md`)
- [ ] Can read report (`cat results/analysis/REPORT.md`)

## Getting Help

1. **Read the docs:**
   - [INDEX.md](INDEX.md) - Framework overview
   - [README.md](README.md) - Detailed usage
   - [PROTOCOL.md](PROTOCOL.md) - Step-by-step guide
   - [HYPOTHESES.md](HYPOTHESES.md) - Research context

2. **Check test output:**
   ```bash
   python test_framework.py 2>&1 | tee test_log.txt
   ```

3. **Verify versions:**
   ```bash
   python --version
   pip list | grep -E "numpy|scipy|jax|thrml"
   ```

## Success Criteria

You're ready to proceed if:
1. ✓ `test_framework.py` shows "ALL TESTS PASSED"
2. ✓ Small experiment completes without errors
3. ✓ `REPORT.md` contains meaningful results
4. ✓ Both MILP and Hybrid produce feasible solutions

## Time to Science! 🚀

You now have a complete experimental framework for rigorous Hybrid vs MILP benchmarking.

**Recommended workflow:**
1. Run small experiments to validate
2. Scale up to medium for initial results
3. Run full experiments for publication
4. Analyze, iterate, publish!

Good luck with your research!
