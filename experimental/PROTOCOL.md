# Experimental Protocol: Step-by-Step Guide

This document provides a detailed step-by-step guide for running the complete experimental protocol.

## Table of Contents

1. [Setup](#setup)
2. [Dataset Generation](#dataset-generation)
3. [Reference MILP Runs](#reference-milp-runs)
4. [Standard-Budget Experiments](#standard-budget-experiments)
5. [Post-Processing](#post-processing)
6. [Ablation Studies](#ablation-studies)
7. [Publication-Ready Outputs](#publication-ready-outputs)

---

## Setup

### 1. Install Dependencies

```bash
cd experimental
pip install -r requirements.txt
```

### 2. Verify Installation

```python
import numpy as np
import scipy
import jax
import thrml
print("All dependencies installed successfully!")
```

### 3. Hardware Documentation

Document your experimental setup for reproducibility:

```python
import platform
import psutil

print(f"OS: {platform.system()} {platform.release()}")
print(f"CPU: {platform.processor()}")
print(f"Cores: {psutil.cpu_count()}")
print(f"RAM: {psutil.virtual_memory().total / 1e9:.1f} GB")
```

Save this to `results/hardware_specs.txt`.

---

## Dataset Generation

### Step 1: Configure Instance Generation

Edit `run_experiments.py` or create custom config:

```python
from data_models import ExperimentConfig

config = ExperimentConfig(
    n_units_list=[20, 50, 100, 200, 400],
    n_periods_list=[24, 96],
    instances_per_scale=10,
    # ... time budgets, etc.
)
```

### Step 2: Generate Instances

```bash
python run_experiments.py --scales full --regenerate
```

Or programmatically:

```python
from instance_generator import UCInstanceGenerator
from pathlib import Path

generator = UCInstanceGenerator()
dataset = generator.generate_dataset(
    n_units_list=[20, 50, 100, 200, 400],
    n_periods_list=[24, 96],
    instances_per_scale=10,
    output_dir=Path("results/instances"),
    base_seed=42
)
```

### Step 3: Verify Instance Quality

Check a few instances manually:

```python
from data_models import UCInstance

instance = UCInstance.load("results/instances/N50_T24/instance_000.json")
print(f"Units: {instance.n_units}, Periods: {instance.n_periods}")
print(f"Peak demand: {instance.demand.max():.0f} MW")
print(f"Total capacity: {instance.p_max.sum():.0f} MW")
print(f"Margin: {(instance.p_max.sum() / instance.demand.max() - 1) * 100:.1f}%")
```

**Expected:** Margin should be 10-30% for feasibility without being trivial.

---

## Reference MILP Runs

### Purpose

Establish J_ref (best-known objective) for each instance with large time budget.

### Step 1: Run Reference Experiments

The experiment runner handles this automatically:

```python
runner.run_full_protocol()  # Includes reference runs
```

Or manually:

```python
from milp_solver import MILPSolver

solver = MILPSolver(config)

for instance_path in dataset["N50_T24"]:
    instance = UCInstance.load(instance_path)
    solution = solver.solve(
        instance,
        time_limit=7200  # 2 hours
    )
    # Save solution...
```

### Step 2: Monitor Progress

```bash
tail -f results/results_reference/N50_T24/instance_000_milp_ref.json
```

### Step 3: Verify Reference Quality

After completion, check:
- What % of instances were solved to optimality?
- What are the MIP gaps for non-optimal instances?

```python
# Check reference quality
import json
from pathlib import Path

ref_dir = Path("results/results_reference")
for scale_dir in ref_dir.iterdir():
    optimal_count = 0
    total_count = 0
    
    for result_file in scale_dir.glob("*.json"):
        with open(result_file) as f:
            data = json.load(f)
        if data.get('optimal', False):
            optimal_count += 1
        total_count += 1
    
    print(f"{scale_dir.name}: {optimal_count}/{total_count} optimal")
```

**Expected:**
- N=20, N=50: 80-100% optimal
- N=100: 50-80% optimal
- N=200, N=400: 10-30% optimal

---

## Standard-Budget Experiments

### Purpose

Compare MILP and Hybrid under fair time constraints.

### Step 1: Run Experiments

```bash
python run_experiments.py --scales full
```

Time budgets (default):
- N ≤ 50: 5 minutes
- 50 < N ≤ 100: 10 minutes
- 100 < N ≤ 200: 30 minutes
- N > 200: 60 minutes

### Step 2: Monitor Both Methods

```bash
# Terminal 1: Watch MILP
watch -n 5 'ls results/results_standard/*/instance_*_milp.json | wc -l'

# Terminal 2: Watch Hybrid
watch -n 5 'ls results/results_standard/*/instance_*_hybrid.json | wc -l'
```

### Step 3: Handle Failures

If a solver fails:

```python
# Rerun specific instance
from experiment_runner import ExperimentRunner

runner = ExperimentRunner(config, Path("results"))

instance = UCInstance.load("path/to/failed/instance.json")

# Try with more time
solution = runner.milp_solver.solve(instance, time_limit=3600)
```

---

## Post-Processing

### Step 1: Compute Metrics

```python
from analysis import ResultsAnalyzer

analyzer = ResultsAnalyzer(Path("results"))
metrics, aggregated = analyzer.analyze_all()
```

### Step 2: Generate Report

```python
analyzer.generate_report()
```

This creates:
- `results/analysis/per_instance_metrics.json`
- `results/analysis/aggregated_metrics.json`
- `results/analysis/REPORT.md`

### Step 3: Review Key Results

```bash
cat results/analysis/REPORT.md
```

Look for:
- At what N does Hybrid start winning on time?
- At what N does MILP struggle with feasibility?
- Cost quality comparison at each scale

---

## Ablation Studies

### 1. Hybrid Without Thermodynamic Sampling

Test if thermodynamic sampling adds value over random sampling:

```python
class RandomHybridSolver(HybridSolver):
    def _sample_commitment_for_period(self, instance, period, ...):
        # Replace Ising sampling with random sampling
        candidates = []
        for _ in range(self.n_samples_per_period):
            u = np.random.randint(0, 2, instance.n_units)
            candidates.append(u)
        return candidates
```

Run and compare with original hybrid.

### 2. Greedy Heuristic Baseline

```python
def greedy_baseline(instance):
    """Greedy: sort by cost, turn on cheapest until demand met."""
    commitment = np.zeros((instance.n_units, instance.n_periods), dtype=int)
    power = np.zeros((instance.n_units, instance.n_periods))
    
    for t in range(instance.n_periods):
        sorted_idx = np.argsort(instance.marginal_cost)
        cumulative = 0
        for i in sorted_idx:
            if cumulative < instance.demand[t]:
                commitment[i, t] = 1
                power[i, t] = min(instance.p_max[i], instance.demand[t] - cumulative)
                cumulative += power[i, t]
    
    # Calculate cost...
    return solution
```

### 3. Temperature Sensitivity

Run hybrid with different temperatures:

```python
temps = [1.0, 3.0, 5.0, 10.0, 20.0]

for temp in temps:
    config_temp = ExperimentConfig(hybrid_temperature=temp)
    solver = HybridSolver(config_temp)
    # Run and compare...
```

---

## Publication-Ready Outputs

### 1. Summary Table

Create table for paper:

```python
import pandas as pd

data = []
for scale_name, metrics in aggregated.items():
    row = {
        'Scale': scale_name,
        'MILP Gap (%)': f"{metrics['milp']['gaps']['mean']:.2f} ± {metrics['milp']['gaps']['std']:.2f}",
        'Hybrid Gap (%)': f"{metrics['hybrid']['gaps']['mean']:.2f} ± {metrics['hybrid']['gaps']['std']:.2f}",
        'Time Ratio': f"{metrics['comparison']['time_ratios']['mean']:.2f}",
        'Hybrid Wins': f"{metrics['comparison']['hybrid_wins_cost_count']}/{metrics['n_instances']}"
    }
    data.append(row)

df = pd.DataFrame(data)
print(df.to_latex())
```

### 2. Runtime vs Scale Plot

```python
import matplotlib.pyplot as plt

n_units = [20, 50, 100, 200, 400]
milp_times = [...]  # Extract from aggregated
hybrid_times = [...]

plt.figure(figsize=(8, 6))
plt.plot(n_units, milp_times, 'o-', label='MILP', linewidth=2)
plt.plot(n_units, hybrid_times, 's-', label='Hybrid', linewidth=2)
plt.xlabel('Number of Units (N)')
plt.ylabel('Average Solve Time (s)')
plt.yscale('log')
plt.legend()
plt.grid(True, alpha=0.3)
plt.savefig('results/runtime_vs_scale.pdf', bbox_inches='tight')
```

### 3. Gap Distribution Boxplot

```python
fig, axes = plt.subplots(1, 5, figsize=(15, 4), sharey=True)

for i, (scale_name, metrics) in enumerate(aggregated.items()):
    milp_gaps = metrics['milp']['gaps']['values']  # Need to store raw values
    hybrid_gaps = metrics['hybrid']['gaps']['values']
    
    axes[i].boxplot([milp_gaps, hybrid_gaps], labels=['MILP', 'Hybrid'])
    axes[i].set_title(scale_name)
    axes[i].set_ylabel('Gap vs Reference (%)')

plt.savefig('results/gap_distributions.pdf', bbox_inches='tight')
```

### 4. Anytime Performance Curve

If checkpoint data is available:

```python
plt.figure(figsize=(10, 6))

for scale in ['N50_T24', 'N200_T24']:
    # Plot gap vs time for both methods
    pass

plt.xlabel('Time (s)')
plt.ylabel('Gap to Best-Known (%)')
plt.legend()
plt.grid(True, alpha=0.3)
plt.savefig('results/anytime_performance.pdf', bbox_inches='tight')
```

---

## Checklist for Complete Experiment

- [ ] Hardware specs documented
- [ ] All instances generated and verified
- [ ] Reference MILP runs complete (J_ref established)
- [ ] Standard MILP runs complete
- [ ] Standard Hybrid runs complete
- [ ] Metrics computed and saved
- [ ] Report generated and reviewed
- [ ] Ablation studies run (optional)
- [ ] Plots generated
- [ ] Tables formatted for publication
- [ ] Results archived with timestamp

---

## Troubleshooting

### MILP Runs Too Slow

- Check if HiGHS is being used (`scipy>=1.9.0`)
- Reduce MIP gap tolerance
- Reduce time budget for testing

### Hybrid Produces Infeasible Solutions

- Check dispatch solver implementation
- Verify greedy fallback is working
- Increase number of samples
- Adjust temperature

### Memory Issues

- Run smaller batches
- Process one scale at a time
- Reduce number of instances per scale

### Inconsistent Results

- Verify random seeds are fixed
- Check for race conditions if using parallelism
- Ensure single-threaded execution

---

## Estimated Runtimes

**Full Protocol (N ∈ {20, 50, 100, 200, 400}, T ∈ {24, 96}, K=10):**

- Instance generation: < 10 minutes
- Reference MILP: ~100 instances × 2 hours = **200 hours** (8-9 days)
- Standard MILP: ~100 instances × 0.5 hours average = **50 hours** (2 days)
- Standard Hybrid: ~100 instances × 0.25 hours average = **25 hours** (1 day)
- Analysis: < 5 minutes

**Total: ~275 hours (~11-12 days) if run sequentially**

Recommendation: Run in parallel on multiple cores (respecting fairness constraints) or use a compute cluster.

---

## Next Steps After Completion

1. Write paper draft
2. Prepare rebuttal for anticipated reviewer questions
3. Archive dataset and results for reproducibility
4. Prepare supplementary materials
5. Submit code to GitHub with DOI (Zenodo)

Good luck with your experiments!
