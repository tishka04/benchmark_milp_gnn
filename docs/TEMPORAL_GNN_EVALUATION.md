# Temporal GNN Performance Evaluation Guide

This guide shows you how to evaluate your temporal GNN models against MILP baselines using the same metrics as `performance_comparison.ipynb`.

## Quick Start

### Option 1: Using the Evaluation Script (Easiest)

```bash
python scripts/evaluate_temporal_gnn.py --save-plots
```

This will:
- ✅ Load your trained temporal GNN model
- ✅ Evaluate on test set (last 20% of scenarios)
- ✅ Compute all MILP comparison metrics
- ✅ Generate comparison plots
- ✅ Print summary statistics

**Output:**
- Console: Summary statistics
- Plots: Displayed interactively
- Files (if `--save-plots`): `outputs/gnn_runs/temporal_hetero/temporal_gnn_comparison.png`

### Option 2: Using Python API

```python
from pathlib import Path
from src.gnn.temporal_evaluation import evaluate_test_set

# Run evaluation
results_df, summary = evaluate_test_set(
    run_dir=Path('outputs/gnn_runs/temporal_hetero'),
    test_split=0.8,  # Use last 20% as test
    device='cpu',
    verbose=True
)

# View results
print(summary)
display(results_df)
```

---

## Metrics Explained

### 1. Cost Gap
**What it measures:** How close GNN dispatch cost is to MILP optimal objective

**Formula:**
```
cost_gap = (GNN_cost - MILP_objective) / MILP_objective
```

**Interpretation:**
- `< 0`: GNN found cheaper solution (may violate constraints)
- `= 0`: Perfect match with MILP
- `> 0`: GNN solution is more expensive
- `< 0.1` (10%): Generally acceptable

**Example:**
```
MILP objective: $1,000,000
GNN cost:       $1,050,000
Cost gap:       5.0%
```

### 2. Dispatch MAE (Mean Absolute Error)
**What it measures:** Average prediction error on generation values

**Formula:**
```
MAE = mean(|predicted_MW - actual_MW|)
```

**Interpretation:**
- Lower is better
- Typical range: 10-50 MW for well-trained models
- Compare to installed capacity to assess relative error

**Example:**
```
Predicted thermal: 850 MW
Actual thermal:    800 MW
Error:             50 MW
```

### 3. Violation Rate
**What it measures:** Fraction of timesteps with constraint violations

**Formula:**
```
violation_rate = (timesteps_with_unserved_energy) / (total_timesteps)
```

**Interpretation:**
- `0.0`: Perfect feasibility (no unserved energy)
- `> 0.5`: High infeasibility
- Only computable if `unserved` is in target variables

### 4. Speedup
**What it measures:** How much faster GNN inference is vs MILP solving

**Formula:**
```
speedup = MILP_runtime_seconds / GNN_inference_seconds
```

**Interpretation:**
- `> 1`: GNN is faster
- Typical range: 10x - 100x faster
- Higher for larger/more complex scenarios

---

## Example Output

```
====================================================================
Summary Statistics
====================================================================
Scenarios evaluated:        100
Mean cost gap:              3.24%
Median cost gap:            1.85%
Mean dispatch MAE:          28.45 MW
Mean violation rate:        12.34%
Mean speedup:               42.7x
Total GNN inference time:   12.34s
Mean MILP runtime:          4.21s
====================================================================
```

**Generated Plots:**

1. **Cost Gap Bar Chart**: Shows per-scenario cost relative to MILP
   - Green bars: GNN cheaper than MILP
   - Orange bars: Within 10% of MILP
   - Red bars: More than 10% above MILP

2. **Dispatch MAE**: Prediction accuracy per scenario

3. **Violation Rate**: Feasibility per scenario

4. **Speedup**: Runtime comparison (log scale)

---

## Advanced Usage

### Evaluate Specific Model

```bash
python scripts/evaluate_temporal_gnn.py \
    --run-dir outputs/gnn_runs/my_experiment \
    --test-split 0.9 \
    --device cuda \
    --save-csv results.csv
```

### Compare Multiple Models

```bash
# Evaluate baseline
python scripts/evaluate_temporal_gnn.py \
    --run-dir outputs/gnn_runs/rgcn_baseline \
    --save-csv rgcn_results.csv

# Evaluate with temporal edges
python scripts/evaluate_temporal_gnn.py \
    --run-dir outputs/gnn_runs/rgcn_with_temporal \
    --save-csv rgcn_temporal_results.csv

# Compare in Python
import pandas as pd
baseline = pd.read_csv('rgcn_results.csv')
temporal = pd.read_csv('rgcn_temporal_results.csv')

print(f"Baseline MAE: {baseline['dispatch_mae'].mean():.2f}")
print(f"Temporal MAE: {temporal['dispatch_mae'].mean():.2f}")
```

### Programmatic Evaluation

```python
from pathlib import Path
from src.gnn.temporal_evaluation import (
    load_model_and_config,
    evaluate_scenario
)

# Load model once
model, args, target_vars = load_model_and_config(
    Path('outputs/gnn_runs/temporal_hetero')
)

# Evaluate single scenario
metrics = evaluate_scenario(
    model,
    graph_path=Path('outputs/temporal_graphs/supra/scenario_00001.npz'),
    report_path=Path('outputs/scenarios_v1/reports/scenario_00001.json'),
    target_vars=target_vars,
    device='cpu'
)

print(f"Cost gap: {metrics['cost_gap']:.2%}")
print(f"Dispatch MAE: {metrics['dispatch_mae']:.2f} MW")
print(f"Speedup: {metrics['speedup']:.1f}x")
```

---

## Comparison with Original `performance_comparison.ipynb`

| Feature | Original Notebook | Temporal Evaluation | Notes |
|---------|------------------|---------------------|-------|
| **Cost Gap** | ✅ | ✅ | Same metric |
| **Dispatch MAE** | ✅ | ✅ | Same metric |
| **Violation Rate** | ✅ | ⚠️ Simplified | Based on unserved energy |
| **Speedup** | ✅ | ✅ | Same metric |
| **Dataset** | Static graphs | Temporal supra-graphs | Different input format |
| **Models** | Custom architectures | Temporal R-GCN / Separated | Different architectures |
| **Decoder** | ✅ Post-processing | ❌ Raw predictions | No feasibility correction |

**Key Differences:**

1. **No Decoder**: Temporal GNN outputs raw predictions without post-processing to ensure feasibility. This may lead to higher violation rates.

2. **Zone-Level Only**: Metrics are computed on zone-level aggregates (11 zones), not individual assets (63 assets).

3. **Simplified Violations**: Violation rate is based on unserved energy predictions, not full constraint checking.

---

## Troubleshooting

### "No checkpoint found"
Train a model first:
```bash
python -m src.gnn.train_temporal_hetero --epochs 50
```

### "Violation rate is 0"
Add `unserved` to target variables when training:
```bash
python -m src.gnn.train_temporal_hetero --target-vars thermal,nuclear,solar,wind,unserved
```

### "Cost gaps are very high"
This is normal without a decoder. Options:
1. Train with more data / longer
2. Add post-processing to enforce constraints
3. Include `unserved` in targets to let model learn feasibility

### "Evaluation is slow"
Use GPU:
```bash
python scripts/evaluate_temporal_gnn.py --device cuda
```

---

## Best Practices

### 1. Always Evaluate on Test Set
Don't evaluate on training data:
```python
test_split = 0.8  # Use last 20% for testing
```

### 2. Track Multiple Metrics
Don't optimize for cost gap alone - check:
- ✅ Dispatch MAE (prediction accuracy)
- ✅ Violation rate (feasibility)
- ✅ Speedup (computational efficiency)

### 3. Compare Configurations
Train multiple models and compare:
```bash
# Baseline (no temporal edges)
python -m src.gnn.build_hetero_graph_dataset ... --temporal-edges ""
python -m src.gnn.train_temporal_hetero --epochs 50
python scripts/evaluate_temporal_gnn.py --save-csv baseline.csv

# With SOC edges only
python -m src.gnn.build_hetero_graph_dataset ... --temporal-edges soc
python -m src.gnn.train_temporal_hetero --epochs 50
python scripts/evaluate_temporal_gnn.py --save-csv soc_only.csv

# With all temporal edges
python -m src.gnn.build_hetero_graph_dataset ... --temporal-edges soc,ramp,dr
python -m src.gnn.train_temporal_hetero --epochs 50
python scripts/evaluate_temporal_gnn.py --save-csv all_temporal.csv
```

### 4. Save Results for Comparison
Always save evaluation results:
```bash
python scripts/evaluate_temporal_gnn.py \
    --save-csv outputs/evaluations/temporal_gnn_v1.csv \
    --save-plots
```

---

## Example Workflow

```bash
# 1. Train model
python -m src.gnn.train_temporal_hetero \
    --model-type rgcn \
    --hidden-dim 128 \
    --num-layers 3 \
    --epochs 50 \
    --target-vars thermal,nuclear,solar,wind,unserved

# 2. Evaluate
python scripts/evaluate_temporal_gnn.py --save-plots --save-csv results.csv

# 3. View training curves
# Open notebooks/temporal_training_visualization.ipynb

# 4. Analyze results
python -c "import pandas as pd; df = pd.read_csv('results.csv'); print(df.describe())"
```

---

## Summary

✅ **Use for:**
- Comparing GNN vs MILP performance
- Tracking model improvements
- Ablation studies (temporal edges, architecture, etc.)
- Reporting results in papers/presentations

❌ **Limitations:**
- No feasibility decoder (higher violation rates)
- Zone-level aggregation only
- Simplified violation checking

🎯 **Key Takeaway:**
The evaluation provides the same core metrics as the original notebook, adapted for temporal supra-graphs. Results are comparable and you can use them to assess model quality and computational efficiency.
