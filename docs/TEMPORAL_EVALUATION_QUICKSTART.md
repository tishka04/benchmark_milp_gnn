# Temporal GNN Evaluation - Quick Reference

## Three Ways to Evaluate Your Temporal GNNs

### 1. **Jupyter Notebook** (Interactive, Visual) ✨ **RECOMMENDED**

```bash
# Open the notebook
jupyter notebook notebooks/temporal_comparison.ipynb
```

**Features:**
- ✅ Interactive exploration
- ✅ All metrics visualized
- ✅ Cost gap, MAE, violations, speedup plots
- ✅ Scatter plots and distribution analysis
- ✅ Export results to CSV
- ✅ Step-by-step execution with explanations

**Perfect for:**
- Analyzing results interactively
- Creating publication-quality plots
- Exploring trade-offs (cost vs feasibility)
- Comparing multiple model configurations

---

### 2. **Command-Line Script** (Fast, Automated)

```bash
python scripts/evaluate_temporal_gnn.py --save-plots
```

**Features:**
- ✅ One command evaluation
- ✅ Automatic plot generation
- ✅ Console summary statistics
- ✅ Save plots to disk
- ✅ CSV export

**Perfect for:**
- Quick evaluation after training
- Automated pipelines
- Batch evaluation of multiple models

---

### 3. **Python API** (Programmatic, Flexible)

```python
from src.gnn.temporal_evaluation import evaluate_test_set

results_df, summary = evaluate_test_set(
    run_dir='outputs/gnn_runs/temporal_hetero',
    test_split=0.8,
    device='cpu'
)

# Use results as needed
print(f"Cost gap: {summary['mean_cost_gap']:.2%}")
```

**Perfect for:**
- Custom analysis workflows
- Integration with other tools
- Batch comparisons
- Research automation

---

## Complete Workflow Example

```bash
# 1. Train model
python -m src.gnn.train_temporal_hetero \
    --hidden-dim 128 \
    --num-layers 3 \
    --epochs 50 \
    --target-vars thermal,nuclear,solar,wind,unserved

# 2. Visualize training curves
jupyter notebook notebooks/temporal_training_visualization.ipynb

# 3. Evaluate vs MILP (choose one)

## Option A: Interactive notebook (recommended)
jupyter notebook notebooks/temporal_comparison.ipynb

## Option B: Command-line script
python scripts/evaluate_temporal_gnn.py --save-plots --save-csv results.csv
```

---

## What You Get

### **Metrics** (Same as original `performance_comparison.ipynb`)

| Metric | Description | Good Performance |
|--------|-------------|------------------|
| **Cost Gap** | `(GNN_cost - MILP_obj) / MILP_obj` | < 10% |
| **Dispatch MAE** | Mean absolute error (MW) | < 50 MW |
| **Violation Rate** | Fraction with unmet demand | < 20% |
| **Speedup** | `MILP_time / GNN_time` | > 10x |

### **Visualizations**

1. **Cost Gap Bar Chart** - Per-scenario cost relative to MILP (color-coded)
2. **Dispatch MAE** - Prediction accuracy distribution
3. **Violation Rate** - Feasibility per scenario
4. **Speedup** - Runtime comparison (log scale)
5. **Cost vs Feasibility Scatter** - Trade-off analysis

### **Outputs**

- 📊 Interactive plots (in notebook)
- 💾 PNG images (if `--save-plots`)
- 📄 CSV file with all results
- 📈 Summary statistics table

---

## Quick Comparison: All Options

| Feature | Notebook | Script | Python API |
|---------|----------|--------|------------|
| **Ease of use** | ⭐⭐⭐ | ⭐⭐⭐ | ⭐⭐ |
| **Interactivity** | ✅ Yes | ❌ No | ✅ Flexible |
| **Visualizations** | ✅ Best | ✅ Auto | ⚠️ Manual |
| **Flexibility** | ⭐⭐⭐ | ⭐ | ⭐⭐⭐ |
| **Speed** | ⭐⭐ | ⭐⭐⭐ | ⭐⭐⭐ |
| **Automation** | ⭐ | ⭐⭐⭐ | ⭐⭐⭐ |

**Recommendation:** 
- **First time?** → Use notebook (`temporal_comparison.ipynb`)
- **Routine checks?** → Use script (`evaluate_temporal_gnn.py`)
- **Custom workflow?** → Use Python API (`temporal_evaluation.py`)

---

## Files You Have

```
notebooks/
├── temporal_comparison.ipynb           ← Main evaluation notebook ⭐
└── temporal_training_visualization.ipynb  ← Training curves

scripts/
└── evaluate_temporal_gnn.py           ← Command-line evaluator

src/gnn/
└── temporal_evaluation.py             ← Core evaluation module

docs/
├── TEMPORAL_GNN_EVALUATION.md         ← Full documentation
└── TEMPORAL_EVALUATION_QUICKSTART.md  ← This file
```

---

## Example Outputs

### Console Output (Script/Notebook)
```
====================================================================
Summary Statistics
====================================================================
Scenarios evaluated:        100
Mean cost gap:              3.24%      ← Lower is better
Median cost gap:            1.85%
Mean dispatch MAE:          28.45 MW   ← Lower is better
Mean violation rate:        12.34%     ← Lower is better
Mean speedup:               42.7x      ← Higher is better
====================================================================
```

### Saved Files
```
outputs/
├── evaluations/
│   └── temporal_gnn_results.csv       ← Detailed results
└── gnn_runs/temporal_hetero/
    ├── temporal_gnn_comparison.png    ← 4-panel comparison plot
    └── temporal_gnn_summary.png       ← Summary bar chart
```

---

## Troubleshooting

**"No checkpoint found"**
```bash
# Train a model first
python -m src.gnn.train_temporal_hetero --epochs 50
```

**"Violation rate is 0"**
```bash
# Include unserved in targets
python -m src.gnn.train_temporal_hetero --target-vars thermal,nuclear,solar,wind,unserved
```

**"Jupyter not installed"**
```bash
pip install jupyter matplotlib seaborn
```

**"Want to evaluate different model"**
```python
# In notebook, change RUN_DIR:
RUN_DIR = PROJECT_ROOT / 'outputs' / 'gnn_runs' / 'my_experiment'
```

---

## Tips for Best Results

1. **Always use test set** - Don't evaluate on training data
2. **Save results** - Export CSV for later comparison
3. **Track experiments** - Use different run directories
4. **Include unserved** - For accurate violation metrics
5. **Compare configurations** - Ablation studies reveal what helps

---

## Next Steps

After evaluation:

1. **Analyze trade-offs** - Cost vs feasibility scatter plot
2. **Identify failure cases** - High cost gap or violation scenarios
3. **Iterate** - Improve model based on insights
4. **Compare** - Train with different configurations
5. **Ablate** - Test importance of temporal edges

---

## Summary

You have **three powerful ways** to evaluate temporal GNNs against MILP:

🎯 **Recommended starter**: `notebooks/temporal_comparison.ipynb`  
⚡ **For automation**: `scripts/evaluate_temporal_gnn.py`  
🔧 **For custom work**: `src.gnn.temporal_evaluation` API

All use the **same metrics** as the original `performance_comparison.ipynb`!
