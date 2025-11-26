# Notebook Compatibility Guide

This document explains how the temporal GNN training system integrates with existing analysis notebooks.

## Overview

You have two separate GNN training pipelines:

### 1. **Static Hetero Graph Training** (Original)
- Dataset: `GraphTemporalDataset` (zone-level, flat graphs)
- Models: Custom architectures in `src.gnn.models`
- Training: `src.gnn.training` module
- Notebooks: ✅ `performance_comparison.ipynb`, ✅ `training_visualization.ipynb`

### 2. **Temporal Supra-Graph Training** (New)
- Dataset: `TemporalGraphDataset` (time-expanded graphs with temporal edges)
- Models: `TemporalRGCN`, `SeparatedTemporalGNN` in `src.gnn.models.temporal_hetero_gnn`
- Training: `src.gnn.train_temporal_hetero`
- Notebooks: ✅ `temporal_training_visualization.ipynb` (new)

---

## Notebook Compatibility Matrix

| Notebook | Static Graphs | Temporal Graphs | Notes |
|----------|--------------|-----------------|-------|
| `training_visualization.ipynb` | ✅ Full support | ⚠️ Partial | Original notebook; expects old format |
| `performance_comparison.ipynb` | ✅ Full support | ❌ Not compatible | Requires old dataset loader |
| **`temporal_training_visualization.ipynb`** | ❌ | ✅ Full support | **New notebook for temporal GNNs** |

---

## Using the Temporal Training Visualization

### Step 1: Train Your Temporal GNN

```bash
python -m src.gnn.train_temporal_hetero \
    --data-dir outputs/temporal_graphs/supra \
    --model-type rgcn \
    --hidden-dim 128 \
    --num-layers 3 \
    --epochs 50
```

This automatically creates:
- ✅ `outputs/gnn_runs/temporal_hetero/best_model.pt` - Best model checkpoint
- ✅ `outputs/gnn_runs/temporal_hetero/training_history.json` - Epoch-level metrics

### Step 2: Visualize Training

Open and run:
```
notebooks/temporal_training_visualization.ipynb
```

You'll see:
- **Training & Validation Loss Curves** - Track convergence
- **Overfitting Analysis** - Train/val gap visualization
- **Learning Rate Schedule** - How LR changes over epochs
- **Model Configuration** - Hyperparameters used

---

## What Gets Logged

The temporal training script now logs:

```json
[
  {
    "epoch": 1,
    "train_loss": 1754.78,
    "val_loss": 1823.45,
    "learning_rate": 0.001,
    "is_best": true
  },
  {
    "epoch": 2,
    "train_loss": 1187.93,
    "val_loss": 1234.56,
    "learning_rate": 0.001,
    "is_best": true
  },
  ...
]
```

Saved to: `outputs/gnn_runs/temporal_hetero/training_history.json`

---

## Why performance_comparison.ipynb Doesn't Work (Yet)

The `performance_comparison.ipynb` notebook requires:

1. **Specific dataset format**: `GraphTemporalDataset` with zone-level flat graphs
2. **Decoder infrastructure**: Post-processing to ensure feasibility
3. **MILP comparison**: Cost gap, violation rate, speedup computation
4. **Evaluation loop**: Inference on test set with dispatch reconstruction

Your temporal GNNs use:
- PyTorch Geometric `Data` objects (different structure)
- No decoder/post-processing (raw predictions)
- No MILP comparison infrastructure

### Future Work: Unified Evaluation

To make `performance_comparison.ipynb` work with temporal GNNs, we'd need:

1. **Adapter layer**: Convert temporal predictions to zone-level dispatch
2. **Decoder**: Project node-level predictions to valid dispatch solutions
3. **MILP comparison**: Compute cost gap against MILP baseline
4. **Violation metrics**: Check demand/capacity constraints

This is substantial work and probably not needed if you're just doing research on temporal GNN architectures.

---

## Recommended Workflow

### For Temporal GNN Development

1. **Train models:**
   ```bash
   python -m src.gnn.train_temporal_hetero --epochs 50
   ```

2. **Visualize training:**
   - Open `notebooks/temporal_training_visualization.ipynb`
   - Check convergence, overfitting, learning rate

3. **Compare models:**
   - Train multiple configurations (different hidden_dim, num_layers, etc.)
   - Compare final val losses
   - Check which temporal edges help most (ablation with `--temporal-edges`)

4. **Manual evaluation:**
   - Load best model
   - Make predictions on test graphs
   - Compute MAE/RMSE per variable
   - Visualize predictions vs ground truth

### For Production Deployment

If you need full MILP comparison:

1. Train temporal GNN
2. Create evaluation script that:
   - Loads temporal graphs
   - Makes predictions
   - Aggregates to zone-level dispatch
   - Applies feasibility corrections
   - Computes cost gap vs MILP
3. Create custom performance notebook

---

## Example: Manual Evaluation

```python
import torch
import numpy as np
from pathlib import Path
from src.gnn.models.temporal_hetero_gnn import TemporalRGCN

# Load model
checkpoint = torch.load('outputs/gnn_runs/temporal_hetero/best_model.pt')
args = checkpoint['args']

model = TemporalRGCN(
    node_feature_dim=args['node_feature_dim'],  # From checkpoint
    hidden_dim=args['hidden_dim'],
    output_dim=args['output_dim'],
    num_layers=args['num_layers'],
)
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

# Load a test graph
test_graph = np.load('outputs/temporal_graphs/supra/scenario_00100.npz', allow_pickle=True)
x = torch.from_numpy(test_graph['node_features']).float()
edge_index = torch.from_numpy(test_graph['edge_index']).long()
edge_type = torch.from_numpy(test_graph['edge_types']).long()
y_true = torch.from_numpy(test_graph['node_labels']).float()

# Predict
with torch.no_grad():
    y_pred = model(x, edge_index, edge_type)

# Evaluate (zone nodes only)
node_types = torch.from_numpy(test_graph['node_types']).long()
zone_mask = node_types == 2

mae = torch.abs(y_pred[zone_mask] - y_true[zone_mask]).mean()
print(f"MAE on zones: {mae:.2f} MW")
```

---

## Summary

✅ **Works Out of the Box:**
- Training temporal GNNs with logging
- Visualizing training curves in new notebook
- Manual evaluation and comparison

❌ **Not Yet Implemented:**
- MILP performance comparison for temporal GNNs
- Feasibility checking and constraint violation rates
- Integration with old `performance_comparison.ipynb`

🎯 **Recommended:**
- Use `temporal_training_visualization.ipynb` for development
- Create custom evaluation scripts for production
- Focus on research/architecture rather than full deployment

---

## Quick Start Commands

```bash
# 1. Train temporal GNN
python -m src.gnn.train_temporal_hetero --epochs 50

# 2. Visualize training
# Open: notebooks/temporal_training_visualization.ipynb

# 3. Compare models
python -m src.gnn.train_temporal_hetero --model-type separated --epochs 50
# Compare training curves in notebook

# 4. Ablation study
python -m src.gnn.train_temporal_hetero --temporal-edges soc --epochs 50
python -m src.gnn.train_temporal_hetero --temporal-edges soc,ramp --epochs 50
python -m src.gnn.train_temporal_hetero --temporal-edges soc,ramp,dr --epochs 50
# Check which edges improve performance
```

---

## Need Full Integration?

If you absolutely need `performance_comparison.ipynb` to work with temporal GNNs, let me know and I can build:

1. Evaluation adapter
2. Dispatch reconstruction
3. MILP comparison metrics
4. Updated performance notebook

This is ~2-3 hours of work but may not be necessary for research purposes.
