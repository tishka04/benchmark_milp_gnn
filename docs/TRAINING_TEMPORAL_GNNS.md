# Training GNNs on Temporal Heterogeneous Graphs

This guide shows you how to train Graph Neural Networks on your newly generated temporal heterogeneous multi-layer grid graphs.

## Quick Start

### 1. Train a Temporal R-GCN

```bash
python -m src.gnn.train_temporal_hetero \
    --data-dir outputs/temporal_graphs/supra \
    --model-type rgcn \
    --hidden-dim 128 \
    --num-layers 3 \
    --epochs 50 \
    --batch-size 4
```

This will:
- Load your temporal supra-graphs
- Train a Relational GCN that learns different message passing for each edge type
- Predict dispatch decisions (thermal, nuclear, solar, wind by default)
- Save the best model to `outputs/gnn_runs/temporal_hetero/best_model.pt`

### 2. Train with Separated Spatial-Temporal Model

```bash
python -m src.gnn.train_temporal_hetero \
    --data-dir outputs/temporal_graphs/supra \
    --model-type separated \
    --hidden-dim 128 \
    --num-layers 3 \
    --epochs 50
```

This uses a model that explicitly separates:
- **Spatial message passing** (edges 0-6): Within same timestep
- **Temporal message passing** (edges 7-9): Across timesteps

## Model Architectures

### Temporal R-GCN (Recommended)

**What it does:**
- Learns relation-specific transformations for each edge type
- Treats all 10 edge types (7 spatial + 3 temporal) uniformly
- Uses basis decomposition to reduce parameters

**When to use:**
- General-purpose baseline
- When you want the model to learn edge importance automatically
- Good for initial experiments

**Code:**
```python
from src.gnn.models.temporal_hetero_gnn import TemporalRGCN

model = TemporalRGCN(
    node_feature_dim=64,    # From your graphs
    hidden_dim=128,
    output_dim=4,           # Predict 4 variables
    num_edge_types=10,      # 7 spatial + 3 temporal
    num_layers=3,
    dropout=0.1,
)
```

### Separated Spatial-Temporal GNN

**What it does:**
- Explicitly separates spatial and temporal message passing
- Fuses information from both domains
- Better interpretability of spatial vs temporal effects

**When to use:**
- You want to understand spatial vs temporal contributions
- Physics-aware inductive bias is important
- Ablation studies (can disable temporal edges)

**Code:**
```python
from src.gnn.models.temporal_hetero_gnn import SeparatedTemporalGNN

model = SeparatedTemporalGNN(
    node_feature_dim=64,
    hidden_dim=128,
    output_dim=4,
    num_spatial_edge_types=7,   # Nation→Region, ... , Transmission
    num_temporal_edge_types=3,  # SOC, Ramp, DR
    num_layers=3,
)
```

## Understanding Your Data

Your temporal supra-graphs have this structure:

```python
import numpy as np

data = np.load("outputs/temporal_graphs/supra/scenario_00001.npz", allow_pickle=True)

# Metadata
meta = data["meta"].item()
N_base = meta["N_base"]  # e.g., 79 base nodes
T = meta["T"]            # e.g., 96 timesteps
# => Total nodes = 79 × 96 = 7,584

# Node features [N*T, F]
node_features = data["node_features"]  # [7584, 64]

# Edge structure [2, E]
edge_index = data["edge_index"]        # [2, 8053]
edge_types = data["edge_types"]        # [8053]

# Edge type distribution:
# 0-6: Spatial edges (replicated T times)
# 7: Temporal SOC edges (battery/storage continuity)
# 8: Temporal Ramp edges (generator ramping)
# 9: Temporal DR edges (demand response cooldown)
```

## Prediction Tasks

### Task 1: Dispatch Prediction (Default)

Predict generator outputs given grid topology and demand:

```bash
python -m src.gnn.train_temporal_hetero \
    --target-vars thermal,nuclear,solar,wind \
    --hidden-dim 128 \
    --epochs 50
```

**Targets:** Thermal, Nuclear, Solar, Wind generation
**Metric:** MSE on MWh predictions

### Task 2: Storage Optimization

Predict battery charging/discharging:

```bash
python -m src.gnn.train_temporal_hetero \
    --target-vars battery_charge,battery_discharge,pumped_charge,pumped_discharge \
    --hidden-dim 128 \
    --epochs 50
```

### Task 3: Full Dispatch + Storage

Predict all decision variables:

```bash
python -m src.gnn.train_temporal_hetero \
    --target-vars thermal,nuclear,solar,wind,hydro_release,battery_charge,battery_discharge \
    --hidden-dim 256 \
    --num-layers 4 \
    --epochs 100
```

## Advanced Usage

### Custom Training Script

```python
import torch
from pathlib import Path
from src.gnn.models.temporal_hetero_gnn import (
    TemporalRGCN,
    TemporalGraphDataset,
)
from torch_geometric.loader import DataLoader

# Load dataset
graph_files = list(Path("outputs/temporal_graphs/supra").glob("scenario_*.npz"))
dataset = TemporalGraphDataset(graph_files, target_indices=[0, 1, 2, 3])

# Create data loader
loader = DataLoader(dataset, batch_size=4, shuffle=True)

# Create model
model = TemporalRGCN(
    node_feature_dim=64,
    hidden_dim=128,
    output_dim=4,
    num_layers=3,
)

# Training loop
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
criterion = torch.nn.MSELoss()

for epoch in range(50):
    for batch in loader:
        # Forward pass
        pred = model(batch.x, batch.edge_index, batch.edge_type, batch.batch)
        
        # Loss
        loss = criterion(pred, batch.y)
        
        # Backward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
```

### Handling Node Types

Your graphs have 5 node types:
- 0: Nation (1 node)
- 1: Region (2 nodes)
- 2: Zone (11 nodes)
- 3: Asset (63 nodes) ← Most important!
- 4: Weather (2 nodes)

**Focus predictions on assets/zones:**

```python
# Filter predictions by node type
node_types = batch.node_type  # [N*T]

# Predict only on assets (type 3)
asset_mask = node_types == 3
pred_assets = pred[asset_mask]
target_assets = batch.y[asset_mask]

loss = criterion(pred_assets, target_assets)
```

### Temporal Masking

Predict future timesteps from past:

```python
# Data has nodes ordered: [t=0, t=1, ..., t=T-1]
N_base = batch.N_base[0].item()  # e.g., 79
T = batch.T[0].item()             # e.g., 96

# Use first 80% of timesteps for input, last 20% for prediction
split_t = int(0.8 * T)
split_idx = split_t * N_base

# Input: t=0 to t=76
input_mask = torch.arange(len(batch.x)) < split_idx

# Predict: t=77 to t=95
pred_mask = ~input_mask

# Forward with masked input
x_input = batch.x.clone()
x_input[pred_mask] = 0  # Zero out future features

pred = model(x_input, batch.edge_index, batch.edge_type, batch.batch)

# Compute loss only on future timesteps
loss = criterion(pred[pred_mask], batch.y[pred_mask])
```

## Understanding Edge Types

Your temporal graphs have these edge semantics:

### Spatial Edges (0-6, replicated per timestep)
- **0**: Nation → Region (hierarchical)
- **1**: Region → Zone (hierarchical)
- **2**: Zone → Asset (hierarchical)
- **3**: Weather → Zone (external influence)
- **4**: Weather → Asset (external influence for renewables)
- **5**: Zone ↔ Zone (transmission lines)
- **6**: Asset self-loops (temporal storage - legacy)

### Temporal Edges (7-9, across timesteps)
- **7**: SOC edges - Battery/storage continuity (t → t+1)
- **8**: Ramp edges - Generator ramping limits (t → t+1)
- **9**: DR edges - Demand response cooldown (t → t+k)

**Ablation study:**

```bash
# Only spatial information (no temporal edges)
python -m src.gnn.build_hetero_graph_dataset ... --temporal-edges ""

# Only storage continuity
python -m src.gnn.build_hetero_graph_dataset ... --temporal-edges soc

# All temporal edges (default)
python -m src.gnn.build_hetero_graph_dataset ... --temporal-edges soc,ramp,dr
```

## Evaluation Metrics

```python
import torch
from sklearn.metrics import mean_absolute_error, r2_score

# Predictions vs ground truth
pred = model(batch.x, batch.edge_index, batch.edge_type).detach().cpu().numpy()
target = batch.y.cpu().numpy()

# Per-variable metrics
for i, var_name in enumerate(["thermal", "nuclear", "solar", "wind"]):
    mae = mean_absolute_error(target[:, i], pred[:, i])
    r2 = r2_score(target[:, i], pred[:, i])
    print(f"{var_name}: MAE={mae:.2f} MWh, R²={r2:.4f}")

# Cost gap (compare to MILP optimal)
# This requires computing the objective function with predicted dispatch
```

## Hyperparameter Tuning

**Start with these:**
```bash
python -m src.gnn.train_temporal_hetero \
    --hidden-dim 128 \
    --num-layers 3 \
    --dropout 0.1 \
    --lr 0.001 \
    --batch-size 4 \
    --epochs 50
```

**If underfitting (high train/val loss):**
- Increase `--hidden-dim` (256, 512)
- Increase `--num-layers` (4, 5)
- Decrease `--dropout` (0.05, 0)
- Train longer `--epochs` (100, 200)

**If overfitting (low train loss, high val loss):**
- Decrease `--hidden-dim` (64, 32)
- Increase `--dropout` (0.2, 0.3)
- Add more data
- Reduce `--num-layers`

## Memory Optimization

Your supra-graphs can be large (7,584 nodes × 64 features). If you run out of memory:

1. **Reduce batch size:**
   ```bash
   --batch-size 1
   ```

2. **Reduce hidden dimension:**
   ```bash
   --hidden-dim 64
   ```

3. **Use gradient accumulation:**
   ```python
   accumulation_steps = 4
   for i, batch in enumerate(loader):
       loss = ...
       loss = loss / accumulation_steps
       loss.backward()
       
       if (i + 1) % accumulation_steps == 0:
           optimizer.step()
           optimizer.zero_grad()
   ```

4. **Use mixed precision (if on GPU):**
   ```python
   from torch.cuda.amp import autocast, GradScaler
   
   scaler = GradScaler()
   
   with autocast():
       pred = model(batch.x, batch.edge_index, batch.edge_type)
       loss = criterion(pred, batch.y)
   
   scaler.scale(loss).backward()
   scaler.step(optimizer)
   scaler.update()
   ```

## Next Steps

1. **Baseline:** Train R-GCN on full dispatch task
2. **Ablation:** Test without temporal edges (`--temporal-edges ""`)
3. **Comparison:** Compare with sequence mode + RNN
4. **Analysis:** Visualize learned edge importance
5. **Production:** Multi-step ahead prediction with temporal masking

## Troubleshooting

**"CUDA out of memory"**
- Reduce `--batch-size` to 1 or 2
- Reduce `--hidden-dim` to 64
- Use CPU with `--device cpu`

**"Training loss not decreasing"**
- Check learning rate (try `--lr 0.0001` or `--lr 0.01`)
- Verify labels are present in dataset
- Check for NaN values in data

**"Validation loss worse than baseline"**
- You may be overfitting - add dropout
- Try simpler model (`--num-layers 2`)
- Check train/val split is correct

**"Slow training"**
- Reduce `--num-layers`
- Reduce graph size by excluding some edge types
- Use GPU if available
- Set `num_workers=4` in DataLoader (Linux/Mac only)

## References

- R-GCN paper: [Modeling Relational Data with Graph Convolutional Networks](https://arxiv.org/abs/1703.06103)
- PyG documentation: [pytorch-geometric.readthedocs.io](https://pytorch-geometric.readthedocs.io)
- Your temporal graphs: `outputs/temporal_graphs/supra/dataset_index.json`
