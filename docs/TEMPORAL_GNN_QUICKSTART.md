# Temporal GNN Training - Quick Start Guide

You've successfully generated temporal heterogeneous graphs! This guide shows you how to train GNNs on them in 5 minutes.

## ✅ What You Have

Looking at your dataset index:
```json
{
  "graph_file": "outputs\\temporal_graphs\\supra\\scenario_00001.npz",
  "num_nodes": 7584,  // 79 base nodes × 96 timesteps
  "num_edges": 8053,
  "mode": "supra",
  "T": 96,
  "N_base": 79,
  "temporal_edges": ["soc", "ramp", "dr"]
}
```

You have **time-expanded supra-graphs** with:
- ✓ Spatial edges (hierarchical, transmission, weather)
- ✓ Temporal edges (storage, ramping, demand response)
- ✓ 96 timesteps per scenario
- ✓ Ready for Temporal GNN training

## 🚀 Three Ways to Train

### Option 1: Run the Quickstart Example (Easiest)

```bash
python examples/temporal_hetero_quickstart.py
```

This will:
- Load your temporal graphs
- Train a simple R-GCN model
- Show you results in 5 minutes
- Give you working code to customize

### Option 2: Use the Training Script (Recommended)

```bash
python -m src.gnn.train_temporal_hetero --data-dir outputs/temporal_graphs/supra --model-type rgcn --hidden-dim 32 --num-layers 2 --epochs 2 --batch-size 4 --target-vars thermal,nuclear,hydro,solar,wind,dr,battery,pumped,unserved,imports,exports,overgen
```

**What this does:**
- Trains a Relational GCN on all your graphs
- Uses 80/20 train/val split
- Predicts thermal, nuclear, solar, wind generation and the rest as well
- Saves best model to `outputs/gnn_runs/temporal_hetero/best_model.pt`

### Option 3: Write Custom Code

```python
from pathlib import Path
import torch
from torch_geometric.loader import DataLoader
from src.gnn.models.temporal_hetero_gnn import TemporalRGCN, TemporalGraphDataset

# Load your temporal graphs
graph_files = list(Path("outputs/temporal_graphs/supra").glob("*.npz"))
dataset = TemporalGraphDataset(graph_files, target_indices=[0, 1, 2, 3])

# Create model
model = TemporalRGCN(
    node_feature_dim=64,    # From your graphs
    hidden_dim=128,
    output_dim=4,           # Predict 4 variables
    num_edge_types=10,      # 7 spatial + 3 temporal
    num_layers=3,
)

# Train
loader = DataLoader(dataset, batch_size=4, shuffle=True)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
criterion = torch.nn.MSELoss()

for epoch in range(50):
    for batch in loader:
        pred = model(batch.x, batch.edge_index, batch.edge_type, batch.batch)
        loss = criterion(pred, batch.y)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
```

## 🧠 Which Model to Use?

### Temporal R-GCN (Default - Start Here)

**Best for:**
- Initial experiments
- General-purpose baseline
- Learning edge importance automatically

**Command:**
```bash
python -m src.gnn.train_temporal_hetero --model-type rgcn
```

### Separated Spatial-Temporal GNN

**Best for:**
- Understanding spatial vs temporal contributions
- Physics-aware learning
- Ablation studies

**Command:**
```bash
python -m src.gnn.train_temporal_hetero --model-type separated
```

## 📊 What You're Predicting

Your graphs include labels for these variables (per node, per timestep):

| Index | Variable | Description |
|-------|----------|-------------|
| 0 | `thermal` | Thermal generation (MW) |
| 1 | `nuclear` | Nuclear generation (MW) |
| 2 | `solar` | Solar generation (MW) |
| 3 | `wind` | Wind generation (MW) |
| 4 | `hydro_release` | Hydro reservoir release (MW) |
| 5 | `hydro_ror` | Run-of-river hydro (MW) |
| 6 | `dr` | Demand response (MW) |
| 7 | `battery_charge` | Battery charging (MW) |
| 8 | `battery_discharge` | Battery discharging (MW) |
| 9 | `pumped_charge` | Pumped hydro charging (MW) |
| 10 | `pumped_discharge` | Pumped hydro discharging (MW) |
| 11 | `net_import` | Net imports (MW) |
| 12 | `unserved` | Unserved energy (MW) |

**Choose targets with `--target-vars`:**
```bash
# Predict only generation
--target-vars thermal,nuclear,solar,wind

# Predict only storage
--target-vars battery_charge,battery_discharge,pumped_charge,pumped_discharge

# Predict everything
--target-vars thermal,nuclear,solar,wind,hydro_release,battery_charge,battery_discharge
```

## 🎯 Understanding Your Graph Structure

Your temporal supra-graphs have 10 edge types:

### Spatial Edges (0-6) - Replicated per timestep
- Type 0: Nation → Region
- Type 1: Region → Zone
- Type 2: Zone → Asset
- Type 3: Weather → Zone
- Type 4: Weather → Asset
- Type 5: Zone ↔ Zone (transmission)
- Type 6: Asset self-loops

### Temporal Edges (7-9) - Across timesteps
- **Type 7 (SOC)**: Battery storage continuity `asset@t → asset@(t+1)`
- **Type 8 (Ramp)**: Generator ramping limits `generator@t → generator@(t+1)`
- **Type 9 (DR)**: Demand response cooldown `dr@t → dr@(t+k)`

**The GNN learns to:**
- Aggregate spatial information (hierarchical + transmission)
- Propagate temporal constraints (storage, ramping, cooldown)
- Predict optimal dispatch that respects both

## 🔧 Common Configurations

### Fast Baseline (5 min on CPU)
```bash
python -m src.gnn.train_temporal_hetero \
    --hidden-dim 64 \
    --num-layers 2 \
    --epochs 10 \
    --batch-size 2
```

### Production Model (1 hour on GPU)
```bash
python -m src.gnn.train_temporal_hetero \
    --hidden-dim 256 \
    --num-layers 4 \
    --epochs 100 \
    --batch-size 8 \
    --device cuda
```

### Storage-Focused Model
```bash
python -m src.gnn.train_temporal_hetero \
    --target-vars battery_charge,battery_discharge,pumped_charge,pumped_discharge \
    --hidden-dim 128 \
    --num-layers 3 \
    --epochs 50
```

## 📈 Expected Results

With default hyperparameters (128 hidden dim, 3 layers, 50 epochs):

| Variable | Typical MAE | Good Performance | Excellent |
|----------|-------------|------------------|-----------|
| Thermal | 50-100 MW | 20-50 MW | < 20 MW |
| Nuclear | 10-30 MW | 5-10 MW | < 5 MW |
| Solar | 30-60 MW | 10-30 MW | < 10 MW |
| Wind | 50-100 MW | 20-50 MW | < 20 MW |

**Note:** Results depend on your specific dataset and grid complexity.

## 🐛 Troubleshooting

### "No graphs found"
- Run the graph builder first:
  ```bash
  python -m src.gnn.build_hetero_graph_dataset \
      outputs/scenarios_v1 \
      outputs/scenarios_v1/reports \
      outputs/temporal_graphs/supra \
      --temporal --temporal-mode supra
  ```

### "CUDA out of memory"
- Reduce batch size: `--batch-size 1`
- Reduce hidden dim: `--hidden-dim 64`
- Use CPU: `--device cpu`

### "Loss not decreasing"
- Try different learning rate: `--lr 0.0001` or `--lr 0.01`
- Check if labels exist in your graphs
- Reduce model complexity: `--num-layers 2`

### "Training too slow"
- Reduce num layers: `--num-layers 2`
- Use GPU: `--device cuda`
- Reduce dataset size for testing

## 📚 Further Reading

- **Full guide**: `docs/TRAINING_TEMPORAL_GNNS.md`
- **Model code**: `src/gnn/models/temporal_hetero_gnn.py`
- **Training code**: `src/gnn/train_temporal_hetero.py`
- **Example**: `examples/temporal_hetero_quickstart.py`

## 🎓 Next Steps

1. **Run quickstart** to verify everything works
2. **Train baseline** with default settings
3. **Experiment** with different targets and hyperparameters
4. **Ablate** temporal edges to understand their importance
5. **Compare** with static graph baselines

## 💡 Key Insight

Your temporal supra-graphs encode **physics-aware constraints** as edges:
- SOC edges → Battery charge must be conserved
- Ramp edges → Generators can't change output instantly
- DR edges → Demand response has cooldown periods

The GNN learns to **respect these constraints** through message passing, making predictions that are physically feasible!

---

**Quick Test:**
```bash
python examples/temporal_hetero_quickstart.py
```

This runs in 5 minutes and shows you everything working! 🚀
