# Temporal Heterogeneous Graph - Usage Examples

Quick start guide with practical examples for building temporal graphs.

## Quick Start

### 1. Build Supra-Graph (Recommended for TGNN)

```bash
python -m src.gnn.build_hetero_graph_dataset outputs/scenarios_v1 outputs/scenarios_v1/reports outputs/temporal_graphs/supra --temporal --temporal-mode supra
```

**What you get:**
- One `.npz` file per scenario (e.g., `scenario_001.npz`)
- Time-expanded graph with `N_base × T` nodes
- Temporal edges connecting storage, generators, and DR across time

### 2. Build Sequence (For RNN/Transformer)

```bash
python -m src.gnn.build_hetero_graph_dataset \
    outputs/scenarios_v1 \
    outputs/scenarios_v1 \
    outputs/temporal_graphs/sequence \
    --temporal \
    --temporal-mode sequence
```

**What you get:**
- Multiple `.npz` files per scenario (e.g., `scenario_001_t0000.npz`, `scenario_001_t0001.npz`, ...)
- Each file is a snapshot at one time step
- Suitable for sequence models

## Advanced Examples

### Example 1: Supra-Graph with Custom Temporal Edges

Only include storage continuity edges (no ramp or DR):

```bash
python -m src.gnn.build_hetero_graph_dataset \
    outputs/scenarios_v1 \
    outputs/scenarios_v1 \
    outputs/temporal_graphs/supra_soc_only \
    --temporal \
    --temporal-mode supra \
    --temporal-edges soc
```

### Example 2: Hour-of-Day Encoding

Use cyclic hour-of-day encoding for better periodic pattern capture:

```bash
python -m src.gnn.build_hetero_graph_dataset \
    outputs/scenarios_v1 \
    outputs/scenarios_v1 \
    outputs/temporal_graphs/supra_cyclic \
    --temporal \
    --temporal-mode supra \
    --time-enc cyclic-hod
```

### Example 3: Sliding Window Sequences

Create overlapping 24-hour windows with 12-hour stride:

```bash
python -m src.gnn.build_hetero_graph_dataset \
    outputs/scenarios_v1 \
    outputs/scenarios_v1 \
    outputs/temporal_graphs/sequence_window \
    --temporal \
    --temporal-mode sequence \
    --window 48 \
    --stride 24
```

Assuming 30-min intervals: `--window 48` = 24 hours, `--stride 24` = 12 hours

### Example 4: Multi-Step Ahead Prediction

Shift labels by 6 steps ahead (3 hours with 30-min intervals):

```bash
python -m src.gnn.build_hetero_graph_dataset \
    outputs/scenarios_v1 \
    outputs/scenarios_v1 \
    outputs/temporal_graphs/supra_horizon \
    --temporal \
    --temporal-mode supra \
    --target-horizon 6
```

### Example 5: Resume Interrupted Build

Skip already-generated graphs to resume:

```bash
python -m src.gnn.build_hetero_graph_dataset \
    outputs/scenarios_v1 \
    outputs/scenarios_v1 \
    outputs/temporal_graphs/supra \
    --temporal \
    --temporal-mode supra \
    --resume
```

## Using Helper Scripts

### PowerShell (Windows)

```powershell
# Build supra-graph
.\scripts\build_temporal_graphs.ps1 -Mode supra

# Build sequence
.\scripts\build_temporal_graphs.ps1 -Mode sequence
```

### Python (Cross-platform)

```bash
# Build supra-graph
python scripts/build_temporal_graphs.py --mode supra

# Build both modes
python scripts/build_temporal_graphs.py --mode both

# With resume
python scripts/build_temporal_graphs.py --mode supra --resume
```

## Loading Temporal Graphs in Python

### Load Supra-Graph

```python
import numpy as np
from pathlib import Path

# Load a supra-graph
graph_path = Path("outputs/temporal_graphs/supra/scenario_001.npz")
data = np.load(graph_path, allow_pickle=True)

# Access temporal structure
N_base = data["meta"].item()["N_base"]  # Base number of nodes
T = data["meta"].item()["T"]            # Number of timesteps
N_total = len(data["node_types"])       # Should equal N_base * T

print(f"Graph structure:")
print(f"  Base nodes: {N_base}")
print(f"  Timesteps: {T}")
print(f"  Total nodes: {N_total}")
print(f"  Total edges: {data['edge_index'].shape[1]}")

# Node features include time encoding
node_features = data["node_features"]  # [N*T, F+Ft]
print(f"  Feature dim: {node_features.shape[1]}")

# Edge types
edge_types = data["edge_types"]
unique_types, counts = np.unique(edge_types, return_counts=True)
print(f"\nEdge type distribution:")
for etype, count in zip(unique_types, counts):
    print(f"  Type {etype}: {count} edges")
```

### Load Sequence Snapshot

```python
import numpy as np
from pathlib import Path

# Load a single snapshot from sequence mode
snapshot_path = Path("outputs/temporal_graphs/sequence/scenario_001_t0010.npz")
data = np.load(snapshot_path, allow_pickle=True)

# Access snapshot info
time_step = data.get("time_step", 0)
time_index = data.get("time_index", "unknown")

print(f"Snapshot at t={time_step} ({time_index}):")
print(f"  Nodes: {len(data['node_types'])}")
print(f"  Edges: {data['edge_index'].shape[1]}")
```

### Iterate Over Sequence

```python
from pathlib import Path
import numpy as np

sequence_dir = Path("outputs/temporal_graphs/sequence")

# Get all snapshots for scenario_001
snapshots = sorted(sequence_dir.glob("scenario_001_t*.npz"))

print(f"Found {len(snapshots)} snapshots for scenario_001")

for snap_path in snapshots[:5]:  # First 5 snapshots
    data = np.load(snap_path, allow_pickle=True)
    t = data.get("time_step", 0)
    print(f"  t={t:04d}: {len(data['node_types'])} nodes, {data['edge_index'].shape[1]} edges")
```

## Inspecting Temporal Graphs

You can use the existing inspection tool (it will work with temporal graphs):

```bash
python -m src.gnn.inspect_hetero_graph \
    outputs/temporal_graphs/supra/scenario_001.npz
```

## Training with Temporal Graphs

### With PyTorch Geometric

```python
import torch
import numpy as np
from torch_geometric.data import Data

def load_temporal_supra_graph(npz_path):
    """Load temporal supra-graph into PyG Data object."""
    data_dict = np.load(npz_path, allow_pickle=True)
    
    # Extract components
    x = torch.from_numpy(data_dict["node_features"]).float()
    edge_index = torch.from_numpy(data_dict["edge_index"]).long()
    edge_attr = torch.from_numpy(data_dict["edge_features"]).float()
    edge_type = torch.from_numpy(data_dict["edge_types"]).long()
    
    # Optional: node types
    node_type = torch.from_numpy(data_dict["node_types"]).long()
    
    # Create PyG Data object
    data = Data(
        x=x,
        edge_index=edge_index,
        edge_attr=edge_attr,
        edge_type=edge_type,
        node_type=node_type,
    )
    
    # Store metadata
    meta = data_dict["meta"].item()
    data.N_base = meta["N_base"]
    data.T = meta["T"]
    data.temporal_edges = meta["temporal_edges"]
    
    return data

# Usage
from pathlib import Path
graph_path = Path("outputs/temporal_graphs/supra/scenario_001.npz")
data = load_temporal_supra_graph(graph_path)
print(data)
```

### Temporal Message Passing

```python
import torch
import torch.nn as nn
from torch_geometric.nn import MessagePassing

class TemporalGNNLayer(MessagePassing):
    """Example temporal GNN layer that distinguishes temporal edges."""
    
    def __init__(self, in_channels, out_channels):
        super().__init__(aggr='add')
        self.lin = nn.Linear(in_channels, out_channels)
        self.temporal_lin = nn.Linear(in_channels, out_channels)
    
    def forward(self, x, edge_index, edge_type):
        # Separate spatial and temporal edges
        spatial_mask = edge_type < 7  # Edge types 0-6 are spatial
        temporal_mask = edge_type >= 7  # Edge types 7+ are temporal
        
        # Aggregate from spatial edges
        out_spatial = self.propagate(
            edge_index[:, spatial_mask],
            x=x
        )
        
        # Aggregate from temporal edges
        out_temporal = self.propagate(
            edge_index[:, temporal_mask],
            x=x,
            is_temporal=True
        )
        
        return self.lin(out_spatial) + self.temporal_lin(out_temporal)
    
    def message(self, x_j, is_temporal=False):
        return x_j

# Usage
layer = TemporalGNNLayer(64, 64)
out = layer(data.x, data.edge_index, data.edge_type)
```

## Comparison: Sequence vs Supra

| Aspect | Sequence Mode | Supra-Graph Mode |
|--------|---------------|------------------|
| **Output** | Multiple files per scenario | Single file per scenario |
| **Memory** | Low (one snapshot at a time) | High (all timesteps) |
| **Model** | RNN, LSTM, Transformer | Temporal GNN (TGNN) |
| **Temporal edges** | None (handled by sequential model) | Explicit edges across time |
| **Storage** | More files, same total size | Fewer files, same total size |
| **Use case** | Sequential prediction | Graph-based temporal reasoning |

## Best Practices

1. **Start with supra-graph** if you want to use temporal GNNs
2. **Use sequence mode** if you're training RNNs/Transformers over graph embeddings
3. **Include all temporal edges** initially, then ablate to find which are important
4. **Use sinusoidal encoding** for general-purpose positional info
5. **Use cyclic-hod encoding** if you have strong diurnal patterns
6. **Resume builds** when experimenting to avoid re-processing

## Troubleshooting

### "No time steps found in report"

Your reports don't have detailed time-series data. Make sure scenarios are solved with:

```bash
python -m src.milp.main --save-json --output-dir outputs/scenarios_v1
```

### Memory issues with supra-graph

The supra-graph can be large (N×T nodes). Solutions:
- Reduce temporal edge types: `--temporal-edges soc`
- Use sequence mode instead
- Process fewer scenarios at once
- Use a machine with more RAM

### Too many files with sequence mode

Each scenario generates T files (e.g., 48 files for 24h with 30-min intervals). Solutions:
- Use sliding windows: `--window 24 --stride 24`
- Use supra-graph mode instead
- Store on fast SSD

## Next Steps

1. **Explore**: Inspect generated graphs with `inspect_hetero_graph.py`
2. **Train**: Build a temporal GNN model for dispatch prediction
3. **Evaluate**: Compare with static graph baselines
4. **Ablate**: Test different temporal edge combinations

See `TEMPORAL_HETERO_GRAPHS.md` for full documentation.
