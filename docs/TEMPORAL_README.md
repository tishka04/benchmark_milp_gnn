# Temporal Heterogeneous Multi-Layer Grid Graphs - Implementation Summary

This document summarizes the temporal heterogeneous graph feature implementation for the MILP benchmark project.

## What Was Implemented

A complete temporal graph generation system with two modes:

### 1. **Sequence Mode**
- Generates a list of graph snapshots, one per time step
- Each snapshot is an independent heterogeneous graph augmented with time encodings
- Lightweight memory footprint
- Ideal for RNN/LSTM/Transformer training on graph embeddings
- Output: `scenario_001_t0000.npz`, `scenario_001_t0001.npz`, ..., `scenario_001_t0047.npz`

### 2. **Supra-Graph Mode** (Time-Expanded Graph)
- Creates a single large graph where nodes are `(node_id, timestep)` pairs
- Spatial edges are replicated across time
- **Temporal edges** explicitly encode physical constraints:
  - **SOC edges**: Battery/storage state-of-charge continuity
  - **Ramp edges**: Generator ramping constraints  
  - **DR edges**: Demand response cooldown periods
- Physics-aware for Temporal GNN (TGNN) architectures
- Output: `scenario_001.npz` (single file with `N × T` nodes)

## Files Modified/Created

### Core Implementation
- **`src/gnn/hetero_graph_dataset.py`**: Added temporal graph builder functions
  - `build_hetero_temporal_record()`: Main temporal graph builder
  - Time encoding functions (sinusoidal & cyclic hour-of-day)
  - Temporal edge builders (SOC, ramp, DR)
  - Helper functions for node/edge replication across time

- **`src/gnn/build_hetero_graph_dataset.py`**: Updated batch converter
  - Added CLI flags for temporal options
  - Integrated temporal builder into batch pipeline
  - Updated index generation with temporal metadata

### Documentation
- **`TEMPORAL_HETERO_GRAPHS.md`**: Full technical documentation
- **`TEMPORAL_USAGE_EXAMPLES.md`**: Practical usage guide with code examples
- **`TEMPORAL_README.md`**: This summary document

### Helper Scripts
- **`scripts/build_temporal_graphs.py`**: Python helper script
- **`scripts/build_temporal_graphs.ps1`**: PowerShell helper script
- **`scripts/test_temporal_build.py`**: Testing script for actual data
- **`scripts/verify_temporal_install.py`**: Installation verification (✓ all tests pass)

### Output Directories
```
outputs/temporal_graphs/
├── sequence/     # Sequence mode outputs
└── supra/        # Supra-graph mode outputs
```

## Command-Line Interface

### Basic Usage

```bash
# Generate supra-graph (time-expanded)
python -m src.gnn.build_hetero_graph_dataset \
    outputs/scenarios_v1 \
    outputs/scenarios_v1 \
    outputs/temporal_graphs/supra \
    --temporal \
    --temporal-mode supra

# Generate sequence (snapshots)
python -m src.gnn.build_hetero_graph_dataset \
    outputs/scenarios_v1 \
    outputs/scenarios_v1 \
    outputs/temporal_graphs/sequence \
    --temporal \
    --temporal-mode sequence
```

### New CLI Arguments

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--temporal` | flag | False | Enable temporal graph building |
| `--temporal-mode` | choice | `supra` | Mode: `sequence` or `supra` |
| `--window` | int | None | Sequence window size (steps) |
| `--stride` | int | 1 | Window stride for sequence mode |
| `--temporal-edges` | str | `soc,ramp,dr` | Comma-separated edge types |
| `--time-enc` | choice | `sinusoidal` | Time encoding: `sinusoidal` or `cyclic-hod` |
| `--target-horizon` | int | 0 | Prediction horizon (future steps) |

## Data Schema

### Supra-Graph Record Structure

```python
{
    "graph_type": "hetero_multi_layer_temporal_supra",
    "node_features": np.ndarray,    # [N*T, F+Ft] features + time encoding
    "node_types": np.ndarray,       # [N*T] node type IDs
    "node_ids": List[str],          # ["Zone1#t=0", "Zone1#t=1", ...]
    "time_index": List[str],        # ["t=0", "t=1", ..., "t=T-1"]
    "time_steps": List[int],        # [0, 1, 2, ...]
    "edge_index": np.ndarray,       # [2, E] spatial + temporal edges
    "edge_types": np.ndarray,       # [E] edge type IDs (0-9)
    "edge_features": np.ndarray,    # [E, Fe] edge attributes
    "meta": {
        "N_base": int,              # Base number of nodes
        "T": int,                   # Number of timesteps
        "temporal_edges": List[str],
        "time_encoding": str,
        "schema_version": "2.0-temporal",
    },
}
```

### Edge Type Enumeration

| Type ID | Name | Description |
|---------|------|-------------|
| 0 | Nation→Region | Hierarchical containment |
| 1 | Region→Zone | Hierarchical containment |
| 2 | Zone→Asset | Hierarchical containment |
| 3 | Weather→Zone | Weather influence |
| 4 | Weather→Asset | Weather influence on RES |
| 5 | Transmission | Zone-to-zone power lines |
| 6 | Temporal Storage | Static storage self-loops |
| **7** | **Temporal SOC** | **Storage continuity across time** |
| **8** | **Temporal Ramp** | **Generator ramping constraints** |
| **9** | **Temporal DR** | **Demand response cooldown** |

## Quick Start Guide

### Step 1: Verify Installation

```bash
python scripts/verify_temporal_install.py
```

Expected output: `✓ All verification tests passed!`

### Step 2: Test with Sample Data (if available)

```bash
python scripts/test_temporal_build.py
```

This tests building graphs with one scenario.

### Step 3: Generate Full Dataset

Using helper script:
```bash
python scripts/build_temporal_graphs.py --mode supra
```

Or directly:
```bash
python -m src.gnn.build_hetero_graph_dataset \
    outputs/scenarios_v1 \
    outputs/scenarios_v1 \
    outputs/temporal_graphs/supra \
    --temporal \
    --temporal-mode supra \
    --temporal-edges soc,ramp,dr \
    --time-enc sinusoidal
```

## Example: Loading and Using Temporal Graphs

```python
import numpy as np
import torch
from torch_geometric.data import Data

# Load supra-graph
data_dict = np.load("outputs/temporal_graphs/supra/scenario_001.npz", allow_pickle=True)

# Extract components
x = torch.from_numpy(data_dict["node_features"]).float()
edge_index = torch.from_numpy(data_dict["edge_index"]).long()
edge_type = torch.from_numpy(data_dict["edge_types"]).long()

# Metadata
meta = data_dict["meta"].item()
N_base = meta["N_base"]  # Base nodes (before time expansion)
T = meta["T"]            # Time steps

print(f"Temporal graph: {N_base} base nodes × {T} timesteps = {N_base * T} total nodes")
print(f"Edges: {edge_index.shape[1]}")

# Create PyG data object
graph = Data(x=x, edge_index=edge_index, edge_type=edge_type)
```

## Design Choices & Rationale

### Why Two Modes?

1. **Sequence mode**: For models that process time sequentially (RNN/Transformer)
2. **Supra-graph mode**: For models that leverage spatial-temporal message passing (TGNN)

Both are valid approaches; the choice depends on your model architecture.

### Time Encoding Methods

1. **Sinusoidal** (default): Transformer-style positional encoding
   - General purpose, works for any sequence length
   - Captures relative temporal positions

2. **Cyclic Hour-of-Day**: Periodic features for diurnal patterns
   - Better for data with strong daily/weekly cycles
   - Requires time-of-day information

### Temporal Edge Types

- **SOC**: Essential for storage physics (battery charge conservation)
- **Ramp**: Generator startup/shutdown constraints
- **DR**: Demand response activation cooldowns

You can select subsets via `--temporal-edges` flag for ablation studies.

## Performance & Scalability

### Memory Usage

| Mode | Memory Complexity | Example (N=1000, T=48, F=20) |
|------|-------------------|------------------------------|
| Sequence | O(N × F) per snapshot | ~80 KB per snapshot |
| Supra | O(N × T × F) | ~4 MB per scenario |

Supra-graphs are memory-intensive but enable powerful TGNN architectures.

### Recommended Practices

- **Start with supra-graph** if training TGNNs
- **Use `--resume`** to avoid re-processing when experimenting
- **Reduce temporal edges** (`--temporal-edges soc`) for memory savings
- **Use sequence mode** for very long time horizons (T > 100)

## Integration with Existing Code

The temporal graphs maintain **backward compatibility**:
- Flat graph compatibility fields are preserved (`node_static`, `node_time`, etc.)
- Existing dataset loaders should work with minor modifications
- Zone-level features and labels are included in the record

## Testing & Validation

All verification tests pass:
- ✓ Function imports
- ✓ Function signatures
- ✓ CLI arguments
- ✓ Directory structure
- ✓ Documentation
- ✓ Helper scripts

Run `python scripts/verify_temporal_install.py` to confirm.

## Next Steps

1. **Generate temporal graphs** for your scenarios
2. **Adapt your GNN model** to handle temporal edges
3. **Train and evaluate** on dispatch prediction tasks
4. **Compare** sequence vs supra-graph performance
5. **Ablate** temporal edge types to understand their importance

## References

- **Static hetero graphs**: `HETERO_GRAPH_QUICKSTART.md`
- **Full temporal docs**: `TEMPORAL_HETERO_GRAPHS.md`
- **Usage examples**: `TEMPORAL_USAGE_EXAMPLES.md`
- **MILP scenarios**: `outputs/scenarios_v1/`

## Support & Issues

- Check that scenarios include `detail` (re-run MILP with `--save-json`)
- Use `--resume` to skip already-processed graphs
- For memory issues, reduce temporal edges or use sequence mode
- Consult documentation files for detailed troubleshooting

---

**Status**: ✓ Implementation complete and verified  
**Version**: 2.0-temporal  
**Date**: November 2024
