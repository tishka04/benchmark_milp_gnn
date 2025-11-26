# Temporal Heterogeneous Graph Generation

This document describes how to build temporal heterogeneous multi-layer grid graphs from MILP scenarios.

## Overview

The temporal graph builder extends the static heterogeneous graph with time-awareness, supporting two modes:

1. **Sequence Mode**: Generates a list of graph snapshots, one per time step
2. **Supra-Graph Mode**: Creates a single time-expanded graph where nodes are replicated across time with temporal edges

## Graph Structure

### Node Types (inherited from static hetero graph)
- **Nation** (0): Top-level aggregation node
- **Region** (1): Administrative regions
- **Zone** (2): Load centers
- **Asset** (3): Individual generators/storage per zone per technology
- **Weather** (4): Weather cells influencing renewables

### Edge Types

#### Spatial Edges (replicated across time)
- Nation → Region (0)
- Region → Zone (1)
- Zone → Asset (2)
- Weather → Zone (3)
- Weather → Asset (4)
- Zone ↔ Zone transmission (5)
- Asset self-loops (6)

#### Temporal Edges (supra-graph only)
- **SOC edges** (7): Storage state-of-charge continuity (battery, pumped hydro, reservoir)
- **Ramp edges** (8): Generator ramping constraints (thermal, nuclear)
- **DR edges** (9): Demand response cooldown constraints

## Usage

### Basic Command

```bash
python -m src.gnn.build_hetero_graph_dataset \
    scenarios/ \
    reports/ \
    outputs/temporal_graphs/supra/ \
    --temporal \
    --temporal-mode supra
```

### Command-Line Arguments

#### Required Arguments
- `scenario_dir`: Directory containing scenario JSON files
- `reports_dir`: Directory containing report JSON files (with detail)
- `output_dir`: Output directory for graph NPZ files

#### Temporal Options
- `--temporal`: Enable temporal graph building (default: False)
- `--temporal-mode {sequence,supra}`: Graph mode (default: supra)
  - `sequence`: List of snapshots, one per time step
  - `supra`: Single time-expanded graph
- `--window N`: Sequence window size in time steps (for sequence mode)
- `--stride K`: Window stride for sliding windows (default: 1)
- `--temporal-edges TYPES`: Comma-separated list of temporal edge types (default: "soc,ramp,dr")
  - `soc`: State-of-charge edges for storage
  - `ramp`: Ramping constraint edges for generators
  - `dr`: Demand response cooldown edges
- `--time-enc {sinusoidal,cyclic-hod}`: Time encoding method (default: sinusoidal)
  - `sinusoidal`: Positional encoding with sin/cos
  - `cyclic-hod`: Hour-of-day cyclic encoding
- `--target-horizon H`: Prediction horizon for labels (default: 0)

#### Other Options
- `--resume`: Skip graphs that already exist

## Examples

### 1. Supra-Graph with All Temporal Edges

```bash
python -m src.gnn.build_hetero_graph_dataset \
    scenarios/ \
    reports/ \
    outputs/temporal_graphs/supra/ \
    --temporal \
    --temporal-mode supra \
    --temporal-edges soc,ramp,dr \
    --time-enc sinusoidal
```

**Output**: Single `.npz` file per scenario with:
- Nodes: `N_base × T` (all base nodes replicated across T timesteps)
- Edges: Spatial edges (replicated per timestep) + temporal edges
- Features: Node features augmented with time encodings

### 2. Sequence Mode (List of Snapshots)

```bash
python -m src.gnn.build_hetero_graph_dataset \
    scenarios/ \
    reports/ \
    outputs/temporal_graphs/sequence/ \
    --temporal \
    --temporal-mode sequence \
    --time-enc cyclic-hod
```

**Output**: Multiple `.npz` files per scenario:
- `scenario_001_t0000.npz`, `scenario_001_t0001.npz`, ..., `scenario_001_t0047.npz`
- Each file contains a single snapshot at time `t`
- Useful for RNN/Transformer training

### 3. Sliding Window Sequence

```bash
python -m src.gnn.build_hetero_graph_dataset \
    scenarios/ \
    reports/ \
    outputs/temporal_graphs/sequence/ \
    --temporal \
    --temporal-mode sequence \
    --window 24 \
    --stride 12
```

**Output**: Overlapping windows of 24 timesteps, sliding by 12 steps

### 4. Only SOC Edges (Minimal Temporal Info)

```bash
python -m src.gnn.build_hetero_graph_dataset \
    scenarios/ \
    reports/ \
    outputs/temporal_graphs/supra/ \
    --temporal \
    --temporal-mode supra \
    --temporal-edges soc
```

## Data Schema

### Supra-Graph Record

```python
{
    "graph_type": "hetero_multi_layer_temporal_supra",
    "node_features": np.ndarray,  # [N*T, F+Ft] - features + time encoding
    "node_types": np.ndarray,     # [N*T] - node type IDs
    "node_ids": List[str],        # ["Zone1#t=0", "Zone1#t=1", ...]
    "time_index": List[str],      # ["t=0", "t=1", ..., "t=T-1"]
    "time_steps": List[int],      # [0, 1, 2, ...]
    "edge_index": np.ndarray,     # [2, E_total] - spatial + temporal edges
    "edge_types": np.ndarray,     # [E_total] - edge type IDs
    "edge_features": np.ndarray,  # [E_total, Fe] - edge attributes
    "meta": {
        "N_base": int,            # Base number of nodes (before time expansion)
        "T": int,                 # Number of timesteps
        "temporal_edges": List[str],  # ["soc", "ramp", "dr"]
        "time_encoding": str,     # "sinusoidal" or "cyclic-hod"
        "target_horizon": int,
        "schema_version": "2.0-temporal",
    },
    # Flat compatibility fields (preserved from static graph)
    "node_static": np.ndarray,
    "node_time": np.ndarray,
    "node_labels": np.ndarray,
    "zone_region_index": np.ndarray,
}
```

### Sequence Record (per snapshot)

```python
{
    # Same structure as static hetero graph, plus:
    "time_index": "t=5",          # Time identifier
    "time_step": 5,               # Timestep index
    "node_features": np.ndarray,  # [N, F+Ft] - includes time encoding
}
```

## Time Encoding

### Sinusoidal (Positional Encoding)
- 4 features: `[sin(t/d1), cos(t/d1), sin(t/d2), cos(t/d2)]`
- Captures positional information
- Good for arbitrary sequence lengths

### Cyclic Hour-of-Day
- 4 features: `[sin(2π·h/24), cos(2π·h/24), sin(dow), cos(dow)]`
- Captures hour-of-day and day-of-week cycles
- Good for datasets with daily/weekly patterns

## Node Features

Each node's feature vector is:
```
[static_features..., time_encoding...]
```

For example, a Zone node at time t:
```
[thermal_cap, solar_cap, wind_cap, ..., sin_t, cos_t, ...]
```

## Edge Interpretation

### Temporal Edge Semantics

1. **SOC Edges** (type 7): `asset_i@t → asset_i@(t+1)`
   - Represents storage continuity
   - Edge attribute: retention rate (currently 1.0)

2. **Ramp Edges** (type 8): `generator_i@t → generator_i@(t+1)`
   - Represents ramping constraints
   - Edge attribute: ramp rate (MW/period)

3. **DR Edges** (type 9): `dr_asset_i@t → dr_asset_i@(t+k)`
   - Represents cooldown periods
   - Edge attribute: cooldown distance k

## Directory Structure

```
outputs/
└── temporal_graphs/
    ├── sequence/          # Sequence mode outputs
    │   ├── scenario_001_t0000.npz
    │   ├── scenario_001_t0001.npz
    │   └── ...
    └── supra/            # Supra-graph mode outputs
        ├── scenario_001.npz
        ├── scenario_002.npz
        └── ...
```

## Performance Considerations

### Memory Usage
- **Sequence mode**: Light, one snapshot at a time
- **Supra-graph mode**: `O(N × T × F + E × T)` memory
  - For T=48, N=1000, this is ~50-100MB per scenario

### Training Recommendations
- **Sequence mode**: Use with RNN/LSTM/Transformer architectures
- **Supra-graph mode**: Use with temporal GNN (TGNN) architectures that propagate messages through time

## Dataset Index

The `dataset_index.json` includes temporal metadata:

```json
{
  "entries": [...],
  "metadata": {
    "total_scenarios": 500,
    "graph_type": "heterogeneous_multi_level_temporal_supra",
    "temporal": true,
    "temporal_mode": "supra",
    "temporal_edges": "soc,ramp,dr",
    "time_encoding": "sinusoidal"
  }
}
```

## Next Steps

1. **Training**: Adapt your GNN model to handle temporal graphs
2. **Evaluation**: Use time-aware metrics (e.g., MAE over time horizon)
3. **Visualization**: Inspect temporal edge connectivity with `inspect_hetero_graph.py`

## Troubleshooting

### Issue: "No time steps found in report"
- Ensure reports include `detail` with `time_steps` and `time_hours`
- Rerun MILP with `--save-json` flag

### Issue: Memory error with supra-graph
- Reduce number of temporal edge types
- Use sequence mode instead
- Process fewer scenarios at once

### Issue: Sequence mode creates too many files
- Increase `--stride` value
- Specify a smaller `--window`

## References

- Static hetero graph documentation: `HETERO_GRAPH_QUICKSTART.md`
- Original MILP documentation: See `outputs/scenarios_v1/`
