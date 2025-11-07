# Heterogeneous Multi-Level Graph Transformation Guide

This guide explains how to transform MILP-solved scenarios into hierarchical heterogeneous graphs as depicted in your presentation slide.

## Overview

The transformation converts the flat zone-based graph into a multi-level heterogeneous graph with explicit hierarchy and weather nodes:

### **Node Types (5 types)**

| Type | ID | Description | Count per Scenario |
|------|----|----|---|
| **Nation** | 0 | Top-level aggregation | 1 |
| **Region** | 1 | Geographical/administrative regions | R (typically 2-5) |
| **Zone** | 2 | Load centers (original zones) | Z (typically 5-20) |
| **Asset** | 3 | Individual generators/storage per zone/tech | A (typically 50-200) |
| **Weather** | 4 | Weather cells influencing renewables | W = R (one per region) |

### **Edge Types (7 types)**

| Type | ID | Description | Directionality |
|------|----|----|---|
| **Nation → Region** | 0 | Containment hierarchy | Directed |
| **Region → Zone** | 1 | Containment hierarchy | Directed |
| **Zone → Asset** | 2 | Containment hierarchy | Directed |
| **Weather → Zone** | 3 | Weather influence on zone demand/RES | Directed |
| **Weather → Asset** | 4 | Direct influence on RES/hydro assets | Directed |
| **Zone ↔ Zone** | 5 | Transmission lines (AC interties) | Bidirectional |
| **Asset → Asset** | 6 | Temporal state dependencies (storage) | Self-loops |

## Graph Structure Example

```
                    [Nation]
                   /    |    \
                  /     |     \
            [Region1] [Region2] [Weather1] [Weather2]
             /   \      /  \        ↓         ↓
            /     \    /    \       ↓         ↓
       [Zone_A] [Zone_B] [Zone_C] ...
          |  \      |   \     |
          |   \     |    \    |
     [Asset] [Asset] ...  [Asset]
      (Thermal) (Solar)   (Battery)
          ↓                  ↓
      (temporal)        (temporal)
```

## Asset Node Types

Assets are created per zone for each technology with non-zero capacity:
- **Thermal**: conventional generators
- **Solar**: solar PV
- **Wind**: wind turbines  
- **Nuclear**: nuclear plants
- **Hydro Reservoir**: dispatchable hydro with storage
- **Hydro RoR**: run-of-river hydro
- **Battery**: battery storage
- **Pumped**: pumped hydro storage
- **DR**: demand response

## Usage

### 1. Generate Heterogeneous Graph Dataset

```bash
# Single scenario
python -m src.gnn.hetero_graph_dataset \
    outputs/scenarios_v1/scenario_00001.json \
    outputs/scenarios_v1/reports/scenario_00001.json \
    outputs/hetero_graphs/scenario_00001.npz

# Batch conversion (create wrapper script)
python -m src.gnn.build_hetero_graph_dataset \
    outputs/scenarios_v1 \
    outputs/scenarios_v1/reports \
    outputs/hetero_graphs
```

### 2. Inspect the Graph Structure

```python
import numpy as np

# Load heterogeneous graph
data = np.load("outputs/hetero_graphs/scenario_00001.npz", allow_pickle=True)

print(f"Total nodes: {len(data['node_types'])}")
print(f"Node type distribution:")
for node_type, count in zip(*np.unique(data['node_types'], return_counts=True)):
    type_names = ['Nation', 'Region', 'Zone', 'Asset', 'Weather']
    print(f"  {type_names[node_type]}: {count}")

print(f"\nTotal edges: {data['edge_index'].shape[1]}")
print(f"Edge type distribution:")
edge_type_names = [
    'Nation→Region', 'Region→Zone', 'Zone→Asset',
    'Weather→Zone', 'Weather→Asset', 'Transmission', 'Temporal'
]
for edge_type, count in zip(*np.unique(data['edge_types'], return_counts=True)):
    print(f"  {edge_type_names[edge_type]}: {count}")
```

### 3. Adapt GNN Models

The existing GNN models already support heterogeneous graphs through:
- `node_type`: node type embeddings
- `edge_type`: typed message passing

#### Option A: Minimal Changes (Use Existing Models)

The `GCNModel`, `GraphSAGEModel`, and `GATModel` already have:
- `typed_message_passing=True`: enables edge-type-specific aggregation
- `node_type_embedding`: adds learnable embeddings per node type

Update training config:

```yaml
model:
  backbone: gcn  # or graphsage, gat
  typed_message_passing: true
  num_edge_types: 7  # changed from 2
  node_type_cardinality: 5  # changed from 4
  type_embedding_dim: 32  # increased for richer types
```

#### Option B: Custom Heterogeneous GNN

For more sophisticated heterogeneous message passing:

```python
# src/gnn/models/hetero_gcn.py
class HeteroGCNLayer(nn.Module):
    """Type-specific message passing with different weights per (src_type, edge_type, dst_type)."""
    
    def __init__(self, in_dim, out_dim, num_node_types, num_edge_types):
        super().__init__()
        # Separate linear transformations for each (src_type, rel_type) pair
        self.transforms = nn.ModuleDict()
        for src_type in range(num_node_types):
            for edge_type in range(num_edge_types):
                key = f"{src_type}_{edge_type}"
                self.transforms[key] = nn.Linear(in_dim, out_dim, bias=False)
    
    def forward(self, x, edge_index, node_types, edge_types):
        # Group edges by (src_type, edge_type)
        # Apply type-specific transformations
        # Aggregate to target nodes
        ...
```

### 4. Training with Heterogeneous Graphs

Create new dataset loader:

```python
# src/gnn/data/hetero_temporal.py
class HeteroGraphTemporalDataset(Dataset):
    """Dataset loader for heterogeneous multi-level graphs."""
    
    def __getitem__(self, idx):
        # Load NPZ
        record = self._read_npz(self.paths[idx])
        
        # Extract all node types
        node_features = torch.from_numpy(record["node_features"])
        node_types = torch.from_numpy(record["node_types"])
        
        # Extract all edge types
        edge_index = torch.from_numpy(record["edge_index"])
        edge_types = torch.from_numpy(record["edge_types"])
        edge_features = torch.from_numpy(record["edge_features"])
        
        # Labels remain at zone level for compatibility
        zone_indices = record["zone_node_indices"]
        zone_labels = record["zone_labels"]  # [T, Z, D]
        
        return HeteroGraphSample(
            node_features=node_features,
            node_types=node_types,
            edge_index=edge_index,
            edge_types=edge_types,
            edge_features=edge_features,
            zone_indices=zone_indices,
            target=torch.from_numpy(zone_labels),
            ...
        )
```

### 5. Prediction and Evaluation

Predictions can be made at:
1. **Zone level** (for compatibility with existing metrics)
2. **Asset level** (for fine-grained dispatch)

```python
# In model forward pass
def forward(self, batch):
    # Message passing over full heterogeneous graph
    x = self.hetero_gnn(batch.node_features, batch.edge_index, 
                        batch.node_types, batch.edge_types)
    
    # Extract zone node representations
    zone_mask = batch.node_types == NODE_TYPE_ZONE
    zone_features = x[zone_mask]
    
    # Predict zone-level dispatch
    zone_dispatch = self.zone_head(zone_features)
    
    # Optionally: extract asset predictions
    asset_mask = batch.node_types == NODE_TYPE_ASSET
    asset_features = x[asset_mask]
    asset_dispatch = self.asset_head(asset_features)
    
    return zone_dispatch, asset_dispatch
```

## Benefits of Heterogeneous Representation

### 1. **Explicit Hierarchy**
- Nation/Region-level aggregations can inform zone predictions
- Hierarchical pooling naturally captures multi-scale patterns
- Better generalization to scenarios with different topologies

### 2. **Interpretable Asset-Level Predictions**
- Direct prediction of individual generator/storage dispatch
- Explicit modeling of asset constraints (ramp rates, min power)
- More granular feasibility checking

### 3. **Weather as First-Class Citizens**
- Weather nodes can aggregate meteorological features
- Direct influence edges to RES/hydro assets
- Potential for weather forecasting integration

### 4. **Temporal Dependencies**
- Explicit temporal edges for storage state transitions
- Can model multi-timestep dependencies (e.g., SOC evolution)
- Clearer representation of inter-temporal constraints

### 5. **Richer Message Passing**
- Information flows: Weather → Asset → Zone → Region → Nation
- Reverse flows: Nation-level policy → Region → Zone → Asset
- Cross-layer skip connections possible

## Data Available in ScenarioData

The MILP scenarios already contain all necessary information:

✅ **Region hierarchy**: `region_of_zone`  
✅ **Weather data**: `region_weather_profile`, `region_weather_spread`  
✅ **Asset capacities**: All technology capacities per zone  
✅ **Transmission topology**: `lines` with `from_zone`, `to_zone`  
✅ **Time series**: Demand, RES availability, hydro inflows per zone  
✅ **Dispatch solutions**: Per-zone per-technology dispatch in `detail`

## Implementation Checklist

- [x] Heterogeneous graph builder (`hetero_graph_dataset.py`)
- [ ] Batch conversion script (`build_hetero_graph_dataset.py`)
- [ ] Heterogeneous dataset loader (`data/hetero_temporal.py`)
- [ ] Update GNN models for multi-level graphs
- [ ] Training script modifications
- [ ] Evaluation metrics at zone and asset levels
- [ ] Visualization tools for hierarchical graphs
- [ ] Documentation and examples

## Next Steps

1. **Test the builder**:
   ```bash
   python -m src.gnn.hetero_graph_dataset \
       outputs/scenarios_v1/scenario_00001.json \
       outputs/scenarios_v1/reports/scenario_00001.json \
       outputs/test_hetero.npz
   ```

2. **Inspect the output** to verify node/edge counts

3. **Create batch conversion script** for all scenarios

4. **Adapt existing models** to handle heterogeneous graphs

5. **Train and compare** heterogeneous vs flat representations

## Potential Extensions

- **Asset-level labels**: Individual generator dispatch targets
- **Weather time series**: Integrate real weather forecasts
- **Cross-scenario weather sharing**: Weather nodes shared across scenarios
- **Hierarchical pooling layers**: Bottom-up aggregation (Asset→Zone→Region→Nation)
- **Attention mechanisms**: Learn which hierarchy levels matter most
- **Multi-task learning**: Predict at zone + asset + region levels simultaneously
