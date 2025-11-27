# Asset-Level Labels in Heterogeneous Multi-Layer Graphs

## Overview

The heterogeneous multi-layer temporal graphs now include **synthetic asset-level labels** decomposed from zone-level MILP solutions. This enables training models that learn asset-level dispatch patterns even when trained on zone-level optimization results.

## Graph Structure Enhancement

### Node Hierarchy
```
Nation (1 node)
  ├── Regions (R nodes)
  │   ├── Zones (Z nodes)
  │   │   ├── Assets (A nodes per zone)
  │   │   │   ├── Thermal
  │   │   │   ├── Nuclear
  │   │   │   ├── Solar
  │   │   │   ├── Wind
  │   │   │   ├── Hydro (reservoir + run-of-river)
  │   │   │   ├── Battery
  │   │   │   ├── Pumped Storage
  │   │   │   └── Demand Response
```

### Asset Node Features

Each asset node has static features:
- **Capacity** (MW)
- **Marginal cost** (EUR/MWh)
- **Minimum power** (fraction)
- **Ramp rate** (MW/period)
- **Energy capacity** (MWh for storage)
- **Efficiency** (round-trip for storage)
- **Parent zone index**

## Synthetic Asset Labels

### Format

Asset labels are stored as: `asset_labels[T, N_assets, 5]`

Where the 5 dimensions are:
1. **dispatch_mw**: Power dispatch (MW) - positive for generation/discharge
2. **commitment_binary**: On/off status (0 or 1)
3. **charge_mw**: Charging power (MW, storage only)
4. **discharge_mw**: Discharging power (MW, storage only)
5. **soc_mwh**: State of charge (MWh, storage only)

### Decomposition Strategy

#### 1. **Thermal Assets**
```python
# Zone aggregate → Single thermal asset
zone_thermal = 120 MW
asset_thermal_dispatch = 120 MW
asset_thermal_commitment = 1 (ON)
```

**Merit order extension** (for multiple thermal units per zone):
```python
# Sort by marginal cost
units_sorted = sorted(units, key=lambda u: u.marginal_cost)

# Allocate zone target to cheapest units first
remaining = zone_thermal
for unit in units_sorted:
    unit_dispatch = min(unit.capacity, remaining)
    unit_commitment = 1 if unit_dispatch > min_power else 0
    remaining -= unit_dispatch
```

#### 2. **Nuclear Assets**
```python
# Typically baseload - always on at capacity
asset_nuclear_dispatch = zone_nuclear
asset_nuclear_commitment = 1 if zone_nuclear > 0 else 0
```

#### 3. **Renewable Assets (Solar/Wind)**
```python
# Direct allocation (one asset per zone per type)
asset_solar_dispatch = zone_solar
asset_wind_dispatch = zone_wind
```

#### 4. **Storage Assets (Battery/Pumped)**
```python
# Full charge/discharge/SOC tracking
asset_battery_dispatch = zone_battery_discharge
asset_battery_charge = zone_battery_charge
asset_battery_discharge = zone_battery_discharge
asset_battery_soc = zone_battery_soc
asset_battery_commitment = 1 if (charge > 0 or discharge > 0) else 0
```

#### 5. **Hydro Assets**
```python
# Reservoir hydro
asset_hydro_res_dispatch = zone_hydro_release

# Run-of-river (fixed generation)
asset_hydro_ror_dispatch = zone_hydro_ror
```

#### 6. **Demand Response**
```python
# DR activation
asset_dr_dispatch = zone_dr_shed
asset_dr_commitment = 1 if zone_dr_shed > 0 else 0
```

## Usage

### Building Graphs with Asset Labels

```bash
# Build hetero temporal graphs with asset nodes and labels
python -m src.gnn.build_hetero_graph_dataset \
    outputs/scenarios_v1 \
    outputs/scenarios_v1/reports \
    outputs/graphs/hetero_temporal_assets \
    --temporal \
    --temporal-mode supra
```

### Loading Asset Labels

```python
import numpy as np

# Load graph
data = np.load('outputs/graphs/hetero_temporal_assets/scenario_00001.npz', allow_pickle=True)

# Extract asset information
asset_labels = data['asset_labels']  # [T, N_assets, 5]
asset_names = data['asset_names']    # [N_assets]
asset_node_indices = data['asset_node_indices']  # [N_assets]

T, N_assets, label_dim = asset_labels.shape
print(f"Time steps: {T}, Assets: {N_assets}")

# Example: Get thermal asset dispatch
thermal_assets = [i for i, name in enumerate(asset_names) if 'thermal' in name.decode()]
for idx in thermal_assets:
    name = asset_names[idx].decode()
    dispatch = asset_labels[:, idx, 0]  # [T]
    commitment = asset_labels[:, idx, 1]  # [T]
    print(f"{name}: avg dispatch = {dispatch.mean():.2f} MW, "
          f"commitment rate = {commitment.mean():.2%}")
```

### Training with Asset Labels

```python
import torch
from torch_geometric.data import HeteroData

# Create PyG hetero data object
hetero_data = HeteroData()

# Node features (all nodes including assets)
hetero_data['node'].x = torch.from_numpy(data['node_features'])
hetero_data['node'].node_type = torch.from_numpy(data['node_types'])

# Asset-specific data
asset_indices = data['asset_node_indices']
asset_features = hetero_data['node'].x[asset_indices]
asset_labels_tensor = torch.from_numpy(data['asset_labels'])

# Training targets
hetero_data['asset'].y = asset_labels_tensor  # [T, N_assets, 5]
hetero_data['asset'].names = asset_names

# Model prediction
model_output = model(hetero_data)
asset_predictions = model_output['asset']  # [T, N_assets, 5]

# Loss computation
loss_dispatch = F.mse_loss(asset_predictions[:, :, 0], asset_labels_tensor[:, :, 0])
loss_commitment = F.binary_cross_entropy(
    torch.sigmoid(asset_predictions[:, :, 1]), 
    asset_labels_tensor[:, :, 1]
)
```

## Comparison: Zone-Level vs Asset-Level

### Zone-Level Labels (Original)
```
node_labels[T, N_zones, 13]:
  - thermal (aggregate)
  - nuclear (aggregate)
  - solar (aggregate)
  - ...
```

**Pros:**
- Directly from MILP optimal solution
- Guaranteed feasible and optimal
- Simpler to train

**Cons:**
- No asset-level detail
- Can't answer: "Which specific units to turn on?"
- Requires post-processing for real operation

### Asset-Level Labels (New)
```
asset_labels[T, N_assets, 5]:
  - dispatch_mw (per asset)
  - commitment_binary (per asset)
  - charge/discharge (storage)
  - soc (storage state)
```

**Pros:**
- Asset-level granularity
- Can learn unit commitment patterns
- Ready for EBM sampling at asset level

**Cons:**
- Synthetic (decomposed, not from asset-level MILP)
- May not represent optimal asset-level solution
- Requires validation against true asset-level UC

## Advanced: Merit Order Extension

For true asset-level UC, extend decomposition with multiple units per zone:

```python
class AdvancedDecomposer:
    def __init__(self, scenario_data):
        # Load detailed unit database
        self.units = self.load_unit_database(scenario_data)
    
    def decompose_thermal_multi_unit(self, zone, zone_thermal_mw):
        """Decompose zone thermal into multiple units."""
        zone_units = [u for u in self.units if u.zone == zone and u.type == 'thermal']
        
        # Sort by economic merit order
        zone_units.sort(key=lambda u: u.marginal_cost)
        
        commitments = []
        dispatches = []
        remaining = zone_thermal_mw
        
        for unit in zone_units:
            if remaining < unit.min_power:
                # Not enough demand to turn on this unit
                commitments.append(0)
                dispatches.append(0.0)
            else:
                # Turn on unit
                dispatch = min(unit.capacity, remaining)
                commitments.append(1)
                dispatches.append(dispatch)
                remaining -= dispatch
        
        return commitments, dispatches
```

## Data Files

After building graphs, the following fields are available:

### Heterogeneous Graph Fields
- `node_features`: [N_total, F] - All nodes (nation, region, zone, asset, weather)
- `node_types`: [N_total] - Node type IDs
- `edge_index`: [2, E] - Edge connectivity
- `edge_types`: [E] - Edge type IDs
- `edge_features`: [E, F_edge] - Edge attributes

### Asset-Specific Fields (NEW)
- **`asset_labels`**: [T, N_assets, 5] - Synthetic asset dispatch labels
- **`asset_node_indices`**: [N_assets] - Indices into node_features for assets
- **`asset_names`**: [N_assets] - Asset identifiers (e.g., "R1Z1_thermal")

### Zone Compatibility Fields (Existing)
- `node_static`: [N_zones, 22] - Zone static features
- `node_time`: [T, N_zones, 18] - Zone temporal features
- `node_labels`: [T, N_zones, 13] - Zone dispatch labels

## Use Cases

### 1. **Hierarchical GNN Training**
Train GNN to predict both zone and asset levels simultaneously:
```python
# Zone-level loss (from MILP optimal)
loss_zone = mse_loss(pred_zone, label_zone)

# Asset-level auxiliary loss (from synthetic decomposition)
loss_asset = mse_loss(pred_asset, label_asset)

# Combined loss
loss = loss_zone + alpha * loss_asset
```

### 2. **EBM Asset-Level Sampling**
Use asset labels as training signal for EBM:
```python
# EBM learns: low energy = good configuration
E_theta(h(x), u_asset_config) → scalar energy

# Train with contrastive learning
# Positive: synthetic asset labels (low energy)
# Negative: infeasible/suboptimal configs (high energy)
```

### 3. **Transfer Learning**
Pre-train on synthetic asset labels, fine-tune with real asset-level data (if available):
```python
# Stage 1: Pre-train on many synthetic examples
model.train(synthetic_asset_labels)  # 10,000 scenarios

# Stage 2: Fine-tune on few real asset-level UC solutions
model.finetune(real_asset_uc_solutions)  # 100 scenarios
```

## Validation

To validate synthetic labels against zone aggregates:

```python
# Check consistency
zone_thermal_from_milp = node_labels[t, zone_idx, 0]
assets_in_zone = [a for a in asset_names if a.startswith(f'{zone}_')]
asset_thermal_sum = sum(asset_labels[t, a_idx, 0] for a_idx in assets_in_zone if 'thermal' in asset_names[a_idx])

assert abs(zone_thermal_from_milp - asset_thermal_sum) < 1e-3, "Decomposition inconsistent!"
```

## Future Extensions

1. **Multi-unit decomposition**: Split single asset into multiple units
2. **Start-up cost modeling**: Track cold/warm/hot starts in labels
3. **Ramping trajectories**: Include ramp-up/down states in labels
4. **Contingency reserves**: Decompose reserve allocation to assets
5. **True asset-level MILP**: Compare synthetic vs optimal asset labels

---

## References

- Heterogeneous graph builder: `src/gnn/hetero_graph_dataset.py`
- Asset decomposition methods: `HeteroGraphBuilder._decompose_*()` 
- Dataset builder: `src/gnn/build_hetero_graph_dataset.py`
- Usage examples: `examples/temporal_hetero_quickstart.py`
