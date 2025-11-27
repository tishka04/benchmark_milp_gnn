# Startup Costs Implementation

## Overview

Startup costs have been added throughout the entire pipeline to model the fixed costs incurred when thermal and nuclear generators start up from an off state.

## Changes Made

### 1. **Configuration (scenario_space.yaml)**

Added startup cost ranges:

```yaml
operation_costs:
  thermal_fuel_eur_per_mwh: [45, 85]
  nuclear_fuel_eur_per_mwh: [8, 16]
  thermal_startup_cost_eur: [1500, 8000]          # NEW: Cold start cost per thermal unit
  nuclear_startup_cost_eur: [15000, 50000]        # NEW: Cold start cost per nuclear unit
  demand_response_cost_eur_per_mwh: [350, 1200]
  ...
```

**Typical Values:**
- **Thermal units**: €1,500 - €8,000 per startup
  - Small CCGT: ~€2,000
  - Large coal: ~€6,000
- **Nuclear units**: €15,000 - €50,000 per startup
  - High cost due to thermal stress and regulatory requirements

### 2. **Scenario Generator (generator_v1.py)**

**OperationCosts Dataclass:**
```python
@dataclass
class OperationCosts:
    thermal_fuel_eur_per_mwh: float
    nuclear_fuel_eur_per_mwh: float
    thermal_startup_cost_eur: float        # NEW
    nuclear_startup_cost_eur: float        # NEW
    ...
```

**Sampling Function:**
```python
def sample_operation_costs(space: Dict[str, Any]) -> OperationCosts:
    cost_cfg = space["operation_costs"]
    return OperationCosts(
        thermal_startup_cost_eur=rand_float(*cost_cfg["thermal_startup_cost_eur"]),
        nuclear_startup_cost_eur=rand_float(*cost_cfg["nuclear_startup_cost_eur"]),
        ...
    )
```

Startup costs are automatically included in scenario JSON via `asdict(cfg.costs)`.

### 3. **Scenario Loader (scenario_loader.py)**

**ScenarioData Fields:**
```python
@dataclass
class ScenarioData:
    ...
    thermal_capacity: Dict[str, float]
    thermal_cost: Dict[str, float]
    thermal_startup_cost: Dict[str, float]      # NEW
    
    nuclear_capacity: Dict[str, float]
    nuclear_cost: Dict[str, float]
    nuclear_startup_cost: Dict[str, float]      # NEW
    ...
```

**Loading Logic:**
```python
# Extract from scenario JSON
thermal_startup_cost_value = costs.get("thermal_startup_cost_eur", 5000.0)
nuclear_startup_cost_value = costs.get("nuclear_startup_cost_eur", 30000.0)

# Assign to each zone
for zone in zones:
    thermal_startup_cost[zone] = thermal_startup_cost_value
    nuclear_startup_cost[zone] = nuclear_startup_cost_value
```

### 4. **MILP Model (model.py)**

**New Parameters:**
```python
m.thermal_startup_cost = Param(m.Z, initialize=data.thermal_startup_cost)
m.nuclear_startup_cost = Param(m.Z, initialize=data.nuclear_startup_cost)
```

**New Variables:**
```python
m.u_thermal = Var(m.Z, m.T, within=Binary)         # Commitment (ON/OFF)
m.v_thermal_startup = Var(m.Z, m.T, within=Binary) # Startup indicator

m.u_nuclear = Var(m.Z, m.T, within=Binary)         # Commitment (ON/OFF)
m.v_nuclear_startup = Var(m.Z, m.T, within=Binary) # Startup indicator
```

**Startup Detection Constraints:**
```python
# Thermal startup: v[t] >= u[t] - u[t-1]
def _thermal_startup_rule(model, z, t):
    if t == model.T.first():
        # If unit turns on at t=0, count as startup
        return model.v_thermal_startup[z, t] >= model.u_thermal[z, t]
    else:
        # Startup when u[t] > u[t-1]
        return model.v_thermal_startup[z, t] >= model.u_thermal[z, t] - model.u_thermal[z, t - 1]

m.thermal_startup_detection = Constraint(m.Z, m.T, rule=_thermal_startup_rule)

# Nuclear startup (similar logic)
```

**Objective Function:**
```python
def _objective_rule(model):
    gen_cost = sum(
        model.thermal_cost[z] * model.p_thermal[z, t]
        + model.nuclear_cost[z] * model.p_nuclear[z, t]
        for z in model.Z for t in model.T
    )
    
    # NEW: Startup costs
    startup_cost = sum(
        model.thermal_startup_cost[z] * model.v_thermal_startup[z, t]
        + model.nuclear_startup_cost[z] * model.v_nuclear_startup[z, t]
        for z in model.Z for t in model.T
    )
    
    return gen_cost + startup_cost + response_cost + ...
```

**Nuclear Baseload Behavior:**
```python
# Nuclear units typically stay on (baseload)
for z in data.zones:
    if data.nuclear_capacity.get(z, 0.0) > 1e-6:
        for t in data.periods:
            m.u_nuclear[z, t].setlb(1.0)  # Force nuclear to stay on
```

### 5. **Heterogeneous Graph Builder (hetero_graph_dataset.py)**

**Asset Node Features:**

Updated feature vector to include startup costs:
```python
# Feature vector: [capacity, marginal_cost, min_power, ramp_rate, 
#                  energy_cap, efficiency, startup_cost, zone_idx]
features = [
    kwargs.get("capacity", 0.0),
    kwargs.get("marginal_cost", 0.0),
    kwargs.get("min_power", 0.0),
    kwargs.get("ramp_rate", 0.0),
    kwargs.get("energy_capacity", 0.0),
    kwargs.get("efficiency", 1.0),
    kwargs.get("startup_cost", 0.0),      # NEW
    float(zone_idx),
]
```

**Thermal Asset Example:**
```python
self._add_asset_node(
    zone, zone_idx, "thermal",
    capacity=thermal_cap,
    marginal_cost=self.data.thermal_cost.get(zone, 0.0),
    min_power=self.data.thermal_min_power.get(zone, 0.0),
    ramp_rate=self.data.thermal_ramp.get(zone, 0.0),
    startup_cost=self.data.thermal_startup_cost.get(zone, 0.0),  # NEW
)
```

**Nuclear Asset Example:**
```python
self._add_asset_node(
    zone, zone_idx, "nuclear",
    capacity=nuclear_cap,
    marginal_cost=self.data.nuclear_cost.get(zone, 0.0),
    min_power=self.data.nuclear_min_power.get(zone, 0.0),
    startup_cost=self.data.nuclear_startup_cost.get(zone, 0.0),  # NEW
)
```

## Impact on Model Behavior

### Unit Commitment Economics

**Without startup costs:**
- Units can cycle on/off freely without penalty
- Optimal solution may have frequent switching

**With startup costs:**
- Economic trade-off between:
  - Running at minimum power (paying fuel cost)
  - Shutting down and restarting later (paying startup cost)
- More realistic dispatch behavior

### Example Scenario

**Zone with 300 MW thermal, €60/MWh fuel cost, €5,000 startup cost:**

| Hour | Demand (MW) | Without Startup Costs | With Startup Costs |
|------|-------------|----------------------|-------------------|
| 1    | 100         | ON (100 MW)          | ON (100 MW)       |
| 2    | 50          | OFF                  | ON (min 90 MW)    |
| 3    | 80          | ON (80 MW)           | ON (90 MW)        |
| 4    | 200         | ON (200 MW)          | ON (200 MW)       |

**Cost comparison:**
- Without: 100×60 + 0 + 80×60 + 200×60 = €22,800
- With: 100×60 + 90×60 + 90×60 + 200×60 = €28,800 (but no shutdown/startup)
- Alternative with shutdown: 100×60 + 0 + 5000 + 80×60 + 200×60 = €27,800

The startup cost prevents economically inefficient cycling!

## Usage

### Generate Scenarios

```bash
# Scenarios now automatically include startup costs
python -m src.generator.generator_v1 \
    --config config/scenario_space.yaml \
    --output outputs/scenarios_v2 \
    --count 100
```

### Solve MILP

```bash
# MILP model now includes startup cost optimization
python -m src.milp.solve \
    outputs/scenarios_v2/scenario_00001.json \
    --save-json \
    --output-dir outputs/scenarios_v2/reports
```

### Inspect MILP Solution

```python
import json

with open('outputs/scenarios_v2/reports/scenario_00001.json') as f:
    report = json.load(f)

# Check startup costs in objective
total_cost = report['mip']['objective']
print(f"Total cost: €{total_cost:,.0f}")

# Startup costs are embedded in the objective
# To extract: compare with LP relaxation cost
```

### Build Graphs

```bash
# Graphs now include startup costs as asset features
python -m src.gnn.build_hetero_graph_dataset \
    outputs/scenarios_v2 \
    outputs/scenarios_v2/reports \
    outputs/graphs/hetero_temporal_v2 \
    --temporal \
    --temporal-mode supra
```

### Inspect Asset Features

```python
import numpy as np

data = np.load('outputs/graphs/hetero_temporal_v2/scenario_00001.npz', allow_pickle=True)

node_features = data['node_features']  # [N_nodes, F]
node_types = data['node_types']        # [N_nodes]

# Asset nodes have type = 3
asset_mask = (node_types == 3)
asset_features = node_features[asset_mask]

# Feature indices:
# [0] capacity
# [1] marginal_cost
# [2] min_power
# [3] ramp_rate
# [4] energy_capacity
# [5] efficiency
# [6] startup_cost  ← NEW
# [7] zone_idx

startup_costs = asset_features[:, 6]
print(f"Asset startup costs: {startup_costs}")
```

## Model Extensions

### Warm vs Cold Starts

Current implementation uses a single cold-start cost. To model warm starts:

```python
# In model.py, add:
m.thermal_warmstart_cost = Param(...)
m.thermal_hours_down = Var(m.Z, m.T, within=NonNegativeIntegers)

# Track hours offline
def _hours_down_rule(model, z, t):
    if t == model.T.first():
        return Constraint.Skip
    return model.thermal_hours_down[z, t] == (
        (1 - model.u_thermal[z, t]) * (1 + model.thermal_hours_down[z, t-1])
    )

# Variable startup cost based on hours offline
def _startup_cost_rule(model, z, t):
    cold_threshold = 8  # hours
    # If down > cold_threshold: cold start, else warm start
    ...
```

### Min Up/Down Time

```python
# Minimum up time constraint
m.thermal_min_up_time = Param(m.Z, default=4)

def _min_up_rule(model, z, t):
    up_periods = model.thermal_min_up_time[z]
    if t < up_periods:
        return Constraint.Skip
    return sum(model.v_thermal_startup[z, tt] for tt in range(t - up_periods + 1, t + 1)) <= model.u_thermal[z, t]
```

### Shutdown Costs

```python
m.v_thermal_shutdown = Var(m.Z, m.T, within=Binary)
m.thermal_shutdown_cost = Param(m.Z, ...)

# Shutdown detection
def _shutdown_rule(model, z, t):
    if t == model.T.first():
        return Constraint.Skip
    return model.v_thermal_shutdown[z, t] >= model.u_thermal[z, t-1] - model.u_thermal[z, t]

# Add to objective
shutdown_cost = sum(
    model.thermal_shutdown_cost[z] * model.v_thermal_shutdown[z, t]
    for z in model.Z for t in model.T
)
```

## Validation

### Check Startup Costs in Scenarios

```python
import json

with open('outputs/scenarios_v2/scenario_00001.json') as f:
    scenario = json.load(f)

costs = scenario['meta']['operation_costs']
print(f"Thermal startup: €{costs['thermal_startup_cost_eur']:.0f}")
print(f"Nuclear startup: €{costs['nuclear_startup_cost_eur']:.0f}")
```

### Verify MILP Variables

```python
from src.milp.scenario_loader import load_scenario_data
from src.milp.model import build_uc_model

data = load_scenario_data('outputs/scenarios_v2/scenario_00001.json')
model = build_uc_model(data)

# Check startup variables exist
assert hasattr(model, 'v_thermal_startup')
assert hasattr(model, 'v_nuclear_startup')

# Check startup costs in objective
obj_expr = model.obj.expr
print(f"Objective includes startup costs: {'thermal_startup_cost' in str(obj_expr)}")
```

### Compare Solutions With/Without Startup Costs

```python
# Solve with startup costs
model_with = build_uc_model(data)
solver.solve(model_with)
cost_with = model_with.obj()

# Solve without (set startup costs to zero)
data_no_startup = copy.deepcopy(data)
data_no_startup.thermal_startup_cost = {z: 0.0 for z in data.zones}
data_no_startup.nuclear_startup_cost = {z: 0.0 for z in data.zones}

model_without = build_uc_model(data_no_startup)
solver.solve(model_without)
cost_without = model_without.obj()

print(f"Cost difference: €{cost_with - cost_without:,.0f}")
print(f"Startup cost premium: {(cost_with / cost_without - 1) * 100:.2f}%")
```

## Summary

✅ **Configuration**: Startup cost ranges added to `scenario_space.yaml`  
✅ **Generator**: Samples and includes startup costs in scenarios  
✅ **Scenario Loader**: Parses and provides startup costs to MILP  
✅ **MILP Model**: Binary startup variables and startup cost in objective  
✅ **Graph Builder**: Startup costs included as asset node features  

Startup costs are now fully integrated throughout the pipeline, enabling more realistic unit commitment modeling and richer graph representations for GNN training!

---

**Next Steps:**
1. Regenerate scenarios with startup costs
2. Re-solve MILP models
3. Rebuild heterogeneous graphs
4. Train GNN/EBM models with startup cost awareness
5. Consider extensions: warm starts, min up/down time, shutdown costs
