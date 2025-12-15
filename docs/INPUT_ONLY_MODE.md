# Input-Only Mode: Eliminating Solution Data Leakage

## Problem Identified

The original graph construction pipeline included **MILP solution data** in node temporal features:
- ✗ Optimal dispatch values (solar, wind, thermal, etc.)
- ✗ Optimal storage states (SOC, charging/discharging)
- ✗ Optimal demand response activation
- ✗ Spills, unserved energy, etc.

This caused **data leakage** during encoder training:
- The encoder learned to recognize solution patterns, not just problem structure
- Embeddings `h` contained hints about optimal solutions
- EBM training became easier (solution-guided) but less generalizable

## Solution Implemented

Added `--input-only` flag to graph builder that uses **only input forecasts**:

### Input-Only Features (Clean)
```
node_time: [T, N_zones, 8]
  - demand_forecast        ✓ Input
  - solar_available        ✓ Input (capacity factor)
  - wind_available         ✓ Input (capacity factor)
  - hydro_inflow          ✓ Input (natural inflow)
  - hydro_ror_generation  ✓ Input (run-of-river)
  - battery_initial_soc   ✓ Boundary condition
  - pumped_initial_level  ✓ Boundary condition
  - dr_limit              ✓ Input (DR availability)
```

### Labels (Still from Solution)
```
node_labels: [T, N_zones, 13]  
  - thermal, nuclear, solar, wind dispatch  ← Target for training
  - battery/pumped charge/discharge         ← Target for training
  - etc.
```

**Key difference**: Features = inputs only, Labels = solution targets. No solution data leaks into encoder.

---

## Usage

### Rebuild Graphs with Input-Only Mode

```bash
# Original command (with solution leakage)
python -m src.gnn.build_hetero_graph_dataset \
    outputs/scenarios_v1 \
    outputs/reports_v1 \
    outputs/graphs_hetero_temporal \
    --temporal \
    --temporal-mode supra

# NEW: Input-only mode (no leakage)
python -m src.gnn.build_hetero_graph_dataset \
    outputs/scenarios_v1 \
    outputs/reports_v1 \
    outputs/graphs_hetero_temporal_input_only \
    --temporal \
    --temporal-mode supra \
    --input-only  # ← New flag
```

### Impact on Feature Dimensions

**Old (with solution):**
- `node_time`: [T, N, 18] (includes all solution variables)

**New (input-only):**
- `node_time`: [T, N, 8] (only forecasts + boundary conditions)

The encoder will need to learn from **structure and forecasts alone**, not from solution patterns.

---

## Expected Results

### Old Pipeline (with leakage)
- ✓ Fast convergence (~3.0 loss)
- ✓ Excellent 0-2% cost gap
- ✗ **Artificially good** - model sees solution hints
- ✗ Unclear true generalization

### New Pipeline (input-only)
- Results will likely be **worse initially**
- But represents **true generalization capability**
- No solution hints during encoder training
- EBM must learn energy landscape from scratch

---

## Next Steps

1. **Rebuild graphs:**
   ```bash
   python -m src.gnn.build_hetero_graph_dataset outputs/scenarios_v1 outputs/scenarios_v1/reports outputs/graphs_input_only --temporal --temporal-mode supra --input-only
   ```

2. **Retrain encoder:**
   ```bash
   python -m src.gnn.pretrain_encoder \
       outputs/graphs_input_only \
       outputs/encoders/hierarchical_temporal_clean \
       --epochs 150
   ```

3. **Retrain EBM:**
   Use new embeddings to train EBM without solution leakage

4. **Re-evaluate:**
   Compare cost gaps between:
   - Old (with leakage): ~0-2%
   - New (clean): TBD (likely higher but more honest)

---

## Code Changes Summary

### Modified Files

1. **`src/gnn/hetero_graph_dataset.py`**
   - Added `_build_flat_compatibility_input_only()` method
   - Modified `HeteroGraphBuilder.__init__()` to accept `use_solution_features`
   - Modified `_to_record()` to conditionally use input-only features
   - Updated `build_hetero_temporal_record()` signature
   - Updated `build_hetero_graph_record()` signature

2. **`src/gnn/build_hetero_graph_dataset.py`**
   - Added `--input-only` CLI flag
   - Passes `use_solution_features=not args.input_only` to builder

### Backward Compatibility

Default behavior unchanged (`use_solution_features=True`). To enable input-only mode, explicitly add `--input-only` flag.

---

## For Presentation

Update your slides to clarify:

**Previous framing (misleading):**
> "Learning-based method that generalizes to new scenarios"

**Correct framing for old results:**
> "Solution-guided hybrid optimization: encoder learns from optimal solution patterns to accelerate future solves"

**Or use new results:**
> "True generalization: encoder learns from problem structure and forecasts alone, without seeing optimal solutions"
