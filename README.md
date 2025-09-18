﻿# Multilayer MILP + GNN Benchmark

This repository bundles three tightly coupled pieces:
- a scenario generator that samples synthetic power-system cases;
- a MILP-based unit commitment solver with optional reporting & plotting; and
- utilities to transform solved scenarios into graph datasets for GNN pipelines.

The code is pure Python (3.10 or newer recommended) built on top of [Pyomo](http://www.pyomo.org/) and the HiGHS LP/MIP solver. The sections below walk you through environment setup and the full workflow end to end.

## 0. Prerequisites

1. **Python**: 3.10+ with a working `python` executable on your PATH.
2. **C/C++ toolchain**: not required, but having one helps if wheels fall back to source builds.
3. **HiGHS solver**: Pyomo can delegate to a system binary or to the `highspy` Python wheel. The easiest path is:
   ```bash
   pip install highspy
   ```
   If you prefer an external binary, ensure the `highs` executable is discoverable via PATH and matches your platform.

## 1. Environment setup

```bash
python -m venv .venv
. .venv/Scripts/activate        # Windows PowerShell: .\.venv\Scripts\Activate.ps1
pip install --upgrade pip
pip install pyomo highspy numpy scipy pandas matplotlib pyarrow tqdm pyyaml torch
```
Optional extras:
- `pip install hvplot` if you want richer plotting backends.
- `pip install black ruff` if you plan to contribute code.
- Install [PyTorch](https://pytorch.org/get-started/locally/) with the build that matches your platform if you prefer GPU acceleration.

## 2. Repository layout primer

```
config/                  # Scenario-space configuration knobs
outputs/                 # Generated scenarios, solve reports, dispatch CSVs, plots, etc.
src/
  generator/             # Scenario generator entry points
  milp/                  # MILP model, single/batch runners, plotting helpers
  gnn/                   # Graph dataset exporter, models, and training utilities
```

Keep everything relative to the project root (`benchmark/` in this checkout). All commands below assume you run them from that directory.

## 3. Generate scenarios

The quickest way to sample a new batch is the convenience wrapper:

```bash
python -m src.generator.run_generator
```

This reads `config/scenario_space.yaml` and writes JSON payloads plus a `manifest.json` under `outputs/scenarios_v1/`. You can customise behaviour by editing the YAML (e.g., horizon length, asset ranges) or by importing `generate_scenarios` from `src.generator.generator_v1` in your own script.

Useful tips:
- Re-run the generator with a different `outputs/<folder>` to keep multiple corpora side-by-side.
- Set `scenario_space.global.target` to control how many scenarios to keep.
- Inspect a sample file with `python -m src.generator.inspect_scenarios outputs/scenarios_v1/scenario_00001.json`.

## 4. Solve scenarios with the MILP

### 4.1 Single scenario

```bash
python -m src.milp.run_milp outputs/scenarios_v1/scenario_00001.json \
    --solver highs \
    --tee \
    --save-json outputs/scenarios_v1/reports/scenario_00001.json \
    --plot --plot-dir outputs/scenarios_v1/plots \
    --export-prefix outputs/scenarios_v1/dispatch/scenario_00001 \
    --export-hdf outputs/scenarios_v1/dispatch/scenario_00001.h5
```

Flags of interest:
- `--solver`: any Pyomo-registered solver (HiGHS by default).
- `--tee`: stream solver output to the terminal.
- `--save-json`: persist the full report (required later for the GNN exporter).
- `--plot`: generate capacity & dispatch visualisations (requires matplotlib).
- `--export-*`: dump time-series CSV/HDF files. HDF export needs pandas + pyarrow.

### 4.2 Batch solving

```bash
python -m src.milp.batch_runner outputs/scenarios_v1 --solver highs --workers 4 --reports-dir outputs/scenarios_v1/reports --dispatch-dir outputs/scenarios_v1/dispatch_batch --plots-dir outputs/scenarios_v1/plots --summary-json outputs/scenarios_v1/batch_summary.json --plot --tee
```

Notes:
- You can mix files and glob patterns in the positional `inputs` (e.g., `outputs/scenarios_v1/scenario_000*.json`).
- Any of `--reports-dir`, `--dispatch-dir`, `--hdf-dir`, or `--plot` triggers capture of high-resolution detail from the MILP.
- `--workers N` toggles a `ProcessPoolExecutor`; keep an eye on RAM usage for large horizons.
- The summary JSON aggregates success/failure counts, objective statistics, and cost totals for quick QA.

Troubleshooting:
- A `FAILED: 'solar_per_site'` message means the JSON predates the solar/wind split. Regenerate reports with the current loaderï¿½all legacy `res_per_site` cases are now auto-split.
- If Pyomo says it cannot locate the solver, confirm `highspy` is installed or add the HiGHS binary folder to PATH.

## 5. Build GNN graph datasets

Once a scenario is solved and you have a detailed JSON report, convert it to an NPZ:

```bash
python -m src.gnn.graph_dataset \
    outputs/scenarios_v1/scenario_00001.json \
    outputs/scenarios_v1/reports/scenario_00001.json \
    outputs/gnn/scenario_00001.npz
```

The exporter expects the report to include the `detail` payload (produced automatically when using `--save-json`, `--reports-dir`, `--dispatch-dir`, etc.). Each NPZ contains:
- `node_static`: per-zone capacities (thermal, solar, wind, storage, demand-response, etc.).
- `node_time`: stacked demand/renewable/hydro/battery state traces.
- `node_labels`: ground-truth dispatch & slack per timestep.
- `edge_*`: topology and flows; `duals_*`: optional dual arrays from the LP relaxation.

To convert an entire directory of scenario/report pairs in one shot:

```bash
python -m src.gnn.build_graph_dataset outputs/scenarios_v1 outputs/scenarios_v1/reports outputs/datasets/graphs
```

The command above mirrors scenario/report stems into `outputs/datasets/graphs`, emitting one compressed NPZ per solved case.

## 6. Train GNN models

With NPZ graph files in hand you can train the baseline dispatch predictors provided in `src/gnn`.

1. **Prepare the dataset index**: Step 5 writes `dataset_index.json` beside the NPZs (default: `outputs/datasets/graphs/dataset_index.json`). If you store datasets elsewhere, update `data.index_path` in the training config accordingly.
2. **Review the training config**: `config/gnn/baseline.yaml` lists the data splits, model backbone (`gcn`, `graphsage`, or `gat`), optimisation hyperparameters, feasibility decoder options, and metric settings. Paths are resolved relative to the repository root unless they start with `./` or `../`, in which case they are relative to the config file.
3. **Launch training**:

```bash
python -m src.gnn.train --config config/gnn/baseline.yaml
```

During training the CLI logs epoch losses and validation metrics, writes the fully-resolved config to the run directory, and saves `best_model.pt`, `final_model.pt`, plus `test_metrics.json` under `outputs/gnn_runs/<run_name>/`. Use `--device cuda` (or set `loop.device: cuda` in the YAML) if you have a GPU-capable PyTorch install.



Each run now also emits: 
- `training_history.json` / `.csv`: epoch-level train/validation metrics with an `is_best` flag for the checkpointed model.
- `training_step_losses.json` / `.csv`: mini-batch loss snapshots logged at the configured interval (default: every `log_every` steps).
- `training_summary.json`: quick metadata overview (best epoch, total steps).
- `test_metrics.json`: nested metric dictionaries (`value` + `details`) for easier downstream analysis.



### 6.1 Analyse GNN vs MILP performance



Open `notebooks/performance_comparison.ipynb` to combine MILP scenario metadata with GNN predictions. The notebook loads speed-up estimates, cost gaps, and feasibility rates using the helpers in `src.analysis.performance`. Speed-up calculations now use the measured MILP solve time stored in each report (falling back to the historical estimates when needed). Update `RUN_DIR` if you want to compare a different experiment.



### 6.2 Inspect training curves



`notebooks/training_visualization.ipynb` reads the history files above and renders epoch-level and step-level diagnostics. If the notebook reports missing logs, re-run training with the updated pipeline to regenerate them.



## 7. Optional tooling & best practices

- **Visual QA**: `src/milp/visualize.py` contains helper functions to overlay capacities and dispatch. The batch runner can auto-save plots if `--plot` is set.
- **Scenario tweaks**: adjust `config/scenario_space.yaml` to change weather regimes, asset distributions, or solver budget guards. The defaults target a moderate-size UC instance solvable in minutes.
- **Reproducibility**: scenario IDs seed the RNG, so keeping the same JSON yields identical profiles and capacities.
- **Performance knobs**: the MILP model supports a `--workers` parallel solve, but also consider tightening horizons or asset counts if solve times spike.
- **Data hygiene**: clear `outputs/` between major runs to avoid mixing corpora with different configuration baselines.

## 8. Frequently asked questions

**Q: Pyomo throws `ApplicationError: No executable found for solver 'highs'`.**
A: Install `highspy` (`pip install highspy`) or download the HiGHS binary and add it to PATH.

**Q: How do I inspect the generated scenarios without solving them?**
A: Use `python -m src.generator.inspect_scenarios outputs/scenarios_v1/scenario_00010.json` to print headline stats.

**Q: Can I add my own solver?**
A: Yesï¿½pass any Pyomo-supported solver via `--solver`. For commercial solvers (CPLEX/Gurobi), ensure licences and bindings are configured.

**Q: Do the NPZs include cost information?**
A: Cost components live in the JSON reports; the NPZ stores structural/time-series tensors. Load the JSON alongside the NPZ in your ML pipeline if you need economics.

---
Happy benchmarking! File issues or ideas to extend the workflowï¿½there is plenty of room for new weather regimes, solver configurations, and richer graph targets.


## 9. License

MIT License





