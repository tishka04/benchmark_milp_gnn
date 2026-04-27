# Coordinating Multilayer Power System Flexibility — MILP / GNN / EBM Benchmark

Companion code for the paper
**"Coordinating Multilayer Power System Flexibility: A Hybrid MILP–GNN–EBM
Framework for Large-Scale Scenario Exploration"** (Coudray & Goutte).

The repository implements every block of the pipeline figure in the paper
(Section 2) plus the deterministic baseline:

| Block | Module | Description |
|------:|--------|-------------|
| 1 | `src/generator/`         | Diversity-maximising scenario generator (LHS + greedy *k*-center) |
| 2 | `src/milp/`              | Multi-layer MILP oracle (Pyomo + HiGHS) and `LPWorkerTwoStage` cascade |
| 3 | `src/gnn/`               | Hierarchical heterogeneous temporal graphs and HTE encoder |
| 4 | `src/gnn/`               | Topology-consistent GNN encoder (zone embeddings) |
| 5 | `src/ebm/`               | Conditional graph EBM with normalized temporal Langevin sampler |
| 6 | `src/ebm/feasibility.py` | Hierarchical feasibility decoder (merit-order projection) |
| 7 | `src/eval/`              | Pipeline runner (Stage 1–5 LP cascade) and metrics |
| 8 | `src/heuristics/`        | **RH-MO+LP** rolling-horizon merit-order baseline |
| 9 | `src/analysis/`          | Criticality index, HDBSCAN clustering, paper figures |

Pure Python (3.10 +). Solver: [HiGHS](https://highs.dev/) via `highspy`.
Deep-learning stack: PyTorch + PyTorch Geometric.

---

## 0. Quick start (single-command paper reproduction)

```powershell
# Windows PowerShell, from the repo root benchmark_milp_gnn/
python -m venv .venv
. .\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt

# Reproduce the three criticality families used in the paper
python -m src.generator.run_generator -v hard -o outputs/high_criticality_scenarios
python -m src.milp.batch_runner outputs/high_criticality_scenarios `
    --solver highs --workers 4 `
    --reports-dir outputs/high_criticality_scenarios/reports `
    --summary-json outputs/high_criticality_scenarios/batch_summary.json

# Heuristic baseline (RH-MO+LP) on the same scenarios
python -m src.heuristics.runner outputs/high_criticality_scenarios `
    -W 6 -o outputs/rhmo_high.json

# Full MILP-GNN-EBM pipeline (requires trained checkpoints; see §6)
python -m src.eval.pipeline_runner --family high
```

> **GPU users**: install a CUDA wheel of `torch` *before* running
> `pip install -r requirements.txt`, and pull the matching
> `torch-scatter` wheel from
> `https://data.pyg.org/whl/torch-<ver>+<cuda>.html`.

---

## 1. Prerequisites

1. **Python 3.10 +** with `python` on `PATH`.
2. **HiGHS solver** — bundled via `pip install highspy` (already in
   `requirements.txt`). For an external binary, ensure `highs` is on
   `PATH`.
3. **Build tools** (Linux/Mac): a recent C++ compiler is helpful when
   wheels for `torch-scatter` or `hdbscan` fall back to source.
4. **Disk**: ≈ 5 GB for a full 200-scenarios-per-family run with detailed
   JSON reports.

---

## 2. Environment setup

```bash
# Use Python 3.11 (torch < 2.5 has no wheels for 3.12+)
py -3.11 -m venv .venv             # Windows
# python3.11 -m venv .venv         # Linux / macOS

source .venv/bin/activate           # Linux / macOS
# .\.venv\Scripts\Activate.ps1      # Windows PowerShell

python -m pip install --upgrade pip
pip install -r requirements.txt
pip install -r requirements-pyg.txt   # torch-scatter from the PyG wheel index
```

The PyG companion wheels (`torch-scatter`) are split into a separate
file because they need a custom index URL. CPU build is the default;
for CUDA, edit the `--find-links` line in `requirements-pyg.txt` to
match your torch / CUDA version (see comments inside).

Verify the install:

```bash
python -c "import torch, pyomo, highspy, torch_geometric, torch_scatter, hdbscan, umap; print('OK')"
```

---

## 3. Repository layout

```
benchmark_milp_gnn/
├── config/                    # Scenario-space and GNN training configs (YAML)
├── configs/                   # EBM training configs
├── outputs/                   # Generated scenarios, reports, results, plots
│   ├── low_criticality_scenarios/
│   ├── medium_criticality_scenarios/
│   └── high_criticality_scenarios/
├── src/
│   ├── generator/             # Scenario sampler (Block 1)
│   ├── milp/                  # MILP oracle + LP worker cascade (Block 2/7)
│   ├── gnn/                   # Hetero temporal graphs + HTE encoder (Block 3/4)
│   ├── ebm/                   # Conditional EBM + Langevin sampler (Block 5/6)
│   ├── heuristics/            # RH-MO+LP rolling-horizon baseline (Block 8)
│   ├── eval/                  # Full-pipeline runner and metrics
│   └── analysis/              # Criticality index, clustering, figures
├── scripts/                   # One-off paper-reproduction scripts
├── notebooks/                 # Result analysis / visualisation
├── pdfs/main.tex              # Paper source
├── requirements.txt
└── README.md
```

All commands assume the repo root (`benchmark_milp_gnn/`) as the working
directory.

---

## 4. Generate scenarios (Block 1)

Three criticality families are used in the paper. The generator
implements LHS sampling, a budget guard, and greedy *k*-center
selection (paper §2.1).

```bash
# Low criticality
python -m src.generator.run_generator -v v2 -o outputs/low_criticality_scenarios

# Medium / hard - controlled by config/scenario_space[_hard].yaml
python -m src.generator.run_generator -v v2  -o outputs/medium_criticality_scenarios
python -m src.generator.run_generator -v hard -o outputs/high_criticality_scenarios
```

Each output folder contains `scenario_*.json` plus a `manifest.json`.

### 4.1 Criticality index & clustering (paper §2.1.4)

```bash
python -m src.analysis.criticality_index outputs/high_criticality_scenarios -a 0.6
python -m src.analysis.criticality_clustering \
    outputs/high_criticality_scenarios/criticality_results.json \
    --min-cluster-size 30 --cluster-epsilon 0.5 --n-components 11
python -m src.analysis.visualize_clusters \
    outputs/high_criticality_scenarios/criticality_results.json \
    -o outputs/high_criticality_scenarios/clusters_viz.png
```

---

## 5. MILP oracle (Block 2 — ground truth)

### 5.1 Single scenario

```bash
python -m src.milp.run_milp outputs/high_criticality_scenarios/scenario_00001.json \
    --solver highs --tee \
    --save-json outputs/high_criticality_scenarios/reports/scenario_00001.json \
    --plot --plot-dir outputs/high_criticality_scenarios/plots
```

### 5.2 Whole family in parallel

```bash
python -m src.milp.batch_runner outputs/high_criticality_scenarios \
    --solver highs --workers 4 \
    --reports-dir outputs/high_criticality_scenarios/reports \
    --dispatch-dir outputs/high_criticality_scenarios/dispatch_batch \
    --summary-json outputs/high_criticality_scenarios/batch_summary.json
```

The `--reports-dir` JSONs contain the *full* MILP solution and are the
ground truth used by every other block.

---

## 6. MILP-GNN-EBM pipeline (Blocks 3–7)

### 6.1 Build hierarchical temporal graphs (Block 3)

```bash
python -m src.gnn.build_hetero_graph_dataset \
    outputs/high_criticality_scenarios outputs/graphs/hetero_temporal_high \
    --temporal
```

### 6.2 Train the HTE encoder (Block 4)

```bash
python -m src.gnn.train_temporal_hetero \
    --data-dir outputs/graphs/hetero_temporal_high \
    --model-type hgt --hidden-dim 128 --num-layers 2 --num-heads 8 \
    --epochs 60 --batch-size 8 --lr 5e-4
```

Outputs land in `outputs/encoders/hierarchical_temporal_v3/`.

### 6.3 Train the conditional EBM (Block 5)

```bash
python -m src.ebm.train_v3 --config configs/ebm_config.yaml
```

Best checkpoint is saved to `outputs/ebm_models/ebm_v3/ebm_v3_final.pt`.

### 6.4 Run the full pipeline + LP worker cascade (Block 6 + 7)

```bash
python -m src.eval.pipeline_runner --family high
```

The runner performs Langevin sampling → feasibility decoding → the same
five-stage LP cascade (`hard_fix → repair_20 → repair_wider →
soft_relaxation → round_and_refix`) used by the heuristic baseline.

---

## 7. Rolling-horizon heuristic baseline (Block 8 — RH-MO+LP)

Deterministic learning-free baseline described in paper §6
(*RH-MO+LP: Rolling-Horizon Merit-Order Heuristic with LP
Reconstruction*). See `src/heuristics/README.md` for algorithm details.

```bash
# Single scenario, with verbose timing
python -m src.heuristics.runner \
    outputs/high_criticality_scenarios/scenario_00001.json -W 6 -v

# Whole family, with custom LP-cascade depth
python -m src.heuristics.runner outputs/medium_criticality_scenarios \
    -W 8 -o outputs/rhmo_medium.json --max-stages 5
```

The output JSON has one record per scenario: timings (`time_heuristic`,
`time_lp_solve`, `time_total`), `lp_objective`, `lp_stage_used`,
`lp_slack`, and `binary_active_fractions`.

---

## 8. Reproducing the paper experiments

The exact commands used to produce Tables/Figures of Section 6 are in
`scripts/`. The recommended end-to-end recipe per criticality family is:

```bash
FAMILY=high_criticality_scenarios

# (1) Generate scenarios
python -m src.generator.run_generator -v hard -o outputs/$FAMILY

# (2) MILP oracle
python -m src.milp.batch_runner outputs/$FAMILY --workers 4 \
    --reports-dir outputs/$FAMILY/reports \
    --summary-json outputs/$FAMILY/batch_summary.json

# (3) Graphs + encoder + EBM (skip if checkpoints already present)
python -m src.gnn.build_hetero_graph_dataset outputs/$FAMILY \
    outputs/graphs/hetero_temporal_${FAMILY%_scenarios} --temporal
python -m src.gnn.train_temporal_hetero --data-dir \
    outputs/graphs/hetero_temporal_${FAMILY%_scenarios}
python -m src.ebm.train_v3 --config configs/ebm_config.yaml

# (4) Full pipeline evaluation
python -m src.eval.pipeline_runner --family ${FAMILY%_criticality_scenarios}

# (5) Heuristic baseline on the same scenarios
python -m src.heuristics.runner outputs/$FAMILY \
    -o outputs/rhmo_${FAMILY%_criticality_scenarios}.json -W 6
```

Then open the analysis notebooks:

- `notebooks/performance_comparison.ipynb` — speed-up, optimality gap,
  feasibility rate vs. MILP and vs. RH-MO+LP.
- `notebooks/training_visualization.ipynb` — encoder / EBM training
  curves.
- `notebooks/criticality_*` — figures of paper §2.1.4.

---

## 9. Reproducibility & determinism

- Scenario IDs seed every random draw (sampler, profile generator).
  Re-running with the same JSON yields bit-identical inputs.
- The MILP is deterministic at the solver level (HiGHS with default
  parameters).
- The heuristic baseline (`src/heuristics/`) is fully deterministic
  given its `HeuristicConfig`.
- Encoder + EBM training fix `torch.manual_seed`; for fully reproducible
  GPU runs set `CUBLAS_WORKSPACE_CONFIG=:4096:8` and pass
  `--deterministic` where supported.

---

## 10. FAQ

**Q. `ApplicationError: No executable found for solver 'highs'`.**
A. `pip install highspy`, or place the HiGHS binary on `PATH`.

**Q. `ModuleNotFoundError: No module named 'numpy'` (or `torch`).**
A. The active interpreter is not the venv. Re-activate with
`. .\.venv\Scripts\Activate.ps1` (Windows) or
`source .venv/bin/activate` (Linux/macOS) and re-run.

**Q. `torch-scatter` fails to install.**
A. Use the prebuilt wheel matching your torch + CUDA version from
`https://data.pyg.org/whl/torch-<ver>+<cuda>.html`.

**Q. Inspect a scenario without solving it.**
A. `python -m src.generator.inspect_scenarios outputs/<folder>/scenario_00010.json`.

**Q. Add a commercial solver (Gurobi / CPLEX).**
A. Install bindings, then pass `--solver gurobi` (or `cplex`) to
`run_milp` / `batch_runner`. The LP-worker cascade also accepts any
Pyomo-supported solver via `--solver`.

---

## 11. Citation

If you use this code, please cite:

```bibtex
@article{coudray2026milpgnnebm,
  title   = {Coordinating Multilayer Power System Flexibility:
             A Hybrid MILP--GNN--EBM Framework for
             Large-Scale Scenario Exploration},
  author  = {Coudray, Th{\'e}otime and Goutte, St{\'e}phane},
  journal = {Working paper, UVSQ -- UMI SOURCE},
  year    = {2026}
}
```

## 12. License

MIT — see `LICENSE`.
