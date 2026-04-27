# RH-MO+LP — Rolling-Horizon Merit-Order Heuristic with LP Reconstruction

Deterministic, learning-free baseline that mirrors the two-stage
structure of the MILP-GNN-EBM pipeline:

- **Stage A — Binary schedule** (`rolling_horizon.py`)
  A handcrafted rolling-horizon merit-order policy produces the same
  `[Z, T, 7]` binary tensor `u^heur` consumed by the LP worker, with
  channel order matching `src.ebm.feasibility.FeasiblePlan.to_tensor`:

  | idx | meaning                  |
  |-----|--------------------------|
  | 0   | battery charging         |
  | 1   | battery discharging      |
  | 2   | pumped-hydro charging    |
  | 3   | pumped-hydro discharging |
  | 4   | demand-response active   |
  | 5   | thermal start-up (off→on)|
  | 6   | thermal commitment       |

- **Stage B — Continuous dispatch** (`runner.py`)
  Delegates to `src.milp.lp_worker_two_stage.LPWorkerTwoStage`, the
  exact LP cascade used by the full pipeline (hard-fix → repair-K →
  wider repair → soft relaxation → round-and-refix). A pass-through
  feasibility decoder builds a warm-start to accelerate the LP solve
  but does **not** modify the heuristic binaries.

The heuristic does not consume MILP labels, GNN embeddings, or EBM
outputs.

## Algorithmic outline (Stage A)

For each time step `t` and look-ahead window `[t, t+W]`:

1. Compute residual load `RL_z,t = D - P^nuc_must - P^ror - P^solar - P^wind`.
2. Resolve deficits in merit order:
   nuclear headroom → reservoir hydro (with reserve `β·max(0, RL^max - RL_t)·dt`)
   → storage discharge (only if `RL_t ≥ q70(RL_window)`)
   → imports (anchor zone)
   → thermal commitment (anti-flapping via `H_keep`)
   → DR (only if `RL_t ≥ q80` and budget hours remain).
3. Resolve surplus: charge storage toward future-stress-driven SOC
   target, then allow thermal decommitment when `RL^max < P^min,th`.

State (`SOC_b`, `SOC_p`, `hydro_e`, `thermal_on_prev`, `dr_used_h`,
`thermal_keep_until`) is propagated forward each step.

## Usage

### From Python

```python
from src.heuristics import HeuristicRunner, HeuristicConfig

runner = HeuristicRunner(
    heuristic_config=HeuristicConfig(window=6),
    solver_name="appsi_highs",
    warm_start=True,
)

# Single scenario
result = runner.evaluate_scenario(
    Path("outputs/high_criticality_scenarios/scenario_00001.json"),
    family="high",
)

# Whole family
results = runner.evaluate_family(
    Path("outputs/high_criticality_scenarios"),
    family_name="high",
)
HeuristicRunner.save_results(results, Path("outputs/rhmo_high.json"))
```

### From CLI

```bash
# Single scenario
python -m src.heuristics.runner outputs/high_criticality_scenarios/scenario_00001.json -W 6 -v

# Whole family
python -m src.heuristics.runner outputs/medium_criticality_scenarios \
    --output outputs/rhmo_medium.json -W 8
```

## Reproducibility

The heuristic is fully deterministic given a `HeuristicConfig`. The
LP-worker results inherit reproducibility properties from the chosen
solver (`appsi_highs` by default).
