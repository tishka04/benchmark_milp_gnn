# Demand response: forward load shifting

The MILP built by `src/milp/model.py` now uses formulation
`load_shifting_v1`. Each zone defers consumption and recovers the same energy
later within the optimisation horizon. This changes the feasible set relative
to the former shedding model; historical results are not results of this model.

## Physical formulation

For each zone and period:

- `dr_shed` is deferred power in MW (the legacy variable name is retained).
- `dr_recover` is additional consumption used to recover deferred load, in MW.
- `dr_rebound` is outstanding deferred energy at the end of the period, in MWh.
- `dr_active = 1` allows deferral; `dr_active = 0` allows recovery or inactivity.

With initial backlog zero, the lossless energy balance is

```text
E[t] = E[t-1] + dt_hours * (dr_shed[t] - dr_recover[t])
E[t] >= 0
E[last] = 0
dt_hours * dr_recover[t] <= E[t-1]
```

The electricity balance includes `dr_shed` on the supply side and
`dr_recover` on the demand side. Therefore the demand after shifting is
`demand - dr_shed + dr_recover`. Summing the energy balance gives exactly
`sum(dt_hours * dr_shed) = sum(dt_hours * dr_recover)` in each zone, up to
solver feasibility tolerances.

Recovery is bounded by

```text
dr_recover[z,t] <= dr_limit[z,t] * dr_recovery_limit_factor * (1 - dr_active[z,t])
```

The factor defaults to 1.0 and may be specified as a finite non-negative number
in the scenario JSON's `tech.dr_recovery_limit_factor`, or on `ScenarioData`.
It is not currently a sampled parameter of the scenario generator. Zero
disables recovery and consequently disables deferral. Zones with no DR assets
have zero DR capacity.

The existing tier costs are charged on deferred power; recovered load must be
supplied through the electricity balance and incurs the corresponding supply
cost. Recovery is not charged a second DR tier fee. The objective's existing
time-step cost convention has not been changed in this revision.

This is **postponement only**: no pre-consumption, no initial debt, no debt left
after the horizon, and no simultaneous deferral and recovery in the integer
model. The LP relaxation retains energy conservation and causality, but relaxes
the binary exclusion as usual. There is no separate maximum delay or daily
reset inside a multi-day horizon. `unserved` still represents involuntary
unserved load and remains penalised separately by VOLL.

## Inputs retained for compatibility

`dr_rebound_decay` and `dr_rebound_tolerance` remain readable in old scenario
files / `ScenarioData`, but **do not enter the new MILP**. Energy no longer
decays, and terminal debt is exactly zero regardless of their values.

The existing deferral capacity, tiered costs, minimum activation power and
deferral ramp constraints remain. As before, `dr_max_events` counts active
deferral **periods**, not event starts, and `dr_min_duration` is not enforced.
Those event-duration semantics are outside this formulation change.

## Outputs and downstream use

The standard solve reports include `dr_formulation: "load_shifting_v1"`.
Detailed JSON and zone/system CSV exports include:

| Field | Meaning | Unit |
| --- | --- | --- |
| `demand_response` | Deferred power, retaining the historical key | MW |
| `dr_recover` | Recovered consumption | MW |
| `dr_rebound` | Outstanding energy | MWh |
| `demand_response_net` | Deferred minus recovered power | MW |

System CSV names have a `_total` suffix. Plots show the demand after shifting
and both deferred and recovered power. Binary-solution exports also include
these fields. The exact LP worker and MILP warm starts carry `dr_recover`.

## GNN, latent EBM and inference pipeline

The source pipeline now propagates recovery through datasets, dispatch heads,
decoders, LP warm starts and evaluation. No training or scenario optimisation
was launched for this migration.

| Interface | Load-shifting schema |
| --- | --- |
| Flat/temporal supervised dispatch labels | 14 channels; `dr_recover` appended at index 13 |
| DispatchGNN | 12 channels; `dr_recover` appended at index 11 |
| EBM binary decisions | 7 channels, unchanged; DR activation remains deferral only |
| UC autoencoder / JEPA / latent EBM decisions | 8 channels including import mode, unchanged |
| Flat graph temporal features | 20 channels; deferral/recovery capacities appended at indices 18/19 |

The extra flat input features are scenario capacities, not observed recovery
targets. Input-only heterogeneous graphs carry the recovery factor, deferral
capacity, activation-period budget and ramp limit. Their DR temporal edges
connect consecutive periods to represent the lossless deferred-energy state.
Graph topology keeps heterogeneous spatial edges separate from the zone-only
compatibility fields.

`src/demand_response.py` centralises the formulation marker, label validation,
trajectory diagnostics and a solver-free scheduling heuristic. Label extraction
requires the complete scenario horizon: silently cropping or padding a trajectory
could destroy terminal repayment. Training data with different horizons must be
prepared with the corresponding horizon configuration.

The hierarchical decoder and rolling-horizon heuristic reserve future recovery
capacity before deferring demand. They currently reserve only forecast surplus
solar, wind and run-of-river production; deferral is conservatively capped by the
ramp limit. This heuristic can therefore miss shifts that an exact LP/MILP can
serve using other resources. It supplies a warm start, not a global UC feasibility
certificate. The exact worker uses the full MILP formulation, including recovery.

The per-period GNN balance decoder includes recovery on the consumption side,
but cannot enforce temporal conservation by itself. Whole-horizon diagnostics
report deferred/recovered MWh, maximum per-zone terminal debt, causality errors,
simultaneous actions and backlog residuals. When activation and scenario limits
are available (in particular in LP-worker results), they also check capacity,
activation, period budget and ramp constraints. `dr_feasible` refers only to the
DR constraints checked, not to the complete UC model.

Raw DispatchGNN predictions have status `predicted_unverified` and an unknown
objective (`NaN`). A forward pass does not certify optimality or physical
feasibility. Evaluations reject LP candidates whose DR diagnostics fail.

## Artifact compatibility and future runs

Reports, graphs, embeddings, labels, checkpoints and pipeline results carry
`dr_formulation: "load_shifting_v1"`. The migrated loaders reject historical or
unversioned artifacts. Do **not** add a new marker to an old dataset/checkpoint:
its labels or learned parameters still correspond to the old problem.

Default UC-JEPA, DispatchGNN, HTE and EBM-v3 output paths use
`outputs/load_shifting_v1/`. The baseline GNN configuration uses the same
namespace. Existing files are not moved. Explicit paths supplied by notebooks
or CLI arguments still override defaults; point those to the new artifacts.
Scenario input files and train/validation/test assignments can be reused.

When computation is authorised, proceed in dependency order:

1. Resolve training/validation MILPs and test references under the new model.
2. Rebuild graphs, embeddings and labels from the new data. The changed HTE
   feature/edge schema requires regenerating its encoder and embeddings too.
3. Train the GNN and the UC autoencoder, JEPA and latent EBM in dependency order.
4. Re-run test inference and baselines, then recompute comparison metrics using
   only references and results from this formulation.

## Validation

Solver-free pipeline checks (synthetic fixtures and small forward passes only;
solver creation is explicitly blocked in this test class):

```powershell
.\.venv\Scripts\python.exe -B -m unittest discover -s tests -p test_pipeline_load_shifting.py -v
```

The separate `tests/test_milp_load_shifting.py` regression suite **does launch
HiGHS solves** and was not run during the pipeline migration. It checks energy
conservation at hourly and subhourly resolutions, genuine increased consumption
on recovery, causality, terminal repayment, absence of decay, exclusion of
simultaneous actions, recovery capacity, zero DR, per-zone conservation, LP and
fixed-binary projection behaviour, exports and warm starts.

Passing the solver-free checks validates interfaces and local accounting;
numerical end-to-end performance remains to be checked in the future runs.

Local validation: 22 solver-free tests passed, Python syntax was checked, and
`git diff --check` passed. In this Windows virtual environment, loading the
optional PyArrow extension after Torch caused a native access violation.
The test module initialises pandas before Torch, which allowed the full suite
to complete. Installed dependencies were not changed. For future local runs,
start a fresh interpreter/kernel and import pandas before Torch; this is an
environment workaround, not a load-shifting model constraint.
