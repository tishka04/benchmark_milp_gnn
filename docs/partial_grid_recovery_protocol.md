# Partial-grid recovery under contingencies — experimental protocol

Operationalizes supervisor comment #3:

> *"Does a smaller part of a grid (i.e. a zone or region) solved means that we can
> recover the whole grid when a contingency happen? Identify if a recovery from a
> part of the grid is possible — i.e. whether we can avoid cascading effects with
> solving only a critical part of the grid."*

The goal of this document is to fix definitions, baselines, metrics, and
expected outputs **before any implementation**. Once the supervisor agrees on
the protocol, the protocol becomes a rectangular workload that can be split
across several runs.

---

## 1. Question, formalized

Let `G = (Z, L)` denote the full grid (zones `Z`, transmission lines / interties
`L`). Let `S* ⊂ Z` be a *critical zone subset* (selection rule below).

**H0 (null hypothesis we want to falsify).** Solving UC only over `S*` and
filling the rest of the grid with a cheap default policy yields, after a
single-line N-1 contingency, a system state with a load-shed and post-fault
overload comparable to the full-grid UC solution.

If H0 holds even on a non-trivial subset of scenarios, then *partial-grid
solving is contingency-recoverable*: the operator can afford to solve only
`S*` quickly (with our pipeline) and still survive the worst N-1 fault.

If H0 fails systematically, we have evidence for cascading risk and the
pipeline cannot replace full-grid UC.

## 2. Critical zone subset `S*`

We propose three selection rules so we can tell whether the result depends
on the rule:

1. **`top-k criticality`** — rank zones by a per-zone criticality proxy
   (peak load / firm capacity, or a thermal-flexibility-deficit score derived
   from the A/B components in `src/analysis/criticality_index.py`) and keep
   the top `k = ⌈p · |Z|⌉` for `p ∈ {0.20, 0.40, 0.60}`.
2. **`graph centrality`** — rank zones by betweenness or by line-flow
   participation factor in the base case; same fractions.
3. **`random baseline`** — uniformly random subset, same `|S*|`.
   Sanity check that the criticality-driven selection actually beats random.

For each rule, `S*` is precomputed *before* solving — no peeking at the
post-contingency outcome.

## 3. Baselines / arms

For each scenario `s` we compute four arms:

| Arm | Commitment on `S*` | Commitment on `Z \ S*` | Cost |
|-----|--------------------|------------------------|------|
| A. **Full MILP**           | full-grid UC MILP                          | full-grid UC MILP                          | reference |
| B. **Partial MILP + fill** | UC MILP solved on the `S*` sub-problem only | heuristic fill (e.g. merit-order LP, or "all-on for must-runs") | cheap |
| C. **Pipeline (full)**     | pipeline output                             | pipeline output                             | already evaluated |
| D. **Pipeline-on-`S*` + fill** | pipeline restricted to `S*`            | heuristic fill                              | cheap |

Arm A is the gold standard, arm B isolates the *sub-region MILP* effect
without the pipeline, arms C and D isolate the *pipeline* effect.

The **heuristic fill** must be specified explicitly. Two options:
- (i) Run an LP economic dispatch on `Z \ S*` with all thermal units forced
  off, only must-runs and renewables on. Cheap, sometimes infeasible.
- (ii) Force unit commitments to their reference (single-zone, no-network)
  myopic merit order. More feasible, less optimal.
We propose (ii) for the main run, (i) for a robustness check.

## 4. Contingency set and post-fault evaluation

For every arm we freeze the resulting commitment vector `u` and run a
**post-fault LP redispatch** under each line in a contingency set `C`. Two
contingency sets:

- `C_N1_lines` — single-line outage for every line in `L` (full N-1).
- `C_critical` — the top-`k` lines by base-case loading, to bound runtime.

The redispatch solves an LP with:
- generation bounded by `u` (commitments fixed; only continuous dispatch
  variables free),
- DC power flow on `L \ {ℓ}`,
- load shedding allowed at penalty cost `VOLL`,
- line overload penalty `c_ovl` (or hard limit, configurable).

The redispatch already exists in lighter form: see how we evaluate the
pipeline's projected dispatch — extend it to handle a single line outage and
report the new flows.

## 5. Metrics

Per (scenario, arm, contingency) row:
- `lol_mwh` — total load shed.
- `n_overloaded_lines` — count of post-fault lines above thermal limit.
- `cascade_risk_score` — sum of `(flow / limit − 1)+` over post-fault lines
  (proxy for cascade probability).
- `redispatch_cost` — LP objective value.
- `feasible` — boolean (LP returned a finite objective without exceeding
  shed cap).

Aggregations per (scenario, arm):
- `worst_case_lol_mwh = max_c lol_mwh`.
- `mean_lol_mwh = mean_c lol_mwh`.
- `n_critical_contingencies = |{c : lol_mwh > τ}|` for some threshold `τ`.

Aggregations per arm (across scenarios):
- median worst-case LOL, % scenarios with `worst_case_lol_mwh = 0`,
  cascade-risk distribution, paired comparisons vs arm A.

## 6. Statistical comparison

For each arm pair `(X, A)` and each metric, paired-by-scenario tests:
- Wilcoxon signed-rank (non-parametric, robust).
- Bootstrap CI on median difference.
- Family-wise correction across the 3 selection rules × 2 contingency sets.

The headline figures are:
1. **Pareto plot**: x = mean solve time per arm; y = median worst-case LOL.
2. **Per-criticality stratification**: split scenarios into criticality
   deciles (we already have those) and plot worst-case LOL vs criticality
   for each arm. This directly answers "does partial-grid solving still
   recover the system as criticity rises?"

## 7. Scenario set and compute envelope

- Reuse the **300 scenarios from `latent_vs_grid_milp_120_caps_k1`** (we
  already have the pipeline runs and reference MILPs). The new compute is
  arms B and D, plus the post-fault LP loop.
- Estimated cost (rough): for each scenario, `|C_N1_lines| ≈ 20-60` LPs at
  ~1 s each → ~1 min/scenario. 300 scenarios × 4 arms = ~20 hours of LP
  redispatch in addition to whatever MILP partial-solves cost.
- A pilot run on **10 scenarios per family (30 total)** is recommended first.

## 8. Implementation plan (when greenlit)

1. **Add zone-subset solver** at `src/milp/lp_worker_two_stage.py`: a
   `solve_partial(scenario, zones_keep)` method that drops UC binaries and
   line / power-balance constraints involving `Z \ zones_keep`, treats
   `Z \ zones_keep` net injection as a fixed parameter from the heuristic
   fill, and returns the same `MIPSummary`-shaped object.
2. **Add post-fault LP redispatch** at `src/eval/contingency_redispatch.py`:
   given a base solution and a single-line outage, solve a DC-LP for
   redispatch with shed and overload penalties, return the metrics in §5.
3. **Wire arms B / D** into a new launcher
   `scripts/run_partial_grid_recovery.py` that takes
   `--scenarios-dir`, `--reports-dir`, `--pipeline-pkl`, `--selection-rule`,
   `--keep-fraction`, `--contingency-set`, and writes
   `outputs/uc_jepa/partial_recovery/<rule>_<frac>/per_scenario.csv`.
4. **Add aggregation + plotting notebook** mirroring the
   `criticality_drivers_analysis` structure.

## 9. Open questions for the supervisor

- Is the LP redispatch model (DC, with shed at VOLL and overload penalty)
  acceptable, or does the contingency model need AC / SCED constraints?
- Should we simulate **cascades** (sequential outages) rather than single
  N-1, since the supervisor explicitly mentioned "cascading effects"? If
  yes, we need a cascade-propagation simulator (one outage triggers
  overload that triggers the next outage, with a configurable threshold
  and a cap on iterations).
- What threshold `τ` defines a "non-recoverable" contingency for the
  binary outcome metric?
- Compute envelope: is the pilot (30 scenarios × full N-1) a fair first
  pass, or do we need to commit to the full 300 from the start?

Once these four points are settled, the protocol is fully specified and
implementation is mechanical.

---

## 10. Implementation (as built)

The protocol is implemented in three modules plus a launcher. Design decisions
that were resolved during implementation are recorded here.

### 10.1 Resolved decisions

- **Network / redispatch model — approximate DC power flow.** The MILP oracle is
  a *transport* model (`m.flow[l,t]` hard-bounded by `±capacity_mw`, no
  reactances). A pure transport model cannot express post-fault overloads or
  cascades, so the redispatch assigns each line a reactance `x_l ∝ distance_km`
  (standard when only line lengths are known) and enforces
  `flow[l,t] = B_l · (θ_from − θ_to)` with `B_l = mean_len / x_l`. Because DC
  flows depend only on *relative* susceptances, the absolute scaling cancels.
  The slack bus is the first zone. Thermal limits are **soft** with an overload
  penalty `c_ovl` so `n_overloaded_lines` and `cascade_risk_score` are
  observable.
- **Frozen commitment = thermal commitment (SCED).** Only `u_thermal` (which
  thermal plants are online) is frozen in the redispatch; storage, DR, and
  cross-border exchange are treated as fast, re-dispatchable continuous
  flexibility (standard security-constrained economic dispatch). Freezing the
  storage/DR mode binaries as well (`--fix-modes`) is supported but can render
  imperfect learned plans spuriously infeasible. Remaining integers are relaxed
  so the redispatch is a pure LP.
- **Contingency dynamics — single N-1 (headline) + cascade robustness.** The
  N-1 sweep solves one redispatch per single-line outage (plus a no-outage base
  case). The cascade pass seeds from the worst single outage (max post-fault
  load shed), then trips any line whose overload exceeds `--cascade-threshold`
  and re-dispatches, up to `--cascade-max-iter` rounds.
- **Arms.** `A` full MILP, `B` partial MILP (binaries on `Z\S*` fixed to the
  heuristic fill, `S*` optimised), `C` pipeline (full), `D` pipeline-on-`S*` +
  fill. `A`/`C` are `S*`-independent and computed once per scenario; `B`/`D`
  are computed per `(rule, keep_fraction)`.
- **Heuristic fill.** Thermal all-on where capacity exists, storage idle, no DR
  (the most-feasible "all-on for must-runs" reference of §3).
- **`S*` selection rules.** `top_k` (per-zone peak/firm), `graph_centrality`
  (capacity-weighted line degree), `random` (seeded).

### 10.2 Code

- `src/eval/contingency_redispatch.py` — DC-PF redispatch LP, `run_n1_sweep`,
  `run_cascade`, and the `Commitment` container.
- `src/eval/partial_grid.py` — `S*` selection, heuristic fill, and per-arm
  commitment producers (`commitment_full_milp`, `commitment_partial_milp`,
  `commitment_from_binary_tensor`, `commitment_pipeline_on_critical`).
- `src/eval/pipeline_runner.py` — `PipelineRunner.extract_binary_candidate`
  returns the pipeline's selected commitment binaries (arms C/D).
- `scripts/run_partial_grid_recovery.py` — launcher and aggregation.

### 10.3 How to run

Pilot, MILP arms only (no GPU/checkpoints needed), 10 scenarios/family:

```powershell
python -m scripts.run_partial_grid_recovery `
  --families low medium high --scenarios-per-family 10 `
  --arms A B --selection-rule top_k --keep-fraction 0.4 `
  --contingency-set critical --contingency-top-k 12 `
  --milp-time-limit 120 --redispatch-time-limit 120 `
  --cascade --cascade-max-iter 4 `
  --out outputs/uc_jepa/partial_recovery
```

Full pilot, all four arms (adds the learned pipeline; `--device cuda` if a GPU
is available):

```powershell
python -m scripts.run_partial_grid_recovery `
  --families low medium high --scenarios-per-family 10 `
  --arms A B C D --selection-rule top_k --keep-fraction 0.4 `
  --contingency-set critical --contingency-top-k 12 `
  --milp-time-limit 120 --redispatch-time-limit 120 `
  --cascade --cascade-max-iter 4 `
  --run-pipeline --device cpu --pipeline-samples 5 `
  --out outputs/uc_jepa/partial_recovery
```

Robustness sweep over selection rules and fractions (§2, §6):

```powershell
python -m scripts.run_partial_grid_recovery `
  --families low medium high --scenarios-per-family 10 `
  --arms A B C D --run-pipeline --device cpu `
  --selection-rule top_k graph_centrality random `
  --keep-fraction 0.2 0.4 0.6 `
  --out outputs/uc_jepa/partial_recovery_sweep
```

Scale to the full eval set with `--scenarios-per-family 100` and
`--contingency-set n1` (full single-line N-1) once the pilot looks sane.

**Per-scenario time budget & resume.** Hard scenarios accumulate many per-solve
time limits (≈ `contingency-top-k × arms × redispatch-time-limit` + commitments),
so a single scenario can run for an hour or more. `--scenario-time-limit`
(default `1200` s) caps the wall-clock spent on one scenario: remaining
contingencies/arms are skipped once the budget passes and each remaining solve is
capped to the leftover budget. Results flush to CSV after every scenario and
`--resume` (default on) skips scenarios already in the output CSVs, so you can
Ctrl+C a stuck run and relaunch the *same command* to continue where it stopped:

```powershell
& "<repo>\.venv\Scripts\python.exe" -m scripts.run_partial_grid_recovery `
  --families low medium high --scenarios-per-family 10 --arms A B C D `
  --run-pipeline --device cuda --pipeline-samples 5 `
  --milp-time-limit 300 --redispatch-time-limit 120 `
  --scenario-time-limit 1200 --cascade --cascade-max-iter 4 `
  --out outputs/uc_jepa/partial_recovery
```

### 10.4 Outputs

Written to `--out`:

- `per_contingency.csv` — one row per `(scenario, arm, rule, frac, outage)`:
  `feasible`, `lol_mwh`, `dr_mwh`, `overload_mwh`, `n_overloaded_lines`,
  `cascade_risk_score`, `redispatch_cost`, `solve_seconds`.
- `per_scenario_arm.csv` — per-arm aggregates: `worst_case_lol_mwh`,
  `mean_lol_mwh`, `n_critical_contingencies`, `feasible_fraction`,
  `max_cascade_risk`, `base_lol_mwh`.
- `cascade.csv` — per-round cascade trace (`round`, `n_tripped`,
  `tripped_lines`, `lol_mwh`, `n_overloaded_lines`).
- `headline_by_arm.csv` — grouped headline: `median_worst_case_lol`,
  `pct_fully_recoverable` (worst-case LOL ≤ `--tau-mwh`), `mean_feasible_fraction`,
  `median_max_cascade_risk`, stratified by family.
- `run_meta.json` — the exact CLI args used.

These feed the two §6 headline figures (Pareto: solve time vs worst-case LOL;
criticality stratification: worst-case LOL vs family) and the §6.5 subsection.

### 10.5 Figures

After a run, build the two headline figures + summary table:

```powershell
python -m scripts.build_partial_recovery_figures --in outputs/uc_jepa/partial_recovery
```

This writes `fig_partial_recovery_pareto.png`,
`fig_partial_recovery_criticality.png`, and `partial_recovery_summary.csv`.

> **Note.** On hard (medium/high) scenarios the commitment MILP for arms A/B can
> hit the time limit; an arm that returns no incumbent is logged and skipped for
> that scenario (it simply contributes fewer rows). Raise `--milp-time-limit`
> for the full run if you want every arm solved on every scenario.
