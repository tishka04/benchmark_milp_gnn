"""Pure, testable schedule. No models, data generation or solver imports."""
from .core import FAMILIES, METHODS, config, seed

CONDITIONS = ("base", "size", "lines", "fleet", "permutation", "horizon48", "horizon72")
SUITES = {"four": METHODS, "ablation": ("pipeline", "perturb16"),
          "gurobi": ("gurobi",), "multi": ("mh_jepa", "mh_pipeline", "mh_perturb16")}
STAGES = ("Check", "Prepare", "AnalyzeExisting", "Estimate", "ReferencesDevelopment",
          "Development", "TrainingLabels", "TrainMultiHorizon", "ValidateMultiHorizon",
          "Lock", "ReferencesFinal", "Final", "Gurobi", "Analyze", "Manuscript")

def selected_repeats(cases):
    out = set()
    for family in FAMILIES:
        candidates = sorted((c for c in cases if c["family"] == family),
                            key=lambda c: seed("repeat-selection", c["group"]))
        out.update(c["key"] for c in candidates[:config()["repeat_cases_per_family"]])
    return out

def schedule(cases, suite, *, extra_repeats=False, training_seed=None, sensitivity=False):
    cfg = config()
    budgets = cfg["repeat_budgets"] if extra_repeats or sensitivity else cfg["extension_budgets"]
    repeats = cfg["extra_repetitions"] if extra_repeats else [0]
    selected = selected_repeats(cases) if extra_repeats else None
    for i, case in enumerate(sorted(cases, key=lambda c: c["key"])):
        if selected is not None and case["key"] not in selected:
            continue
        if sensitivity and case["condition"] not in ("base", "horizon48", "horizon72"):
            continue
        for budget in budgets:
            for repetition in repeats:
                methods = SUITES[suite]
                # Counterbalance method ordering within cases and repeated blocks.
                offset = (i + repetition) % len(methods)
                for method in methods[offset:] + methods[:offset]:
                    if case["expected_horizon"] > 24 and method in ("jepa", "pipeline", "perturb16"):
                        continue
                    item = dict(key=case["key"], method=method, budget=budget,
                                repetition=repetition, seed=seed(case["group"], repetition), suite=suite)
                    if method.startswith("mh_"):
                        item["training_seed"] = training_seed or cfg["training_seeds"][0]
                    yield case, item

def planned_blocks():
    """Counts and sums of timed allocations; excludes loading and training."""
    cfg=config(); nf=len(FAMILIES); n=cfg["new_cases_per_family"]*nf
    c=cfg["controlled_cases_per_family"]*nf; b=cfg["extension_budgets"]
    repeat_n=cfg["repeat_cases_per_family"]*nf
    values=[("independent_four",2*n*4*len(b),2*n*4*sum(b)),
            ("energy_ablation",2*n*2*len(b),2*n*2*sum(b)),
            ("structural_frozen",2*c*(5*4+2*2)*len(b),2*c*24*sum(b)),
            ("multihorizon_primary",2*c*7*3*len(b),2*c*7*3*sum(b)),
            ("training_seed_sensitivity",c*3*3*2*2,c*3*3*2*sum(cfg["repeat_budgets"])),
            ("timing_repeats",repeat_n*6*2*2,repeat_n*6*2*sum(cfg["repeat_budgets"])),
            ("gurobi",(120+n)*len(b),(120+n)*sum(b))]
    refs=2*n+2*c*7+cfg["training_bases_per_family"]*nf*3
    return dict(blocks=[dict(block=name,trials=count,allocated_hours=seconds/3600)
                        for name,count,seconds in values],
                trials=sum(v[1] for v in values), references=refs,
                allocated_trial_hours=sum(v[2] for v in values)/3600,
                reference_cap_hours=refs*cfg["reference_wall_cap_seconds"]/3600,
                caveat="Sums of allocations, not elapsed forecasts. Loading, training and analysis are additional.")
