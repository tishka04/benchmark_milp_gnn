"""Read-only inventory, identity-based splits and frozen campaign order."""
from __future__ import annotations

from collections import Counter
from pathlib import Path
import platform
from .common import (ROOT, PHYSICS, PROTOCOL, BUDGETS, METHODS, FAMILIES, digest,
                     read_json, sha, seed_for)


def source_signature():
    paths = sorted(set(ROOT.glob("src/**/*.py")) | set(ROOT.glob("configs/uc_jepa/*.yaml")) |
                   {ROOT / "scripts/run_operational_exit.ps1", ROOT / "scripts/recompute_thermal_references.py",
                    ROOT / "scripts/benchmark_matched_budgets.py"})
    return {str(p.relative_to(ROOT)).replace("\\", "/"): sha(p) for p in paths}


def versions():
    import importlib.metadata as md
    return dict(python=platform.python_version(), platform=platform.platform(),
                **{name: md.version(name) for name in ("pyomo", "highspy", "torch", "numpy", "psutil")})


def case_info(path, family):
    data = read_json(path)
    graph = data["graph"]
    zones = graph["zones_per_region"]
    zones = sum(zones) if isinstance(zones, list) else zones * graph["regions"]
    assets = sum(sum(v) for k, v in data["assets"].items() if k.endswith("_per_zone") and isinstance(v, list))
    T = round(data["horizon_hours"] * 60 / data["dt_minutes"])
    if T != 24:
        raise ValueError(f"Horizon changed: {path}")
    return dict(key=f"{family}/{path.stem}", family=family, scenario_id=path.stem,
                uuid=data["id"], file=str(path.resolve()), scenario_sha256=sha(path),
                periods=T, dt_hours=data["dt_minutes"] / 60,
                graph_size=(1 + graph["regions"] + zones + assets) * T)


def size_selection(cases, count):
    cases = sorted(cases, key=lambda c: (c["graph_size"], c["key"]))
    if len(cases) < count:
        raise ValueError("Insufficient cases for a size-spanning selection")
    return [cases[round(i * (len(cases) - 1) / (count - 1))]["key"] for i in range(count)]


def freeze_splits(train, source):
    """Reconstruct the historical sorted-label/random-permutation split BEFORE reclassification."""
    import torch
    old = sorted((source / "uc_jepa/dataset_train").rglob("*_uc.npz"))
    old_keys = ["scenarios_v3/" + p.name.removesuffix("_uc.npz") for p in old]
    if len(old_keys) != len(set(old_keys)):
        raise ValueError("Historical label identifiers are duplicated")
    n = len(old_keys)
    n_train = max(1, n - max(1, round(n * .1)))
    perm = torch.randperm(n, generator=torch.Generator().manual_seed(42)).tolist()
    historic = {old_keys[i]: "train" if j < n_train else "validation" for j, i in enumerate(perm)}
    assignments = {c["key"]: historic.get(c["key"], "validation" if seed_for(c["key"]) % 10 == 0 else "train") for c in train}
    return dict(seed=42, validation_fraction=.1, assignments=assignments,
                source="historical sorted UCLabelDataset files + torch.randperm(seed42); new IDs hash(seed42) mod 10",
                historical_files=[str(p.relative_to(source)) for p in old],
                historical_count=n, counts=dict(Counter(assignments.values())))


def inventory(source):
    source = Path(source).resolve()
    historical = read_json(source / "evaluation/budgets_v1/manifest.json")
    cohort = [case_info(source / c["family"] / (c["scenario_id"] + ".json"), c["family"])
              for c in historical["cohort"]]
    if Counter(c["family"] for c in cohort) != Counter({f: 100 for f in FAMILIES}):
        raise ValueError("Expected exactly 100 scenarios per evaluation family")
    train = [case_info(p, "scenarios_v3") for p in sorted((source / "scenarios_v3").glob("scenario_*.json"))]
    if len(train) != 5000:
        raise ValueError("Expected unchanged 5000 scenarios_v3")
    for values in (train, cohort):
        if len({c["uuid"] for c in values}) != len(values):
            raise ValueError("Repeated scenario UUID")
    if {c["uuid"] for c in train} & {c["uuid"] for c in cohort}:
        raise ValueError("Evaluation/training leakage by UUID")
    if {c["scenario_sha256"] for c in train} & {c["scenario_sha256"] for c in cohort}:
        raise ValueError("Evaluation/training duplicate content")
    split = freeze_splits(train, source)
    validation = [c for c in train if split["assignments"][c["key"]] == "validation"]
    return dict(protocol=PROTOCOL, physics=PHYSICS, source=str(source), cohort=cohort, train=train,
                splits=split, budgets=list(BUDGETS), methods=list(METHODS), k=16,
                calibration_cases=size_selection(validation, 30),
                variability_cases=sum((size_selection([c for c in cohort if c["family"] == f], 5) for f in FAMILIES), []),
                real_audit_cases=sum((size_selection([c for c in cohort if c["family"] == f], 2) for f in FAMILIES), []),
                physical_tolerance=1e-3, integer_tolerance=1e-5, reference_solver_seconds=600,
                training_workers=4, training_threads=4, reference_threads=16,
                repair=dict(max_stages=3, slack_tol_mwh=1.0, deviation_penalty=10000.,
                            flip_budget_20=100, flip_budget_100=1000),
                versions=versions(), sources=source_signature(),
                hte_sha256=sha(source / "encoders/hierarchical_temporal_v3/best_encoder.pt"))


def schedule(manifest, extra=False):
    """Balanced Latin rotation inside family-stratified blocks; strictly serial."""
    selected = set(manifest["variability_cases"]) if extra else None
    families = [[c for c in manifest["cohort"] if c["family"] == f and (selected is None or c["key"] in selected)] for f in FAMILIES]
    for group in families:
        group.sort(key=lambda c: seed_for(c["key"]))
    for repetition in ((1, 2) if extra else (0,)):
        for index in range(len(families[0])):
            for f in range(6):
                c = families[(f + index + repetition) % 6][index]
                for j in range(7):
                    budget = BUDGETS[(j + index + f) % 7]
                    for k in range(4):
                        method = METHODS[(k + index + j + f + repetition) % 4]
                        yield dict(key=c["key"], method=method, budget=budget, repetition=repetition,
                                   seed=seed_for(c["key"], repetition))


def assert_unchanged(manifest):
    if manifest["sources"] != source_signature() or manifest["versions"] != versions():
        raise ValueError("Code/configuration/runtime changed: use a new output directory")
    for c in manifest["train"] + manifest["cohort"]:
        if sha(c["file"]) != c["scenario_sha256"]:
            raise ValueError(f"Scenario changed: {c['key']}")
