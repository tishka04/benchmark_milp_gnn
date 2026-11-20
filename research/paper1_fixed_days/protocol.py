"""Freeze the day designs, invariance checks and pre-solve forecasts before any solve."""
from __future__ import annotations

import platform
from importlib.metadata import version
from pathlib import Path

from src.operational_exit.common import ROOT, atomic_json, digest, read_json, sealed, sha, verify

HOME = Path(__file__).resolve().parent
DEFAULT_RUN = HOME / "runs" / "pilot_v1"
SCHEMA = "paper1_fixed_days_v1"
FROZEN_DEADLINES = (5, 10, 30, 60, 120)
PROFILE_NAMES = ("equal4", "cost_priority", "co2_priority", "time_priority", "service4_co2")
ANALYSIS_FILES = ("research/paper1_fixed_days/analysis.py", "research/paper1_revision/future_analysis.py")
SOURCE_FILES = (
    "research/paper1_fixed_days/protocol.py",
    "research/paper1_fixed_days/design.py",
    "research/paper1_fixed_days/predict.py",
    "research/paper1_fixed_days/runner.py",
    "research/paper1_fixed_days/worker.py",
    "research/paper1_revision/realized/features.py",
    "research/paper1_revision/realized/models.py",
    "research/paper1_revision/campaign.py",
    "research/paper1_weights/data.py",
    "research/paper1_weights/protocol.json",
    "src/milp/scenario_loader.py",
    "src/milp/profiles.py",
    "src/milp/defaults.py",
    "src/milp/model.py",
    "src/demand_response.py",
    "src/thermal.py",
    "src/operational_exit/common.py",
    "src/operational_exit/physics.py",
    "src/operational_exit/solving.py",
    "src/operational_exit/supervisor.py",
)


def read_manifest(run_dir):
    path = Path(run_dir) / "manifest.json"
    if not path.is_file():
        raise ValueError(f"No manifest in {run_dir}. Use the prepare command first.")
    manifest = verify(read_json(path))
    if manifest.get("schema") != SCHEMA:
        raise ValueError(f"Incompatible fixed-day manifest: {path}")
    return manifest


def verify_inputs(manifest):
    """Check inputs used to regenerate and solve days; predictions are embedded in the manifest."""
    verify(manifest)
    if sha(manifest["anchor"]) != manifest["anchor_sha256"]:
        raise ValueError("The source system JSON has changed since preparation.")
    for filename, fingerprint in manifest["source_hashes"].items():
        if sha(ROOT / filename) != fingerprint:
            raise ValueError(f"Experimental code or configuration changed since preparation: {filename}")
    for name, expected in manifest["packages"].items():
        if version(name) != expected:
            raise ValueError(f"{name} version changed since preparation (expected {expected}).")


def _design_options(anchor, n_days, deadlines, reference_seconds, quota, seed, solver_seed, threads):
    if n_days < 2 or not 1 <= quota < n_days:
        raise ValueError("At least two days are required, with an inspection quota from 1 to days - 1.")
    if not deadlines or len(set(deadlines)) != len(deadlines) or not set(deadlines) <= set(FROZEN_DEADLINES):
        raise ValueError(f"Use distinct deadlines from the frozen forecasts: {FROZEN_DEADLINES}.")
    if reference_seconds < 1 or threads < 1:
        raise ValueError("Reference time and number of threads must be positive.")
    if not 0 <= solver_seed < 2**31 or not 0 <= seed < 2**31:
        raise ValueError("Seeds must be nonnegative 31-bit integers.")
    return dict(anchor=str(Path(anchor).resolve(strict=True)), n_days=n_days,
                deadlines=sorted(deadlines), reference_seconds=reference_seconds, quota=quota,
                day_seed=seed, order_seed=seed, solver_seed=solver_seed, threads=threads)


def prepare(run_dir, *, anchor, n_days=30, deadlines=(30, 120), reference_seconds=600,
            quota=6, seed=20261119, solver_seed=42, threads=16):
    from .design import day_parameters, inspect_day, load_anchor, load_day
    from .predict import load_frozen, predict_day
    from .runner import campaign_lock

    options = _design_options(anchor, n_days, deadlines, reference_seconds, quota, seed, solver_seed, threads)
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    with campaign_lock(run_dir):
        path = run_dir / "manifest.json"
        if path.exists():
            manifest = read_manifest(run_dir)
            if any(manifest[key] != value for key, value in options.items()):
                raise ValueError("This directory has a different frozen design. Choose another --run-dir.")
            verify_inputs(manifest)
            return manifest
        if any((run_dir / name).exists() for name in ("results", "attempts", "analysis")):
            raise ValueError(f"Existing run artifacts without a manifest in {run_dir}; use another directory.")

        models, model_hash = load_frozen()
        baseline = inspect_day(load_anchor(Path(options["anchor"])))
        days = []
        profile_hashes = set()
        structural_predictions = None
        for index in range(1, n_days + 1):
            day_id = f"day_{index:03d}"
            day_seed = int(digest([SCHEMA, seed, index])[:8], 16) % (2**31 - 1)
            features = inspect_day(load_day(Path(options["anchor"]), day_seed))
            for key in ("static_sha256", "compact_trio", "model_dimensions"):
                if features[key] != baseline[key]:
                    raise ValueError(f"{day_id}: {key} differs from the fixed source system.")
            if features["profiles_sha256"] in profile_hashes:
                raise ValueError(f"{day_id}: duplicate operating profiles.")
            profile_hashes.add(features["profiles_sha256"])
            predictions = predict_day(models, features, options["deadlines"])
            constant = {(p["representation"], p["budget"], p["target"]): p["prediction_raw"]
                        for p in predictions if p["representation"] in ("realized_compact", "model_size")}
            if structural_predictions is not None and constant != structural_predictions:
                raise ValueError(f"{day_id}: supposedly fixed structural forecasts vary.")
            structural_predictions = constant
            days.append(dict(day_id=day_id, seed=day_seed, regime=day_parameters(day_seed),
                             features=features, predictions=predictions))
            print(f"Prepared {index}/{n_days}: {day_id}", flush=True)

        manifest = sealed(dict(schema=SCHEMA, **options, anchor_sha256=sha(options["anchor"]),
                               baseline=baseline, days=days, profiles=list(PROFILE_NAMES),
                               frozen_models_sha256=model_hash,
                               source_hashes={f: sha(ROOT / f) for f in SOURCE_FILES},
                               analysis_source_hashes={f: sha(ROOT / f) for f in ANALYSIS_FILES},
                               packages={n: version(n) for n in ("highspy", "pyomo", "numpy")},
                               host=dict(platform=platform.platform(), processor=platform.processor(),
                                         python=platform.python_version())))
        atomic_json(path, manifest)
        return manifest
