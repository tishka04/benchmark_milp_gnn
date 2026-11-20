"""Preserve completed training while repairing a diagnosed pilot I/O failure.

The original failed pilot remains immutable. A sealed runtime revision gives
the corrected pilot different identities; no timeout or success is retried.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
from pathlib import Path

from .common import DEFAULT_OUT, digest, load_record, read_json, save_record, sha

RECEIPT_NAME = "pilot_io_amendment.json"
REPAIR_KIND = "windows_long_path_io_v1"
REPAIR_SOURCES = frozenset({"src/operational_exit/common.py", "src/operational_exit/protocol.py",
                            "src/operational_exit/campaign.py", "src/operational_exit/pilot_io_repair.py"})


def check_sources(before, after):
    changed = {p for p in before.keys() | after.keys() if before.get(p) != after.get(p)}
    if not changed <= REPAIR_SOURCES or before.keys() - after.keys():
        raise ValueError("Changes outside the pilot I/O repair; preserve frozen training/runtime")
    return sorted(changed)


def verify_repair(manifest, out, before, current, *, check_artifacts=False):
    receipt = load_record(Path(out) / RECEIPT_NAME) if out is not None else None
    if (not receipt or not receipt.get("complete") or receipt.get("kind") != REPAIR_KIND
            or receipt.get("manifest_hash") != digest(manifest)
            or receipt.get("previous_sources_digest") != digest(before)
            or receipt.get("sources") != current
            or receipt.get("changed_sources") != check_sources(before, current)):
        raise ValueError("Code/configuration changed without a compatible pilot I/O amendment")
    if check_artifacts:
        for relative, expected in receipt["retained_files"].items():
            path = Path(relative)
            if path.is_absolute() or ".." in path.parts or sha(Path(out) / path) != expected:
                raise ValueError(f"Retained campaign artifact changed: {relative}")
        from .campaign import model_files
        if {k: sha(v) for k, v in model_files(manifest, out).items()} != receipt["models"]:
            raise ValueError("Models changed after pilot I/O repair")
    return receipt


def calibration_revision(out, models):
    receipt = load_record(Path(out) / RECEIPT_NAME)
    if receipt is None:
        return None
    if not receipt.get("complete") or receipt.get("kind") != REPAIR_KIND or receipt.get("models") != models:
        raise ValueError("Pilot I/O revision/model stack changed")
    return receipt["sha256"]


def failed_io_observations(manifest, out):
    """Only permit the precise publication failure observed before calibration."""
    out = Path(out).resolve()
    observations = []
    for path in sorted((out / "calibration_runs").glob("*.json")):
        record = load_record(path)
        identity = record.get("identity", {}) if record else {}
        if (not record or not record.get("complete") or record.get("success")
                or record.get("status") != "worker_error" or record.get("forced")
                or not record.get("all_processes_exited") or record.get("result") is not None
                or record.get("returncode") in (None, 0)
                or identity.get("manifest_hash") != digest(manifest)
                or identity.get("purpose") != "calibration"
                or identity.get("key") not in manifest["calibration_cases"]):
            raise ValueError("Pilot I/O repair cannot supersede a success, timeout or unrelated failure")
        attempt = Path(record["attempt"]).resolve()
        attempt.relative_to(out / "calibration_runs_attempts")
        log = attempt / "worker.log"
        text = log.read_text(encoding="utf-8")
        if (not all(token in text for token in ("FileNotFoundError", "atomic_json", "final.json.", ".tmp"))
                or len(str(attempt / "final.json")) + 37 <= 259
                or (attempt / "final.json").exists()):
            raise ValueError("Pilot failure is not the diagnosed long temporary result path")
        observations.append(dict(record=str(path.relative_to(out)), record_sha256=sha(path),
                                 log=str(log.relative_to(out)), log_sha256=sha(log), identity=identity))
    if not observations:
        raise ValueError("No diagnosed pilot I/O failure to supersede")
    return observations


def register_repair(manifest, out):
    from .protocol import source_signature, versions, hte_runtime_sources, effective_sources
    from .campaign import model_files
    out = Path(out)
    if load_record(out / RECEIPT_NAME) is not None:
        effective_sources(manifest, out, check_artifacts=True)
        return
    before, current = hte_runtime_sources(manifest, out), source_signature()
    changed = check_sources(before, current)
    if not changed or manifest["versions"] != versions():
        raise ValueError("Expected only the I/O repair with the unchanged runtime")
    state = read_json(out / "state.json")
    if any(state["steps"].get(str(s)) != "completed" for s in range(1, 9)):
        raise ValueError("Pilot I/O repair requires completed stages 1 through 8")
    if (any(state["steps"].get(str(s)) == "completed" for s in range(9, 13))
            or any((out / name).exists() for name in ("calibration.json", "calibration.json.bak"))
            or any(p.is_file() for area in ("experiments", "experiments_attempts")
                   for p in (out / area).rglob("*"))):
        raise ValueError("Cannot amend a calibrated or evaluated campaign")
    for name in ("audit.json", "dataset.json"):
        record = load_record(out / name)
        if not record or not record.get("complete") or record.get("manifest_hash") != digest(manifest):
            raise ValueError(f"Incomplete/incompatible retained stage: {name}")
    observations = failed_io_observations(manifest, out)
    for case in manifest["train"] + manifest["cohort"]:
        if sha(case["file"]) != case["scenario_sha256"]:
            raise ValueError("Frozen scenario changed")
    models = {k: sha(v) for k, v in model_files(manifest, out).items()}
    if models["hte"] != manifest["hte_sha256"]:
        raise ValueError("Frozen HTE changed")
    retained = set(p for name in ("manifest", "audit", "dataset", "hte_feature_contract", "runtime_amendment")
                   for p in (out / (name + ".json"), out / (name + ".json.bak")) if p.exists())
    retained.update(p for area in ("configs", "training") for p in (out / area).rglob("*") if p.is_file())
    for observation in observations:
        retained.update((out / observation["record"], out / (observation["record"] + ".bak"), out / observation["log"]))
    retained_files = {str(p.relative_to(out)): sha(p) for p in sorted(retained)}
    if source_signature() != current:
        raise ValueError("Code changed during pilot I/O repair registration")
    save_record(out / RECEIPT_NAME, dict(complete=True, kind=REPAIR_KIND,
        created_at=datetime.now(timezone.utc).isoformat(), manifest_hash=digest(manifest),
        reason="Windows temporary result path exceeds MAX_PATH; use extended filesystem paths",
        previous_sources_digest=digest(before), sources=current, changed_sources=changed,
        models=models, retained_files=retained_files, superseded_pilots=observations,
        retained_steps=list(range(1, 9)), physics_changed=False, training_changed=False,
        timing_boundary_changed=False, original_failures_preserved=True))
    effective_sources(manifest, out, check_artifacts=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()
    from scripts.recompute_thermal_references import exclusive_lock
    from scripts.benchmark_matched_budgets import campaign_lock
    out = args.output_dir.resolve()
    manifest = load_record(out / "manifest.json")
    if not manifest:
        raise ValueError("Missing frozen manifest")
    with campaign_lock(Path(manifest["source"])), exclusive_lock(out / "run.lock"):
        register_repair(manifest, out)
    print("Pilot I/O repair registered; stages 1-8 and original failed pilot preserved.", flush=True)


if __name__ == "__main__":
    main()
