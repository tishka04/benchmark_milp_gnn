"""Record the correction preserving physical fixed decisions during repair.

Retain models and all pilot history, then recalibrate with a single runtime.
Registration is restricted to the diagnosed zero-capacity repair failure before
any completed calibration or evaluation.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
from pathlib import Path

from .common import DEFAULT_OUT, digest, load_record, read_json, save_record, sha
from .repair_integrality import RECEIPT_NAME as INTEGER_RECEIPT, calibration_revision as integer_revision

RECEIPT_NAME = "repair_structural_fixes_amendment.json"
KIND = "repair_preserve_structural_fixes_v1"
REPAIR_SOURCES = frozenset({"src/operational_exit/solving.py", "src/operational_exit/protocol.py",
    "src/operational_exit/campaign.py", "src/operational_exit/repair_structural_fixes.py"})
ZERO_VIOLATIONS = frozenset({"zero_capacity:u_thermal", "zero_capacity:v_thermal_startup",
    "zero_capacity:b_charge_mode", "zero_capacity:pumped_charge_mode", "zero_dr", "zero_import"})


def check_sources(before, after):
    changed = {p for p in before.keys() | after.keys() if before.get(p) != after.get(p)}
    if not changed <= REPAIR_SOURCES or before.keys() - after.keys():
        raise ValueError("Changes outside structural-fix repair; retain physics, validator and training")
    return sorted(changed)


def verify_repair(manifest, out, before, current, *, check_artifacts=False):
    receipt = load_record(Path(out) / RECEIPT_NAME)
    if (not receipt or not receipt.get("complete") or receipt.get("kind") != KIND
            or receipt.get("manifest_hash") != digest(manifest)
            or receipt.get("previous_sources_digest") != digest(before)
            or receipt.get("previous_integrality_receipt_sha256") != sha(Path(out) / INTEGER_RECEIPT)
            or receipt.get("sources") != current
            or receipt.get("changed_sources") != check_sources(before, current)):
        raise ValueError("Missing/incompatible structural-fix repair amendment")
    if check_artifacts:
        for relative, expected in receipt["retained_files"].items():
            path = Path(relative)
            if path.is_absolute() or ".." in path.parts or sha(Path(out) / path) != expected:
                raise ValueError(f"Retained structural-fix pilot changed: {relative}")
        from .campaign import model_files
        if {k: sha(v) for k, v in model_files(manifest, out).items()} != receipt["models"]:
            raise ValueError("Models changed after structural-fix repair")
    return receipt


def calibration_revision(out, models):
    previous = integer_revision(out, models)
    receipt = load_record(Path(out) / RECEIPT_NAME)
    if receipt is None:
        return previous
    if (not receipt.get("complete") or receipt.get("kind") != KIND or receipt.get("models") != models
            or receipt.get("previous_integrality_receipt_sha256") != sha(Path(out) / INTEGER_RECEIPT)):
        raise ValueError("Structural-fix revision/model stack changed")
    return receipt["sha256"]


def check_pilots(manifest, out, previous_revision):
    records = [load_record(p) for p in sorted((Path(out) / "calibration_runs").glob("*.json"))]
    active = [r for r in records if r and r.get("identity", {}).get("runtime_revision") == previous_revision]
    found = False
    for record in active:
        ident, result = record["identity"], record.get("result") or {}
        if (ident.get("manifest_hash") != digest(manifest) or ident.get("purpose") != "calibration"
                or ident.get("key") not in manifest["calibration_cases"] or not record.get("complete")
                or record.get("forced") or not record.get("all_processes_exited")
                or record.get("status") not in ("success", "invalid_dispatch")):
            raise ValueError("Cannot supersede timeout/unrelated pilot failures")
        if record["status"] == "invalid_dispatch":
            diagnostics = result.get("candidate_diagnostics", [])
            if ident.get("method") not in ("pipeline", "jepa", "rh") or not diagnostics:
                raise ValueError("Missing diagnosed structural-fix pilot")
            for diagnostic in diagnostics:
                validation = diagnostic.get("validation") or {}
                violations = set(validation.get("violations", {}))
                if (diagnostic.get("status") != "invalid_dispatch"
                        or diagnostic.get("repair_stage") not in ("SolveStage.REPAIR_20", "SolveStage.REPAIR_100")
                        or validation.get("valid") is not False or not violations or not violations <= ZERO_VIOLATIONS
                        or validation.get("max_integrality_residual", 1) > validation.get("integer_tolerance", 1e-5)):
                    raise ValueError("Pilot does not match the diagnosed structural-fix failure")
            found = True
    if not found:
        raise ValueError("Missing diagnosed structural-fix pilot")
    return [r["identity"] for r in active]


def register_repair(manifest, out):
    from .protocol import source_signature, hte_runtime_sources, effective_sources, versions
    from .pilot_io_repair import verify_repair as verify_io, RECEIPT_NAME as IO_RECEIPT
    from .repair_integrality import verify_repair as verify_integrality
    from .campaign import model_files
    out = Path(out)
    if load_record(out / RECEIPT_NAME):
        effective_sources(manifest, out, check_artifacts=True)
        return
    previous, io = load_record(out / INTEGER_RECEIPT), load_record(out / IO_RECEIPT)
    if not previous or not io:
        raise ValueError("Expected the recorded previous pilot runtimes")
    verify_io(manifest, out, hte_runtime_sources(manifest, out), io["sources"], check_artifacts=True)
    verify_integrality(manifest, out, io["sources"], previous["sources"], check_artifacts=True)
    current = source_signature()
    changed = check_sources(previous["sources"], current)
    if not changed or versions() != manifest["versions"]:
        raise ValueError("Runtime/environment changed outside structural-fix repair")
    state = read_json(out / "state.json")
    if (any(state["steps"].get(str(s)) != "completed" for s in range(1, 9))
            or any(state["steps"].get(str(s)) == "completed" for s in range(9, 13))
            or any((out / n).exists() for n in ("calibration.json", "calibration.json.bak"))
            or any(p.is_file() for area in ("experiments", "experiments_attempts") for p in (out / area).rglob("*"))):
        raise ValueError("Repair requires completed training and no calibrated/evaluated campaign")
    superseded = check_pilots(manifest, out, previous["sha256"])
    for case in manifest["train"] + manifest["cohort"]:
        if sha(case["file"]) != case["scenario_sha256"]:
            raise ValueError("Frozen scenario changed")
    models = {k: sha(v) for k, v in model_files(manifest, out).items()}
    if models != previous["models"]:
        raise ValueError("Frozen trained models changed")
    retained = {out / INTEGER_RECEIPT, out / (INTEGER_RECEIPT + ".bak")}
    retained.update(p for area in ("calibration_runs", "calibration_runs_attempts", "diagnostics")
                    for p in (out / area).rglob("*") if p.is_file())
    if source_signature() != current:
        raise ValueError("Code changed during structural-fix repair certification")
    save_record(out / RECEIPT_NAME, dict(complete=True, kind=KIND,
        created_at=datetime.now(timezone.utc).isoformat(), manifest_hash=digest(manifest),
        previous_sources_digest=digest(previous["sources"]), previous_integrality_receipt_sha256=sha(out / INTEGER_RECEIPT),
        sources=current, changed_sources=changed, models=models,
        retained_files={str(p.relative_to(out)): sha(p) for p in sorted(retained)}, superseded_pilots=superseded,
        reason="Retain physical fixed decisions and normalize impossible decoder targets in the shared repair cascade",
        retained_steps=list(range(1, 9)), training_changed=False, validator_changed=False,
        timing_boundary_changed=False, original_pilots_preserved=True,
        policy="Recalibrate all four methods with the corrected runtime; no best-of-revisions selection"))
    effective_sources(manifest, out, check_artifacts=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()
    from scripts.benchmark_matched_budgets import campaign_lock
    from scripts.recompute_thermal_references import exclusive_lock
    out = args.output_dir.resolve()
    manifest = load_record(out / "manifest.json")
    with campaign_lock(Path(manifest["source"])), exclusive_lock(out / "run.lock"):
        register_repair(manifest, out)
    print("Structural-fix repair registered; trained models and previous pilots preserved.", flush=True)


if __name__ == "__main__":
    main()
