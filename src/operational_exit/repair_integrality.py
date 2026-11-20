"""Certify the correction of fractional decisions in the common repair cascade.

Keep the full pre-correction pilot history. Restart validation calibration with
one fixed, corrected runtime; no evaluation result or trained model is changed.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
from pathlib import Path

from .common import ROOT, DEFAULT_OUT, digest, load_record, read_json, save_record, sha, atomic_json
from .pilot_io_repair import RECEIPT_NAME as IO_RECEIPT, calibration_revision as io_revision

RECEIPT_NAME = "repair_integrality_amendment.json"
KIND = "repair_original_integer_domains_v1"
REPAIR_SOURCES = frozenset({"src/operational_exit/solving.py", "src/operational_exit/worker.py",
    "src/operational_exit/protocol.py", "src/operational_exit/campaign.py", "src/operational_exit/repair_integrality.py"})


def check_sources(before, after):
    changed = {p for p in before.keys() | after.keys() if before.get(p) != after.get(p)}
    if not changed <= REPAIR_SOURCES or before.keys() - after.keys():
        raise ValueError("Changes outside repair integrality/diagnostics; frozen models and validator must be retained")
    return sorted(changed)


def verify_repair(manifest, out, before, current, *, check_artifacts=False):
    receipt = load_record(Path(out) / RECEIPT_NAME)
    if (not receipt or not receipt.get("complete") or receipt.get("kind") != KIND
            or receipt.get("manifest_hash") != digest(manifest)
            or receipt.get("previous_sources_digest") != digest(before)
            or receipt.get("previous_io_receipt_sha256") != sha(Path(out) / IO_RECEIPT)
            or receipt.get("sources") != current
            or receipt.get("changed_sources") != check_sources(before, current)):
        raise ValueError("Missing/incompatible repair integrality amendment")
    if check_artifacts:
        for relative, expected in receipt["retained_files"].items():
            path = Path(relative)
            if path.is_absolute() or ".." in path.parts or sha(Path(out) / path) != expected:
                raise ValueError(f"Retained repair diagnostic/pilot changed: {relative}")
        from .campaign import model_files
        if {k: sha(v) for k, v in model_files(manifest, out).items()} != receipt["models"]:
            raise ValueError("Models changed after repair integrality correction")
    return receipt


def calibration_revision(out, models):
    previous = io_revision(out, models)
    receipt = load_record(Path(out) / RECEIPT_NAME)
    if receipt is None:
        return previous
    if (not receipt.get("complete") or receipt.get("kind") != KIND or receipt.get("models") != models
            or receipt.get("previous_io_receipt_sha256") != sha(Path(out) / IO_RECEIPT)):
        raise ValueError("Repair integrality revision/model stack changed")
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
            if (ident["method"] != "pipeline" or result.get("generated_candidates") != 16
                    or len(diagnostics) != 16
                    or any(d.get("status") != "invalid_dispatch" or d.get("repair_status") != "optimal"
                           or d.get("repair_stage") != "SolveStage.HARD_FIX" for d in diagnostics)):
                raise ValueError("Pilot does not match the diagnosed hard-fix integrality failure")
            found = True
    if not found:
        raise ValueError("Missing diagnosed fractional hard-fix pilot")
    return [r["identity"] for r in active]


def check_diagnostics(manifest, before_path, after_path):
    from .physics import validate_dispatch
    from src.milp.scenario_loader import load_scenario_data
    before, after = read_json(before_path), read_json(after_path)
    case = next((c for c in manifest["train"] if c["key"] == before["case"]["key"]), None)
    if (not case or case != before["case"] or case != after["case"]
            or case["key"] not in manifest["calibration_cases"] or sha(case["file"]) != case["scenario_sha256"]):
        raise ValueError("Diagnostic must use the unchanged internal validation case")
    data = load_scenario_data(Path(case["file"]))
    old_check, new_check = [validate_dispatch(data, d["dispatch"]) for d in (before, after)]
    if (old_check.get("valid") or old_check.get("violations") != {}
            or old_check.get("max_integrality_residual", 0) <= old_check.get("integer_tolerance", 1e-5)
            or not new_check.get("valid")):
        raise ValueError("Expected fractional legacy dispatch and independently valid corrected dispatch")
    return before, after


def register_repair(manifest, out, before_path, after_path):
    from .protocol import source_signature, hte_runtime_sources, effective_sources, versions
    from .pilot_io_repair import verify_repair as verify_io
    from .campaign import model_files
    out = Path(out)
    if load_record(out / RECEIPT_NAME):
        effective_sources(manifest, out, check_artifacts=True)
        return
    previous = load_record(out / IO_RECEIPT)
    if not previous:
        raise ValueError("Expected the recorded previous pilot runtime")
    verify_io(manifest, out, hte_runtime_sources(manifest, out), previous["sources"], check_artifacts=True)
    current = source_signature()
    changed = check_sources(previous["sources"], current)
    if not changed or versions() != manifest["versions"]:
        raise ValueError("Runtime/environment changed outside the integrality correction")
    state = read_json(out / "state.json")
    if (any(state["steps"].get(str(s)) != "completed" for s in range(1, 9))
            or any(state["steps"].get(str(s)) == "completed" for s in range(9, 13))
            or any((out / n).exists() for n in ("calibration.json", "calibration.json.bak"))
            or any(p.is_file() for area in ("experiments", "experiments_attempts") for p in (out / area).rglob("*"))):
        raise ValueError("Repair requires completed training and no calibrated/evaluated campaign")
    superseded = check_pilots(manifest, out, previous["sha256"])
    before, after = check_diagnostics(manifest, before_path, after_path)
    for case in manifest["train"] + manifest["cohort"]:
        if sha(case["file"]) != case["scenario_sha256"]:
            raise ValueError("Frozen scenario changed")
    models = {k: sha(v) for k, v in model_files(manifest, out).items()}
    if models != previous["models"]:
        raise ValueError("Frozen trained models changed")
    for suffix, body in (("before", before), ("after", after)):
        atomic_json(out / "diagnostics" / ("repair_integrality_" + suffix + ".json"), body)
    retained = {out / IO_RECEIPT, out / (IO_RECEIPT + ".bak")}
    retained.update(p for area in ("calibration_runs", "calibration_runs_attempts", "diagnostics")
                    for p in (out / area).rglob("*") if p.is_file())
    if source_signature() != current:
        raise ValueError("Code changed during integrality repair certification")
    save_record(out / RECEIPT_NAME, dict(complete=True, kind=KIND,
        created_at=datetime.now(timezone.utc).isoformat(), manifest_hash=digest(manifest),
        previous_sources_digest=digest(previous["sources"]), previous_io_receipt_sha256=sha(out / IO_RECEIPT),
        sources=current, changed_sources=changed, models=models,
        retained_files={str(p.relative_to(out)): sha(p) for p in sorted(retained)}, superseded_pilots=superseded,
        reason="Restore original integer domains before solving the shared repair cascade",
        retained_steps=list(range(1, 9)), training_changed=False, validator_changed=False,
        timing_boundary_changed=False, original_pilots_preserved=True,
        policy="Recalibrate all four methods with the corrected runtime; no best-of-revisions selection"))
    effective_sources(manifest, out, check_artifacts=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--before", type=Path, default=ROOT / "tmp/pipeline_dispatch_before.json")
    parser.add_argument("--after", type=Path, default=ROOT / "tmp/pipeline_dispatch_after.json")
    args = parser.parse_args()
    from scripts.benchmark_matched_budgets import campaign_lock
    from scripts.recompute_thermal_references import exclusive_lock
    out = args.output_dir.resolve()
    manifest = load_record(out / "manifest.json")
    with campaign_lock(Path(manifest["source"])), exclusive_lock(out / "run.lock"):
        register_repair(manifest, out, args.before, args.after)
    print("Repair integrality correction registered; trained models and previous pilots preserved.", flush=True)


if __name__ == "__main__":
    main()
