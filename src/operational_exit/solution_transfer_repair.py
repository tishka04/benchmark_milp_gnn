"""Certify linear native-solution transfer before restarting calibration.

Timed-out pilots remain failures. Their raw files are inspected only to diagnose
post-solve overhead, never recovered as successful benchmark observations.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
from pathlib import Path

from .common import ROOT, DEFAULT_OUT, digest, load_record, read_json, save_record, sha, atomic_json, verify
from .repair_structural_fixes import RECEIPT_NAME as STRUCTURAL_RECEIPT, calibration_revision as structural_revision

RECEIPT_NAME = "solution_transfer_amendment.json"
KIND = "native_solution_single_column_copy_v1"
REPAIR_SOURCES = frozenset({"src/operational_exit/solving.py", "src/operational_exit/protocol.py",
    "src/operational_exit/campaign.py", "src/operational_exit/solution_transfer_repair.py"})


def check_sources(before, after):
    changed = {p for p in before.keys() | after.keys() if before.get(p) != after.get(p)}
    if not changed <= REPAIR_SOURCES or before.keys() - after.keys():
        raise ValueError("Changes outside solution transfer repair; retain budgets, supervision, physics and training")
    return sorted(changed)


def verify_repair(manifest, out, before, current, *, check_artifacts=False):
    receipt = load_record(Path(out) / RECEIPT_NAME)
    if (not receipt or not receipt.get("complete") or receipt.get("kind") != KIND
            or receipt.get("manifest_hash") != digest(manifest)
            or receipt.get("previous_sources_digest") != digest(before)
            or receipt.get("previous_structural_receipt_sha256") != sha(Path(out) / STRUCTURAL_RECEIPT)
            or receipt.get("sources") != current
            or receipt.get("changed_sources") != check_sources(before, current)):
        raise ValueError("Missing/incompatible solution transfer amendment")
    if check_artifacts:
        for relative, expected in receipt["retained_files"].items():
            path = Path(relative)
            if path.is_absolute() or ".." in path.parts or sha(Path(out) / path) != expected:
                raise ValueError(f"Retained solution transfer diagnostic/pilot changed: {relative}")
        from .campaign import model_files
        if {k: sha(v) for k, v in model_files(manifest, out).items()} != receipt["models"]:
            raise ValueError("Models changed after solution transfer repair")
    return receipt


def calibration_revision(out, models):
    previous = structural_revision(out, models)
    receipt = load_record(Path(out) / RECEIPT_NAME)
    if receipt is None:
        return previous
    if (not receipt.get("complete") or receipt.get("kind") != KIND or receipt.get("models") != models
            or receipt.get("previous_structural_receipt_sha256") != sha(Path(out) / STRUCTURAL_RECEIPT)):
        raise ValueError("Solution transfer revision/model stack changed")
    return receipt["sha256"]


def check_pilots(manifest, out, previous_revision):
    from .physics import validate_dispatch
    from src.milp.scenario_loader import load_scenario_data
    records = [load_record(p) for p in sorted((Path(out) / "calibration_runs").glob("*.json"))]
    active = [r for r in records if r and r.get("identity", {}).get("runtime_revision") == previous_revision]
    cases = {c["key"]: c for c in manifest["train"]}
    failures = []
    for record in active:
        ident = record["identity"]
        if (ident.get("manifest_hash") != digest(manifest) or ident.get("purpose") != "calibration"
                or ident.get("key") not in manifest["calibration_cases"] or not record.get("complete")
                or not record.get("all_processes_exited") or record.get("status") not in ("success", "forced_timeout")):
            raise ValueError("Cannot supersede unrelated pilot failures")
        if record["status"] == "forced_timeout":
            if ident.get("method") != "highs" or not record.get("forced") or record.get("result") is not None:
                raise ValueError("Expected unrecovered HiGHS timeout")
            raw_path = Path(record["attempt"]) / "final.json"
            if not raw_path.is_file():
                raise ValueError("Missing post-solve diagnostic; cannot attribute timeout to solution transfer")
            # Diagnostic only. The supervisor's complete failure record is kept
            # byte-for-byte, and none of this raw result enters calibration.
            raw = verify(read_json(raw_path))
            data = load_scenario_data(Path(cases[ident["key"]]["file"]))
            if (raw.get("identity") != ident or raw.get("status") != "solution"
                    or not 0 < raw.get("solver_seconds", 0) <= ident["budget"]
                    or not validate_dispatch(data, raw.get("dispatch", {})).get("valid")):
                raise ValueError("Timeout does not match diagnosed post-solve transfer overhead")
            failures.append(ident["key"])
        elif record.get("forced"):
            raise ValueError("Inconsistent successful pilot")
    if not failures:
        raise ValueError("Missing diagnosed HiGHS post-solve timeout")
    return [r["identity"] for r in active], failures


def register_repair(manifest, out, diagnostic_path):
    from .protocol import source_signature, hte_runtime_sources, effective_sources, versions
    from .pilot_io_repair import verify_repair as verify_io, RECEIPT_NAME as IO_RECEIPT
    from .repair_integrality import verify_repair as verify_integrality, RECEIPT_NAME as INTEGER_RECEIPT
    from .repair_structural_fixes import verify_repair as verify_structural
    from .campaign import model_files
    out = Path(out)
    if load_record(out / RECEIPT_NAME):
        effective_sources(manifest, out, check_artifacts=True)
        return
    previous, integer, io = [load_record(out / name) for name in (STRUCTURAL_RECEIPT, INTEGER_RECEIPT, IO_RECEIPT)]
    if not previous or not integer or not io:
        raise ValueError("Expected the recorded previous pilot runtimes")
    verify_io(manifest, out, hte_runtime_sources(manifest, out), io["sources"], check_artifacts=True)
    verify_integrality(manifest, out, io["sources"], integer["sources"], check_artifacts=True)
    verify_structural(manifest, out, integer["sources"], previous["sources"], check_artifacts=True)
    current = source_signature()
    changed = check_sources(previous["sources"], current)
    if not changed or versions() != manifest["versions"]:
        raise ValueError("Runtime/environment changed outside solution transfer repair")
    state = read_json(out / "state.json")
    if (any(state["steps"].get(str(s)) != "completed" for s in range(1, 9))
            or any(state["steps"].get(str(s)) == "completed" for s in range(9, 13))
            or any((out / n).exists() for n in ("calibration.json", "calibration.json.bak"))
            or any(p.is_file() for area in ("experiments", "experiments_attempts") for p in (out / area).rglob("*"))):
        raise ValueError("Repair requires completed training and no calibrated/evaluated campaign")
    superseded, failures = check_pilots(manifest, out, previous["sha256"])
    diagnostic = read_json(diagnostic_path)
    observations = diagnostic["observations"]
    if (diagnostic["case"] not in manifest["train"] or diagnostic["case"]["key"] not in failures
            or len(observations) != 2 or [r["cached"] for r in observations] != [False, True]
            or observations[0]["values_digest"] != observations[1]["values_digest"]
            or observations[0]["columns"] != observations[1]["columns"] or observations[0]["columns"] <= 1):
        raise ValueError("Expected identical column-transfer diagnostic on the failing validation model")
    for case in manifest["train"] + manifest["cohort"]:
        if sha(case["file"]) != case["scenario_sha256"]:
            raise ValueError("Frozen scenario changed")
    models = {k: sha(v) for k, v in model_files(manifest, out).items()}
    if models != previous["models"]:
        raise ValueError("Frozen trained models changed")
    atomic_json(out / "diagnostics/solution_transfer.json", diagnostic)
    retained = {out / STRUCTURAL_RECEIPT, out / (STRUCTURAL_RECEIPT + ".bak")}
    retained.update(p for area in ("calibration_runs", "calibration_runs_attempts", "diagnostics")
                    for p in (out / area).rglob("*") if p.is_file())
    if source_signature() != current:
        raise ValueError("Code changed during solution transfer certification")
    save_record(out / RECEIPT_NAME, dict(complete=True, kind=KIND,
        created_at=datetime.now(timezone.utc).isoformat(), manifest_hash=digest(manifest),
        previous_sources_digest=digest(previous["sources"]), previous_structural_receipt_sha256=sha(out / STRUCTURAL_RECEIPT),
        sources=current, changed_sources=changed, models=models,
        retained_files={str(p.relative_to(out)): sha(p) for p in sorted(retained)}, superseded_pilots=superseded,
        reason="Copy native HiGHS column values once, avoiding quadratic post-solve transfer",
        retained_steps=list(range(1, 9)), training_changed=False, validator_changed=False,
        timing_boundary_changed=False, budgets_changed=False, original_pilots_preserved=True,
        policy="Recalibrate all four methods with the corrected runtime; no recovery or best-of-revisions selection"))
    effective_sources(manifest, out, check_artifacts=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--diagnostic", type=Path, default=ROOT / "tmp/solution_transfer_diagnostic.json")
    args = parser.parse_args()
    from scripts.benchmark_matched_budgets import campaign_lock
    from scripts.recompute_thermal_references import exclusive_lock
    out = args.output_dir.resolve()
    manifest = load_record(out / "manifest.json")
    with campaign_lock(Path(manifest["source"])), exclusive_lock(out / "run.lock"):
        register_repair(manifest, out, args.diagnostic)
    print("Solution transfer repair registered; models and all previous pilot outcomes preserved.", flush=True)


if __name__ == "__main__":
    main()
