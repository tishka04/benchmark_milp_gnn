"""Record deadline-as-failure calibration policy without rerunning observations.

Only campaign bookkeeping changes. Keep the existing execution revision, solver
budgets, supervisor, and completed pilot identities (including all failures).
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
from pathlib import Path

from .common import DEFAULT_OUT, digest, load_record, read_json, save_record, sha
from .solution_transfer_repair import RECEIPT_NAME as PREVIOUS_RECEIPT

RECEIPT_NAME = "calibration_outcomes_amendment.json"
KIND = "calibration_deadlines_are_failures_v1"
POLICY_SOURCES = frozenset({"src/operational_exit/campaign.py", "src/operational_exit/protocol.py",
                          "src/operational_exit/calibration_outcomes.py"})


def check_sources(before, after):
    changed = {p for p in before.keys() | after.keys() if before.get(p) != after.get(p)}
    if not changed <= POLICY_SOURCES or before.keys() - after.keys():
        raise ValueError("Calibration policy cannot change workers, solvers, budgets, models or validation")
    return sorted(changed)


def verify_policy(manifest, out, before, current, *, check_artifacts=False):
    receipt = load_record(Path(out) / RECEIPT_NAME)
    previous = load_record(Path(out) / PREVIOUS_RECEIPT)
    if (not receipt or not previous or not receipt.get("complete") or receipt.get("kind") != KIND
            or receipt.get("manifest_hash") != digest(manifest)
            or receipt.get("previous_sources_digest") != digest(before)
            or receipt.get("previous_receipt_sha256") != sha(Path(out) / PREVIOUS_RECEIPT)
            or receipt.get("execution_runtime_revision") != previous["sha256"]
            or receipt.get("sources") != current
            or receipt.get("changed_sources") != check_sources(before, current)):
        raise ValueError("Missing/incompatible calibration outcome policy amendment")
    if check_artifacts:
        for relative, expected in receipt["retained_files"].items():
            path = Path(relative)
            if path.is_absolute() or ".." in path.parts or sha(Path(out) / path) != expected:
                raise ValueError(f"Retained calibration outcome changed: {relative}")
        from .campaign import model_files
        if {k: sha(v) for k, v in model_files(manifest, out).items()} != receipt["models"]:
            raise ValueError("Models changed after calibration outcome policy")
    return receipt


def register_policy(manifest, out):
    from .protocol import source_signature, verify_source_history, effective_sources, versions
    from .campaign import model_files
    out = Path(out)
    if load_record(out / RECEIPT_NAME):
        effective_sources(manifest, out, check_artifacts=True)
        return
    previous = load_record(out / PREVIOUS_RECEIPT)
    if not previous:
        raise ValueError("Missing execution runtime receipt")
    verify_source_history(manifest, out, previous["sources"], check_artifacts=True)
    current = source_signature()
    changed = check_sources(previous["sources"], current)
    if not changed or versions() != manifest["versions"]:
        raise ValueError("Runtime/environment changed outside calibration outcome policy")
    state = read_json(out / "state.json")
    if (any(state["steps"].get(str(s)) != "completed" for s in range(1, 9))
            or any(state["steps"].get(str(s)) == "completed" for s in range(9, 13))
            or any((out / n).exists() for n in ("calibration.json", "calibration.json.bak"))
            or any(p.is_file() for area in ("experiments", "experiments_attempts") for p in (out / area).rglob("*"))):
        raise ValueError("Outcome policy requires completed training and no calibrated/evaluated campaign")
    for case in manifest["train"] + manifest["cohort"]:
        if sha(case["file"]) != case["scenario_sha256"]:
            raise ValueError("Frozen scenario changed")
    models = {k: sha(v) for k, v in model_files(manifest, out).items()}
    if models != previous["models"]:
        raise ValueError("Frozen trained models changed")
    retained = {out / PREVIOUS_RECEIPT, out / (PREVIOUS_RECEIPT + ".bak")}
    retained.update(p for area in ("calibration_runs", "calibration_runs_attempts", "diagnostics")
                    for p in (out / area).rglob("*") if p.is_file())
    if source_signature() != current:
        raise ValueError("Code changed during calibration outcome certification")
    save_record(out / RECEIPT_NAME, dict(complete=True, kind=KIND,
        created_at=datetime.now(timezone.utc).isoformat(), manifest_hash=digest(manifest),
        previous_sources_digest=digest(previous["sources"]), previous_receipt_sha256=sha(out / PREVIOUS_RECEIPT),
        sources=current, changed_sources=changed, models=models,
        execution_runtime_revision=previous["sha256"],
        retained_files={str(p.relative_to(out)): sha(p) for p in sorted(retained)},
        reason="Count deadline failures and continue calibration, as requested; no retries to obtain success",
        training_changed=False, worker_changed=False, budgets_changed=False, validator_changed=False,
        timing_boundary_changed=False, original_pilots_preserved=True, pilot_identities_unchanged=True,
        policy="Reuse all complete current-runtime outcomes, including failures; never recover forced raw results"))
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
        register_policy(manifest, out)
    print("Deadline failures are retained; calibration resumes without retrying completed pilots.", flush=True)


if __name__ == "__main__":
    main()
