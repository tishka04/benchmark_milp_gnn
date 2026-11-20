"""Budget-scaled finalization reserve and lossless, identity-based migration."""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import math
from pathlib import Path
import shutil

from .common import DEFAULT_OUT, BUDGETS, digest, load_record, read_json, save_record, sha, atomic_json

RECEIPT_NAME = "budget_reserve_amendment.json"
PREVIOUS_RECEIPT = "calibration_outcomes_amendment.json"
KIND = "budget_scaled_finalization_reserve_v1"
POLICY = dict(kind=KIND, calibration_budget_s=120., minimum_reserve_s=.05, maximum_budget_fraction=.5)
ALLOWED_SOURCES = frozenset({"src/operational_exit/budget_reserve.py", "src/operational_exit/campaign.py",
    "src/operational_exit/protocol.py", "src/operational_exit/analysis.py"})


def effective_reserve(budget, calibrated_reserve):
    """Scale the 120 s pilot reserve; always leave at least half for computation."""
    budget, calibrated_reserve = float(budget), float(calibrated_reserve)
    if not math.isfinite(budget) or budget <= 0 or not math.isfinite(calibrated_reserve) or calibrated_reserve < 0:
        raise ValueError("Budget must be positive and calibrated reserve nonnegative and finite")
    scaled = calibrated_reserve * min(1., budget / POLICY["calibration_budget_s"])
    return min(POLICY["maximum_budget_fraction"] * budget, max(POLICY["minimum_reserve_s"], scaled))


def evaluation_schedule(manifest, calibration, amendment=None, *, extra=False):
    """New identities only where the compute allowance actually changes.

    Identical 120 s trials, including failures, retain their exact identities.
    Changed-budget trials are all superseded, without selecting by outcome.
    Legacy schedules remain readable for historical analysis.
    """
    from .protocol import schedule
    enabled = amendment is not None or calibration.get("reserve_policy") == POLICY
    if amendment is not None and (amendment.get("kind") != KIND or amendment.get("policy") != POLICY
                                  or amendment.get("manifest_hash") != digest(manifest)):
        raise ValueError("Incompatible budget reserve policy")
    revision = (amendment or calibration).get("sha256")
    for item in schedule(manifest, extra=extra):
        reserve = effective_reserve(item["budget"], calibration["reserve_s"]) if enabled else calibration["reserve_s"]
        if reserve != calibration["reserve_s"]:
            if not revision:
                raise ValueError("Missing sealed timing revision")
            item = dict(item, reserve_policy=KIND, timing_revision=revision, reserve_s=reserve)
        yield item


def check_sources(before, after):
    changed = {p for p in before.keys() | after.keys() if before.get(p) != after.get(p)}
    if not changed <= ALLOWED_SOURCES or before.keys() - after.keys():
        raise ValueError("Reserve correction cannot change models, workers, solvers or physical validation")
    return sorted(changed)


def verify_policy(manifest, out, before, current, *, check_artifacts=False):
    out = Path(out)
    receipt = load_record(out / RECEIPT_NAME)
    if (not receipt or not receipt.get("complete") or receipt.get("kind") != KIND
            or receipt.get("policy") != POLICY or receipt.get("manifest_hash") != digest(manifest)
            or receipt.get("previous_sources_digest") != digest(before)
            or receipt.get("previous_receipt_sha256") != sha(out / PREVIOUS_RECEIPT)
            or receipt.get("calibration_sha256") != sha(out / "calibration.json")
            or receipt.get("sources") != current
            or receipt.get("changed_sources") != check_sources(before, current)):
        raise ValueError("Missing/incompatible budget reserve amendment")
    if check_artifacts:
        for relative, expected in receipt["retained_files"].items():
            path = Path(relative)
            if path.is_absolute() or ".." in path.parts or sha(out / path) != expected:
                raise ValueError(f"Retained pre-correction result changed: {relative}")
        from .campaign import model_files
        if {k: sha(v) for k, v in model_files(manifest, out).items()} != receipt["models"]:
            raise ValueError("Models changed during reserve correction")
    return receipt


def register_policy(manifest, out):
    """Caller owns both campaign locks. No old completed outcome is overwritten."""
    from .protocol import source_signature, verify_source_history, effective_sources, versions, schedule
    from .campaign import model_files
    out = Path(out)
    if load_record(out / RECEIPT_NAME):
        effective_sources(manifest, out, check_artifacts=True)
        return load_record(out / RECEIPT_NAME)
    previous = load_record(out / PREVIOUS_RECEIPT)
    if not previous:
        raise ValueError("Missing predecessor runtime policy")
    verify_source_history(manifest, out, previous["sources"], check_artifacts=True)
    current = source_signature()
    changed = check_sources(previous["sources"], current)
    if not changed or versions() != manifest["versions"]:
        raise ValueError("Runtime/environment changed outside reserve correction")
    state = read_json(out / "state.json")
    if any(state["steps"].get(str(s)) != "completed" for s in range(1, 10)):
        raise ValueError("Reserve migration requires completed steps 1 through 9")
    calibration = load_record(out / "calibration.json")
    if not calibration or not calibration.get("complete") or calibration["manifest_hash"] != digest(manifest):
        raise ValueError("Missing frozen calibration")
    models = {k: sha(v) for k, v in model_files(manifest, out).items()}
    if models != calibration["models"] or models != previous["models"]:
        raise ValueError("Frozen model stack changed")
    for case in manifest["train"] + manifest["cohort"]:
        if sha(case["file"]) != case["scenario_sha256"]:
            raise ValueError("Frozen scenario changed")

    archive = out / "archives" / KIND
    archive.mkdir(parents=True, exist_ok=True)
    state_copy = archive / "state_before.json"
    if not state_copy.exists():
        shutil.copy2(out / "state.json", state_copy)
    retained = {state_copy, out / "calibration.json", out / "calibration.json.bak",
                out / PREVIOUS_RECEIPT, out / (PREVIOUS_RECEIPT + ".bak")}
    counts, records = Counter(), []
    originals = set()
    for item in list(schedule(manifest)) + list(schedule(manifest, extra=True)):
        ident = dict(item, manifest_hash=digest(manifest))
        path = out / "experiments" / (digest(ident) + ".json")
        originals.add(path.name)
        old = load_record(path, ident)
        if old is None:
            continue
        unchanged = effective_reserve(item["budget"], calibration["reserve_s"]) == calibration["reserve_s"]
        decision = "reuse" if unchanged and old.get("complete") else "superseded" if old.get("complete") else "interrupted"
        counts[decision] += 1
        counts[f"{decision}_{old.get('status', 'unknown')}"] += 1
        records.append(dict(identity=ident, record_file=str(path.relative_to(out)),
                            record_sha256=old["sha256"], decision=decision, status=old.get("status")))
        for original in (path, Path(str(path) + ".bak")):
            if not original.exists():
                continue
            if old.get("complete"):
                retained.add(original)
            else:
                # Only an interrupted marker may be replaced by resume; keep its
                # original bytes too. The old attempt directory is never reused.
                target = archive / original.name
                if target.exists() and sha(target) != sha(original):
                    raise ValueError("Interrupted record archive already differs")
                if not target.exists():
                    shutil.copy2(original, target)
                retained.add(target)
    unexpected = [p.name for p in (out / "experiments").glob("*.json") if p.name not in originals]
    if unexpected:
        raise ValueError("Unexpected experiment revisions already exist")
    if source_signature() != current:
        raise ValueError("Sources changed during reserve registration")
    save_record(out / RECEIPT_NAME, dict(complete=True, kind=KIND, policy=POLICY,
        created_at=datetime.now(timezone.utc).isoformat(), manifest_hash=digest(manifest),
        previous_sources_digest=digest(previous["sources"]), previous_receipt_sha256=sha(out / PREVIOUS_RECEIPT),
        calibration_sha256=sha(out / "calibration.json"), sources=current, changed_sources=changed, models=models,
        calibrated_reserve_s=calibration["reserve_s"],
        reserve_by_budget={str(b): effective_reserve(b, calibration["reserve_s"]) for b in BUDGETS},
        retained_files={str(p.relative_to(out)): sha(p) for p in sorted(retained)},
        original_records=records, counts=dict(counts), retained_steps=list(range(1, 10)),
        training_changed=False, worker_changed=False, validator_changed=False, timing_boundary_changed=False,
        policy_note="Reuse every unchanged-budget outcome, including failures. Supersede every changed-budget outcome, including successes. Never recover killed raw results."))
    receipt = load_record(out / RECEIPT_NAME)
    for step in (10, 11, 12):
        state["steps"].pop(str(step), None)
    state.setdefault("protocol_amendments", []).append(dict(kind=KIND, sha256=receipt["sha256"]))
    atomic_json(out / "state.json", state)
    effective_sources(manifest, out, check_artifacts=True)
    return receipt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()
    from scripts.benchmark_matched_budgets import campaign_lock
    from scripts.recompute_thermal_references import exclusive_lock
    out = args.output_dir.resolve()
    manifest = load_record(out / "manifest.json")
    with campaign_lock(Path(manifest["source"])), exclusive_lock(out / "run.lock"):
        receipt = register_policy(manifest, out)
    import json
    print(json.dumps(dict(registered=True, counts=receipt["counts"], reserve_by_budget=receipt["reserve_by_budget"],
                          revision=receipt["sha256"]), ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
