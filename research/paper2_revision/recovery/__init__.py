"""Resume preparation without regenerating verified inputs or changing the freeze.

The original writer compares Python tuples to lists reloaded from JSON. Both
serialize identically, but compare unequal. Only a still-incomplete preparation
needs the JSON normalization adapter below. Completed inputs are read-only.
This package is outside the scientific source inventory; no solver is changed.
"""
from collections import Counter
import json
from pathlib import Path

from .. import core
from ..design import planned_blocks


def json_value(value):
    """Use the same representation on both sides of an immutable JSON write."""
    return json.loads(json.dumps(value, ensure_ascii=False, allow_nan=False))


def publish_case(path, value):
    # The original writer still rejects every real change and owns publication.
    return core.put(path, json_value(value))


def verify_cases(run, snap):
    run = Path(run)
    manifest = core.verify(core.read(run / "cases.json"))
    if manifest["freeze"] != snap["sha256"]:
        raise ValueError("Case manifest belongs to a different frozen protocol")
    entries = manifest["entries"]
    if len({entry["key"] for entry in entries}) != len(entries):
        raise ValueError("Duplicate case keys in prepared manifest")
    cfg = core.config()
    nf = len(core.FAMILIES)
    expected = dict(development=nf * cfg["new_cases_per_family"],
                    final=nf * cfg["new_cases_per_family"],
                    controlled_development=nf * cfg["controlled_cases_per_family"] * 7,
                    controlled_test=nf * cfg["controlled_cases_per_family"] * 7,
                    mh_training=nf * cfg["training_bases_per_family"] * 3, gurobi120=120)
    actual = Counter(entry["partition"] for entry in entries)
    if actual != Counter(expected):
        raise ValueError(f"Prepared cohort counts differ from the frozen plan: {dict(actual)}")
    for entry in entries:
        if core.sha(entry["file"]) != entry["scenario_sha256"]:
            raise ValueError(f"Prepared scenario changed or is missing: {entry['key']}")
    return manifest


def verify_audit(run, snap):
    run = Path(run)
    audit = core.verify(core.read(run / "input_audit.json"))
    if audit["freeze"] != snap["sha256"] or audit["cases_sha256"] != core.sha(run / "cases.json"):
        raise ValueError("Input audit no longer matches the frozen scenarios")
    return audit


def verify_prepared(run):
    """Read-only verification, including source, environment and model guards."""
    run = Path(run)
    snap = core.frozen(run)
    manifest = verify_cases(run, snap)
    verify_audit(run, snap)
    counts = run / "planned_counts.json"
    if counts.exists() and core.verify(core.read(counts)) != core.sealed(planned_blocks()):
        raise ValueError("Saved campaign counts differ from the frozen plan")
    return dict(status="verified", cases=len(manifest["entries"]), freeze=snap["sha256"])


def prepare(run):
    """Idempotent Prepare; retain the original lock, freeze and input audit."""
    run = Path(run)
    with core.lock(run):
        if (run / "freeze.json").exists():
            snap = core.frozen(run)
        else:
            if (run / "cases.json").exists() or (run / "input_audit.json").exists():
                raise ValueError("Prepared inputs exist without their freeze; no files were replaced")
            snap = core.freeze(run)
        if (run / "cases.json").exists():
            manifest = verify_cases(run, snap)
            print(f"Reusing {len(manifest['entries'])} verified scenarios; no regeneration.", flush=True)
        else:
            if (run / "input_audit.json").exists():
                raise ValueError("Input audit exists but its case manifest is missing; restore the manifest")
            from .. import cases
            original_put = cases.put
            try:
                cases.put = publish_case
                cases.make_cases(run)
            finally:
                cases.put = original_put
            verify_cases(run, snap)
        if (run / "input_audit.json").exists():
            verify_audit(run, snap)
            print("Reusing the verified input audit.", flush=True)
        else:
            from ..audit import input_audit
            input_audit(run)
            verify_audit(run, snap)
        core.put(run / "planned_counts.json", core.sealed(planned_blocks()))
    print("Completed stage Prepare (existing results preserved).", flush=True)
