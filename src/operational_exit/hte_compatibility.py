"""Recover the HTE's categorical vocabulary from training inputs, never labels.

The historical builder used Python's process-randomized hash(profile) % 100.
Freezing its observed training vocabulary reproduces the encoder inputs exactly
while allowing fresh, scenario-only graph construction in every timed worker.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
from pathlib import Path
import time

import numpy as np

from .common import DEFAULT_OUT, digest, load_record, read_json, save_record, sha

CONTRACT_NAME = "hte_feature_contract.json"
AMENDMENT_NAME = "runtime_amendment.json"
AMENDMENT_KIND = "hte_weather_encoding_v1"
REPAIR_SOURCES = frozenset({
    "src/operational_exit/hte_compatibility.py", "src/operational_exit/embedding.py",
    "src/operational_exit/generation.py", "src/operational_exit/workflows.py",
    "src/operational_exit/campaign.py", "src/operational_exit/protocol.py",
})
INPUT_ARRAYS = ("node_features", "edge_index", "edge_types",
                "node_types", "node_ids", "zone_region_index")


def weather_rows(record, data):
    meta = record["meta"]
    meta = meta.item() if hasattr(meta, "item") else meta
    n, periods = int(meta["N_base"]), int(meta["T"])
    names = np.asarray(record["node_ids"])
    types = np.asarray(record["node_types"])
    expected = {f"Weather_{region}#t=0": profile for region, profile in data.region_weather_profile.items()}
    rows = np.flatnonzero(types[:n] == 4)
    if len(rows) != len(expected) or len(names) != n * periods:
        raise ValueError("HTE weather node axes changed")
    result = []
    for row in rows:
        name = str(names[row])
        if name not in expected:
            raise ValueError(f"Unknown weather node: {name}")
        indices = row + n * np.arange(periods)
        expected_names = [name.removesuffix("#t=0") + f"#t={t}" for t in range(periods)]
        if not np.array_equal(names[indices], expected_names) or not np.all(types[indices] == 4):
            raise ValueError("HTE temporal weather node axes changed")
        result.append((indices, expected[name]))
    return result


def recover_weather_codes(record, data, codes):
    """Reject mixed historical encodings instead of choosing one per scenario."""
    for indices, profile in weather_rows(record, data):
        values = np.asarray(record["node_features"])[indices, 1]
        code = float(values[0])
        if not np.isfinite(values).all() or not np.all(values == code) or code != int(code) or not 0 <= code < 100:
            raise ValueError(f"Invalid historical weather code: {profile}")
        if profile in codes and codes[profile] != int(code):
            raise ValueError(f"Conflicting historical weather encodings for {profile}; HTE reuse refused")
        codes[profile] = int(code)


def apply_weather_codes(record, data, codes):
    for indices, profile in weather_rows(record, data):
        if profile not in codes:
            raise ValueError(f"Weather profile absent from frozen HTE training vocabulary: {profile}")
        record["node_features"][indices, 1] = codes[profile]
    return record


def assert_graph_equal(old, new, context):
    for key in INPUT_ARRAYS:
        if not np.array_equal(old[key], new[key]):
            raise ValueError(f"HTE reuse refused: {key} changed in {context}")
    # Neither TemporalGraphDatasetV3, graph_data nor HTE.forward consumes
    # edge_features. Historical line-distance metadata can differ; all edge
    # topology, line capacities and other auxiliary attributes still agree.
    a, b = np.asarray(old["edge_features"]), np.asarray(new["edge_features"])
    if a.shape != b.shape:
        raise ValueError(f"HTE reuse refused: edge_features axes changed in {context}")
    distance = (np.asarray(old["edge_types"])[:, None] == 5) & (np.arange(a.shape[1])[None, :] == 1)
    if np.any((a != b) & ~distance):
        raise ValueError(f"HTE reuse refused: edge_features outside unused line distance changed in {context}")


def load_feature_contract(path, checkpoint, manifest_hash=None):
    contract = load_record(path)
    if (not contract or not contract.get("complete") or contract.get("kind") != AMENDMENT_KIND
            or contract.get("hte_sha256") != sha(checkpoint)
            or (manifest_hash is not None and contract.get("manifest_hash") != manifest_hash)):
        raise ValueError("Missing/incompatible frozen HTE feature contract")
    codes = contract.get("weather_codes", {})
    if not codes or any(not isinstance(p, str) or type(c) is not int or not 0 <= c < 100 for p, c in codes.items()):
        raise ValueError("Invalid frozen HTE weather vocabulary")
    return contract


def ensure_feature_contract(manifest, out, progress):
    out, source = Path(out), Path(manifest["source"])
    checkpoint = source / "encoders/hierarchical_temporal_v3/best_encoder.pt"
    target = out / CONTRACT_NAME
    if target.exists() or Path(str(target) + ".bak").exists():
        return load_feature_contract(target, checkpoint, digest(manifest))
    if sha(checkpoint) != manifest["hte_sha256"]:
        raise ValueError("HTE changed since campaign preparation")
    from src.demand_response import require_current_formulation
    from src.gnn.hetero_graph_dataset import build_hetero_temporal_record
    from src.milp.scenario_loader import load_scenario_data
    started = time.perf_counter()
    codes, evidence = {}, []
    for i, case in enumerate(manifest["train"], 1):
        if case["family"] != "scenarios_v3" or sha(case["file"]) != case["scenario_sha256"]:
            raise ValueError("HTE vocabulary audit requires unchanged training inputs only")
        data = load_scenario_data(Path(case["file"]))
        graph = source / "graphs/scenarios_v3" / (case["scenario_id"] + ".npz")
        graph_hash = sha(graph)
        new = build_hetero_temporal_record(data, mode="supra", temporal_edges=("soc", "ramp", "dr"), time_encoding="sinusoidal")
        with np.load(graph, allow_pickle=True) as old:
            require_current_formulation(old, graph)
            recover_weather_codes(old, data, codes)
            apply_weather_codes(new, data, codes)
            assert_graph_equal(old, new, graph)
        if sha(graph) != graph_hash:
            raise ValueError("Historical graph changed during compatibility audit")
        evidence.append(dict(key=case["key"], scenario_sha256=case["scenario_sha256"], graph_sha256=graph_hash))
        progress(i, len(manifest["train"]), "HTE input compatibility: " + case["key"])
    if sha(checkpoint) != manifest["hte_sha256"]:
        raise ValueError("HTE changed during compatibility audit")
    save_record(target, dict(complete=True, kind=AMENDMENT_KIND, manifest_hash=digest(manifest),
        hte_sha256=manifest["hte_sha256"], weather_codes=codes, checked_graphs=len(evidence), evidence=evidence,
        exact_input_arrays=list(INPUT_ARRAYS), ignored_auxiliary_metadata="transmission line distance; never consumed by HTE",
        source="historical training graph weather nodes; no labels or evaluation graphs",
        elapsed_s=time.perf_counter() - started))
    return load_feature_contract(target, checkpoint, digest(manifest))


def check_repair_sources(manifest, sources):
    before = manifest["sources"]
    changed = {p for p in before.keys() | sources.keys() if before.get(p) != sources.get(p)}
    if not changed <= REPAIR_SOURCES or before.keys() - sources.keys():
        raise ValueError("Changes outside the HTE compatibility repair; refusing to reuse frozen results")
    return sorted(changed)


def register_runtime_repair(manifest, out, progress):
    """One documented amendment before embeddings/training/evaluation exist.

    Keep the original manifest byte-for-byte: label/reference identities remain
    valid, and the receipt records exactly which runtime sources supersede it.
    """
    from .protocol import source_signature, versions, effective_sources
    out = Path(out)
    if (out / AMENDMENT_NAME).exists():
        effective_sources(manifest, out)
        return
    sources = source_signature()
    changed = check_repair_sources(manifest, sources)
    if not changed or manifest["versions"] != versions():
        raise ValueError("Expected only the HTE repair with the unchanged runtime")
    state = read_json(out / "state.json")
    if any(state["steps"].get(str(s)) != "completed" for s in (2, 3, 4)):
        raise ValueError("Repair requires completed audit, labels and references")
    if any(state["steps"].get(str(s)) == "completed" for s in range(5, 13)):
        raise ValueError("Cannot amend a campaign after embeddings/training/evaluation completed")
    if any((out / p).exists() for p in ("dataset.json", "training", "calibration.json", "calibration_runs", "experiments")):
        raise ValueError("Cannot amend a campaign with downstream results")
    if list((out / "dataset").rglob("*_uc.npz")):
        raise ValueError("Cannot mix previously generated embeddings with the HTE repair")
    for case in manifest["train"] + manifest["cohort"]:
        if sha(case["file"]) != case["scenario_sha256"]:
            raise ValueError("Frozen scenario changed")
    ensure_feature_contract(manifest, out, progress)
    if source_signature() != sources:
        raise ValueError("Code changed during compatibility audit")
    save_record(out / AMENDMENT_NAME, dict(complete=True, kind=AMENDMENT_KIND,
        created_at=datetime.now(timezone.utc).isoformat(), manifest_hash=digest(manifest),
        reason="Replace randomized Python weather hashes by the exact historical training vocabulary",
        sources=sources, changed_sources=changed, feature_contract_sha256=sha(out / CONTRACT_NAME),
        retained_steps=[2, 3, 4], physics_changed=False, original_manifest_unchanged=True))
    effective_sources(manifest, out)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()
    from scripts.recompute_thermal_references import exclusive_lock
    from scripts.benchmark_matched_budgets import campaign_lock
    manifest = load_record(args.output_dir / "manifest.json")
    if manifest is None:
        raise ValueError("Missing frozen manifest")
    def progress(i, n, text):
        if i == 1 or i % 100 == 0 or i == n:
            print(f"{i}/{n} {text}", flush=True)
    with campaign_lock(Path(manifest["source"])), exclusive_lock(args.output_dir / "run.lock"):
        register_runtime_repair(manifest, args.output_dir, progress)
    print("HTE compatibility repair registered; labels, references and original manifest retained.", flush=True)


if __name__ == "__main__":
    main()
