"""Read sealed historical tables; write only to this package's own run directory."""
from __future__ import annotations
import contextlib
from functools import lru_cache
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import platform
import uuid

HOME = Path(__file__).resolve().parent
ROOT = HOME.parents[1]
DEFAULT_SOURCE = ROOT/"research/three_papers/runs/v2_service_by_deadline"
DEFAULT_RUN = HOME/"runs/components_v1"
SIZE = ["n_zones", "n_steps", "n_lines", "zone_steps"]
COMPONENTS = [
    "A1_vre_penetration", "A2_residual_volatility", "A3_peak_to_valley",
    "A4_short_term_variability", "A5_demand_scale", "A6_peak_to_firm",
    "A7_inv_storage_power", "A8_inv_storage_energy", "A9_inv_thermal_flex",
    "A10_inv_dr_headroom", "A11_trade_reliance", "A12_inv_congestion",
    "B1_n_zones", "B2_horizon", "B3_binary_est", "B4_thermal_density",
    "B5_min_gen_tight", "B6_startup_intensity", "B7_ramp_tightness",
    "B8_soc_tightness", "B9_interconn_density", "B10_network_hetero"]
INTERPRETABLE = ["A2_residual_volatility", "A6_peak_to_firm", "A8_inv_storage_energy",
                 "A10_inv_dr_headroom", "B3_binary_est", "B7_ramp_tightness"]
DESCRIPTORS = SIZE + ["criticality", "stress", "hardness"] + COMPONENTS
FEATURE_AUDIT = ["raw_"+c for c in COMPONENTS] + ["sat_"+c for c in COMPONENTS]

def digest(obj):
    return hashlib.sha256(json.dumps(obj, sort_keys=True, allow_nan=False,
                                    separators=(",", ":")).encode()).hexdigest()

def sealed(obj):
    body = {k: v for k, v in obj.items() if k != "sha256"}
    return dict(body, sha256=digest(body))

def verify(obj):
    if not isinstance(obj, dict) or obj.get("sha256") != sealed(obj)["sha256"]:
        raise ValueError("Invalid sealed JSON")
    return obj

def read(p):
    return json.loads(Path(p).read_text(encoding="utf-8-sig"))

def sha(p):
    h = hashlib.sha256()
    with Path(p).open("rb") as stream:
        for block in iter(lambda: stream.read(1024*1024), b""):
            h.update(block)
    return h.hexdigest()

@lru_cache(maxsize=1)
def protocol():
    cfg = read(HOME/"protocol.json")
    if cfg["deadlines_s"] != [5, 10, 30, 60, 120]:
        raise ValueError("This implementation declares exactly five deadlines")
    if any(c not in COMPONENTS for c in cfg["targeted_components"]):
        raise ValueError("Unknown targeted component")
    pairs = cfg["interactions"]
    if len({tuple(sorted(p)) for p in pairs}) != len(pairs):
        raise ValueError("Duplicate interaction")
    if any(len(p) != 2 or p[0] == p[1] or any(c not in COMPONENTS for c in p) for p in pairs):
        raise ValueError("Invalid interaction parents")
    if cfg["outer_folds"] < 2 or cfg["inner_folds"] < 2 or min(cfg["regularization_C"]) <= 0:
        raise ValueError("Invalid fitting protocol")
    return cfg

def seed(*parts):
    return int(digest([protocol()["seed"], *map(str, parts)])[:8], 16)

def writable(p):
    p = Path(p).resolve()
    if not p.is_relative_to((HOME/"runs").resolve()):
        raise ValueError("Paper-1 writes must stay in research/paper1_components/runs")
    return p

def put(p, obj):
    p = writable(p)
    if p.exists():
        if read(p) != obj:
            raise ValueError(f"Immutable output differs; choose a new paper-1 run: {p}")
        return
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_name(p.name+"."+uuid.uuid4().hex+".tmp")
    with tmp.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(obj, stream, ensure_ascii=False, allow_nan=False, sort_keys=True, indent=2)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    tmp.rename(p)

@contextlib.contextmanager
def lock(run):
    p = writable(Path(run)/".writer.lock")
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("a+b") as stream:
        stream.seek(0)
        if os.name == "nt":
            import msvcrt
            if p.stat().st_size == 0:
                stream.write(b"0"); stream.flush(); stream.seek(0)
            msvcrt.locking(stream.fileno(), msvcrt.LK_NBLCK, 1)
        else:
            import fcntl
            fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        try:
            yield
        finally:
            stream.seek(0)
            if os.name == "nt":
                msvcrt.locking(stream.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                fcntl.flock(stream, fcntl.LOCK_UN)

def environment():
    names = ["numpy", "pandas", "scikit-learn", "matplotlib", "scipy"]
    return dict(python=platform.python_version(),
                packages={n: importlib.metadata.version(n) for n in names})

def source_hashes():
    files = list(HOME.glob("*.py")) + list((HOME/"tests").glob("*.py")) + [HOME/"protocol.json"]
    return {p.relative_to(HOME).as_posix(): sha(p) for p in files}

def finite(x):
    return isinstance(x, (int, float)) and not isinstance(x, bool) and math.isfinite(x)

def labels(row, cfg):
    """Known failures dominate unknown active conditions; never relabel an allocation."""
    if row.get("observed") is not True:
        return {e: None for e in cfg["endpoints"]}
    valid = row.get("feasible")
    valid = int(valid) if isinstance(valid, bool) else None
    t, b = row.get("exit_s"), row["budget"]
    timely = int(0 <= t <= b) if finite(t) else None
    cost, ref, ens = row.get("cost"), row.get("reference_cost"), row.get("ens_mwh")
    quality = None
    if finite(cost) and finite(ref):
        delta = cost-ref
        quality = int(delta <= (cfg["cost_tolerance"]*abs(ref) if abs(ref) > cfg["reference_zero_eur"]
                                else cfg["near_zero_absolute_tolerance_eur"]))
    eps = cfg["zero_ens_tolerance_mwh"]
    service = int(ens <= eps) if finite(ens) and ens >= -eps else None
    def conjunction(flags):
        return 0 if 0 in flags else None if None in flags else 1
    return dict(economic_5=conjunction([valid, timely, quality]),
                economic_5_ens=conjunction([valid, timely, quality, service]))

def select_rows(rows, cfg):
    import numpy as np
    import pandas as pd
    selected = []
    for row in rows:
        if row["method"] != cfg["method"] or row["repetition"] != cfg["repetition"]:
            continue
        if row["budget"] not in cfg["deadlines_s"]:
            continue
        if row["partition"] != "historical_development" or row["condition"] != "historical":
            raise ValueError("Only the original historical development scenarios are accepted")
        if row.get("observed") is not True:
            raise ValueError("The requested completed historical grid has missing observations")
        values = {k: row.get(k) for k in DESCRIPTORS+FEATURE_AUDIT}
        selected.append(dict(key=row["key"], group=row["group"], family=row["family"],
                             budget=row["budget"], **values, **labels(row, cfg)))
    df = pd.DataFrame(selected)
    if df.empty or df.duplicated(["key", "budget"]).any():
        raise ValueError("Empty or duplicate primary observations")
    groups = df[["key", "group", "family"]].drop_duplicates()
    if (len(groups) != cfg["expected_scenarios"] or groups.key.duplicated().any()
            or groups.group.duplicated().any()):
        raise ValueError("Expected exactly 600 independent scenario groups with stable families")
    family_counts = groups.groupby("family").size()
    if len(family_counts) != cfg["expected_families"] or (family_counts != cfg["expected_per_family"]).any():
        raise ValueError("Historical six-family support changed")
    if len(df) != cfg["expected_scenarios"]*len(cfg["deadlines_s"]):
        raise ValueError("Incomplete deadline grid")
    if any(set(q.budget) != set(cfg["deadlines_s"]) for _, q in df.groupby("key")):
        raise ValueError("Each scenario needs its own observations at all five deadlines")
    for c in DESCRIPTORS+FEATURE_AUDIT:
        df[c] = pd.to_numeric(df[c], errors="raise")
        if np.isinf(df[c]).any():
            raise ValueError(f"Infinite descriptor: {c}")
        if df.groupby("key")[c].nunique(dropna=False).max() > 1:
            raise ValueError(f"Pre-solve feature varies across deadlines: {c}")
    for c in COMPONENTS:
        if ((df[c] < -1e-9) | (df[c] > 1+1e-9)).any():
            raise ValueError(f"Normalized component outside [0,1]: {c}")
    return df.sort_values(["key", "budget"]).reset_index(drop=True)

def group_folds(df, count, salt):
    """Outcome-independent balanced assignment within family; one base never splits."""
    import pandas as pd
    units = df[["group", "family"]].drop_duplicates()
    if units.group.duplicated().any():
        raise ValueError("A group belongs to multiple families")
    mapping = {}
    for family, part in units.groupby("family"):
        order = sorted(part.group.tolist(), key=lambda g: (seed(salt, family, g), g))
        for i, g in enumerate(order):
            mapping[g] = i % count
    return pd.Series(df.group.map(mapping).to_numpy(), index=df.index, dtype=int)

def prepare(run, source):
    import pandas as pd
    run, source = writable(run), Path(source).resolve()
    if (run/"manifest.json").exists():
        frozen(run)
        if read(run/"manifest.json")["source_run"] != str(source):
            raise ValueError("Source changed")
        print("Existing paper-1 preparation verified.", flush=True)
        return
    cfg = protocol()
    source_files = [source/"freeze.json", source/"main_rows.json", source/"routing/model_receipt.json"]
    hashes = {str(p): sha(p) for p in source_files}
    snap, obj, router = [verify(read(p)) for p in source_files]
    if obj["freeze"] != snap["sha256"] or router["freeze"] != snap["sha256"]:
        raise ValueError("Historical extraction and router do not share the frozen campaign")
    if router["input_sha256"] != hashes[str(source/"main_rows.json")]:
        raise ValueError("Historical extraction differs from router development input")
    # Read-only integrity verification; do not import or invoke frozen project modules.
    for rel, h in snap["sources"].items():
        if sha(ROOT/rel) != h:
            raise ValueError(f"Frozen source has changed: {rel}")
    df = select_rows(obj["rows"], cfg)
    del obj
    df["outer_fold"] = group_folds(df, cfg["outer_folds"], "paper1-outer")
    data_path = run/"data.json"
    clean = df.astype(object).where(pd.notna(df), None).to_dict("records")
    put(data_path, sealed(dict(rows=clean)))
    if any(sha(p) != h for p, h in hashes.items()):
        raise ValueError("Source changed during preparation")
    receipt = sealed(dict(schema=cfg["schema"], protocol=cfg, sources=source_hashes(),
        environment=environment(), source_run=str(source), input_files=hashes,
        source_freeze=snap["sha256"], historical_sources=snap["sources"],
        data_sha256=sha(data_path), rows=len(df), scenarios=int(df.group.nunique()),
        disclosure=cfg["interpretation"]))
    put(run/"manifest.json", receipt)
    print(f"Prepared {receipt['scenarios']} scenarios, {len(df)} observations; no new solves.", flush=True)

def frozen(run):
    run = writable(run)
    m = verify(read(run/"manifest.json"))
    if m["protocol"] != protocol() or m["sources"] != source_hashes():
        raise ValueError("Paper-1 code/config changed: use a new paper-1 run, preserve the old one")
    if m["environment"] != environment():
        raise ValueError("Paper-1 analysis environment changed")
    for p, h in m["input_files"].items():
        if sha(p) != h:
            raise ValueError(f"Read-only historical input changed: {p}")
    for rel, h in m["historical_sources"].items():
        if sha(ROOT/rel) != h:
            raise ValueError(f"Frozen router/campaign source changed: {rel}")
    if sha(run/"data.json") != m["data_sha256"]:
        raise ValueError("Prepared observations changed")
    return m

def data(run):
    import pandas as pd
    return pd.DataFrame(verify(read(Path(run)/"data.json"))["rows"])

def output_directory(run, name):
    run = writable(run)
    final = run/name
    receipt = final/"receipt.json"
    if receipt.exists():
        r = verify(read(receipt))
        if r["manifest_sha256"] != sha(run/"manifest.json"):
            raise ValueError("Output manifest changed")
        for rel, h in r["outputs"].items():
            if sha(final/rel) != h:
                raise ValueError(f"Output changed: {rel}")
        return final, None
    if final.exists():
        raise ValueError("Unpublished output exists; preserve it and use a new run")
    tmp = writable(run/"attempts"/(name+"_"+uuid.uuid4().hex))
    (tmp/"tables").mkdir(parents=True)
    (tmp/"figures").mkdir()
    return final, tmp

def publish(run, final, tmp):
    outputs = {p.relative_to(tmp).as_posix(): sha(p) for p in tmp.rglob("*") if p.is_file()}
    put(tmp/"receipt.json", sealed(dict(manifest_sha256=sha(Path(run)/"manifest.json"), outputs=outputs)))
    tmp.rename(final)
