"""Read-only access to historical records; immutable, isolated new artifacts."""
from __future__ import annotations
import contextlib
import hashlib
import importlib.metadata as md
import json
import math
import os
from pathlib import Path
import platform
import uuid
from functools import lru_cache

ROOT = Path(__file__).resolve().parents[2]
HOME = Path(__file__).resolve().parent
METHODS = ("highs", "rh", "jepa", "pipeline")
FAMILIES = ("low_criticality_scenarios", "medium_criticality_scenarios",
            "high_criticality_scenarios", "persona_vre_battery_scenarios",
            "persona_network_operator_scenarios", "persona_mathematician_scenarios")

def digest(obj):
    return hashlib.sha256(json.dumps(obj, sort_keys=True, allow_nan=False,
                                    separators=(",", ":")).encode()).hexdigest()

def io(path):
    p = str(Path(path).resolve())
    return Path(("\\\\?\\UNC\\" + p[2:] if p.startswith("\\\\") else "\\\\?\\" + p)
                if os.name == "nt" and not p.startswith("\\\\?\\") else p)

def sha(path):
    h = hashlib.sha256()
    with io(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()

def read(path):
    return json.loads(io(path).read_text(encoding="utf-8-sig"))

def sealed(obj):
    body = {k: v for k, v in obj.items() if k != "sha256"}
    return dict(body, sha256=digest(body))

def verify(obj):
    if not isinstance(obj, dict) or obj.get("sha256") != sealed(obj)["sha256"]:
        raise ValueError("Invalid sealed record")
    return obj

def record(path, identity=None, optional=False):
    """Never repair, rename or write a source. Reject inconsistent copies."""
    copies = []
    for candidate in (Path(path), Path(str(path) + ".bak")):
        if io(candidate).exists():
            obj = verify(read(candidate))
            if identity is not None and obj.get("identity") != identity:
                raise ValueError(f"Identity mismatch: {candidate}")
            copies.append(obj)
    if not copies:
        if optional:
            return None
        raise FileNotFoundError(path)
    if any(x != copies[0] for x in copies[1:]):
        raise ValueError(f"Copies disagree, possibly an active write: {path}. Retry after campaign completion.")
    return copies[0]

def writable(path):
    raw = str(path)
    if raw.startswith("\\\\?\\UNC\\"):
        raw = "\\\\" + raw[8:]
    elif raw.startswith("\\\\?\\"):
        raw = raw[4:]
    p = Path(raw).resolve()
    if not p.is_relative_to(HOME):
        raise ValueError(f"Writes are restricted to {HOME}: {p}")
    return io(p)

def put(path, obj):
    """Immutable JSON publication. An identical restart is a no-op."""
    p = writable(path)
    if p.exists():
        if read(p) != obj:
            raise ValueError(f"Immutable output changed; use a new run directory: {path}")
        return
    p.parent.mkdir(parents=True, exist_ok=True)
    temp = p.with_name(p.name + "." + uuid.uuid4().hex + ".tmp")
    with temp.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(obj, stream, ensure_ascii=False, allow_nan=False, sort_keys=True, indent=2)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    # Windows rename does not overwrite. A single writer lock protects POSIX too.
    temp.rename(p)

@contextlib.contextmanager
def lock(run):
    """OS-owned advisory lock: released on crash; file is intentionally retained."""
    p = writable(Path(run) / ".writer.lock")
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

@lru_cache(maxsize=1)
def config():
    return read(HOME / "config.json")

def campaign():
    return ROOT / config()["campaign"]

def seed(*parts):
    return int(digest([config()["seed"], *parts])[:8], 16) % (2**31 - 1)

def environment():
    names = ("numpy", "pandas", "scipy", "scikit-learn", "matplotlib", "torch",
             "torch-geometric", "highspy", "pyomo", "psutil", "PyYAML")
    result = {"python": platform.python_version(), "platform": platform.platform()}
    # Gurobi is an optional, separately sealed comparator, not a dependency
    # of the open-method experiment freeze.
    for name in names:
        try:
            result[name] = md.version(name)
        except md.PackageNotFoundError:
            result[name] = None
    return result

def source_hashes():
    from src.operational_exit.protocol import source_signature
    paths = list(HOME.glob("*.py")) + list((HOME / "tests").glob("*.py")) + [HOME / "config.json"]
    paths += [ROOT / "config/scenario_space.yaml", ROOT / "scripts/generate_eval_families.py",
              ROOT / "scripts/generate_persona_scenarios.py"]
    return dict(source_signature(), **{p.relative_to(ROOT).as_posix(): sha(p) for p in paths})

def model_paths(manifest):
    c = campaign()
    return dict(hte=str(Path(manifest["source"]) / "encoders/hierarchical_temporal_v3/best_encoder.pt"),
                hte_features=str(c / "hte_feature_contract.json"),
                ae=str(c / "training/autoencoder/uc_ae_best.pt"),
                jepa=str(c / "training/jepa/jepa_best.pt"),
                ebm=str(c / "training/latent_ebm/latent_ebm_best.pt"))

def scheduled():
    from src.operational_exit.budget_reserve import evaluation_schedule
    c = campaign()
    m = verify(read(c / "manifest.json"))
    cal = record(c / "calibration.json")
    amend = record(c / "budget_reserve_amendment.json")
    for extra in (False, True):
        for item in evaluation_schedule(m, cal, amend, extra=extra):
            identity = dict(item, manifest_hash=digest(m))
            yield m, item, identity, c / "experiments" / (digest(identity) + ".json")

def inventory():
    from collections import Counter
    states = Counter(); issues = []; files = {}
    m = verify(read(campaign() / "manifest.json"))
    for _, item, identity, p in scheduled():
        try:
            r = record(p, identity, optional=True)
            state = "not_run" if r is None else "incomplete" if not r.get("complete") else "observed"
            if r is not None:
                files[str(p)] = r["sha256"]
        except (ValueError, OSError) as exc:
            state = "unreadable"; issues.append(dict(file=str(p), error=str(exc)))
        states[f"rep{item['repetition']}:{state}"] += 1
    return dict(protocol=config()["protocol"], manifest_hash=digest(m), states=dict(states),
                expected=18480, state_file=read(campaign() / "state.json"), issues=issues, records=files)

def freeze(run):
    run = Path(run)
    inv = inventory()
    if sum(inv["states"].values()) != 18480 or any(not k.endswith(":observed") for k in inv["states"]):
        raise ValueError("Campaign incomplete or changing; inventory is available, final freeze must wait")
    steps = inv["state_file"]["steps"]
    if any(steps.get(str(s)) != "completed" for s in (10, 11)):
        raise ValueError("Campaign has not declared steps 10 and 11 complete")
    m = verify(read(campaign() / "manifest.json"))
    paths = model_paths(m)
    from src.operational_exit.protocol import source_signature
    amendment = record(campaign()/"budget_reserve_amendment.json")
    calibration = record(campaign()/"calibration.json")
    if amendment["sources"] != source_signature() or amendment["manifest_hash"] != digest(m):
        raise ValueError("Current runtime does not match the sealed reserve amendment")
    if calibration["manifest_hash"] != digest(m) or calibration["models"] != {k:sha(p) for k,p in paths.items()}:
        raise ValueError("Calibration/model provenance mismatch")
    current_environment = environment()
    for name, version in m["versions"].items():
        if current_environment.get(name) != version:
            raise ValueError(f"Campaign runtime version changed: {name}; preserve its environment for matched extensions")
    for case in m["train"] + m["cohort"]:
        if sha(case["file"]) != case["scenario_sha256"]:
            raise ValueError("Historical scenario changed")
    refs = {}
    for case in m["cohort"]:
        p = campaign() / "references" / case["family"] / (case["scenario_id"] + ".json")
        refs[str(p)] = record(p)["sha256"]
    snap = sealed(dict(config=config(), sources=source_hashes(), environment=current_environment,
                       manifest_hash=digest(m), model_paths=paths,
                       models={k: sha(p) for k, p in paths.items()}, references=refs,
                       records=inv["records"], calibration=calibration,
                       reserve_amendment=amendment))
    put(run / "freeze.json", snap)
    return snap

def frozen(run):
    snap = verify(read(Path(run) / "freeze.json"))
    if snap["sources"] != source_hashes() or snap["config"] != config():
        raise ValueError("Source/configuration changed after freeze; choose a new run")
    if snap["environment"] != environment():
        raise ValueError("Environment changed after freeze; install prerequisites BEFORE freeze")
    if any(sha(p) != snap["models"][k] for k, p in snap["model_paths"].items()):
        raise ValueError("Frozen model changed")
    return snap

def gap(cost, reference, cfg=None):
    cfg = cfg or config()
    if cost is None or reference is None or not all(math.isfinite(x) for x in (cost, reference)):
        return None, None
    delta = cost - reference
    return (delta / abs(reference) if abs(reference) > cfg["reference_zero_eur"] else None), delta

def economic(success, cost, reference, threshold, cfg=None):
    cfg = cfg or config()
    if success is None:
        return None
    if not success:
        return 0
    if reference is None or not math.isfinite(reference):
        return None
    rel, delta = gap(cost, reference, cfg)
    if delta is None:
        return None
    return int(rel <= threshold if rel is not None else delta <= cfg["near_zero_absolute_tolerance_eur"])
