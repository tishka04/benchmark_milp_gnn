"""Durable, checksummed records; no favourable retry of experimental failures."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import time
import uuid

PROTOCOL = "operational_exit_v1"
PHYSICS = "thermal_transitions_v1+load_shifting_v1+energy_cost_dt_v1"
BUDGETS = (1, 2, 5, 10, 30, 60, 120)
METHODS = ("highs", "pipeline", "jepa", "rh")
FAMILIES = ("low_criticality_scenarios", "medium_criticality_scenarios",
            "high_criticality_scenarios", "persona_vre_battery_scenarios",
            "persona_network_operator_scenarios", "persona_mathematician_scenarios")
ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUT = ROOT / "outputs/thermal_transitions_v1/operational_exit_v1"
THREAD_ENV = ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
              "NUMEXPR_NUM_THREADS", "BLIS_NUM_THREADS")


def digest(obj):
    return hashlib.sha256(json.dumps(obj, sort_keys=True, allow_nan=False,
                                     separators=(",", ":")).encode()).hexdigest()


def io_path(path):
    """Use Windows' extended path syntax without changing registry settings.

    An otherwise valid result path can exceed MAX_PATH once the atomic write
    appends its unique temporary suffix. Keep the temporary in the same folder.
    """
    path = Path(path)
    if os.name != "nt":
        return path
    absolute = os.path.abspath(path)
    if absolute.startswith("\\\\?\\"):
        return Path(absolute)
    if absolute.startswith("\\\\"):
        return Path("\\\\?\\UNC\\" + absolute[2:])
    return Path("\\\\?\\" + absolute)


def sha(path):
    h = hashlib.sha256()
    with io_path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def retry_io(fn, deadline=None):
    until = time.perf_counter() + 2 if deadline is None else deadline
    while True:
        try:
            return fn()
        except PermissionError:
            if time.perf_counter() >= until:
                raise
            time.sleep(min(.02, max(0, until - time.perf_counter())))


def read_json(path):
    return json.loads(retry_io(lambda: io_path(path).read_text(encoding="utf-8")))


def atomic_json(path, obj, *, deadline=None):
    path = io_path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    raw = json.dumps(obj, sort_keys=True, allow_nan=False, ensure_ascii=False).encode("utf-8")
    temp = path.with_name(path.name + "." + uuid.uuid4().hex + ".tmp")
    with temp.open("xb") as f:
        f.write(raw)
        f.flush()
        os.fsync(f.fileno())
    retry_io(lambda: os.replace(temp, path), deadline)


def sealed(obj):
    body = {k: v for k, v in obj.items() if k != "sha256"}
    return dict(body, sha256=digest(body))


def verify(obj):
    if not isinstance(obj, dict) or obj.get("sha256") != sealed(obj)["sha256"]:
        raise ValueError("Invalid checkpoint checksum")
    return obj


def save_record(path, obj):
    # Two independent copies of the SAME terminal outcome, including failures.
    # A crash between writes can only recover that outcome, never an older try.
    payload = sealed(obj)
    atomic_json(str(path) + ".bak", payload)
    atomic_json(path, payload)


def load_record(path, identity=None):
    path = io_path(path)
    damaged, copies = [], {}
    for candidate in (path, Path(str(path) + ".bak")):
        if not candidate.exists():
            continue
        try:
            result = verify(read_json(candidate))
        except (ValueError, UnicodeError):
            damaged.append(candidate)
            continue
        if identity is not None and result.get("identity") != identity:
            raise ValueError(f"Incompatible checkpoint: {candidate}")
        copies[candidate] = result
    # Permission failures deliberately propagate: never retry a locked result.
    for bad in damaged:
        retry_io(lambda bad=bad: bad.rename(str(bad) + ".corrupt-" + uuid.uuid4().hex))
    # The backup is published FIRST, so it can contain the terminal outcome
    # while the primary still contains the preceding incomplete marker.
    result = copies.get(Path(str(path) + ".bak"), copies.get(path))
    if result is not None and copies.get(path) != result:
        atomic_json(path, result)
    return result


def seed_for(case, repetition=0, candidate=None):
    # Deliberately independent of budget and method.
    return int(digest([42, case, repetition, candidate])[:8], 16) % (2**31 - 1)


def finite(value):
    import math
    return isinstance(value, (int, float)) and math.isfinite(value)
