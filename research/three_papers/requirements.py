"""Pure requirement semantics. No solver, training, I/O mutation or imports of models."""
from __future__ import annotations
import math
from numbers import Real
from functools import lru_cache
from .core import config, economic, digest

SCHEMA = "service_routing_v2_by_deadline"

def finite(value):
    return value is not None and not isinstance(value, bool) and isinstance(value, Real) and math.isfinite(value)

@lru_cache(maxsize=1)
def _configured_profiles():
    # Configuration is immutable for a frozen process; return copies to callers.
    return profiles(config())

def profiles(cfg=None):
    if cfg is None:
        return [dict(p) for p in _configured_profiles()]
    settings = cfg["service_routing"]
    eps = settings["zero_ens_tolerance_mwh"]
    if not finite(eps) or eps < 0:
        raise ValueError("ENS numerical tolerance must be finite and nonnegative (MWh)")
    result = []
    for raw in settings["profiles"]:
        p = dict(raw)
        p["ens_cap_mwh"] = eps if p["ens_cap_mwh"] == "zero" else p["ens_cap_mwh"]
        for name in ("cost_tolerance", "cost_ceiling_eur", "ens_cap_mwh"):
            v = p.get(name)
            if v is not None and (not finite(v) or (name != "cost_ceiling_eur" and v < 0)):
                raise ValueError(f"Invalid {name}: {v}")
        if p.get("cost_tolerance") is not None and p.get("cost_ceiling_eur") is not None:
            raise ValueError("Use either a relative tolerance or an absolute cost ceiling")
        if not isinstance(p.get("id"), str) or not p["id"] or p["priority"] not in ("cost", "time"):
            raise ValueError("Invalid profile identifier or priority")
        if not finite(p["alpha"]) or not 0 <= p["alpha"] < 1:
            raise ValueError("alpha must belong to [0,1)")
        p.setdefault("cost_ceiling_eur", None)
        p["zero_ens_tolerance_mwh"] = eps
        result.append(p)
    if not result or len({p["id"] for p in result}) != len(result):
        raise ValueError("Profiles must be nonempty and have unique identifiers")
    return result

def profile(profile_id, cfg=None):
    found = [p for p in profiles(cfg) if p["id"] == profile_id]
    if len(found) != 1:
        raise ValueError(f"Unknown request profile: {profile_id}")
    return found[0]

def request_for(profile_id, deadline_s, cfg=None):
    if not finite(deadline_s) or deadline_s <= 0:
        raise ValueError("Deadline must be positive and finite")
    return dict(profile(profile_id, cfg), deadline_s=float(deadline_s))

def check_request(request, cfg=None):
    expected = request_for(request["id"], request["deadline_s"], cfg)
    if request != expected:
        raise ValueError("Request differs from the frozen profile")
    return expected

@lru_cache(maxsize=1)
def _configured_deadlines():
    return tuple(deadlines(config()))

def deadlines(cfg=None):
    if cfg is None:
        return list(_configured_deadlines())
    grid = cfg["service_routing"]["deadlines_s"]
    if not grid or any(not finite(b) or b <= 0 for b in grid):
        raise ValueError("Deadlines must be positive and finite")
    if list(grid) != sorted(set(grid)):
        raise ValueError("Deadlines must be unique and increasing")
    return list(grid)

def request_key(profile_id, deadline_s):
    if not finite(deadline_s) or deadline_s not in deadlines():
        raise ValueError("Deadline is outside the declared evaluation grid")
    return f"{profile_id}@{float(deadline_s):g}s"

def profile_signature(cfg=None):
    return digest(dict(schema=SCHEMA, profiles=profiles(cfg), deadlines_s=deadlines(cfg)))

def outcome(row, request, cfg=None):
    """Three-valued conjunction; known failure dominates unknown conditions.
    Never turn a 120-s allocation into a 30-s observation.
    """
    cfg = cfg or config()
    b = request["deadline_s"]
    if not finite(row.get("budget")) or float(row["budget"]) != b:
        raise ValueError("Independent allocation mismatch; do not relabel budgets")
    if row.get("observed") not in (True, 1):
        return dict(joint_success=None, valid_exit_ok=None, deadline_ok=None,
                    cost_ok=None, ens_ok=None)
    delivered = row.get("feasible")
    delivered = None if delivered is None or (isinstance(delivered, float) and math.isnan(delivered)) else bool(delivered)
    t = row.get("exit_s")
    deadline = int(0 <= t <= b) if finite(t) else None
    cost, ens, ref = row.get("cost"), row.get("ens_mwh"), row.get("reference_cost")
    tau, ceiling, cap = request["cost_tolerance"], request.get("cost_ceiling_eur"), request["ens_cap_mwh"]
    if tau is None and ceiling is None:
        quality = 1
    elif ceiling is not None:
        quality = int(cost <= ceiling) if finite(cost) else None
    else:
        quality = economic(True, cost if finite(cost) else None,
                           ref if finite(ref) else None, tau, cfg)
    if cap is None:
        service = 1
    elif not finite(ens) or ens < -request["zero_ens_tolerance_mwh"]:
        service = None
    else:
        service = int(ens <= cap)
    flags = [None if delivered is None else int(delivered), deadline, quality, service]
    joint = 0 if 0 in flags else None if None in flags else 1
    return dict(joint_success=joint, valid_exit_ok=flags[0], deadline_ok=deadline,
                cost_ok=quality, ens_ok=service)

def request_features(request, allowance):
    """Only operator preferences; no realized objective/reference/dispatch."""
    return dict(budget=float(allowance),
                cost_active=float(request["cost_tolerance"] is not None),
                threshold=float(request["cost_tolerance"] or 0.),
                ceiling_active=float(request.get("cost_ceiling_eur") is not None),
                cost_ceiling=float(request.get("cost_ceiling_eur") or 0.),
                ens_active=float(request["ens_cap_mwh"] is not None),
                ens_cap=float(request["ens_cap_mwh"] or 0.))
