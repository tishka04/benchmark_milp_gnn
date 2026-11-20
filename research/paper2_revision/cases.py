"""Deterministic new cohorts and physically paired, isolated transformations."""
from __future__ import annotations
import copy
from dataclasses import asdict, fields, replace
from pathlib import Path
import uuid
from .core import ROOT, FAMILIES, config, seed, read, put, sha, sealed, frozen, digest

def describe(payload, cfg):
    from src.generator.generator_v1 import (estimate_milp_size, estimate_solve_time_hours,
        build_meta, compute_flexibility_metrics, compute_difficulty_indicators)
    from src.generator.generator_v1 import ScenarioConfig, GraphSpec, AssetSpec, EconPolicy, TechScalers
    from src.generator.generator_v1 import OperationCosts, ExogenousSpec, UnitCapacities, TransportCapacities
    # cfg is the actual sampled dataclass, avoiding stale derived metadata.
    nv, nc, nb = estimate_milp_size(cfg)
    meta = build_meta(cfg, nv, nc, 0.)
    payload.update(meta={k: v for k, v in meta.items() if k != "estimates"},
                   estimates=meta["estimates"], flexibility_metrics=compute_flexibility_metrics(cfg),
                   difficulty_indicators=compute_difficulty_indicators(cfg, nv, nc, 0.))
    return payload

def draw(family, partition, number, max_zones=300):
    """New generator draws; no solve, historical solution or test outcome is consulted."""
    from src.generator import generator_v1 as g
    from scripts.generate_eval_families import (make_low_criticality_space,
        make_medium_criticality_space, make_high_criticality_space)
    from scripts.generate_persona_scenarios import SPACE_BUILDERS
    space = g.load_space(str(ROOT / "config/scenario_space.yaml"))
    builders = dict(zip(FAMILIES[:3], (make_low_criticality_space,
                    make_medium_criticality_space, make_high_criticality_space)))
    builders.update({"persona_"+k+"_scenarios": v for k, v in SPACE_BUILDERS.items()})
    space = builders[family](copy.deepcopy(space))
    g.set_seed(seed("new", partition, family))
    result = []
    for attempt in range(number * 200):
        graph = g.sample_graph(space)
        if sum(graph.zones_per_region) > max_zones:
            continue
        cfg = g.ScenarioConfig(
            id=str(uuid.uuid5(uuid.NAMESPACE_URL, f"{config()['protocol']}/{partition}/{family}/{attempt}")),
            horizon_hours=24, dt_minutes=60, graph=graph, assets=g.sample_assets(space, graph),
            econ_policy=g.sample_econ(space), tech=g.sample_tech(space),
            costs=g.sample_operation_costs(space), exogenous=g.sample_exogenous(space, graph),
            unit_capacities=g.sample_unit_capacities(space),
            transport_capacities=g.sample_transport_capacities(space), mip_gap_target_pct=g.sample_mip_gap(space))
        payload = asdict(cfg)
        payload["operation_costs"] = payload.pop("costs")
        # Match historical family JSON omission: loaders use their documented unit defaults.
        # The sampled capacity parameters remain recorded in metadata/provenance.
        payload["_sampling_audit"] = dict(unused_unit_capacities=payload.pop("unit_capacities"),
                                          unused_transport_capacities=payload.pop("transport_capacities"))
        payload = describe(payload, cfg)
        result.append(payload)
        if len(result) == number:
            return result
    raise ValueError(f"Not enough compatible draws for {family}; no silent truncation")

def make_cases(run):
    run = Path(run); snap = frozen(run); cfg = config()
    manifest = read(ROOT / cfg["campaign"] / "manifest.json")
    historical = {c["uuid"] for c in manifest["cohort"] + manifest["train"]}
    entries = []
    def save(payload, partition, family, index, condition="base", base=None):
        key = f"{partition}/{family}/{index:04d}/{condition}"
        group = f"{partition}/{family}/{index:04d}"
        p = run / "cases" / partition / family / f"{index:04d}_{condition}.json"
        if base:
            payload = copy.deepcopy(payload)
            payload["_three_papers"] = dict(base_file=base["file"], base_sha256=base["scenario_sha256"],
                                            condition=condition, seed=seed(group, condition))
        put(p, payload)
        e = dict(key=key, group=group, family=family, partition=partition, condition=condition,
                 file=str(p.resolve()), scenario_sha256=sha(p), uuid=payload["id"],
                 expected_horizon=int(condition.removeprefix("horizon")) if condition.startswith("horizon") else 24)
        entries.append(e)
        return e
    for partition, number, cap in (("development", cfg["new_cases_per_family"], 300),
                                   ("final", cfg["new_cases_per_family"], 300),
                                   ("controlled_development", cfg["controlled_cases_per_family"], 150),
                                   ("controlled_test", cfg["controlled_cases_per_family"], 150),
                                   ("mh_training", cfg["training_bases_per_family"], 300)):
        for family in FAMILIES:
            for i, raw in enumerate(draw(family, partition, number, cap)):
                if raw["id"] in historical:
                    raise ValueError("New scenario UUID collides with historical data")
                base = save(raw, partition, family, i)
                if partition.startswith("controlled"):
                    for condition in ("size", "lines", "fleet", "permutation", "horizon48", "horizon72"):
                        save(raw, partition, family, i, condition, base)
                elif partition == "mh_training":
                    for condition in ("horizon12", "horizon48"):
                        save(raw, partition, family, i, condition, base)
    # Ex ante external solver subset: no result-dependent ranking.
    external = []
    for family in FAMILIES:
        selected = sorted((c for c in manifest["cohort"] if c["family"] == family),
                          key=lambda c: seed("gurobi120", c["key"]))[:20]
        external.extend(dict(c, group=c["key"], partition="gurobi120", condition="historical", expected_horizon=24)
                        for c in selected)
    put(run / "cases.json", sealed(dict(freeze=snap["sha256"], entries=entries + external,
        generation="independent family-space draws v1; no criticality ranking or budget-guard filtering",
        distribution_caveat="Same family parameter spaces, different selection from the historical cohort. Not an IID replication.",
        pairing="All transformed cases share their base group; keep the whole group in one fold.")))

def rebuild_lines(data):
    data.lines_from_index = {z: tuple(k for k, v in data.lines.items() if v.from_zone == z) for z in data.zones}
    data.lines_to_index = {z: tuple(k for k, v in data.lines.items() if v.to_zone == z) for z in data.zones}

def transform(data, condition, random_seed):
    """Transform realized physics; never redraw a supposedly paired control."""
    import numpy as np
    from src.milp.scenario_loader import NetworkLine
    d = copy.deepcopy(data)
    rng = np.random.default_rng(random_seed)
    if condition == "permutation":
        mapping = dict(zip(d.zones, rng.permutation(d.zones).tolist()))
        for f in fields(d):
            v = getattr(d, f.name)
            if isinstance(v, dict) and f.name not in ("lines", "lines_from_index", "lines_to_index"):
                def key(k):
                    return tuple(mapping.get(a, a) for a in k) if isinstance(k, tuple) else mapping.get(k, k)
                setattr(d, f.name, {key(k): value for k, value in v.items()})
        d.import_anchor_zone = mapping[d.import_anchor_zone]
        d.lines = {k: replace(line, from_zone=mapping[line.from_zone], to_zone=mapping[line.to_zone])
                   for k, line in d.lines.items()}
        # Keep the original canonical list: the physical zones now occupy new indices.
    elif condition == "lines":
        order = rng.permutation(d.zones).tolist()
        pairs = {tuple(sorted((order[i], order[i+1]))) for i in range(len(order)-1)}
        available = [tuple(sorted((a, b))) for i, a in enumerate(d.zones) for b in d.zones[i+1:]
                     if tuple(sorted((a, b))) not in pairs]
        rng.shuffle(available)
        pairs.update(available[:max(0, len(d.lines)-len(pairs))])
        old = list(d.lines.values())
        if not old:
            raise ValueError("Topology control needs an existing capacity distribution")
        d.lines = {f"L{i}": NetworkLine(f"L{i}", a, b, old[i % len(old)].capacity_mw,
                                       old[i % len(old)].distance_km)
                   for i, (a, b) in enumerate(sorted(pairs))}
    elif condition == "fleet":
        for name in ("thermal_capacity", "thermal_min_power", "thermal_ramp", "thermal_initial_output",
                     "thermal_startup_limit", "thermal_shutdown_limit"):
            setattr(d, name, {k: .8*v for k, v in getattr(d, name).items()})
        for name in ("battery_power", "battery_energy", "battery_initial", "battery_final_min", "battery_final_max"):
            setattr(d, name, {k: 1.2*v for k, v in getattr(d, name).items()})
    elif condition == "size":
        if len(d.zones)*2 > 300:
            raise ValueError("Frozen decoder limit exceeded")
        regions = sorted(set(d.region_of_zone.values()))
        rmap = {r: f"R{len(regions)+i+1}" for i, r in enumerate(regions)}
        zmap = {z: rmap[d.region_of_zone[z]] + "Z" + z.split("Z")[-1] for z in d.zones}
        for f in fields(d):
            v = getattr(d, f.name)
            if isinstance(v, dict) and f.name not in ("lines", "lines_from_index", "lines_to_index", "dr_block_cost"):
                def key(k):
                    return tuple(zmap.get(a, a) for a in k) if isinstance(k, tuple) else zmap.get(k, rmap.get(k, k))
                extra = {key(k): rmap.get(value, value) if isinstance(value, str) else value for k, value in v.items()}
                setattr(d, f.name, dict(v, **extra) if all(isinstance(k, str) for k in extra) else {**v, **extra})
        d.zones += [zmap[z] for z in data.zones]
        d.lines.update({"copy_"+k: replace(line, name="copy_"+k, from_zone=zmap[line.from_zone], to_zone=zmap[line.to_zone])
                        for k, line in data.lines.items()})
        d.lines["bridge"] = NetworkLine("bridge", data.import_anchor_zone, zmap[data.import_anchor_zone],
                                        max(1., data.import_capacity), 100.)
        d.import_capacity *= 2
    elif condition.startswith("horizon"):
        n = int(condition.removeprefix("horizon")); old = len(d.periods)
        if old != 24 or d.dt_hours != 1:
            raise ValueError("Hourly 24-step base required")
        for f in fields(d):
            v = getattr(d, f.name)
            if isinstance(v, dict) and v and all(isinstance(k, tuple) and len(k) >= 2 for k in v):
                expanded = {}
                for k, value in v.items():
                    for t in range(k[1], n, old):
                        # Preserve first day; bounded daily availability and demand changes.
                        factor = (1., .95, 1.05)[t//24] if f.name in ("demand", "dr_limit", "dr_block_limit") else (
                                 (1., .9, .85)[t//24] if f.name in ("solar_available", "wind_available", "res_available") else 1.)
                        expanded[(k[0], t, *k[2:])] = value * factor
                setattr(d, f.name, expanded)
        d.periods = list(range(n))
        d.peak_demand = {z: max(d.demand[z,t] for t in d.periods) for z in d.zones}
        d.dr_max_events = {z: min(n, max(1, round(v*n/24))) for z, v in d.dr_max_events.items()}
    elif condition != "base":
        raise ValueError(condition)
    rebuild_lines(d)
    return d

# Keep the original callable before installing a process-local adapter.
from src.milp.scenario_loader import load_scenario_data as _native_load

def load_data(path):
    path = Path(path); raw = read(path); directive = raw.get("_three_papers")
    if directive:
        base = Path(directive["base_file"])
        if sha(base) != directive["base_sha256"]:
            raise ValueError("Paired base changed")
        data = transform(_native_load(base), directive["condition"], directive["seed"])
    else:
        data = _native_load(path)
    return data

def install_loader():
    """Only the new worker process is adapted; no source file is modified."""
    import sys
    import src.milp.scenario_loader as module
    module.load_scenario_data = load_data
    old = sys.modules.get("src.milp.lp_worker_two_stage")
    if old is not None:
        old.load_scenario_data = load_data

def select(run, partition):
    receipt = read(Path(run) / "cases.json")
    from .core import verify
    verify(receipt)
    if receipt["freeze"] != frozen(run)["sha256"]:
        raise ValueError("Case manifest belongs to a different frozen protocol")
    values = [c for c in receipt["entries"] if c["partition"] == partition]
    if not values:
        raise ValueError(f"Unknown or empty case partition: {partition}")
    return values
