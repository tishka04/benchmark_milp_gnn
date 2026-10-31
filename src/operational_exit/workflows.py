"""Explicit future-only audit, reference, label and dataset stages."""
from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor, as_completed
import json
import math
import os
from pathlib import Path
import time
import uuid

from .common import PHYSICS, ROOT, digest, read_json, sha, save_record, load_record, atomic_json, THREAD_ENV
from .physics import from_report, validate_dispatch, apply_cost_units


def report_path(out, case, stage="labels"):
    return Path(out) / stage / case["family"] / (case["scenario_id"] + ".json")


def solve_case(case, threads, relaxation=False):
    for name in THREAD_ENV:
        os.environ[name] = "1"
    os.environ["BENCHMARK_HIGHS_THREADS"] = str(threads)
    from src.milp.scenario_loader import load_scenario_data
    from .solving import solve_data
    if sha(case["file"]) != case["scenario_sha256"]:
        raise ValueError("Changed scenario input")
    start = time.perf_counter()
    result = solve_data(load_scenario_data(Path(case["file"])), threads=threads, seconds=600, relaxation=relaxation)
    result["wall_s"] = time.perf_counter() - start
    return result


def compatible_reference(case, manifest):
    """Only the corrected campaign, never budgets_v1 costs. Validate every reused dispatch."""
    from src.milp.scenario_loader import load_scenario_data
    from src.milp.batch_runner import _report_hash
    from scripts.recompute_thermal_references import digest as old_digest
    root = ROOT / "outputs/thermal_transitions_v1/reference_criticality"
    if not (root / "manifest.json").exists():
        return None
    old = read_json(root / "manifest.json")
    signature = old["solver"]
    if (signature["mip_limit_seconds"] != 600 or signature["threads"] != 16
            or signature["highs_version"] != manifest["versions"]["highspy"]
            or signature["pyomo_version"] != manifest["versions"]["pyomo"]):
        return None
    if any(sha(ROOT / p) != h for p, h in signature["sources"].items()):
        return None
    match = [c for c in old["cases"] if c["family"] == case["family"] and c["scenario_id"] == case["scenario_id"]]
    if len(match) != 1 or match[0]["scenario_sha256"] != case["scenario_sha256"]:
        return None
    path = root / "references" / case["family"] / "reports" / (case["scenario_id"] + ".json")
    if not path.exists():
        return None
    old_report = read_json(path)
    cp = old_report.get("batch_checkpoint", {})
    if (cp.get("complete") is not True or cp.get("report_sha256") != _report_hash(old_report)
            or old_report.get("reference_campaign_sha256") != old_digest(old)):
        return None
    if (old_report.get("thermal_formulation") != "thermal_transitions_v1"
            or old_report.get("dr_formulation") != "load_shifting_v1"):
        return None
    data = load_scenario_data(Path(case["file"]))
    # The old energy objective is provably identical only for hourly data.
    if data.dt_hours != 1:
        return None
    try:
        dispatch = from_report(old_report["detail"], data)
        check = validate_dispatch(data, dispatch)
        if not check["valid"] or not math.isclose(check["cost"], old_report["mip"]["objective"], rel_tol=1e-9, abs_tol=1e-3):
            return None
    except (KeyError, ValueError):
        return None
    return dict(complete=True, valid=True, status="reused_corrected_reference", physics=PHYSICS,
                objective=check["cost"], validation=check, dispatch=dispatch,
                proven_optimal=old_report["mip"]["termination"] == "optimal",
                solver_seconds=old_report["mip"].get("solve_seconds"), solver_limit=600, threads=16,
                bound=old_report["mip"].get("bound"), gap=old_report["mip"].get("gap"),
                reused_from=str(path), reused_sha256=sha(path), compatibility="identical sources and hourly objective; full dispatch revalidated")


def audit_real(manifest, out, progress):
    """Future gate: two size-spanning cases per family; fixed-binary LP comparison."""
    from src.milp.scenario_loader import load_scenario_data
    from src.milp.model import build_uc_model
    from pyomo.environ import SolverFactory, Var, value
    from pyomo.core import TransformationFactory
    from .physics import extract_dispatch, BINARY_VARS
    rows = []
    chosen = [c for c in manifest["cohort"] if c["key"] in manifest["real_audit_cases"]]
    for i, case in enumerate(chosen):
        result = compatible_reference(case, manifest) or solve_case(case, 16)
        if not result.get("valid"):
            raise ValueError(f"No validated audit incumbent for {case['key']}")
        data = load_scenario_data(Path(case["file"]))
        model = build_uc_model(data)
        apply_cost_units(model)
        v = result["dispatch"]["values"]
        for name in BINARY_VARS:
            comp = getattr(model, name)
            for idx in comp:
                actual = v[name][data.zones.index(idx[0])][idx[1]] if isinstance(idx, tuple) else v[name][idx]
                comp[idx].fix(round(actual))
        TransformationFactory("core.relax_integer_vars").apply_to(model)
        solver = SolverFactory("highs")
        solver.options.update({"threads": 16, "time_limit": 600})
        res = solver.solve(model, load_solutions=False)
        if not len(res.solution):
            raise ValueError("Fixed-binary audit LP produced no incumbent")
        model.solutions.load_from(res)
        check = validate_dispatch(data, extract_dispatch(model))
        if not check["valid"] or check["cost"] > result["validation"]["cost"] + 1e-3 + abs(check["cost"]) * 1e-9:
            raise ValueError(f"Unexplained MILP/fixed-LP discrepancy: {case['key']}")
        if result.get("proven_optimal") and not math.isclose(check["cost"], result["validation"]["cost"], abs_tol=1e-3, rel_tol=1e-7):
            raise ValueError("Proven optimum and fixed LP costs disagree")
        rows.append(dict(key=case["key"], reference_cost=result["validation"]["cost"], fixed_lp_cost=check["cost"], validation=check))
        progress(i + 1, len(chosen), case["key"])
    save_record(Path(out) / "audit.json", dict(complete=True, passed=True, manifest_hash=digest(manifest), cases=rows))


def compute_reports(manifest, out, *, training, with_lp=False, progress):
    cases = manifest["train"] if training else manifest["cohort"]
    stage = "labels" if training else "references"
    pending, count = [], 0
    for case in cases:
        identity = dict(manifest_hash=digest(manifest), key=case["key"], solver_seconds=600,
                        threads=4 if training else 16, extra_lp=with_lp if training else False)
        path = report_path(out, case, stage)
        old = load_record(path, identity)
        if old and old.get("complete"):
            count += 1
            continue  # Includes completed no-incumbent; never a favourable retry.
        if not training:
            reused = compatible_reference(case, manifest)
            if reused:
                save_record(path, dict(identity=identity, complete=True, result=reused))
                count += 1
                progress(count, len(cases), "reference compatible: " + case["key"])
                continue
        pending.append((case, path, identity))
    if not pending:
        progress(count, len(cases), "all compatible checkpoints retained")
        return
    # Native HiGHS threads are fixed for each process for its whole lifetime.
    with ProcessPoolExecutor(max_workers=4 if training else 1) as pool:
        futures = {pool.submit(solve_case, c, 4 if training else 16, with_lp if training else False): (c, p, ident)
                   for c, p, ident in pending}
        for future in as_completed(futures):
            case, path, identity = futures[future]
            try:
                result = future.result()
            except Exception as exc:
                save_record(path, dict(identity=identity, complete=False, status="infrastructure_error", error=repr(exc)))
                for f in futures:
                    f.cancel()
                raise
            if result.get("dispatch") and not result.get("valid"):
                save_record(path, dict(identity=identity, complete=False, status="unexplained_invalid_dispatch", result=result))
                raise ValueError(f"Physical audit required for {case['key']}")
            save_record(path, dict(identity=identity, complete=True, result=result))
            count += 1
            progress(count, len(cases), case["key"] + ": " + result["status"])


def build_dataset(manifest, out, progress):
    import numpy as np
    from .embedding import OnlineHTE
    from src.milp.scenario_loader import load_scenario_data
    from src.uc_jepa.data.extract_uc_labels import extract_u_star
    from .physics import to_report
    out = Path(out)
    source = Path(manifest["source"])
    if sha(source / "encoders/hierarchical_temporal_v3/best_encoder.pt") != manifest["hte_sha256"]:
        raise ValueError("Frozen HTE changed")
    encoder = OnlineHTE(source / "encoders/hierarchical_temporal_v3/best_encoder.pt")
    entries, excluded, graph_checks = [], [], 0
    for i, case in enumerate(manifest["train"]):
        stored = load_record(report_path(out, case))
        if not stored or not stored.get("complete"):
            raise ValueError("Labels stage incomplete")
        r = stored["result"]
        if not r.get("valid"):
            excluded.append(dict(key=case["key"], status=r["status"]))
            continue
        kind = "gold" if r["proven_optimal"] else "silver"
        target = out / "dataset" / kind / (case["scenario_id"] + "_uc.npz")
        if target.exists():
            with np.load(target, allow_pickle=False) as saved:
                if str(saved["training_manifest"].item()) != digest(manifest) or str(saved["label_sha256"].item()) != stored["sha256"]:
                    raise ValueError("Existing dataset file incompatible")
        else:
            data = load_scenario_data(Path(case["file"]))
            # Full graph comparison proves that the unchanged contrastive HTE
            # and its normalization still receive exactly the same features.
            graph = source / "graphs/scenarios_v3" / (case["scenario_id"] + ".npz")
            h = encoder.encode(data, check_graph=graph)[0].cpu().numpy()
            graph_checks += 1
            extracted = extract_u_star(dict(zones=data.zones, detail=to_report(r["dispatch"], data)))
            if extracted is None or extracted[1] != data.zones or h.shape[:2] != extracted[0].shape[:2]:
                raise ValueError("UC/HTE zone ordering mismatch")
            target.parent.mkdir(parents=True, exist_ok=True)
            temp = target.with_name(target.name + "." + uuid.uuid4().hex + ".tmp")
            with temp.open("xb") as f:
                np.savez_compressed(f, u_star=extracted[0].numpy(), h_scenario=h,
                    scenario_id=case["scenario_id"], family=kind, criticality=0.,
                    milp_objective=r["validation"]["cost"], dr_formulation="load_shifting_v1",
                    physics=PHYSICS, training_manifest=digest(manifest), label_sha256=stored["sha256"])
                f.flush()
                os.fsync(f.fileno())
            os.replace(temp, target)
        entries.append(dict(key=case["key"], partition=manifest["splits"]["assignments"][case["key"]],
                            classification=kind, file=str(target), sha256=sha(target), label_sha256=stored["sha256"]))
        progress(i + 1, len(manifest["train"]), case["key"])
    save_record(out / "dataset.json", dict(complete=True, manifest_hash=digest(manifest), entries=entries,
                excluded=excluded, hte_reuse="graph feature/hierarchy arrays compared before each new embedding",
                graph_checks=len(entries), hte_sha256=manifest["hte_sha256"]))


def prepare_configs(manifest, out):
    import yaml
    out = Path(out)
    paths = {"ae": out / "training/autoencoder/uc_ae_best.pt", "jepa": out / "training/jepa/jepa_best.pt"}
    for name, epochs in (("autoencoder", 60), ("jepa", 50), ("latent_ebm", 20)):
        cfg = yaml.safe_load((ROOT / f"configs/uc_jepa/{name}.yaml").read_text(encoding="utf-8"))
        cfg["data"].update(roots=[str(out / "dataset")], split_manifest=str(out / "manifest.json"), batch_size=8, num_workers=0)
        cfg["optim"].update(epochs=epochs, operational=True, manifest_hash=digest(manifest))
        cfg["logging"].update(output_dir=str(out / "training" / name), save_every=0)
        if name == "latent_ebm":
            # Inherit the newly tuned internal-validation JEPA thresholds,
            # never historical hand-tuned thresholds from another label regime.
            cfg["eval"]["feature_thresholds"] = {}
        for key in ("ae", "jepa"):
            if key in cfg:
                cfg[key]["checkpoint"] = str(paths[key])
        # YAML accepts JSON; atomic publication uses the same robust I/O path.
        path = out / "configs" / (name + ".yaml")
        if path.exists():
            old = read_json(path)
            comparable = json.loads(json.dumps(old))
            for key in ("dataset_index", "dataset_digest"):
                comparable["data"].pop(key, None)
            comparable["optim"].pop("dependency_sha256", None)
            if comparable != cfg:
                raise ValueError("Prepared training configuration changed")
        else:
            atomic_json(path, cfg)


def bind_training_inputs(out, name):
    out = Path(out)
    path = out / "configs" / (name + ".yaml")
    cfg = read_json(path)
    index_path = out / "dataset.json"
    fields = dict(dataset_index=str(index_path), dataset_digest=digest(read_json(index_path)))
    deps = {k: sha(cfg[k]["checkpoint"]) for k in ("ae", "jepa") if k in cfg}
    for k, v in fields.items():
        if k in cfg["data"] and cfg["data"][k] != v:
            raise ValueError("Training dataset changed")
    if "dependency_sha256" in cfg["optim"] and cfg["optim"]["dependency_sha256"] != deps:
        raise ValueError("Frozen AE/JEPA changed")
    cfg["data"].update(fields)
    cfg["optim"]["dependency_sha256"] = deps
    atomic_json(path, cfg)
