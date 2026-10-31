"""Isolated, resumable 600-scenario reference campaign. No neural models.

Only --preview is read-only. A normal invocation solves the missing references,
then launches the statistical audit. Importing this module never starts work.
"""
from __future__ import annotations

import argparse
from collections import Counter
from contextlib import contextmanager
import hashlib
from importlib.metadata import version
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
FAMILIES = ("low_criticality_scenarios", "medium_criticality_scenarios", "high_criticality_scenarios",
            "persona_vre_battery_scenarios", "persona_network_operator_scenarios", "persona_mathematician_scenarios")
PROTOCOL = "corrected_thermal_references_v1"
THERMAL = "thermal_transitions_v1"
DR = "load_shifting_v1"
SOLVER_SOURCES = ("src/demand_response.py", "src/thermal.py", "src/milp/model.py",
                  "src/milp/solve.py", "src/milp/scenario_loader.py", "src/milp/defaults.py",
                  "src/milp/profiles.py", "src/milp/solver_resources.py", "src/milp/batch_runner.py",
                  "scripts/recompute_thermal_references.py")
ANALYSIS_SOURCES = ("scripts/analyze_criticality_existing.py", "scripts/analyze_corrected_criticality.py",
                    "src/analysis/criticality_index.py", "scripts/generate_persona_scenarios.py")


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def digest(payload):
    return hashlib.sha256(json.dumps(payload, sort_keys=True, allow_nan=False).encode()).hexdigest()


def read_json(path):
    # OneDrive may hold a short-lived lock. Never equate it with a damaged file.
    for attempt in range(8):
        try:
            return json.loads(Path(path).read_text(encoding="utf-8"))
        except PermissionError:
            if attempt == 7:
                raise
            time.sleep(.25 * (attempt + 1))


def atomic_json(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + f".{os.getpid()}.tmp")
    with tmp.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2, allow_nan=False)
        f.flush()
        os.fsync(f.fileno())
    for attempt in range(8):
        try:
            os.replace(tmp, path)
            return
        except PermissionError:
            if attempt == 7:
                raise
            time.sleep(.25 * (attempt + 1))


def optional_checkpoint(path):
    """Non-authoritative progress/cache metadata can be rebuilt from references."""
    try:
        return read_json(path) if path.exists() else {}
    except (json.JSONDecodeError, UnicodeDecodeError):
        return {}


@contextmanager
def exclusive_lock(path):
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        f = path.open("a+b")
    except OSError as exc:
        raise RuntimeError(f"Une autre campagne utilise {path}") from exc
    locked = False
    try:
        f.seek(0)
        if os.name == "nt":
            import msvcrt
            msvcrt.locking(f.fileno(), msvcrt.LK_NBLCK, 1)
        else:
            import fcntl
            fcntl.flock(f, fcntl.LOCK_EX | fcntl.LOCK_NB)
        locked = True
        yield
    finally:
        if locked and os.name == "nt":
            f.seek(0)
            msvcrt.locking(f.fileno(), msvcrt.LK_UNLCK, 1)
        f.close()


def validate_paths(source, out):
    if source == out or source in out.parents or out in source.parents:
        raise ValueError("Le dossier de sortie doit etre separe des anciens resultats.")


def inventory(source):
    """Freeze the exact original matched-budget cohort, not a new file glob."""
    mp = source / "evaluation/budgets_v1/manifest.json"
    original = read_json(mp)
    cohort = original["cohort"]
    if Counter(x["family"] for x in cohort) != Counter({f: 100 for f in FAMILIES}):
        raise ValueError("La cohorte doit contenir exactement 100 scenarios de chacune des six familles.")
    expected = {k.replace("\\", "/"): v for k, v in original["inputs_sha256"].items()}
    cases, identities = [], set()
    for item in sorted(cohort, key=lambda x: (FAMILIES.index(x["family"]), x["scenario_id"])):
        f, sid = item["family"], item["scenario_id"]
        if not sid.startswith("scenario_") or not sid[9:].isdigit() or (f, sid) in identities:
            raise ValueError("Identite de scenario invalide ou dupliquee")
        identities.add((f, sid))
        scenario, old_report = f"{f}/{sid}.json", f"{f}/reports/{sid}.json"
        hashes = {name: sha(source/name) for name in (scenario, old_report)}
        if any(hashes[name] != expected[name] for name in hashes):
            raise ValueError(f"Scenario/reference historique modifie : {f}/{sid}")
        old = read_json(source/old_report)
        provenance = old["batch_checkpoint"]["provenance"]
        if (old["dr_formulation"] != DR or old.get("thermal_formulation") is not None
                or provenance["solver"] != "highs" or provenance["highs_threads"] != 16
                or provenance["time_limit_seconds"] != 600
                or provenance["scenario_sha256"] != hashes[scenario]
                or not math.isfinite(old["mip"]["objective"])):
            raise ValueError(f"Reference historique incompatible : {old_report}")
        cases.append(dict(family=f, scenario_id=sid, scenario_path=scenario, old_report_path=old_report,
                          scenario_sha256=hashes[scenario], old_report_sha256=hashes[old_report]))
    return dict(source_root=str(source), original_manifest_sha256=sha(mp), cases=cases)


def solver_signature():
    return dict(protocol=PROTOCOL, thermal_formulation=THERMAL, dr_formulation=DR,
                threads=16, workers=1, mip_limit_seconds=600, lp_limit_seconds=600,
                highs_version=version("highspy"), pyomo_version=version("pyomo"),
                sources={p: sha(ROOT/p) for p in SOLVER_SOURCES})


def prepare_manifest(source, out, resume):
    current = dict(**inventory(source), solver=solver_signature())
    path = out/"manifest.json"
    if path.exists():
        if not resume:
            raise ValueError("Campagne existante : utiliser -Resume.")
        saved = read_json(path)
        if saved != current:
            changed = [k for k in current if current[k] != saved.get(k)]
            raise ValueError("Reprise incompatible (" + ", ".join(changed) +
                             "). Conserver ce dossier et utiliser un nouveau dossier pour un autre protocole.")
        return saved
    if resume:
        raise ValueError("Aucun manifeste de reprise. Lancer une premiere fois sans -Resume.")
    if out.exists() and any(p.name != "run.lock" for p in out.iterdir()):
        raise ValueError("Dossier non vide sans manifeste : choisir un nouveau dossier.")
    atomic_json(path, current)
    return current


def report_path(out, case):
    return out/"references"/case["family"]/"reports"/(case["scenario_id"]+".json")


def thermal_check(report):
    from src.thermal import thermal_metrics
    d, zones = report["detail"], report["zones"]
    arrays = lambda name: [d[name][z] for z in zones]
    param = lambda name: [d["thermal_parameters"]["thermal_"+name][z] for z in zones]
    metrics = thermal_metrics(arrays("thermal"), arrays("thermal_commitment"), arrays("thermal_startup"),
        capacity=param("capacity"), minimum=param("min"), ramp=param("ramp"),
        startup_limit=param("startup_limit"), shutdown_limit=param("shutdown_limit"),
        initial_power=param("initial"), initial_on=param("initial_on"))
    if not metrics["thermal_feasible"]:
        raise ValueError("Solution ne respectant pas les transitions thermiques")
    return metrics


def read_completed(out, case, manifest):
    from src.milp.batch_runner import _report_hash, _validate_completed_report, _provenance
    path = report_path(out, case)
    if not path.exists():
        return None
    try:
        r = read_json(path)
        cp = r["batch_checkpoint"]
        if (cp["complete"] is not True or cp["report_sha256"] != _report_hash(r)
                or cp["provenance"] != _provenance(Path(manifest["source_root"])/case["scenario_path"], "highs")
                or r["reference_campaign_sha256"] != digest(manifest)
                or r["thermal_formulation"] != THERMAL):
            return None
        _validate_completed_report(r)
        thermal_check(r)
        return r
    except PermissionError:
        raise  # A persistent lock must not cause a valid reference to be overwritten.
    except (ValueError, KeyError, TypeError, IndexError):
        return None


def worker(out, index):
    from src.milp import batch_runner as batch
    from src.milp.scenario_loader import load_scenario_data
    if batch.DEFAULT_TIME_LIMIT_SECONDS != 600:
        raise ValueError("Le producteur de references doit conserver son budget de 600 s")
    manifest = read_json(out/"manifest.json")
    if manifest["solver"] != solver_signature():
        raise ValueError("Code ou solveur modifie depuis la preparation")
    os.environ["BENCHMARK_HIGHS_THREADS"] = "16"
    case = manifest["cases"][index]
    source = Path(manifest["source_root"])
    path = source/case["scenario_path"]
    if sha(path) != case["scenario_sha256"]:
        raise ValueError("Entree modifiee")
    target = report_path(out, case)
    started = time.perf_counter()
    result = batch._run_single(path, "highs", True, True, target, None, None, False, None)
    if not result.success:
        raise RuntimeError(result.message)
    payload = read_json(target)
    payload["thermal_validation"] = thermal_check(payload)
    # The legacy breakdown omits startup; the model objective already includes it.
    data = load_scenario_data(path)
    payload["cost_components"]["thermal_startup"] = sum(
        data.thermal_startup_cost[z]*sum(payload["detail"]["thermal_startup"][z]) for z in payload["zones"])
    if not math.isclose(sum(payload["cost_components"].values()), payload["mip"]["objective"], rel_tol=1e-8, abs_tol=1e-4):
        raise ValueError("Decomposition des couts incoherente")
    payload["reference_campaign_sha256"] = digest(manifest)
    payload["reference_wall_seconds"] = time.perf_counter()-started
    if sha(path) != case["scenario_sha256"] or manifest["solver"] != solver_signature():
        raise ValueError("Entree/code modifie pendant la resolution")
    payload["batch_checkpoint"]["report_sha256"] = batch._report_hash(payload)
    atomic_json(target, payload)


def progress(stage, completed, total, elapsed, operation):
    print(f"PROGRESS|{stage}|{completed}|{total}|{elapsed:.1f}|{operation}", flush=True)


def call_child(command, log_path, env, heartbeat, timeout=None):
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w", encoding="utf-8") as log:
        child = subprocess.Popen(command, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT)
        started = time.perf_counter()
        try:
            while child.poll() is None:
                heartbeat(time.perf_counter()-started)
                if timeout and time.perf_counter()-started > timeout:
                    raise TimeoutError(f"Delai de surveillance depasse : {log_path}")
                try:
                    child.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    pass
            if child.returncode:
                raise RuntimeError(f"Processus en echec (code {child.returncode}) : {log_path}")
        finally:
            if child.poll() is None:
                child.terminate()
                try:
                    child.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    child.kill()
                    child.wait()


def analyze_signature(manifest, out, repeats):
    return dict(manifest_sha256=digest(manifest), repeats=repeats,
        sources={p: sha(ROOT/p) for p in ANALYSIS_SOURCES},
        versions={p: version(p) for p in ("numpy", "pandas", "scipy", "scikit-learn", "matplotlib")},
        references={str(report_path(out, c).relative_to(out)): sha(report_path(out, c)) for c in manifest["cases"]})


def run(args):
    source, out = args.source_root.resolve(), args.output_dir.resolve()
    validate_paths(source, out)
    if args.preview:
        inv = inventory(source)
        print(json.dumps(dict(n=len(inv["cases"]), solver=solver_signature(), output=str(out),
            steps=["600 references: 1 scenario x 16 threads, MILP 600 s puis relaxation LP <=600 s",
                   "Audit avant/apres : taille, criticite, stress/hardness, composantes, graphiques HTML"],
            neural_training=False, neural_inference=False), indent=2))
        return
    with exclusive_lock(source/"_pipeline/run.lock"), exclusive_lock(out/"run.lock"):
        manifest = prepare_manifest(source, out, args.resume or args.start_step == 2)
        env = dict(os.environ, BENCHMARK_HIGHS_THREADS="16", OMP_NUM_THREADS="1",
                   OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1", PYTHONIOENCODING="utf-8")
        old_threads = os.environ.get("BENCHMARK_HIGHS_THREADS")
        os.environ["BENCHMARK_HIGHS_THREADS"] = "16"
        state_path = out/"state.json"
        prior = optional_checkpoint(state_path)
        state = dict(status="running", stage=args.start_step, completed_references=0,
                     stage_timings=prior.get("stage_timings", []))
        stage_started = time.perf_counter()
        try:
            print("Verification des references deja sauvegardees...", flush=True)
            done = [read_completed(out, c, manifest) is not None for c in manifest["cases"]]
            state["completed_references"] = sum(done)
            if args.start_step == 2 and not all(done):
                raise ValueError("Analyse seule refusee : references manquantes ou invalides. Utiliser -Resume.")
            if args.start_step == 1:
                for i, c in enumerate(manifest["cases"]):
                    if done[i]:
                        continue
                    path = report_path(out, c)
                    if path.exists():
                        archived = path.with_name(path.name+f".invalid-{time.time_ns()}")
                        path.rename(archived)
                    label = f"{c['family']}/{c['scenario_id']}"
                    state.update(stage=1, active_scenario=label)
                    atomic_json(state_path, state)
                    log = out/"logs"/c["family"]/(c["scenario_id"]+f"-{time.time_ns()}.log")
                    call_child([sys.executable, "-B", "-u", "-m", "scripts.recompute_thermal_references",
                                "--output-dir", str(out), "--worker", str(i)], log, env,
                        lambda seconds: progress(1, sum(done), 600, time.perf_counter()-stage_started,
                                                 label+f" ({seconds:.0f} s)"), timeout=1500)
                    if read_completed(out, c, manifest) is None:
                        raise RuntimeError(f"Reference non validee : {label}")
                    done[i] = True
                    state["completed_references"] = sum(done)
                    atomic_json(state_path, state)
                    progress(1, sum(done), 600, time.perf_counter()-stage_started, label+" OK")
                state["stage_timings"].append(dict(stage=1, status="completed", seconds=time.perf_counter()-stage_started))
            # Verify original inputs/references again before accepting the cohort.
            if inventory(source) != {k: manifest[k] for k in ("source_root", "original_manifest_sha256", "cases")}:
                raise ValueError("Les entrees historiques ont change")
            if solver_signature() != manifest["solver"]:
                raise ValueError("Le code solveur a change pendant la campagne")
            stage_started = time.perf_counter()
            state.update(stage=2, status="running")
            atomic_json(state_path, state)
            signature = analyze_signature(manifest, out, args.repeats)
            marker = out/"analysis_complete.json"
            cached = optional_checkpoint(marker)
            reusable = cached.get("signature") == signature and bool(cached.get("outputs"))
            if reusable:
                reusable = all((out/p).is_file() and sha(out/p) == h for p, h in cached["outputs"].items())
            if not reusable or args.start_step == 2:
                marker.unlink(missing_ok=True)
                log = out/"logs"/f"analysis-{time.time_ns()}.log"
                call_child([sys.executable, "-B", "-u", "-m", "scripts.analyze_corrected_criticality",
                            "--campaign", str(out), "--repeats", str(args.repeats)], log, env,
                    lambda seconds: progress(2, 0, 1, seconds, "Analyse statistique; details dans "+str(log)))
                if analyze_signature(manifest, out, args.repeats) != signature:
                    raise ValueError("Sources/resultats modifies pendant l'analyse")
                outputs = {str(p.relative_to(out)): sha(p) for p in (out/"analysis").rglob("*") if p.is_file()}
                if not (out/"analysis/rapport.html").is_file():
                    raise ValueError("Rapport final absent")
                atomic_json(marker, dict(signature=signature, outputs=outputs))
            state["stage_timings"].append(dict(stage=2, status="completed", seconds=time.perf_counter()-stage_started))
            state.update(status="completed", stage=3, completed_references=600)
            atomic_json(state_path, state)
            progress(2, 1, 1, time.perf_counter()-stage_started, "Termine")
            print("Rapport :", out/"analysis/rapport.html", flush=True)
        except BaseException as exc:
            state["stage_timings"].append(dict(stage=state["stage"], status="failed", seconds=time.perf_counter()-stage_started))
            state.update(status="failed", error=str(exc))
            atomic_json(state_path, state)
            raise
        finally:
            if old_threads is None:
                os.environ.pop("BENCHMARK_HIGHS_THREADS", None)
            else:
                os.environ["BENCHMARK_HIGHS_THREADS"] = old_threads


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, default=ROOT/"outputs/load_shifting_v1")
    parser.add_argument("--output-dir", type=Path, default=ROOT/"outputs/thermal_transitions_v1/reference_criticality")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--preview", action="store_true")
    parser.add_argument("--start-step", type=int, choices=(1, 2), default=1)
    parser.add_argument("--repeats", type=int, default=10)
    parser.add_argument("--worker", type=int)
    args = parser.parse_args()
    if args.repeats < 1:
        parser.error("--repeats must be positive")
    if args.worker is not None:
        worker(args.output_dir.resolve(), args.worker)
    else:
        run(args)


if __name__ == "__main__":
    main()
