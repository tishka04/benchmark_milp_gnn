"""Dedicated campaign CLI. --preview never solves, trains, infers or writes."""
from __future__ import annotations

import argparse
from contextlib import ExitStack
import json
import os
from pathlib import Path
import subprocess
import sys
import threading
import time
import uuid

from .common import (DEFAULT_OUT, ROOT, BUDGETS, METHODS, PHYSICS, THREAD_ENV, digest,
                     read_json, atomic_json, save_record, load_record, sha, seed_for)
from .protocol import inventory, assert_unchanged, schedule, effective_sources
from .budget_reserve import POLICY as RESERVE_POLICY, RECEIPT_NAME as RESERVE_RECEIPT, evaluation_schedule

STEPS = {
    1: "Inventaire, partitions et configurations figées (sans résolution)",
    2: "Audit physique : 2 scénarios réels par famille et LP à binaires fixés",
    3: "5000 labels UC : 4 processus x 4 threads, 600 s solveur",
    4: "600 références : validation/réutilisation ou HiGHS 600 s, 16 threads",
    5: "Contrôle HTE/normalisation, embeddings et nouveaux labels classés",
    6: "AE depuis nouvelle initialisation ; mémoire et reprise par époque",
    7: "JEPA depuis nouvelle initialisation ; mémoire et reprise par époque",
    8: "Latent EBM depuis nouvelle initialisation ; mémoire et reprise par époque",
    9: "Pilote et calibration sur 30 validations internes ; réserve figée",
    10: "Comparaison principale : 600 x 7 x 4 = 16800 expériences",
    11: "Variabilité : 30 x 7 x 4 x 2 = 1680 expériences",
    12: "Contrôles finaux, rapport HTML, PNG/SVG et CSV",
}


class Progress:
    def __init__(self, step):
        self.step, self.count, self.total, self.operation = step, 0, 1, STEPS[step]
        self.start = time.perf_counter()
        self.stop = threading.Event()
        self.thread = threading.Thread(target=self.loop, daemon=True)
        self.thread.start()

    def __call__(self, count, total, operation):
        self.count, self.total, self.operation = count, total, operation
        self.emit()

    def emit(self):
        print(f"PROGRESS|{self.step}|{self.count}|{self.total}|{time.perf_counter()-self.start:.3f}|{self.operation}", flush=True)

    def loop(self):
        while not self.stop.wait(5):
            self.emit()

    def close(self):
        self.stop.set()
        self.thread.join()


def model_files(manifest, out):
    out = Path(out)
    return dict(hte=str(Path(manifest["source"]) / "encoders/hierarchical_temporal_v3/best_encoder.pt"),
                hte_features=str(out / "hte_feature_contract.json"),
                ae=str(out / "training/autoencoder/uc_ae_best.pt"),
                jepa=str(out / "training/jepa/jepa_best.pt"),
                ebm=str(out / "training/latent_ebm/latent_ebm_best.pt"))


def settings(manifest, out, method, reserve, models):
    return dict(method=method, reserve_s=reserve, manifest_hash=digest(manifest), repair=manifest["repair"],
                source_hashes=effective_sources(manifest, out), model_hashes=models,
                **model_files(manifest, out))


def one_run(manifest, out, case, item, reserve, area, models):
    from .supervisor import run_experiment
    identity = dict(item, manifest_hash=digest(manifest))
    checkpoint = Path(out) / area / (digest(identity) + ".json")
    record = load_record(checkpoint, identity)
    if record and record.get("complete"):
        return record
    attempt = Path(out) / (area + "_attempts") / digest(identity) / uuid.uuid4().hex
    save_record(checkpoint, dict(complete=False, status="external_interruption_if_not_completed", identity=identity, attempt=str(attempt)))
    request = dict(item, identity=identity, file=case["file"], scenario_sha256=case["scenario_sha256"])
    try:
        record = run_experiment(settings(manifest, out, item["method"], reserve, models), request, attempt)
    except BaseException as exc:
        # A supervisor interruption is not an observed timeout. Preserve its
        # attempt directory; never recover its raw result as a success.
        save_record(checkpoint, dict(complete=False, status="external_or_infrastructure_interruption", identity=identity,
                                    attempt=str(attempt), error=repr(exc)))
        raise
    record["attempt"] = str(attempt)
    record["reserve_s"] = reserve
    save_record(checkpoint, record)
    return record


def calibrate(manifest, out, progress):
    from .solution_transfer_repair import calibration_revision
    path = Path(out) / "calibration.json"
    fingerprint = {k: sha(v) for k, v in model_files(manifest, out).items()}
    revision = calibration_revision(out, fingerprint)
    if fingerprint["hte"] != manifest["hte_sha256"]:
        raise ValueError("HTE changed after dataset certification")
    existing = load_record(path)
    if existing:
        if (existing["manifest_hash"] != digest(manifest) or existing["models"] != fingerprint
                or existing.get("runtime_revision") != revision):
            raise ValueError("Frozen calibration/model stack changed; use a new output directory")
        return
    if list((Path(out) / "experiments").glob("*.json")):
        raise ValueError("Cannot recalibrate after evaluation started")
    cases = {c["key"]: c for c in manifest["train"]}
    records = []
    for key in manifest["calibration_cases"]:
        if manifest["splits"]["assignments"][key] != "validation":
            raise ValueError("Calibration must use internal validation only")
        for method in METHODS:
            item = dict(key=key, method=method, budget=120, repetition=0, seed=seed_for(key), purpose="calibration")
            if revision is not None:
                item["runtime_revision"] = revision
            record = one_run(manifest, out, cases[key], item, 30., "calibration_runs", fingerprint)
            if record["status"] in {"worker_error", "repair_error", "repair_crash", "invalid_dispatch", "missing_final_result"}:
                raise ValueError(f"Pilot {record['status']} for {key} / {method}; "
                                 f"inspect {Path(record['attempt']) / 'worker.log'}")
            if not record["all_processes_exited"]:
                raise ValueError(f"Calibration failed to observe clean finalization for {key} / {method} "
                                 f"({record['status']}, exit={record['exit_s']:.3f}s); "
                                 f"inspect {Path(record['attempt']) / 'worker.log'}")
            if method == "pipeline" and record["preload_metadata"].get("repair_processes") != 16:
                raise ValueError("Pilot did not preload K16 repair pool")
            # Deadline failures are benchmark outcomes, not a reason to retry
            # or abort calibration. Forced/late results remain inaccessible.
            if record["status"] in {"forced_timeout", "late_exit", "cooperative_deadline"}:
                if record.get("success") or (record["status"] == "forced_timeout" and not record["forced"]):
                    raise ValueError("Inconsistent deadline failure in pilot")
                records.append(record)
                progress(len(records), 120, key + " / " + method + " : " + record["status"])
                continue
            if record["forced"] or record["finalization_s"] is None:
                raise ValueError(f"Missing natural finalization for {key} / {method}")
            payload = record.get("result") or {}
            if method != "highs" and payload.get("generated_candidates") != (16 if method == "pipeline" else 1):
                raise ValueError("Pilot did not finish the required candidate generation")
            if method in ("pipeline", "jepa") and not payload.get("gpu_peak_bytes"):
                raise ValueError("Missing GPU memory observation in pilot")
            records.append(record)
            progress(len(records), 120, key + " / " + method)
    # A killed or late run has no eligible finalization measurement; never
    # substitute zero or inspect its raw file to recover an observation.
    finalizations = [r["finalization_s"] for r in records if not r["forced"]
                     and r["status"] != "late_exit" and r["finalization_s"] is not None]
    maximum = max(finalizations, default=None)
    from collections import Counter
    save_record(path, dict(complete=True, manifest_hash=digest(manifest), models=fingerprint,
                runtime_revision=revision,
                scenarios=manifest["calibration_cases"], observations=len(records), maximum_finalization_s=maximum,
                outcome_policy="deadlines_are_failures_v1", status_counts=dict(Counter(r["status"] for r in records)),
                finalization_observations=len(finalizations), unobserved_finalizations=len(records) - len(finalizations),
                reserve_s=30. if maximum is None else max(.05, 2 * maximum),
                reserve_policy=RESERVE_POLICY,
                rule="max(0.050 seconds, 2 * maximum observed finalization); retain provisional 30 seconds if none observed",
                provisional_reserve_s=30, evaluation_used=False))


def run_comparison(manifest, out, progress, extra=False):
    calibration = load_record(Path(out) / "calibration.json")
    if not calibration or calibration["manifest_hash"] != digest(manifest):
        raise ValueError("Missing frozen validation calibration")
    if calibration["models"] != {k: sha(v) for k, v in model_files(manifest, out).items()}:
        raise ValueError("Model/thresholds changed after calibration")
    cases = {c["key"]: c for c in manifest["cohort"]}
    amendment = load_record(Path(out) / RESERVE_RECEIPT)
    if amendment is None and calibration.get("reserve_policy") != RESERVE_POLICY:
        raise ValueError("Legacy fixed reserve: register src.operational_exit.budget_reserve before resuming evaluation")
    runs = list(evaluation_schedule(manifest, calibration, amendment, extra=extra))
    for i, item in enumerate(runs):
        reserve = item.get("reserve_s", calibration["reserve_s"])
        record = one_run(manifest, out, cases[item["key"]], item, reserve, "experiments", calibration["models"])
        progress(i + 1, len(runs), f"{item['key']} {item['method']} {item['budget']}s : {record['status']}")


def require_receipt(out, name, mh):
    r = load_record(Path(out) / name)
    if not r or not r.get("complete") or r.get("manifest_hash") != mh:
        raise ValueError(f"Required stage not complete: {name}")
    return r


def run(args):
    invocation_started = time.perf_counter()
    out, source = args.output_dir.resolve(), args.source_root.resolve()
    if out == source or source in out.parents or out in source.parents:
        raise ValueError("Output must be isolated from historical inputs/results")
    if args.preview:
        manifest = inventory(source)
        print(json.dumps(dict(output=str(out), train=len(manifest["train"]), evaluation=len(manifest["cohort"]),
              frozen_split_counts=manifest["splits"]["counts"], methods=METHODS, budgets=BUDGETS,
              main_experiments=16800, extra_experiments=1680, main_budget_hours=152, extra_budget_hours=15.2,
              note="Sum of decision budgets only; excludes preloads, labels, training and pilot. NO computation/write.",
              steps=STEPS, selected_steps=[args.start_step, args.end_step]), indent=2, ensure_ascii=False))
        return
    if args.analyze_only:
        from .analysis import analyze
        analyze(out)
        return
    from scripts.recompute_thermal_references import exclusive_lock
    from scripts.benchmark_matched_budgets import campaign_lock
    with ExitStack() as stack:
        stack.enter_context(campaign_lock(source))
        out.mkdir(parents=True, exist_ok=True)
        stack.enter_context(exclusive_lock(out / "run.lock"))
        manifest_path = out / "manifest.json"
        if manifest_path.exists() or Path(str(manifest_path) + ".bak").exists():
            if not args.resume and args.start_step == 1:
                raise ValueError("Campaign already prepared; use -Resume")
            manifest = load_record(manifest_path)
            if manifest is None:
                raise ValueError("Frozen manifest has no valid copy")
            assert_unchanged(manifest, out)
            if manifest["source"] != str(source) or manifest["labels_extra_lp"] != args.with_label_relaxation:
                raise ValueError("Source/label relaxation settings changed")
        else:
            if args.start_step != 1:
                raise ValueError("Prepare step 1 first")
            if any(p.name != "run.lock" for p in out.iterdir()):
                raise ValueError("Refusing nonempty output without a manifest")
            manifest = inventory(source)
            manifest["labels_extra_lp"] = args.with_label_relaxation
            save_record(manifest_path, manifest)
            manifest = load_record(manifest_path)
        mh = digest(manifest)
        state = read_json(out / "state.json") if (out / "state.json").exists() else dict(steps={}, timings=[])
        from .workflows import prepare_configs, audit_real, compute_reports, build_dataset, bind_training_inputs
        prepare_configs(manifest, out)
        for step in range(args.start_step, args.end_step + 1):
            progress = Progress(step)
            started = invocation_started if step == 1 else time.perf_counter()
            progress.start = started
            try:
                if step >= 3:
                    audit = require_receipt(out, "audit.json", mh)
                    if not audit.get("passed"):
                        raise ValueError("Physical audit gate failed")
                if step >= 6:
                    require_receipt(out, "dataset.json", mh)
                if step == 1:
                    pass  # Prepared and checked before executing any selected stage.
                elif step == 2:
                    old = load_record(out / "audit.json")
                    if not old or not old.get("passed") or old.get("manifest_hash") != mh:
                        audit_real(manifest, out, progress)
                elif step in (3, 4):
                    compute_reports(manifest, out, training=step == 3, with_lp=args.with_label_relaxation, progress=progress)
                elif step == 5:
                    build_dataset(manifest, out, progress)
                elif step in (6, 7, 8):
                    name = {6: "autoencoder", 7: "jepa", 8: "latent_ebm"}[step]
                    bind_training_inputs(out, name)
                    env = dict(os.environ, **{k: "1" for k in THREAD_ENV}, PYTHONDONTWRITEBYTECODE="1")
                    subprocess.run([sys.executable, "-B", "-u", "-m", f"src.uc_jepa.training.train_{name}",
                                    "--config", str(out / "configs" / (name + ".yaml")), "--device", "cuda:0"],
                                   cwd=ROOT, env=env, check=True)
                elif step == 9:
                    calibrate(manifest, out, progress)
                elif step in (10, 11):
                    run_comparison(manifest, out, progress, extra=step == 11)
                elif step == 12:
                    from .analysis import analyze
                    analyze(out)
                state["steps"][str(step)] = "completed"
                state["timings"].append(dict(step=step, status="completed", seconds=time.perf_counter() - started))
                atomic_json(out / "state.json", state)
                progress(1, 1, STEPS[step] + " : terminé")
            except BaseException as exc:
                state["steps"][str(step)] = "interrupted_or_failed"
                state["timings"].append(dict(step=step, status="interrupted_or_failed", seconds=time.perf_counter() - started, error=repr(exc)))
                atomic_json(out / "state.json", state)
                raise
            finally:
                progress.close()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--source-root", type=Path, default=ROOT / "outputs/load_shifting_v1")
    p.add_argument("--output-dir", type=Path, default=DEFAULT_OUT)
    p.add_argument("--preview", action="store_true")
    p.add_argument("--resume", action="store_true")
    p.add_argument("--start-step", type=int, choices=range(1, 13), default=1)
    p.add_argument("--end-step", type=int, choices=range(1, 13), default=12)
    p.add_argument("--analyze-only", action="store_true")
    p.add_argument("--with-label-relaxation", action="store_true", help="Optional extra LP (off for UC labels by default)")
    args = p.parse_args()
    if args.start_step > args.end_step:
        p.error("start-step must be <= end-step")
    run(args)


if __name__ == "__main__":
    main()
