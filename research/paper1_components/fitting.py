"""Resumable fitting on historical observations only; no solver calls."""
from contextlib import closing
from pathlib import Path
from .dataio import data, frozen, protocol, put, read, sealed, sha, verify
from .models import specifications, tune_fit
from .parallel import results, worker_count

_FIT_STATE = None


def tasks(df, cfg):
    for endpoint in cfg["endpoints"]:
        for budget in cfg["deadlines_s"]:
            q = df[df.budget == budget]
            for control in cfg["family_controls"]:
                for fold in range(cfg["outer_folds"]):
                    for spec in specifications(cfg):
                        identity = dict(endpoint=endpoint, budget=budget, family_control=control,
                                        fold=fold, description=spec.name)
                        yield identity, q, spec


def checkpoint_path(run, identity):
    r = identity
    return Path(run)/"checkpoints"/(
        f"{r['endpoint']}_b{r['budget']}_family{int(r['family_control'])}_"
        f"fold{r['fold']}_{r['description']}.json")


def _initialize_fitting(frame, cfg, manifest):
    global _FIT_STATE
    # One copy of the small prepared table per process, not one copy per fit.
    _FIT_STATE = ({b: q for b, q in frame.groupby("budget")}, cfg, manifest)


def _fit_task(job):
    """Pure computation: never open or write a checkpoint from a worker."""
    import pandas as pd
    identity, spec = job
    frames, cfg, manifest = _FIT_STATE
    q = frames[identity["budget"]]
    test = q[q.outer_fold == identity["fold"]]
    train = q[(q.outer_fold != identity["fold"]) & q[identity["endpoint"]].notna()]
    if test.empty or set(train.group) & set(test.group):
        raise ValueError("Empty outer test fold or group leakage")
    if train.empty:
        meta = dict(status="no_known_training_labels", active_terms=[],
                    dropped_constant_terms=[], inactive_interactions=[], coefficients=[])
        tuning = dict(C=None, inner_brier=None, inner_scores=[], tuning_status="unavailable")
        predicted = [None]*len(test)
    else:
        model, tuning = tune_fit(train, identity["endpoint"], spec,
                                identity["family_control"], cfg)
        meta = model.metadata()
        predicted = model.predict(test).tolist()
    rows = [dict(key=r.key, group=r.group, family=r.family,
                 y=None if pd.isna(r[identity["endpoint"]]) else int(r[identity["endpoint"]]),
                 p=p) for (_, r), p in zip(test.iterrows(), predicted)]
    return sealed(dict(manifest=manifest, identity=identity,
        training_groups=int(train.group.nunique()), test_groups=int(test.group.nunique()),
        train_positives=int((train[identity["endpoint"]] == 1).sum()),
        train_negatives=int((train[identity["endpoint"]] == 0).sum()),
        metadata=meta, tuning=tuning, predictions=rows))


def fit(run, workers=1):
    workers = worker_count(workers)
    run = Path(run)
    m = frozen(run); cfg = protocol(); df = data(run)
    jobs = [(identity, spec) for identity, _, spec in tasks(df, cfg)]
    expected = len(jobs)
    outputs, pending = {}, []
    seen = set()
    # Verify every reusable checkpoint before starting any new calculation.
    for identity, spec in jobs:
        dest = checkpoint_path(run, identity)
        if dest in seen:
            raise ValueError("Duplicate fit/checkpoint identity")
        seen.add(dest)
        if dest.exists():
            obj = verify(read(dest))
            if obj["manifest"] != m["sha256"] or obj["identity"] != identity:
                raise ValueError("Checkpoint provenance mismatch")
            outputs[dest.relative_to(run).as_posix()] = sha(dest)
        else:
            pending.append((identity, spec))
    active_workers = min(workers, len(pending)) if pending else 0
    print(f"Paper 1: {len(outputs)}/{expected} fits reused; "
          f"{len(pending)} pending, {active_workers} worker(s)", flush=True)
    if pending:
        with closing(results(_fit_task, pending, workers=active_workers,
                             initializer=_initialize_fitting,
                             initargs=(df, cfg, m["sha256"]))) as completed:
            for index, obj in completed:
                verify(obj)
                identity = pending[index][0]
                if obj["manifest"] != m["sha256"] or obj["identity"] != identity:
                    raise ValueError("Worker returned a mismatched fit")
                dest = checkpoint_path(run, identity)
                # Only this parent process writes; the CLI holds the run lock.
                put(dest, obj)
                outputs[dest.relative_to(run).as_posix()] = sha(dest)
                if len(outputs) % 20 == 0 or len(outputs) == expected:
                    print(f"Paper 1: {len(outputs)}/{expected} outer fits saved or reused",
                          flush=True)
    if len(outputs) != expected:
        raise ValueError("Incomplete fit grid; no final receipt may be published")
    frozen(run)
    # Worker count and completion order do not alter the scientific receipt.
    put(run/"fit_receipt.json", sealed(dict(manifest=m["sha256"], checkpoints=outputs,
        warning="Nested out-of-fold development predictions; no independent final validation.")))

def predictions(run):
    import pandas as pd
    run = Path(run); m = frozen(run); receipt = verify(read(run/"fit_receipt.json"))
    if receipt["manifest"] != m["sha256"]:
        raise ValueError("Fit receipt does not belong to this preparation")
    rows, models, coefficients = [], [], []
    for rel, h in receipt["checkpoints"].items():
        if sha(run/rel) != h:
            raise ValueError(f"Checkpoint changed: {rel}")
        obj = verify(read(run/rel)); identity = obj["identity"]
        if obj["manifest"] != m["sha256"]:
            raise ValueError("Checkpoint manifest changed")
        rows.extend(dict(**identity, **r) for r in obj["predictions"])
        meta = obj["metadata"]
        models.append(dict(**identity, training_groups=obj["training_groups"],
            train_positives=obj["train_positives"], train_negatives=obj["train_negatives"],
            test_groups=obj["test_groups"], status=meta["status"],
            selected_C=obj["tuning"]["C"], inner_brier=obj["tuning"]["inner_brier"],
            tuning_status=obj["tuning"]["tuning_status"],
            inactive_interactions=str(meta["inactive_interactions"]),
            dropped_constant_terms=str(meta["dropped_constant_terms"])))
        coefficients.extend(dict(**identity, **r) for r in meta["coefficients"])
    df = pd.DataFrame(rows)
    keys = ["endpoint", "budget", "family_control", "description", "key"]
    if df.duplicated(keys).any():
        raise ValueError("Duplicate held-out predictions")
    return df, pd.DataFrame(models), pd.DataFrame(coefficients)
