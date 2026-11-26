"""Checkpoint I/O recovery for the unchanged, frozen multi-horizon trainer.

Serialize each state once. Publish both backup and primary atomically, retrying
only access/sharing violations. An exhausted retry keeps the prior checkpoint
and the candidate temporary file. No uncommitted file is promoted on resume.
"""
import hashlib
import io
import math
import os
from pathlib import Path
import shutil
import time
import uuid

from .. import core


RETRY_DELAYS = (0.1, 0.2, 0.5, 1., 2., 3., 5., 5., 5., 5.)
ADAPTER = "atomic_checkpoint_retry_v1"


def retry_io(operation, description):
    """At most 26.8 seconds of waiting per operation; other I/O errors surface."""
    for attempt in range(len(RETRY_DELAYS) + 1):
        try:
            return operation()
        except OSError as exc:
            transient = isinstance(exc, PermissionError) or getattr(exc, "winerror", None) in (5, 32, 33)
            if not transient:
                raise
            if attempt == len(RETRY_DELAYS):
                exc.add_note(f"{description}: retries exhausted; checkpoint files were retained for inspection.")
                raise
            if attempt == 0:
                print(f"Checkpoint file access denied; retrying {description}.", flush=True)
            time.sleep(RETRY_DELAYS[attempt])


def _copy_backup(source, target):
    # A failed copy can only damage our unique staging file, never the old .bak.
    with source.open("rb") as old, target.open("wb") as copy:
        shutil.copyfileobj(old, copy)
        copy.flush()
        os.fsync(copy.fileno())


def resilient_checkpoint(path, state):
    import torch
    p = core.writable(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    token = uuid.uuid4().hex
    candidate = p.with_name(p.name + "." + token + ".tmp")
    with candidate.open("xb") as stream:
        torch.save(state, stream)
        stream.flush()
        os.fsync(stream.fileno())
    if p.exists():
        backup = p.with_suffix(p.suffix + ".bak")
        backup_candidate = p.with_name(p.name + "." + token + ".bak.tmp")
        retry_io(lambda: _copy_backup(p, backup_candidate), f"backup copy for {p.name}")
        retry_io(lambda: os.replace(backup_candidate, backup), f"backup publication for {p.name}")
    retry_io(lambda: os.replace(candidate, p), f"publication of {p.name}")


def run_training(run):
    """Only the checkpoint publisher is adapted; preserve all scientific guards."""
    from .. import multihorizon
    run = Path(run)
    with core.lock(run):
        snap = core.frozen(run)
        if (run / "design_lock.json").exists():
            raise ValueError("Final design is locked")
        source_sha = core.sha(__file__)
        core.put(run / "recovery" / f"checkpoint_writer_{source_sha}.json", core.sealed(dict(
            adapter=ADAPTER, source_sha256=source_sha, freeze=snap["sha256"],
            retry_delays_seconds=list(RETRY_DELAYS),
            scope="Checkpoint publication only; frozen trainer, models, optimizer and random states unchanged")))
        print(f"Checkpoint writer: {ADAPTER} ({source_sha[:12]}).", flush=True)
        original = multihorizon.checkpoint
        try:
            multihorizon.checkpoint = resilient_checkpoint
            for training_seed in core.config()["training_seeds"]:
                for part in ("ae", "jepa", "ebm"):
                    multihorizon.train(run, part, training_seed)
        finally:
            multihorizon.checkpoint = original
    print("Completed stage TrainMultiHorizon", flush=True)


def inspect_training(run):
    """CPU-only validation of resume metadata; no training or file modification."""
    # Preserve the project's pandas-before-torch DLL initialization order.
    from .. import multihorizon
    torch = multihorizon.torch
    run = Path(run)
    snap = core.frozen(run)
    dataset = core.verify(core.read(run / "multi_horizon" / "dataset.json"))
    if dataset["freeze"] != snap["sha256"]:
        raise ValueError("Training dataset belongs to a different freeze")
    n_train = sum(entry["split"] == "train" for entry in dataset["entries"])
    if not n_train:
        raise ValueError("No training entries")
    result = []
    for training_seed in core.config()["training_seeds"]:
        folder = run / "multi_horizon" / str(training_seed)
        cfg = dict(core.config()["multi_horizon"], seed=training_seed)
        for part in ("ae", "jepa", "ebm"):
            receipt = folder / f"{part}_receipt.json"
            state_path = folder / f"{part}_last.pt"
            summary = dict(seed=training_seed, part=part, status="not_started")
            if receipt.exists():
                obj = core.verify(core.read(receipt))
                if (obj["freeze"] != snap["sha256"] or obj["dataset"] != dataset["sha256"] or
                        obj["training_seed"] != training_seed or
                        obj["model_sha256"] != core.sha(folder / f"{part}_best.pt")):
                    raise ValueError(f"Training receipt mismatch: {receipt}")
                summary["status"] = "completed"
            elif state_path.exists():
                dependencies = {name: core.sha(folder / f"{name}_best.pt") for name in ("ae", "jepa")
                                if (part == "jepa" and name == "ae") or part == "ebm"}
                expected = core.digest(dict(freeze=snap["sha256"], dataset=dataset["sha256"],
                                            stage=part, cfg=cfg, dependencies=dependencies))
                # Load and hash the same snapshot if the writer advances meanwhile.
                checkpoint_bytes = core.io(state_path).read_bytes()
                state = torch.load(io.BytesIO(checkpoint_bytes), map_location="cpu", weights_only=False)
                required = {"identity", "model", "optimizer", "epoch", "cursor", "steps", "best", "best_state",
                            "history", "torch_rng", "cuda_rng", "numpy_rng", "python_rng"}
                if not required.issubset(state) or state["identity"] != expected:
                    raise ValueError(f"Training resume identity/structure mismatch: {state_path}")
                epoch, cursor, steps = state["epoch"], state["cursor"], state["steps"]
                expected_steps = epoch * math.ceil(n_train / cfg["accumulation"]) + math.ceil(cursor / cfg["accumulation"])
                if (not all(isinstance(x, int) for x in (epoch, cursor, steps)) or
                        not 0 <= epoch <= cfg["epochs"][part] or not 0 <= cursor <= n_train or
                        (cursor % cfg["accumulation"] and cursor != n_train) or
                        (epoch == cfg["epochs"][part] and cursor != 0) or
                        steps != expected_steps or len(state["history"]) != epoch):
                    raise ValueError(f"Inconsistent training progress: {state_path}")
                if (not state["model"] or not {"state", "param_groups"}.issubset(state["optimizer"]) or
                        any(not torch.isfinite(v).all().item() for v in state["model"].values())):
                    raise ValueError(f"Invalid model/optimizer state: {state_path}")
                summary.update(status="resumable", completed_epochs=epoch, cursor=cursor,
                               training_entries=n_train, optimizer_steps=steps,
                               checkpoint_sha256=hashlib.sha256(checkpoint_bytes).hexdigest())
            result.append(summary)
    return dict(freeze=snap["sha256"], training_entries=n_train, checkpoints=result,
                policy="Resume verified primary checkpoints; never promote .tmp or .bak automatically")
