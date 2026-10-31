"""Opt-in training contracts; old configuration files retain their behaviour."""
from __future__ import annotations

from dataclasses import asdict
import gc
import os
from pathlib import Path
import random
import shutil
import uuid

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset
from .common import PHYSICS, atomic_json, digest, read_json, sha, retry_io


def training_device(cfg, requested):
    if cfg.optim.get("operational"):
        if requested and requested not in ("cuda", "cuda:0"):
            raise ValueError("Operational training requires RTX 4050")
        from .embedding import require_gpu
        return require_gpu()
    return torch.device(requested or ("cuda" if torch.cuda.is_available() else "cpu"))


def frozen_loaders(data_cfg, include_dataset=False):
    from src.uc_jepa.data.uc_label_dataset import UCLabelDataset, uc_collate
    manifest = read_json(data_cfg["split_manifest"])
    assignments = manifest["splits"]["assignments"]
    expected = digest(manifest)
    if data_cfg.get("dataset_index"):
        index = read_json(data_cfg["dataset_index"])
        if digest(index) != data_cfg["dataset_digest"]:
            raise ValueError("Label dataset index changed")
        for entry in index["entries"]:
            if sha(entry["file"]) != entry["sha256"]:
                raise ValueError("Training data changed after dataset freeze")
    class CertifiedDataset(UCLabelDataset):
        def __getitem__(self, idx):
            with np.load(self.files[idx], allow_pickle=False) as a:
                if str(a["physics"].item()) != PHYSICS or str(a["training_manifest"].item()) != expected:
                    raise ValueError("Uncertified training label")
            return super().__getitem__(idx)
    ds = CertifiedDataset(data_cfg["roots"])
    keys = ["scenarios_v3/" + Path(p).name.removesuffix("_uc.npz") for p in ds.files]
    if len(keys) != len(set(keys)) or not set(keys) <= assignments.keys():
        raise ValueError("Unknown/duplicate training label identifier")
    indices = [[i for i, k in enumerate(keys) if assignments[k] == part] for part in ("train", "validation")]
    if any(not values for values in indices):
        raise ValueError("Both frozen partitions must contain valid labels")
    loaders = tuple(DataLoader(Subset(ds, idx), batch_size=8, shuffle=(n == 0), num_workers=0,
                              collate_fn=uc_collate, pin_memory=False) for n, idx in enumerate(indices))
    return (*loaders, ds) if include_dataset else loaders


def rng_state():
    return dict(python=random.getstate(), numpy=np.random.get_state(), torch=torch.get_rng_state(),
                cuda=torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [])


def restore_rng(state):
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"].cpu())
    if state["cuda"]:
        torch.cuda.set_rng_state_all([x.cpu() for x in state["cuda"]])


def atomic_torch_save(payload, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + "." + uuid.uuid4().hex + ".tmp")
    with tmp.open("xb") as f:
        torch.save(payload, f)
        f.flush()
        os.fsync(f.fileno())
    retry_io(lambda: os.replace(tmp, path))


def atomic_copy(source, target):
    target = Path(target)
    tmp = target.with_name(target.name + "." + uuid.uuid4().hex + ".tmp")
    retry_io(lambda: shutil.copyfile(source, tmp))
    with tmp.open("r+b") as f:
        f.flush()
        os.fsync(f.fileno())
    retry_io(lambda: os.replace(tmp, target))


def save_training_checkpoint(payload, path):
    cfg = payload.get("config", {})
    if cfg.get("optim", {}).get("operational"):
        payload = dict(payload, physics=PHYSICS, training_manifest=cfg["optim"]["manifest_hash"])
        atomic_torch_save(payload, path)
    else:
        torch.save(payload, path)


def backward_step(loss, optimizer, parameters, grad_clip, batch_size, step, loader):
    params = list(parameters)
    effective = getattr(optimizer, "_op_effective", None)
    if effective is None:
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        if grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(params, grad_clip)
        optimizer.step()
        return
    micro = loader.batch_size
    accumulation = effective // micro
    if step % accumulation == 0:
        optimizer.zero_grad(set_to_none=True)
    group_start = step // accumulation * accumulation * micro
    group_size = min(effective, len(loader.dataset) - group_start)
    (loss * batch_size / group_size).backward()
    if (step + 1) % accumulation == 0 or step + 1 == len(loader):
        if grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(params, grad_clip)
        if not getattr(optimizer, "_op_probe", False):
            optimizer.step()


class TrainingRuntime:
    def __init__(self, cfg, model, optimizer, scheduler, folder):
        self.active = bool(cfg.optim.get("operational"))
        self.model, self.optimizer, self.scheduler = model, optimizer, scheduler
        self.folder, self.cfg = Path(folder), cfg
        self.state, self.start_epoch, self.micro = {}, 0, 8
        self.probed = not self.active
        self.path = self.folder / "resume.pt"
        self.signature = digest(asdict(cfg))
        if not self.active:
            return
        if int(cfg.data.get("batch_size", 8)) != 8:
            raise ValueError("Effective batch must remain 8")
        optimizer._op_effective = 8
        commit_path = self.folder / "resume.json"
        if self.path.exists() and not commit_path.exists():
            # A first epoch interrupted before its commit is not resumable.
            # Preserve it for diagnosis and restart from the seeded new model.
            retry_io(lambda: self.path.rename(str(self.path) + ".uncommitted-" + uuid.uuid4().hex))
        if commit_path.exists():
            commit = read_json(commit_path)
            if not self.path.exists() or sha(self.path) != commit["sha256"]:
                backup = self.folder / "resume.previous.pt"
                if not backup.exists() or sha(backup) != commit["sha256"]:
                    raise ValueError("Interrupted/corrupt training checkpoint without valid committed backup")
                atomic_copy(backup, self.path)
            state = torch.load(self.path, map_location="cpu", weights_only=False)
            if state["signature"] != self.signature:
                raise ValueError("Training configuration/partitions/labels changed")
            model.load_state_dict(state["model_state_dict"])
            optimizer.load_state_dict(state["optimizer_state_dict"])
            scheduler.load_state_dict(state["scheduler_state_dict"])
            self.state = state
            self.start_epoch = state["epoch"] + 1
            if state["bad_epochs"] >= int(cfg.optim["patience"]):
                self.start_epoch = int(cfg.optim["epochs"])
            self.micro = state["microbatch"]
            self.probed = True
            restore_rng(state["rng"])

    def values(self, best, bad, history):
        return self.state.get("best_val", best), self.state.get("bad_epochs", bad), self.state.get("history", history)

    def thresholds(self, default):
        return self.state.get("thresholds", default)

    def loader(self, original, *, shuffle):
        if not self.active:
            return original
        return DataLoader(original.dataset, batch_size=self.micro, shuffle=shuffle, num_workers=0,
                          collate_fn=original.collate_fn, pin_memory=False)

    def train_epoch(self, callback, original):
        if not self.probed:
            from .embedding import require_gpu
            require_gpu()
            saved_rng = rng_state()
            # Largest internal train/validation example, repeated to stress the full
            # padding size. No evaluation-family data is read by this probe.
            internal = original.dataset.dataset if isinstance(original.dataset, Subset) else original.dataset
            sizes = [(internal[i].n_zones, i) for i in range(len(internal))]
            biggest = max(sizes)[1]
            self.optimizer._op_probe = True
            trials = []
            try:
                for micro in (8, 4, 2, 1):
                    self.micro = micro
                    reserve = None
                    try:
                        torch.cuda.reset_peak_memory_stats()
                        # Reserve Adam moments on the first epoch before its
                        # optimizer has allocated them.
                        reserve = [torch.zeros_like(p) for p in self.model.parameters() for _ in range(2)] if not self.optimizer.state else []
                        probe = DataLoader(Subset(internal, [biggest] * 8), batch_size=micro,
                                           collate_fn=original.collate_fn, num_workers=0)
                        callback(probe)
                        torch.cuda.synchronize()
                        trials.append(dict(microbatch=micro, success=True, peak_bytes=torch.cuda.max_memory_allocated()))
                        self.probed = True
                        break
                    except torch.cuda.OutOfMemoryError:
                        trials.append(dict(microbatch=micro, success=False))
                    finally:
                        del reserve
                        self.optimizer.zero_grad(set_to_none=True)
                        gc.collect()
                        torch.cuda.empty_cache()
                if not self.probed:
                    raise RuntimeError("RTX 4050 cannot fit microbatch 1; no CPU fallback")
            finally:
                self.optimizer._op_probe = False
                restore_rng(saved_rng)
                atomic_json(self.folder / "memory_probe.json", dict(trials=trials, effective_batch=8, largest_zones=max(sizes)[0]))
        return callback(self.loader(original, shuffle=True))

    def save(self, epoch, best, bad, history, thresholds=None, **extra):
        if not self.active:
            return
        payload = dict(signature=self.signature, physics=PHYSICS, training_manifest=self.cfg.optim["manifest_hash"],
                       model_state_dict=self.model.state_dict(), optimizer_state_dict=self.optimizer.state_dict(),
                       scheduler_state_dict=self.scheduler.state_dict(), epoch=epoch, best_val=best,
                       bad_epochs=bad, history=history, rng=rng_state(), thresholds=thresholds,
                       microbatch=self.micro, effective_batch=8, **extra)
        # Preserve the last COMMITTED epoch while publishing the next one.
        if self.path.exists():
            atomic_copy(self.path, self.folder / "resume.previous.pt")
        atomic_torch_save(payload, self.path)
        atomic_json(self.folder / "resume.json", dict(epoch=epoch, sha256=sha(self.path)))
