"""Train the UC autoencoder (Phase 1 MVP).

Usage:
    python -m src.uc_jepa.training.train_autoencoder \
        --config configs/uc_jepa/autoencoder.yaml

The config is YAML with sections ``data``, ``model``, ``loss``, ``optim``, ``logging``.
"""
from __future__ import annotations

from src.demand_response import DR_FORMULATION, require_current_formulation

import argparse
import json
import math
import os
import random
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader, Subset

from src.uc_jepa import N_UC_FEATURES, UC_FEATURE_NAMES
from src.uc_jepa.data.uc_label_dataset import UCLabelDataset, uc_collate
from src.uc_jepa.models.uc_autoencoder import (
    UCAELossWeights,
    UCAutoencoder,
    uc_autoencoder_loss,
)


@dataclass
class TrainConfig:
    data: Dict
    model: Dict
    loss: Dict
    optim: Dict
    logging: Dict


def _load_config(path: str) -> TrainConfig:
    with open(path, "r") as f:
        cfg = yaml.safe_load(f)
    return TrainConfig(
        data=cfg.get("data", {}),
        model=cfg.get("model", {}),
        loss=cfg.get("loss", {}),
        optim=cfg.get("optim", {}),
        logging=cfg.get("logging", {}),
    )


def _seed_all(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _build_dataloaders(data_cfg: Dict) -> Tuple[DataLoader, DataLoader, UCLabelDataset]:
    dataset = UCLabelDataset(
        roots=data_cfg["roots"],
        families=data_cfg.get("families"),
        max_samples=data_cfg.get("max_samples"),
    )
    n = len(dataset)
    val_split = float(data_cfg.get("val_split", 0.1))
    n_val = max(1, int(round(n * val_split)))
    n_train = max(1, n - n_val)
    seed = int(data_cfg.get("seed", 42))
    g = torch.Generator().manual_seed(seed)
    perm = torch.randperm(n, generator=g).tolist()
    train_idx = perm[:n_train]
    val_idx = perm[n_train : n_train + n_val]

    train_ds = Subset(dataset, train_idx)
    val_ds = Subset(dataset, val_idx)

    bs = int(data_cfg.get("batch_size", 8))
    nw = int(data_cfg.get("num_workers", 0))
    train_loader = DataLoader(
        train_ds, batch_size=bs, shuffle=True, num_workers=nw,
        collate_fn=uc_collate, drop_last=False, pin_memory=False,
    )
    val_loader = DataLoader(
        val_ds, batch_size=bs, shuffle=False, num_workers=nw,
        collate_fn=uc_collate, drop_last=False, pin_memory=False,
    )
    print(f"  train={len(train_ds)} val={len(val_ds)} (total={n})")
    return train_loader, val_loader, dataset


def _build_model(model_cfg: Dict) -> UCAutoencoder:
    sd = model_cfg.get("scenario_dim", None)
    return UCAutoencoder(
        n_features=int(model_cfg.get("n_features", 8)),
        hidden_dim=int(model_cfg.get("hidden_dim", 256)),
        n_tokens=int(model_cfg.get("n_tokens", 16)),
        latent_dim=int(model_cfg.get("latent_dim", 128)),
        n_heads=int(model_cfg.get("n_heads", 4)),
        gru_layers=int(model_cfg.get("gru_layers", 1)),
        dropout=float(model_cfg.get("dropout", 0.1)),
        n_zones_max=int(model_cfg.get("n_zones_max", 64)),
        n_timesteps_max=int(model_cfg.get("n_timesteps_max", 48)),
        scenario_dim=int(sd) if sd is not None else None,
        decoder_layers=int(model_cfg.get("decoder_layers", 1)),
    )


def _move_batch(batch: Dict, device: torch.device) -> Dict:
    out = {}
    for k, v in batch.items():
        if isinstance(v, torch.Tensor):
            out[k] = v.to(device, non_blocking=True)
        else:
            out[k] = v
    return out


def _compute_pos_weight(
    loader: DataLoader,
    n_features: int,
    cap: float = 50.0,
    eps: float = 1e-6,
) -> torch.Tensor:
    """Estimate per-feature ``pos_weight = (neg / pos)`` from the train loader.

    Capped at ``cap`` to avoid extreme weights for ultra-rare features
    (e.g. ``pumped_charge_mode`` at 0.3% positives would otherwise yield 333).
    """
    pos = np.zeros(n_features, dtype=np.float64)
    neg = np.zeros(n_features, dtype=np.float64)
    for batch in loader:
        u = batch["u_star"]
        mask = batch["zone_mask"]
        m = mask.unsqueeze(-1).unsqueeze(-1).expand_as(u).numpy()
        u_np = u.numpy()
        valid = m[..., 0]                                 # [B, Z, T]
        n_valid = valid.sum()
        for f in range(n_features):
            p_f = (u_np[..., f] * m[..., f]).sum()
            pos[f] += p_f
            neg[f] += (n_valid - p_f)
    ratio = neg / np.maximum(pos, eps)
    ratio = np.minimum(ratio, cap)
    ratio = np.maximum(ratio, 1.0)  # never down-weight positives
    return torch.tensor(ratio, dtype=torch.float32)


def _macro_metrics(
    pred_bin: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
) -> Dict[str, float]:
    """Compute macro-F1 / mean recall / mean precision over features."""
    m = mask.unsqueeze(-1).unsqueeze(-1).expand_as(target)
    f1s, recs, precs = [], [], []
    F_ = target.shape[-1]
    for f in range(F_):
        tp = float(((pred_bin[..., f] == 1) & (target[..., f] == 1) & (m[..., f] == 1)).sum().item())
        fp = float(((pred_bin[..., f] == 1) & (target[..., f] == 0) & (m[..., f] == 1)).sum().item())
        fn = float(((pred_bin[..., f] == 0) & (target[..., f] == 1) & (m[..., f] == 1)).sum().item())
        pr = tp / max(1.0, tp + fp)
        rc = tp / max(1.0, tp + fn)
        f1 = 2 * pr * rc / max(1e-9, pr + rc)
        precs.append(pr); recs.append(rc); f1s.append(f1)
    return {
        "macro_f1": float(np.mean(f1s)),
        "macro_recall": float(np.mean(recs)),
        "macro_precision": float(np.mean(precs)),
    }


def _epoch_pass(
    model: UCAutoencoder,
    loader: DataLoader,
    device: torch.device,
    weights: UCAELossWeights,
    optimizer: Optional[torch.optim.Optimizer],
    grad_clip: float,
    log_every: int,
    epoch: int,
    is_train: bool,
    pos_weight: Optional[torch.Tensor] = None,
    track_macro: bool = False,
) -> Dict[str, float]:
    if is_train:
        model.train()
    else:
        model.eval()

    totals = {"loss": 0.0, "bce": 0.0, "temporal": 0.0, "consistency": 0.0, "hamming": 0.0}
    if track_macro:
        # Per-feature TP/FP/FN tallied over the epoch (computed from binarized predictions).
        F_ = model.n_features
        tp = np.zeros(F_); fp = np.zeros(F_); fn = np.zeros(F_)
    n_seen = 0
    t0 = time.time()

    ctx = torch.enable_grad() if is_train else torch.no_grad()
    with ctx:
        for step, batch in enumerate(loader):
            batch = _move_batch(batch, device)
            u = batch["u_star"]
            mask = batch["zone_mask"]
            h = batch.get("h_scenario") if model.scenario_dim is not None else None
            out = model(u, zone_mask=mask, h_scenario=h)
            losses = uc_autoencoder_loss(
                out["logits"], u, zone_mask=mask,
                weights=weights, pos_weight=pos_weight,
            )

            if is_train:
                optimizer.zero_grad(set_to_none=True)
                losses["loss"].backward()
                if grad_clip and grad_clip > 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                optimizer.step()

            B = u.shape[0]
            n_seen += B
            for k in totals:
                totals[k] += float(losses[k].item()) * B

            if track_macro:
                with torch.no_grad():
                    pred = (torch.sigmoid(out["logits"]) >= 0.5).float()
                    m = mask.unsqueeze(-1).unsqueeze(-1).expand_as(u)
                    for f in range(F_):
                        tp[f] += float(((pred[..., f] == 1) & (u[..., f] == 1) & (m[..., f] == 1)).sum().item())
                        fp[f] += float(((pred[..., f] == 1) & (u[..., f] == 0) & (m[..., f] == 1)).sum().item())
                        fn[f] += float(((pred[..., f] == 0) & (u[..., f] == 1) & (m[..., f] == 1)).sum().item())

            if is_train and log_every > 0 and step % log_every == 0:
                lr = optimizer.param_groups[0]["lr"] if optimizer else float("nan")
                print(
                    f"  [ep{epoch:03d} step{step:04d}] loss={losses['loss'].item():.4f} "
                    f"bce={losses['bce'].item():.4f} ham={losses['hamming'].item():.4f} "
                    f"lr={lr:.2e}"
                )

    dt = time.time() - t0
    avg = {k: (v / max(1, n_seen)) for k, v in totals.items()}
    avg["seconds"] = dt
    avg["n"] = n_seen
    if track_macro:
        f1s, recs, precs = [], [], []
        for f in range(F_):
            pr = tp[f] / max(1.0, tp[f] + fp[f])
            rc = tp[f] / max(1.0, tp[f] + fn[f])
            f1 = 2 * pr * rc / max(1e-9, pr + rc)
            precs.append(pr); recs.append(rc); f1s.append(f1)
        avg["macro_f1"] = float(np.mean(f1s))
        avg["macro_recall"] = float(np.mean(recs))
        avg["macro_precision"] = float(np.mean(precs))
        avg["per_feature_f1"] = [float(x) for x in f1s]
    return avg


def _save_checkpoint(
    model: UCAutoencoder,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    metrics: Dict,
    cfg: TrainConfig,
    path: Path,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "dr_formulation": DR_FORMULATION,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "epoch": epoch,
        "metrics": metrics,
        "config": asdict(cfg),
    }, path)


def _make_lr_lambda(warmup_epochs: int, total_epochs: int):
    def fn(epoch: int) -> float:
        if warmup_epochs > 0 and epoch < warmup_epochs:
            return (epoch + 1) / float(warmup_epochs)
        # Cosine after warmup.
        t = max(1, total_epochs - warmup_epochs)
        e = epoch - warmup_epochs
        return 0.5 * (1.0 + math.cos(math.pi * e / t))
    return fn


def train(cfg: TrainConfig, device: Optional[str] = None) -> Dict:
    seed = int(cfg.data.get("seed", 42))
    _seed_all(seed)

    dev = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    print(f"Device: {dev}")

    train_loader, val_loader, dataset = _build_dataloaders(cfg.data)
    model = _build_model(cfg.model).to(dev)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"UCAutoencoder params: {n_params:,}")

    weights = UCAELossWeights(
        bce=float(cfg.loss.get("bce", 1.0)),
        temporal=float(cfg.loss.get("temporal", 0.05)),
        consistency=float(cfg.loss.get("consistency", 0.1)),
    )

    # ─── pos_weight (class imbalance correction) ────────────────────────────
    pw_cfg = cfg.loss.get("pos_weight", "auto")
    pos_weight: Optional[torch.Tensor] = None
    if pw_cfg in (None, "none", False):
        print("pos_weight: disabled (raw BCE).")
    elif isinstance(pw_cfg, (list, tuple)):
        pos_weight = torch.tensor([float(x) for x in pw_cfg], dtype=torch.float32)
        print(f"pos_weight: manual = {pos_weight.tolist()}")
    elif pw_cfg in ("auto", True):
        cap = float(cfg.loss.get("pos_weight_cap", 50.0))
        print(f"pos_weight: computing from train set (cap={cap})...")
        pos_weight = _compute_pos_weight(train_loader, model.n_features, cap=cap)
        for name, w in zip(UC_FEATURE_NAMES, pos_weight.tolist()):
            print(f"    {name:<22} pos_weight={w:.2f}")
    else:
        raise ValueError(f"Unknown loss.pos_weight={pw_cfg!r}; expected 'auto', 'none' or a list.")
    if pos_weight is not None:
        pos_weight = pos_weight.to(dev)

    lr = float(cfg.optim.get("lr", 3e-4))
    wd = float(cfg.optim.get("weight_decay", 1e-4))
    epochs = int(cfg.optim.get("epochs", 60))
    patience = int(cfg.optim.get("patience", 10))
    grad_clip = float(cfg.optim.get("grad_clip", 1.0))
    warmup_epochs = int(cfg.optim.get("warmup_epochs", 0))

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lr_lambda=_make_lr_lambda(warmup_epochs, epochs),
    )

    out_dir = Path(cfg.logging.get("output_dir", "outputs/uc_jepa/autoencoder"))
    out_dir.mkdir(parents=True, exist_ok=True)
    save_every = int(cfg.logging.get("save_every", 5))
    log_every = int(cfg.logging.get("log_every", 50))

    history: List[Dict] = []
    best_metric_name = str(cfg.optim.get("best_metric", "macro_f1"))
    higher_is_better = best_metric_name in {"macro_f1", "macro_recall", "macro_precision"}
    best_val = -float("inf") if higher_is_better else float("inf")
    bad_epochs = 0
    best_path = out_dir / "uc_ae_best.pt"

    for epoch in range(epochs):
        train_metrics = _epoch_pass(
            model, train_loader, dev, weights, optimizer, grad_clip,
            log_every, epoch, is_train=True,
            pos_weight=pos_weight, track_macro=False,
        )
        val_metrics = _epoch_pass(
            model, val_loader, dev, weights, optimizer=None, grad_clip=0.0,
            log_every=0, epoch=epoch, is_train=False,
            pos_weight=pos_weight, track_macro=True,
        )
        scheduler.step()

        record = {
            "epoch": epoch,
            "lr": optimizer.param_groups[0]["lr"],
            "train": train_metrics,
            "val": val_metrics,
        }
        history.append(record)
        print(
            f"Epoch {epoch+1}/{epochs} ({train_metrics['seconds']:.1f}s+"
            f"{val_metrics['seconds']:.1f}s) | "
            f"train loss={train_metrics['loss']:.4f} ham={train_metrics['hamming']:.4f} | "
            f"val loss={val_metrics['loss']:.4f} ham={val_metrics['hamming']:.4f} "
            f"F1={val_metrics.get('macro_f1', float('nan')):.3f} "
            f"R={val_metrics.get('macro_recall', float('nan')):.3f} "
            f"P={val_metrics.get('macro_precision', float('nan')):.3f} | "
            f"lr={optimizer.param_groups[0]['lr']:.2e}"
        )

        cur = float(val_metrics.get(best_metric_name, val_metrics["loss"]))
        improved = (cur > best_val + 1e-6) if higher_is_better else (cur < best_val - 1e-6)
        if improved:
            best_val = cur
            bad_epochs = 0
            _save_checkpoint(model, optimizer, epoch, val_metrics, cfg, best_path)
            print(f"  >> new best val {best_metric_name}={best_val:.4f} -> {best_path}")
        else:
            bad_epochs += 1
            if bad_epochs >= patience:
                print(f"  Early stopping at epoch {epoch+1} (no improvement for {patience} epochs).")
                break

        if save_every > 0 and (epoch + 1) % save_every == 0:
            _save_checkpoint(
                model, optimizer, epoch, val_metrics, cfg,
                out_dir / f"uc_ae_epoch_{epoch+1}.pt",
            )

    history_path = out_dir / "history.json"
    with open(history_path, "w") as f:
        json.dump(history, f, indent=2)
    print(f"History -> {history_path}")
    print(f"Best val {best_metric_name}: {best_val:.4f} ({best_path})")
    return {"best_metric": best_metric_name, "best_val": best_val, "best_ckpt": str(best_path), "history": history}


def _parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train the UC autoencoder (Phase 1 MVP).")
    p.add_argument("--config", required=True, help="YAML config path.")
    p.add_argument("--device", default=None, help="cuda or cpu (auto if unset).")
    return p.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = _parse_args(argv)
    cfg = _load_config(args.config)
    train(cfg, device=args.device)
    return 0


if __name__ == "__main__":
    sys.exit(main())
