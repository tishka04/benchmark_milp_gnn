# ==============================================================================
# DISPATCH GNN TRAINER - Training loop and evaluation for DispatchGNN
# ==============================================================================

from __future__ import annotations

import time
import torch
import torch.nn as nn
import numpy as np
from typing import Dict, List, Optional, Tuple
from torch.utils.data import DataLoader
from dataclasses import dataclass, field


@dataclass
class TrainConfig:
    """Training configuration for DispatchGNN."""
    # Model
    n_binary_features: int = 7
    embed_dim: int = 128
    hidden_dim: int = 256
    n_dispatch: int = 11
    gru_layers: int = 2
    n_gat_layers: int = 3
    n_heads: int = 8
    dropout: float = 0.1

    # Training
    epochs: int = 60
    lr: float = 3e-4
    weight_decay: float = 1e-4
    warmup_epochs: int = 5
    grad_clip: float = 1.0
    batch_size: int = 16

    # Loss weights per dispatch channel (higher = more important)
    # Order: thermal, nuclear, solar, wind, batt_ch, batt_dis,
    #        pump_ch, pump_dis, dr, unserved, hydro
    channel_weights: Optional[List[float]] = None

    # Data
    n_timesteps: int = 24
    val_split: float = 0.1
    num_workers: int = 2
    seed: int = 42
    max_scenarios: Optional[int] = None

    # Paths (Colab defaults)
    repo_path: str = '/content/drive/MyDrive/benchmark'
    reports_dir: str = 'outputs/scenarios_v3/reports'
    embeddings_dir: str = 'outputs/encoders/hierarchical_temporal_v3/embeddings_v3'
    output_dir: str = 'outputs/gnn_dispatch'

    device: str = 'cuda'


class MaskedMSELoss(nn.Module):
    """
    Per-channel weighted MSE loss with zone masking.

    Computes MSE only on valid (non-padded) zones, with optional
    per-channel weights to emphasize important dispatch variables.
    """

    def __init__(
        self,
        channel_weights: Optional[torch.Tensor] = None,
        normalize_mean: Optional[torch.Tensor] = None,
        normalize_std: Optional[torch.Tensor] = None,
    ):
        super().__init__()
        self.register_buffer('channel_weights', channel_weights)
        self.register_buffer('normalize_mean', normalize_mean)
        self.register_buffer('normalize_std', normalize_std)

    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        zone_mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            pred:      [B, Z, T, C] predicted dispatch
            target:    [B, Z, T, C] ground truth dispatch
            zone_mask: [B, Z] valid zone indicator (1=valid, 0=pad)

        Returns:
            scalar loss
        """
        # Normalize targets if stats provided
        if self.normalize_std is not None:
            std = self.normalize_std.view(1, 1, 1, -1)
            mean = self.normalize_mean.view(1, 1, 1, -1) if self.normalize_mean is not None else 0
            target_norm = (target - mean) / std
            pred_norm = (pred - mean) / std
        else:
            target_norm = target
            pred_norm = pred

        # Squared error [B, Z, T, C]
        se = (pred_norm - target_norm) ** 2

        # Apply channel weights
        if self.channel_weights is not None:
            w = self.channel_weights.view(1, 1, 1, -1)
            se = se * w

        # Apply zone mask [B, Z, 1, 1]
        mask = zone_mask.view(zone_mask.shape[0], zone_mask.shape[1], 1, 1).float()
        se = se * mask

        # Mean over valid entries
        n_valid = mask.sum() * se.shape[2] * se.shape[3]
        loss = se.sum() / n_valid.clamp(min=1.0)

        return loss


def train_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    loss_fn: MaskedMSELoss,
    device: str,
    scaler: Optional[torch.amp.GradScaler] = None,
    grad_clip: float = 1.0,
) -> Dict[str, float]:
    """Train one epoch. Returns dict of metrics."""
    model.train()
    total_loss = 0.0
    n_batches = 0

    for batch in loader:
        u_zt = batch["u_zt"].to(device)
        h_zt = batch["h_zt"].to(device)
        y_zt = batch["y_zt"].to(device)
        zone_mask = batch["zone_mask"].to(device)

        optimizer.zero_grad(set_to_none=True)

        with torch.amp.autocast('cuda', enabled=scaler is not None):
            pred = model(u_zt, h_zt, zone_mask)
            loss = loss_fn(pred, y_zt, zone_mask)

        if scaler is not None:
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()

        total_loss += loss.item()
        n_batches += 1

    return {"train_loss": total_loss / max(n_batches, 1)}


@torch.no_grad()
def validate(
    model: nn.Module,
    loader: DataLoader,
    loss_fn: MaskedMSELoss,
    device: str,
    channel_names: Optional[List[str]] = None,
) -> Dict[str, float]:
    """Validate and compute per-channel MAE metrics."""
    from src.gnn.dispatch_model import DISPATCH_CHANNELS, N_DISPATCH

    model.eval()
    total_loss = 0.0
    n_batches = 0

    # Per-channel MAE accumulator
    channel_mae_sum = torch.zeros(N_DISPATCH, device=device)
    channel_count = torch.zeros(N_DISPATCH, device=device)

    for batch in loader:
        u_zt = batch["u_zt"].to(device)
        h_zt = batch["h_zt"].to(device)
        y_zt = batch["y_zt"].to(device)
        zone_mask = batch["zone_mask"].to(device)

        pred = model(u_zt, h_zt, zone_mask)
        loss = loss_fn(pred, y_zt, zone_mask)

        total_loss += loss.item()
        n_batches += 1

        # Per-channel MAE (valid zones only)
        mask = zone_mask.unsqueeze(-1).unsqueeze(-1).float()  # [B, Z, 1, 1]
        ae = (pred - y_zt).abs() * mask                       # [B, Z, T, C]
        for c in range(N_DISPATCH):
            channel_mae_sum[c] += ae[:, :, :, c].sum()
            channel_count[c] += mask.squeeze(-1).squeeze(-1).sum() * y_zt.shape[2]

    metrics = {"val_loss": total_loss / max(n_batches, 1)}

    # Per-channel MAE
    names = channel_names or DISPATCH_CHANNELS
    for c in range(N_DISPATCH):
        mae = (channel_mae_sum[c] / channel_count[c].clamp(min=1)).item()
        metrics[f"mae_{names[c]}"] = mae

    # Overall MAE
    metrics["mae_overall"] = (
        channel_mae_sum.sum() / channel_count.sum().clamp(min=1)
    ).item()

    return metrics


def run_training(
    cfg: TrainConfig,
    train_loader: DataLoader,
    val_loader: DataLoader,
    channel_mean: Optional[torch.Tensor] = None,
    channel_std: Optional[torch.Tensor] = None,
) -> Tuple[nn.Module, List[Dict]]:
    """
    Full training loop for DispatchGNN.

    Returns:
        model: trained DispatchGNN
        history: list of per-epoch metric dicts
    """
    from src.gnn.dispatch_model import DispatchGNN, DISPATCH_CHANNELS
    import os, json
    from pathlib import Path

    device = cfg.device

    # Build model
    model = DispatchGNN(
        n_binary_features=cfg.n_binary_features,
        embed_dim=cfg.embed_dim,
        hidden_dim=cfg.hidden_dim,
        n_dispatch=cfg.n_dispatch,
        gru_layers=cfg.gru_layers,
        n_gat_layers=cfg.n_gat_layers,
        n_heads=cfg.n_heads,
        dropout=cfg.dropout,
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters())
    print(f"DispatchGNN: {n_params:,} parameters")

    # Channel weights
    if cfg.channel_weights:
        cw = torch.tensor(cfg.channel_weights, dtype=torch.float32, device=device)
    else:
        # Default: upweight thermal, nuclear, unserved (expensive resources)
        cw = torch.ones(cfg.n_dispatch, device=device)
        cw[0] = 2.0   # thermal
        cw[1] = 2.0   # nuclear
        cw[9] = 3.0   # unserved (high penalty)

    # Loss function
    loss_fn = MaskedMSELoss(
        channel_weights=cw,
        normalize_mean=channel_mean.to(device) if channel_mean is not None else None,
        normalize_std=channel_std.to(device) if channel_std is not None else None,
    ).to(device)

    # Optimizer + scheduler
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=cfg.epochs - cfg.warmup_epochs, eta_min=cfg.lr * 0.01,
    )

    # Warmup
    warmup_scheduler = None
    if cfg.warmup_epochs > 0:
        warmup_scheduler = torch.optim.lr_scheduler.LinearLR(
            optimizer, start_factor=0.1, total_iters=cfg.warmup_epochs,
        )

    # Mixed precision
    use_amp = device == 'cuda' and torch.cuda.is_available()
    scaler = torch.amp.GradScaler('cuda') if use_amp else None

    # Output dir
    output_dir = Path(cfg.repo_path) / cfg.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    history = []
    best_val_loss = float('inf')
    best_epoch = 0

    print(f"\n{'='*60}")
    print(f"Training DispatchGNN for {cfg.epochs} epochs")
    print(f"  Device: {device}, AMP: {use_amp}")
    print(f"  LR: {cfg.lr}, WD: {cfg.weight_decay}")
    print(f"{'='*60}\n")

    for epoch in range(1, cfg.epochs + 1):
        t0 = time.time()

        # Train
        train_metrics = train_epoch(
            model, train_loader, optimizer, loss_fn, device,
            scaler=scaler, grad_clip=cfg.grad_clip,
        )

        # Validate
        val_metrics = validate(model, val_loader, loss_fn, device)

        # Scheduler step
        if warmup_scheduler is not None and epoch <= cfg.warmup_epochs:
            warmup_scheduler.step()
        else:
            scheduler.step()

        elapsed = time.time() - t0
        lr_now = optimizer.param_groups[0]['lr']

        # Merge metrics
        metrics = {**train_metrics, **val_metrics, "epoch": epoch, "lr": lr_now, "time": elapsed}
        history.append(metrics)

        # Save best
        if val_metrics["val_loss"] < best_val_loss:
            best_val_loss = val_metrics["val_loss"]
            best_epoch = epoch
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_loss": best_val_loss,
                "config": {
                    "n_binary_features": cfg.n_binary_features,
                    "embed_dim": cfg.embed_dim,
                    "hidden_dim": cfg.hidden_dim,
                    "n_dispatch": cfg.n_dispatch,
                    "gru_layers": cfg.gru_layers,
                    "n_gat_layers": cfg.n_gat_layers,
                    "n_heads": cfg.n_heads,
                    "dropout": cfg.dropout,
                },
                "channel_mean": channel_mean,
                "channel_std": channel_std,
            }, output_dir / "dispatch_gnn_best.pt")

        # Log
        marker = " *" if epoch == best_epoch else ""
        print(
            f"Epoch {epoch:3d}/{cfg.epochs} | "
            f"Train {train_metrics['train_loss']:.4f} | "
            f"Val {val_metrics['val_loss']:.4f} | "
            f"MAE {val_metrics['mae_overall']:.1f} MW | "
            f"LR {lr_now:.2e} | "
            f"{elapsed:.1f}s{marker}"
        )

        # Every 10 epochs, print per-channel MAE
        if epoch % 10 == 0 or epoch == cfg.epochs:
            print("  Per-channel MAE (MW):")
            for c, name in enumerate(DISPATCH_CHANNELS):
                print(f"    {name:20s}: {val_metrics.get(f'mae_{name}', 0):.1f}")

    # Save final
    torch.save({
        "epoch": cfg.epochs,
        "model_state_dict": model.state_dict(),
        "config": {
            "n_binary_features": cfg.n_binary_features,
            "embed_dim": cfg.embed_dim,
            "hidden_dim": cfg.hidden_dim,
            "n_dispatch": cfg.n_dispatch,
            "gru_layers": cfg.gru_layers,
            "n_gat_layers": cfg.n_gat_layers,
            "n_heads": cfg.n_heads,
            "dropout": cfg.dropout,
        },
        "channel_mean": channel_mean,
        "channel_std": channel_std,
        "history": history,
    }, output_dir / "dispatch_gnn_final.pt")

    # Save history as JSON
    with open(output_dir / "training_history.json", "w") as f:
        json.dump(history, f, indent=2)

    print(f"\nBest val_loss: {best_val_loss:.4f} at epoch {best_epoch}")
    print(f"Models saved to {output_dir}")

    # Reload best weights
    best_ckpt = torch.load(output_dir / "dispatch_gnn_best.pt", map_location=device, weights_only=False)
    model.load_state_dict(best_ckpt["model_state_dict"])

    return model, history
