"""
Training and validation loops for the Hierarchical Temporal Encoder.
"""

from __future__ import annotations

import math
import time
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
from torch_geometric.utils import add_self_loops
from tqdm.auto import tqdm

from src.gnn.embeddings.dataset import get_hierarchy_from_batch
from src.gnn.embeddings.loss import contrastive_loss_multilag


# ---------------------------------------------------------------------------
# Cosine schedule with linear warm-up
# ---------------------------------------------------------------------------

def get_cosine_schedule_with_warmup(optimizer, num_warmup_epochs, num_epochs):
    """Cosine annealing with linear warmup."""

    def lr_lambda(epoch):
        if epoch < num_warmup_epochs:
            return float(epoch + 1) / float(max(1, num_warmup_epochs))
        progress = float(epoch - num_warmup_epochs) / float(
            max(1, num_epochs - num_warmup_epochs)
        )
        return max(0.0, 0.5 * (1.0 + math.cos(math.pi * progress)))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


# ---------------------------------------------------------------------------
# Single training epoch
# ---------------------------------------------------------------------------

def train_epoch(
    model: nn.Module,
    loader,
    optimizer: torch.optim.Optimizer,
    device: str,
    train_mean: torch.Tensor,
    train_std: torch.Tensor,
    *,
    accumulation_steps: int = 8,
    grad_clip: float = 0.5,
    lags=(1, 4, 8),
    neg_sample_ratio: float = 0.25,
    max_nodes: int = 160,
    temperature: float = 0.10,
) -> float:
    """Run one training epoch and return mean loss."""
    model.train()
    total_loss = 0.0
    num_batches = 0

    optimizer.zero_grad(set_to_none=True)

    pbar = tqdm(loader, desc="Training", leave=False)
    for i, batch in enumerate(pbar):
        batch = batch.to(device)

        # Normalise features
        if hasattr(batch, "x") and batch.x is not None:
            batch.x = (batch.x - train_mean) / train_std

        N_total = batch.x.size(0)
        N_base = batch.N_base[0].item() if batch.N_base.dim() > 0 else batch.N_base.item()
        T = batch.T[0].item() if batch.T.dim() > 0 else batch.T.item()

        hierarchy = get_hierarchy_from_batch(batch, device)

        edge_index_sl, _ = add_self_loops(batch.edge_index, num_nodes=N_total)

        with torch.autocast(device_type="cuda", enabled=torch.cuda.is_available()):
            embeddings = model(
                batch.x,
                edge_index_sl,
                batch.node_type if hasattr(batch, "node_type") else None,
                N_base,
                T,
                hierarchy_mapping=hierarchy,
                return_sequence=True,
            )

        E = embeddings["assets"].float() if isinstance(embeddings, dict) else embeddings.float()

        if torch.isnan(E).any() or torch.isinf(E).any():
            raise RuntimeError("NaN/Inf detected in embeddings during training")

        loss = contrastive_loss_multilag(
            E,
            lags=lags,
            neg_sample_ratio=neg_sample_ratio,
            max_nodes=max_nodes,
            temperature=temperature,
        )
        loss = loss / accumulation_steps

        if not torch.isfinite(loss):
            raise RuntimeError(f"Non-finite loss: {loss.item()}")

        loss.backward()

        if (i + 1) % accumulation_steps == 0:
            nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)

        total_loss += loss.item() * accumulation_steps
        num_batches += 1
        pbar.set_postfix(loss=f"{total_loss / num_batches:.4f}")

    # Final step if mid-accumulation
    remainder = (i + 1) % accumulation_steps
    if remainder != 0:
        nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)

    return total_loss / max(1, num_batches)


# ---------------------------------------------------------------------------
# Validation epoch
# ---------------------------------------------------------------------------

@torch.no_grad()
def validate_epoch(
    model: nn.Module,
    loader,
    device: str,
    train_mean: torch.Tensor,
    train_std: torch.Tensor,
    *,
    lags=(1, 4, 8),
    neg_sample_ratio: float = 0.10,
    max_nodes: int = 160,
    temperature: float = 0.20,
) -> float:
    """Run one validation epoch and return mean loss."""
    model.eval()
    total_loss = 0.0
    num_batches = 0

    for batch in loader:
        batch = batch.to(device)

        if hasattr(batch, "x") and batch.x is not None:
            batch.x = (batch.x - train_mean) / train_std

        if not hasattr(batch, "N_base") or not hasattr(batch, "T"):
            continue

        N_base = batch.N_base[0].item() if batch.N_base.dim() > 0 else batch.N_base.item()
        T = batch.T[0].item() if batch.T.dim() > 0 else batch.T.item()

        hierarchy = get_hierarchy_from_batch(batch, device)
        num_nodes = batch.x.size(0)
        edge_index_sl, _ = add_self_loops(batch.edge_index, num_nodes=num_nodes)

        with torch.amp.autocast(
            device_type="cuda", dtype=torch.bfloat16,
            enabled=torch.cuda.is_available(),
        ):
            embeddings = model(
                batch.x,
                edge_index_sl,
                batch.node_type if hasattr(batch, "node_type") else None,
                N_base,
                T,
                hierarchy_mapping=hierarchy,
                return_sequence=True,
            )

        E = embeddings["assets"].float() if isinstance(embeddings, dict) else embeddings.float()

        loss = contrastive_loss_multilag(
            E,
            lags=lags,
            neg_sample_ratio=neg_sample_ratio,
            max_nodes=max_nodes,
            temperature=temperature,
        )

        if not torch.isfinite(loss):
            return float("nan")

        total_loss += loss.item()
        num_batches += 1

    return total_loss / max(1, num_batches)


# ---------------------------------------------------------------------------
# Full training driver
# ---------------------------------------------------------------------------

def run_training(
    model: nn.Module,
    train_loader,
    val_loader,
    config,
    train_mean: torch.Tensor,
    train_std: torch.Tensor,
    hierarchy_mapping: Optional[dict] = None,
) -> dict:
    """
    Full training loop with early stopping, checkpointing, and LR scheduling.

    Returns:
        history dict with keys train_loss, val_loss, lr, epoch_time.
    """
    device = config.device
    save_dir = Path(config.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=config.lr, weight_decay=config.weight_decay,
        betas=(0.9, 0.999),
    )
    scheduler = get_cosine_schedule_with_warmup(
        optimizer, config.warmup_epochs, config.epochs,
    )

    history = {"train_loss": [], "val_loss": [], "lr": [], "epoch_time": []}
    best_val_loss = float("inf")
    patience_counter = 0

    print(f"\nStarting training for {config.epochs} epochs ...")
    print(f"  Early Stopping Patience : {config.early_stopping_patience}")
    print(f"  Gradient Accumulation   : {config.accumulation_steps}")
    print(f"  Effective Batch Size    : {config.effective_batch_size}")

    for epoch in range(1, config.epochs + 1):
        t0 = time.time()

        train_loss = train_epoch(
            model, train_loader, optimizer, device,
            train_mean, train_std,
            accumulation_steps=config.accumulation_steps,
            grad_clip=config.grad_clip,
            lags=config.lags,
            neg_sample_ratio=config.neg_sample_ratio,
            max_nodes=config.max_nodes,
            temperature=config.temperature,
        )

        val_loss = validate_epoch(
            model, val_loader, device,
            train_mean, train_std,
            lags=config.lags,
            neg_sample_ratio=config.neg_sample_ratio_val,
            max_nodes=config.max_nodes,
            temperature=config.temperature_val,
        )

        scheduler.step()
        # Cap LR
        for pg in optimizer.param_groups:
            pg["lr"] = min(pg["lr"], config.max_lr)

        epoch_time = time.time() - t0
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["lr"].append(optimizer.param_groups[0]["lr"])
        history["epoch_time"].append(epoch_time)

        print(
            f"Epoch {epoch:3d}/{config.epochs} | "
            f"Train: {train_loss:.4f} | Val: {val_loss:.4f} | "
            f"LR: {optimizer.param_groups[0]['lr']:.6f} | "
            f"Time: {epoch_time:.1f}s"
        )

        # Checkpointing
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "scheduler_state_dict": scheduler.state_dict(),
                    "val_loss": val_loss,
                    "config": config,
                    "hierarchy_mapping": hierarchy_mapping,
                    "train_mean": train_mean.cpu(),
                    "train_std": train_std.cpu(),
                },
                save_dir / "best_encoder.pt",
            )
            print(f"  -> Saved BEST model (val_loss={best_val_loss:.4f})")
        else:
            patience_counter += 1
            print(
                f"  -> No improvement "
                f"({patience_counter}/{config.early_stopping_patience})"
            )

        if patience_counter >= config.early_stopping_patience:
            print(f"\nEarly stopping triggered after {epoch} epochs.")
            break

        if epoch % config.save_freq == 0:
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "scheduler_state_dict": scheduler.state_dict(),
                    "val_loss": val_loss,
                    "history": history,
                    "config": config,
                },
                save_dir / f"checkpoint_epoch_{epoch}.pt",
            )

    print(f"\nTraining complete!  Best val loss: {best_val_loss:.6f}")
    print(f"Model saved to: {save_dir}")
    return history
