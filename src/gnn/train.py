from __future__ import annotations

import argparse
from dataclasses import asdict
import random
import json
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import torch
import yaml
from torch import nn
from torch.optim import Optimizer
from torch.utils.data import DataLoader

from src.gnn.data import GraphTemporalDataset, collate_graph_samples
from src.gnn.models import build_model
from src.gnn.training import (
    LoopConfig,
    MetricSuite,
    TrainingConfig,
    build_decoder,
    load_training_config,
)



def _to_primitive(obj):
    if isinstance(obj, dict):
        return {k: _to_primitive(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_to_primitive(v) for v in obj]
    if isinstance(obj, tuple):
        return tuple(_to_primitive(v) for v in obj)
    if isinstance(obj, Path):
        return str(obj)
    return obj


def _set_seed(seed: Optional[int]) -> None:
    if seed is None:
        return
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _build_dataloader(dataset: GraphTemporalDataset, cfg: TrainingConfig, split: str) -> DataLoader:
    shuffle = cfg.data.shuffle if split == cfg.data.train_split else False
    return DataLoader(
        dataset,
        batch_size=cfg.data.batch_size,
        shuffle=shuffle,
        num_workers=cfg.data.num_workers,
        pin_memory=cfg.data.pin_memory,
        collate_fn=collate_graph_samples,
    )


def _build_optimizer(model: nn.Module, cfg: TrainingConfig) -> Optimizer:
    opt_name = cfg.optimizer.name.lower()
    params = [p for p in model.parameters() if p.requires_grad]
    if opt_name == "adamw":
        betas = tuple(cfg.optimizer.betas) if cfg.optimizer.betas else (0.9, 0.999)
        return torch.optim.AdamW(params, lr=cfg.optimizer.lr, weight_decay=cfg.optimizer.weight_decay, betas=betas)
    if opt_name == "adam":
        betas = tuple(cfg.optimizer.betas) if cfg.optimizer.betas else (0.9, 0.999)
        return torch.optim.Adam(params, lr=cfg.optimizer.lr, weight_decay=cfg.optimizer.weight_decay, betas=betas)
    if opt_name == "sgd":
        return torch.optim.SGD(params, lr=cfg.optimizer.lr, weight_decay=cfg.optimizer.weight_decay, momentum=0.9)
    raise ValueError(f"Unsupported optimizer '{cfg.optimizer.name}'")


def _build_loss(loop_cfg: LoopConfig) -> nn.Module:
    loss_name = loop_cfg.loss.lower()
    if loss_name == "mse":
        return nn.MSELoss()
    if loss_name in {"mae", "l1"}:
        return nn.L1Loss()
    if loss_name == "huber":
        return nn.SmoothL1Loss()
    raise ValueError(f"Unsupported loss '{loop_cfg.loss}'")



def _evaluate(model: nn.Module, loader: DataLoader, decoder, metrics: MetricSuite, device: torch.device) -> Dict[str, float]:
    model.eval()
    results: Dict[str, float] = {}
    counts: Dict[str, int] = {}
    with torch.no_grad():
        for batch in loader:
            batch_gpu = batch.to(device)
            pred = model(batch_gpu)
            decoded = decoder(batch_gpu, pred)
            batch_cpu = batch_gpu.to(torch.device("cpu"))
            metric_values = metrics.evaluate(batch_cpu, decoded.cpu(), batch_cpu.target)
            for key, metric in metric_values.items():
                results[key] = results.get(key, 0.0) + metric.value
                counts[key] = counts.get(key, 0) + 1
    for key in results:
        results[key] /= max(1, counts[key])
    return results


def train(config: TrainingConfig) -> None:
    output_dir = config.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    resolved_config_path = output_dir / "training_config_resolved.yaml"
    resolved_config_path.write_text(yaml.safe_dump(_to_primitive(asdict(config)), sort_keys=False), encoding="utf-8")

    _set_seed(config.loop.seed)

    train_dataset = GraphTemporalDataset(
        config.data.index_path,
        split=config.data.train_split,
        include_duals=config.data.include_duals,
        preload=config.data.preload,
    )
    val_dataset = GraphTemporalDataset(
        config.data.index_path,
        split=config.data.val_split,
        include_duals=config.data.include_duals,
        preload=config.data.preload,
    ) if config.data.val_split else None

    train_loader = _build_dataloader(train_dataset, config, config.data.train_split)
    val_loader = _build_dataloader(val_dataset, config, config.data.val_split) if val_dataset else None

    model = build_model(config.model, train_dataset)
    device = torch.device(config.loop.device)
    model.to(device)

    optimizer = _build_optimizer(model, config)
    loss_fn = _build_loss(config.loop)
    decoder = build_decoder(config.decoder)
    metrics = MetricSuite(
        dispatch_cfg=config.metrics.dispatch_error,
        cost_cfg=config.metrics.cost_gap,
        violation_cfg=config.metrics.constraint_violation_rate,
    )

    best_val = float("inf")
    best_path = output_dir / "best_model.pt"

    global_step = 0
    for epoch in range(1, config.loop.epochs + 1):
        model.train()
        running_loss = 0.0
        for batch_idx, batch in enumerate(train_loader, start=1):
            batch_gpu = batch.to(device)
            optimizer.zero_grad()
            pred = model(batch_gpu)
            loss = loss_fn(pred, batch_gpu.target)
            loss.backward()
            if config.loop.gradient_clip_norm is not None:
                torch.nn.utils.clip_grad_norm_(model.parameters(), config.loop.gradient_clip_norm)
            optimizer.step()
            running_loss += loss.item()
            global_step += 1
            if global_step % config.loop.log_every == 0:
                avg_loss = running_loss / config.loop.log_every
                print(f"[Epoch {epoch} | Step {global_step}] loss={avg_loss:.6f}")
                running_loss = 0.0
        if config.loop.eval_every and epoch % config.loop.eval_every == 0 and val_loader is not None:
            eval_metrics = _evaluate(model, val_loader, decoder, metrics, device)
            score = eval_metrics.get("dispatch_error", float("inf"))
            print(f"[Epoch {epoch}] validation metrics: {eval_metrics}")
            if score < best_val:
                best_val = score
                torch.save(model.state_dict(), best_path)
                print(f"  New best model saved at {best_path}")
    final_path = output_dir / "final_model.pt"
    if best_path.exists():
        state_dict = torch.load(best_path, map_location=device)
        model.load_state_dict(state_dict)
        torch.save(state_dict, final_path)
        print(f"Training complete. Restored best checkpoint to {final_path}")
    else:
        torch.save(model.state_dict(), final_path)
        print(f"Training complete. Final weights saved to {final_path}")

    if config.data.test_split:
        test_dataset = GraphTemporalDataset(
            config.data.index_path,
            split=config.data.test_split,
            include_duals=config.data.include_duals,
            preload=config.data.preload,
        )
        test_loader = _build_dataloader(test_dataset, config, config.data.test_split)
        test_metrics = _evaluate(model, test_loader, decoder, metrics, device)
        (output_dir / "test_metrics.json").write_text(json.dumps(test_metrics, indent=2), encoding="utf-8")
        print(f"Test metrics: {test_metrics}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train GNN dispatch models from configuration")
    parser.add_argument("--config", type=Path, required=True, help="Path to YAML training config")
    parser.add_argument("--device", type=str, default=None, help="Override device (cpu/cuda)")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_training_config(args.config)
    if args.device:
        cfg.loop.device = args.device
    train(cfg)


if __name__ == "__main__":
    main()
