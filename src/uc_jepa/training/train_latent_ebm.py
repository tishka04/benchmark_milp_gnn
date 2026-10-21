"""Train a latent-space EBM on top of frozen UC-JEPA components.

Pipeline:
    h_scenario --[frozen JEPA]--> z_anchor
    u_star     --[frozen AE enc]--> z_pos
    z_anchor --[latent Langevin + EBM]--> z_neg

The EBM is optimized with a contrastive-divergence style loss so that
``E(z_pos | z_anchor, h) < E(z_neg | z_anchor, h)``.
"""
from __future__ import annotations

from src.demand_response import DR_FORMULATION, require_current_formulation

import argparse
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import yaml
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset

from src.uc_jepa.data.uc_label_dataset import UCLabelDataset, uc_collate
from src.uc_jepa.latent_utils import (
    accumulate_confusion,
    apply_thresholds,
    estimate_latent_prior,
    f1_from_counts,
    load_frozen_ae,
    load_frozen_jepa,
    move_batch,
    pairwise_hamming_mean,
    seed_all,
    threshold_dict,
    threshold_vector,
)
from src.uc_jepa.models.latent_ebm import LatentEBM
from src.uc_jepa.sampling.latent_langevin import LatentLangevinConfig, LatentLangevinSampler


@dataclass
class LatentEBMTrainConfig:
    data: Dict[str, Any]
    ae: Dict[str, Any]
    jepa: Dict[str, Any]
    model: Dict[str, Any]
    sampler: Dict[str, Any]
    loss: Dict[str, Any]
    optim: Dict[str, Any]
    eval: Dict[str, Any]
    logging: Dict[str, Any]


def _load_config(path: str) -> LatentEBMTrainConfig:
    with open(path, "r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle)
    return LatentEBMTrainConfig(
        data=raw.get("data", {}),
        ae=raw.get("ae", {}),
        jepa=raw.get("jepa", {}),
        model=raw.get("model", {}),
        sampler=raw.get("sampler", {}),
        loss=raw.get("loss", {}),
        optim=raw.get("optim", {}),
        eval=raw.get("eval", {}),
        logging=raw.get("logging", {}),
    )


def _build_dataloaders(data_cfg: Dict[str, Any]) -> Tuple[DataLoader, DataLoader]:
    dataset = UCLabelDataset(
        roots=data_cfg["roots"],
        families=data_cfg.get("families"),
        max_samples=data_cfg.get("max_samples"),
    )
    n_total = len(dataset)
    val_split = float(data_cfg.get("val_split", 0.1))
    n_val = max(1, int(round(n_total * val_split)))
    n_train = max(1, n_total - n_val)
    seed = int(data_cfg.get("seed", 42))
    generator = torch.Generator().manual_seed(seed)
    perm = torch.randperm(n_total, generator=generator).tolist()
    train_ds = Subset(dataset, perm[:n_train])
    val_ds = Subset(dataset, perm[n_train : n_train + n_val])

    batch_size = int(data_cfg.get("batch_size", 8))
    num_workers = int(data_cfg.get("num_workers", 0))
    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        collate_fn=uc_collate,
        drop_last=False,
        pin_memory=False,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=uc_collate,
        drop_last=False,
        pin_memory=False,
    )
    print(f"  train={len(train_ds)} val={len(val_ds)} (total={n_total})")
    return train_loader, val_loader


def _make_lr_lambda(warmup_epochs: int, total_epochs: int):
    def fn(epoch: int) -> float:
        if warmup_epochs > 0 and epoch < warmup_epochs:
            return (epoch + 1) / float(warmup_epochs)
        span = max(1, total_epochs - warmup_epochs)
        e = epoch - warmup_epochs
        return 0.5 * (1.0 + math.cos(math.pi * e / span))

    return fn


def _contrastive_loss(
    e_pos: torch.Tensor,
    e_neg: torch.Tensor,
    objective: str,
    energy_reg: float,
    energy_margin: float,
    cd_clamp: float,
    rank_margin: float,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    raw_cd = e_pos.mean() - e_neg.mean()
    objective_key = str(objective or "clamped_cd").lower()
    if objective_key == "softplus":
        cd = F.softplus(raw_cd)
    elif objective_key == "margin":
        cd = torch.relu(rank_margin + raw_cd)
    elif objective_key == "clamped_cd":
        cd = torch.clamp(raw_cd, -cd_clamp, cd_clamp)
    else:
        raise ValueError(f"Unknown latent EBM loss.objective={objective!r}")
    reg = energy_reg * (
        torch.clamp(e_pos.abs() - energy_margin, min=0.0).pow(2).mean()
        + torch.clamp(e_neg.abs() - energy_margin, min=0.0).pow(2).mean()
    )
    loss = cd + reg
    return loss, {
        "loss": float(loss.item()),
        "cd_loss": float(cd.item()),
        "reg_loss": float(reg.item()),
        "raw_cd": float(raw_cd.item()),
        "E_pos_mean": float(e_pos.mean().item()),
        "E_neg_mean": float(e_neg.mean().item()),
        "E_gap": float((e_neg.mean() - e_pos.mean()).item()),
    }


def _decode_diag_metrics(
    ae,
    ebm: LatentEBM,
    sampler: LatentLangevinSampler,
    h_scenario: torch.Tensor,
    zone_mask: torch.Tensor,
    u_star: torch.Tensor,
    z_anchor: torch.Tensor,
    thresholds: torch.Tensor,
    prior_mean: torch.Tensor,
    prior_std: torch.Tensor,
    n_samples: int,
) -> Dict[str, float]:
    with torch.no_grad():
        anchor_logits = ae.decode(
            z_anchor,
            n_zones=u_star.shape[1],
            n_timesteps=u_star.shape[2],
            h_scenario=h_scenario,
        )
        anchor_bin = apply_thresholds(torch.sigmoid(anchor_logits), thresholds)

    z_samples, sample_energies, _ = sampler.sample(
        ebm=ebm,
        z_anchor=z_anchor,
        h_scenario=h_scenario,
        zone_mask=zone_mask,
        n_samples=n_samples,
        prior_mean=prior_mean,
        prior_std=prior_std,
    )
    z_best, best_energy, _ = sampler.select_best(z_samples, sample_energies)

    with torch.no_grad():
        flat_z = z_samples.reshape(
            z_samples.shape[0] * z_samples.shape[1],
            z_samples.shape[2],
            z_samples.shape[3],
        )
        h_expanded = h_scenario.unsqueeze(1).expand(-1, n_samples, -1, -1, -1).reshape(
            flat_z.shape[0],
            h_scenario.shape[1],
            h_scenario.shape[2],
            h_scenario.shape[3],
        )
        sample_logits = ae.decode(
            flat_z,
            n_zones=u_star.shape[1],
            n_timesteps=u_star.shape[2],
            h_scenario=h_expanded,
        ).reshape(
            u_star.shape[0],
            n_samples,
            u_star.shape[1],
            u_star.shape[2],
            u_star.shape[3],
        )
        sample_bin = apply_thresholds(torch.sigmoid(sample_logits), thresholds)
        best_idx = sample_energies.argmin(dim=1)
        gather_idx = best_idx.view(-1, 1, 1, 1, 1).expand(
            -1, 1, sample_bin.shape[2], sample_bin.shape[3], sample_bin.shape[4],
        )
        best_bin = sample_bin.gather(dim=1, index=gather_idx).squeeze(1)

    feat_count = u_star.shape[-1]
    tp_anchor = np.zeros(feat_count)
    fp_anchor = np.zeros(feat_count)
    fn_anchor = np.zeros(feat_count)
    tp_best = np.zeros(feat_count)
    fp_best = np.zeros(feat_count)
    fn_best = np.zeros(feat_count)
    accumulate_confusion(tp_anchor, fp_anchor, fn_anchor, anchor_bin, u_star, zone_mask)
    accumulate_confusion(tp_best, fp_best, fn_best, best_bin, u_star, zone_mask)
    anchor_metrics = f1_from_counts(tp_anchor, fp_anchor, fn_anchor)
    best_metrics = f1_from_counts(tp_best, fp_best, fn_best)

    diversity = float(
        np.mean([pairwise_hamming_mean(sample_bin[b]) for b in range(sample_bin.shape[0])]),
    )
    return {
        "anchor_macro_f1": float(anchor_metrics["macro_f1"]),
        "sample_macro_f1": float(best_metrics["macro_f1"]),
        "sample_diversity": diversity,
        "sample_energy_mean": float(sample_energies.mean().item()),
        "sample_energy_std": float(sample_energies.std(unbiased=False).item()),
        "best_energy_mean": float(best_energy.mean().item()),
    }


def _epoch_pass(
    ebm: LatentEBM,
    ae,
    predictor,
    sampler: LatentLangevinSampler,
    loader: DataLoader,
    device: torch.device,
    optimizer: Optional[torch.optim.Optimizer],
    grad_clip: float,
    is_train: bool,
    epoch: int,
    log_every: int,
    prior_mean: torch.Tensor,
    prior_std: torch.Tensor,
    thresholds: torch.Tensor,
    loss_cfg: Dict[str, Any],
    eval_cfg: Dict[str, Any],
) -> Dict[str, float]:
    if is_train:
        ebm.train()
    else:
        ebm.eval()

    energy_reg = float(loss_cfg.get("energy_reg", 0.01))
    energy_margin = float(loss_cfg.get("energy_margin", 30.0))
    cd_clamp = float(loss_cfg.get("cd_clamp", 10.0))
    objective = str(loss_cfg.get("objective", "clamped_cd"))
    rank_margin = float(loss_cfg.get("rank_margin", 1.0))
    neg_init_noise = float(loss_cfg.get("neg_init_noise", 0.10))
    neg_from_pos_prob = float(loss_cfg.get("neg_from_pos_prob", 0.25))

    totals: Dict[str, float] = {
        "loss": 0.0,
        "cd_loss": 0.0,
        "reg_loss": 0.0,
        "E_pos_mean": 0.0,
        "E_neg_mean": 0.0,
        "E_gap": 0.0,
        "anchor_mse": 0.0,
        "anchor_cos": 0.0,
        "anchor_energy": 0.0,
    }
    diag_totals: Dict[str, float] = {}
    n_batches = 0
    n_diag_batches = 0

    for step, batch in enumerate(loader):
        batch = move_batch(batch, device)
        u_star = batch["u_star"]
        h_scenario = batch["h_scenario"]
        zone_mask = batch["zone_mask"]

        with torch.no_grad():
            z_pos = ae.encode(u_star, zone_mask=zone_mask)
            z_anchor = predictor(h_scenario, zone_mask=zone_mask)["z_pred"]

        if torch.rand(1).item() < neg_from_pos_prob:
            init_z = z_pos + neg_init_noise * torch.randn_like(z_pos)
        else:
            init_z = z_anchor + neg_init_noise * torch.randn_like(z_anchor)

        z_neg_samples, _, _ = sampler.sample(
            ebm=ebm,
            z_anchor=z_anchor,
            h_scenario=h_scenario,
            zone_mask=zone_mask,
            n_samples=1,
            init_z=init_z,
            prior_mean=prior_mean,
            prior_std=prior_std,
        )
        z_neg = z_neg_samples.squeeze(1)

        if is_train:
            optimizer.zero_grad(set_to_none=True)
            e_pos = ebm(z_pos, z_anchor, h_scenario=h_scenario, zone_mask=zone_mask)
            e_neg = ebm(z_neg.detach(), z_anchor, h_scenario=h_scenario, zone_mask=zone_mask)
            loss, metrics = _contrastive_loss(
                e_pos,
                e_neg,
                objective=objective,
                energy_reg=energy_reg,
                energy_margin=energy_margin,
                cd_clamp=cd_clamp,
                rank_margin=rank_margin,
            )
            loss.backward()
            if grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(ebm.parameters(), grad_clip)
            optimizer.step()
        else:
            with torch.no_grad():
                e_pos = ebm(z_pos, z_anchor, h_scenario=h_scenario, zone_mask=zone_mask)
                e_neg = ebm(z_neg.detach(), z_anchor, h_scenario=h_scenario, zone_mask=zone_mask)
                loss, metrics = _contrastive_loss(
                    e_pos,
                    e_neg,
                    objective=objective,
                    energy_reg=energy_reg,
                    energy_margin=energy_margin,
                    cd_clamp=cd_clamp,
                    rank_margin=rank_margin,
                )

        with torch.no_grad():
            anchor_energy = ebm(
                z_anchor,
                z_anchor,
                h_scenario=h_scenario,
                zone_mask=zone_mask,
            )
            anchor_mse = (z_pos - z_anchor).pow(2).mean()
            anchor_cos = torch.nn.functional.cosine_similarity(
                z_pos.reshape(z_pos.shape[0], -1),
                z_anchor.reshape(z_anchor.shape[0], -1),
                dim=-1,
            ).mean()

        metrics["anchor_mse"] = float(anchor_mse.item())
        metrics["anchor_cos"] = float(anchor_cos.item())
        metrics["anchor_energy"] = float(anchor_energy.mean().item())

        for key, value in metrics.items():
            totals[key] = totals.get(key, 0.0) + float(value)
        n_batches += 1

        run_diag = (
            not is_train
            and int(eval_cfg.get("decode_num_samples", 0)) > 0
            and (
                int(eval_cfg.get("decode_eval_batches", 0)) <= 0
                or step < int(eval_cfg.get("decode_eval_batches", 0))
            )
        )
        if run_diag:
            diag = _decode_diag_metrics(
                ae=ae,
                ebm=ebm,
                sampler=sampler,
                h_scenario=h_scenario,
                zone_mask=zone_mask,
                u_star=u_star,
                z_anchor=z_anchor,
                thresholds=thresholds,
                prior_mean=prior_mean,
                prior_std=prior_std,
                n_samples=int(eval_cfg.get("decode_num_samples", 4)),
            )
            for key, value in diag.items():
                diag_totals[key] = diag_totals.get(key, 0.0) + float(value)
            n_diag_batches += 1

        if is_train and log_every > 0 and step % log_every == 0:
            lr = optimizer.param_groups[0]["lr"] if optimizer is not None else float("nan")
            print(
                f"  [ep{epoch:03d} step{step:04d}] loss={metrics['loss']:.4f} "
                f"gap={metrics['E_gap']:.4f} anchor_mse={metrics['anchor_mse']:.4f} "
                f"anchor_cos={metrics['anchor_cos']:.3f} lr={lr:.2e}",
            )

    averaged = {key: value / max(1, n_batches) for key, value in totals.items()}
    if n_diag_batches > 0:
        averaged.update(
            {key: value / n_diag_batches for key, value in diag_totals.items()},
        )
    averaged["n_batches"] = float(n_batches)
    return averaged


def _save_ckpt(
    ebm: LatentEBM,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    metrics: Dict[str, Any],
    cfg: LatentEBMTrainConfig,
    path: Path,
    prior_mean: torch.Tensor,
    prior_std: torch.Tensor,
    thresholds: torch.Tensor,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    cfg_dict = asdict(cfg)
    cfg_dict.setdefault("model", {})
    cfg_dict["model"]["n_tokens"] = int(ebm.n_tokens)
    cfg_dict["model"]["latent_dim"] = int(ebm.latent_dim)
    cfg_dict["model"]["scenario_dim"] = int(ebm.scenario_dim)
    cfg_dict.setdefault("eval", {})
    cfg_dict["eval"]["feature_thresholds"] = threshold_dict(thresholds, thresholds.numel())
    torch.save(
        {
            "dr_formulation": DR_FORMULATION,
            "model_state_dict": ebm.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "epoch": epoch,
            "metrics": metrics,
            "config": cfg_dict,
            "latent_prior_mean": prior_mean.detach().cpu(),
            "latent_prior_std": prior_std.detach().cpu(),
            "thresholds": threshold_dict(thresholds, thresholds.numel()),
            "ae_checkpoint": cfg.ae["checkpoint"],
            "jepa_checkpoint": cfg.jepa["checkpoint"],
        },
        path,
    )


def train(cfg: LatentEBMTrainConfig, device: Optional[str] = None) -> Dict[str, Any]:
    seed = int(cfg.data.get("seed", 42))
    seed_all(seed)

    dev = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    print(f"Device: {dev}")
    train_loader, val_loader = _build_dataloaders(cfg.data)

    ae = load_frozen_ae(cfg.ae["checkpoint"], dev)
    predictor, jepa_cfg = load_frozen_jepa(cfg.jepa["checkpoint"], dev)
    print(
        f"Frozen AE: K={ae.n_tokens} D={ae.latent_dim} "
        f"params={sum(p.numel() for p in ae.parameters()):,}",
    )
    print(
        f"Frozen JEPA: K={predictor.n_tokens} D={predictor.latent_dim} "
        f"params={sum(p.numel() for p in predictor.parameters()):,}",
    )

    model_cfg = dict(cfg.model)
    model_cfg.setdefault("n_tokens", ae.n_tokens)
    model_cfg.setdefault("latent_dim", ae.latent_dim)
    model_cfg.setdefault("scenario_dim", int(cfg.jepa.get("scenario_dim", jepa_cfg.get("model", {}).get("scenario_dim", 128))))
    ebm = LatentEBM(
        latent_dim=int(model_cfg["latent_dim"]),
        n_tokens=int(model_cfg["n_tokens"]),
        scenario_dim=int(model_cfg.get("scenario_dim", 128)),
        hidden_dim=int(model_cfg.get("hidden_dim", 256)),
        token_hidden_dim=model_cfg.get("token_hidden_dim"),
        dropout=float(model_cfg.get("dropout", 0.1)),
        use_scenario_context=bool(model_cfg.get("use_scenario_context", True)),
        energy_bound=float(model_cfg.get("energy_bound", 50.0)),
    ).to(dev)
    print(f"LatentEBM params: {sum(p.numel() for p in ebm.parameters()):,}")

    sampler_cfg = LatentLangevinConfig(
        num_steps=int(cfg.sampler.get("num_steps", 30)),
        step_size=float(cfg.sampler.get("step_size", 0.05)),
        noise_scale=float(cfg.sampler.get("noise_scale", 0.01)),
        temperature=float(cfg.sampler.get("temperature", 1.0)),
        clip_grad=float(cfg.sampler.get("clip_grad", 0.0)),
        normalize_grad=bool(cfg.sampler.get("normalize_grad", True)),
        init_mode=str(cfg.sampler.get("init_mode", "anchor_noise")),
        init_noise_std=float(cfg.sampler.get("init_noise_std", 0.25)),
        drift_weight=float(cfg.sampler.get("drift_weight", 0.10)),
        prior_mix_prob=float(cfg.sampler.get("prior_mix_prob", 0.25)),
        anneal_schedule=str(cfg.sampler.get("anneal_schedule", "cosine")),
        latent_clip=cfg.sampler.get("latent_clip", 6.0),
    )
    sampler = LatentLangevinSampler(sampler_cfg)
    print(
        "Latent sampler: "
        f"steps={sampler_cfg.num_steps} step_size={sampler_cfg.step_size} "
        f"noise={sampler_cfg.noise_scale} drift={sampler_cfg.drift_weight} "
        f"init={sampler_cfg.init_mode}",
    )

    print("Estimating latent prior from train positives...")
    prior_mean, prior_std = estimate_latent_prior(
        ae=ae,
        loader=train_loader,
        device=dev,
        max_batches=int(cfg.eval.get("prior_estimation_batches", 0)),
    )
    prior_mean = prior_mean.to(dev)
    prior_std = prior_std.to(dev)
    print(
        f"  prior mean abs={prior_mean.abs().mean().item():.4f} "
        f"std mean={prior_std.mean().item():.4f}",
    )

    default_threshold = float(
        cfg.eval.get(
            "threshold",
            (jepa_cfg.get("eval", {}) or {}).get("threshold", 0.5),
        ),
    )
    threshold_cfg = dict((jepa_cfg.get("eval", {}) or {}).get("feature_thresholds", {}) or {})
    threshold_cfg.update(dict(cfg.eval.get("feature_thresholds", {}) or {}))
    thresholds = threshold_vector(default_threshold, threshold_cfg, ae.n_features).to(dev)
    print(
        "Decode thresholds: "
        + ", ".join(
            f"{name}={thr:.2f}"
            for name, thr in zip(
                threshold_dict(thresholds, ae.n_features).keys(),
                thresholds.detach().cpu().tolist(),
            )
        ),
    )

    lr = float(cfg.optim.get("lr", 3e-4))
    weight_decay = float(cfg.optim.get("weight_decay", 1e-4))
    epochs = int(cfg.optim.get("epochs", 20))
    patience = int(cfg.optim.get("patience", 6))
    grad_clip = float(cfg.optim.get("grad_clip", 1.0))
    warmup_epochs = int(cfg.optim.get("warmup_epochs", 2))
    best_metric_name = str(cfg.optim.get("best_metric", "E_gap"))
    higher_is_better = best_metric_name in {
        "E_gap",
        "anchor_cos",
        "anchor_macro_f1",
        "sample_macro_f1",
    }

    optimizer = torch.optim.AdamW(ebm.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lr_lambda=_make_lr_lambda(warmup_epochs, epochs),
    )

    output_dir = Path(cfg.logging.get("output_dir", "outputs/uc_jepa/latent_ebm"))
    output_dir.mkdir(parents=True, exist_ok=True)
    best_path = output_dir / "latent_ebm_best.pt"
    save_every = int(cfg.logging.get("save_every", 5))
    log_every = int(cfg.logging.get("log_every", 50))

    best_val = -float("inf") if higher_is_better else float("inf")
    bad_epochs = 0
    history: List[Dict[str, Any]] = []

    for epoch in range(epochs):
        train_metrics = _epoch_pass(
            ebm=ebm,
            ae=ae,
            predictor=predictor,
            sampler=sampler,
            loader=train_loader,
            device=dev,
            optimizer=optimizer,
            grad_clip=grad_clip,
            is_train=True,
            epoch=epoch,
            log_every=log_every,
            prior_mean=prior_mean,
            prior_std=prior_std,
            thresholds=thresholds,
            loss_cfg=cfg.loss,
            eval_cfg=cfg.eval,
        )
        val_metrics = _epoch_pass(
            ebm=ebm,
            ae=ae,
            predictor=predictor,
            sampler=sampler,
            loader=val_loader,
            device=dev,
            optimizer=None,
            grad_clip=0.0,
            is_train=False,
            epoch=epoch,
            log_every=0,
            prior_mean=prior_mean,
            prior_std=prior_std,
            thresholds=thresholds,
            loss_cfg=cfg.loss,
            eval_cfg=cfg.eval,
        )
        scheduler.step()

        history.append(
            {
                "epoch": epoch,
                "lr": optimizer.param_groups[0]["lr"],
                "train": train_metrics,
                "val": val_metrics,
            },
        )
        print(
            f"Epoch {epoch+1}/{epochs} | "
            f"train loss={train_metrics['loss']:.4f} gap={train_metrics['E_gap']:.4f} "
            f"anchor_mse={train_metrics['anchor_mse']:.4f} cos={train_metrics['anchor_cos']:.3f} | "
            f"val loss={val_metrics['loss']:.4f} gap={val_metrics['E_gap']:.4f} "
            f"anchor_mse={val_metrics['anchor_mse']:.4f} cos={val_metrics['anchor_cos']:.3f} "
            f"sampleF1={val_metrics.get('sample_macro_f1', float('nan')):.3f} "
            f"div={val_metrics.get('sample_diversity', float('nan')):.3f} | "
            f"lr={optimizer.param_groups[0]['lr']:.2e}",
        )

        current = float(val_metrics.get(best_metric_name, val_metrics["loss"]))
        improved = (current > best_val + 1e-6) if higher_is_better else (current < best_val - 1e-6)
        if improved:
            best_val = current
            bad_epochs = 0
            _save_ckpt(
                ebm=ebm,
                optimizer=optimizer,
                epoch=epoch,
                metrics=val_metrics,
                cfg=cfg,
                path=best_path,
                prior_mean=prior_mean,
                prior_std=prior_std,
                thresholds=thresholds,
            )
            print(f"  >> new best val {best_metric_name}={best_val:.4f} -> {best_path}")
        else:
            bad_epochs += 1
            if bad_epochs >= patience:
                print(f"  Early stopping at epoch {epoch+1} (patience={patience}).")
                break

        if save_every > 0 and (epoch + 1) % save_every == 0:
            _save_ckpt(
                ebm=ebm,
                optimizer=optimizer,
                epoch=epoch,
                metrics=val_metrics,
                cfg=cfg,
                path=output_dir / f"latent_ebm_epoch_{epoch+1}.pt",
                prior_mean=prior_mean,
                prior_std=prior_std,
                thresholds=thresholds,
            )

    with (output_dir / "history.json").open("w", encoding="utf-8") as handle:
        json.dump(history, handle, indent=2)
    print(f"Best val {best_metric_name}: {best_val:.4f} ({best_path})")
    return {"best_metric": best_metric_name, "best_val": best_val, "best_ckpt": str(best_path)}


def _parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the latent UC-JEPA EBM.")
    parser.add_argument("--config", required=True, help="YAML config path.")
    parser.add_argument("--device", default=None, help="cuda or cpu (auto if unset).")
    parser.add_argument("--max-samples", type=int, default=None, help="Optional override for data.max_samples.")
    parser.add_argument("--epochs", type=int, default=None, help="Optional override for optim.epochs.")
    parser.add_argument("--output-dir", default=None, help="Optional override for logging.output_dir.")
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = _parse_args(argv)
    cfg = _load_config(args.config)
    if args.max_samples is not None:
        cfg.data["max_samples"] = int(args.max_samples)
    if args.epochs is not None:
        cfg.optim["epochs"] = int(args.epochs)
    if args.output_dir is not None:
        cfg.logging["output_dir"] = str(args.output_dir)
    train(cfg, device=args.device)


if __name__ == "__main__":
    main()
