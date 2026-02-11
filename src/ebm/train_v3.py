# ==============================================================================
# EBM V3 TRAINING - Two-Step Pipeline
# ==============================================================================
# Step A: Gold pre-training with Contrastive Divergence
# Step B: Silver fine-tuning with CD + Preference loss (LP oracle)
# ==============================================================================

from __future__ import annotations

import os
import time
import json
import torch
import torch.nn as nn
from typing import Dict, Optional, Tuple, List
from pathlib import Path

from .config_v3 import EBMv3Config
from .model_v3 import TrajectoryZonalEBM
from .sampler_v3 import NormalizedTemporalLangevinSampler
from .loss_v3 import ContrastiveDivergenceLoss, CombinedLoss
from .dataset_v3 import build_dataloaders


# ==============================================================================
# STEP A: GOLD PRE-TRAINING
# ==============================================================================

def _generate_random_negatives(
    u_pos: torch.Tensor,
    zone_mask: Optional[torch.Tensor],
    sparsity: float = 0.025,
) -> torch.Tensor:
    """Generate random Bernoulli negatives matching u_pos shape, masked."""
    u_rand = torch.bernoulli(torch.full_like(u_pos, sparsity))
    if zone_mask is not None:
        mask_4d = zone_mask.unsqueeze(-1).unsqueeze(-1).float()
        u_rand = u_rand * mask_4d
    return u_rand


def _generate_corruption_negatives(
    u_pos: torch.Tensor,
    zone_mask: Optional[torch.Tensor],
    flip_rate: float = 0.05,
) -> torch.Tensor:
    """
    Generate negatives by corrupting real positives: flip ~flip_rate of bits.
    This produces structurally similar negatives that force the model to learn
    fine-grained structure rather than a density shortcut.
    """
    flip_mask = torch.bernoulli(torch.full_like(u_pos, flip_rate))
    # XOR: where flip_mask=1, toggle the bit (0->1 or 1->0)
    u_corrupt = u_pos * (1 - flip_mask) + (1 - u_pos) * flip_mask
    if zone_mask is not None:
        mask_4d = zone_mask.unsqueeze(-1).unsqueeze(-1).float()
        u_corrupt = u_corrupt * mask_4d
    return u_corrupt


def _get_langevin_train_steps(epoch: int, total_epochs: int, config: EBMv3Config) -> int:
    """Curriculum: linearly ramp Langevin steps from start to end over training."""
    s = getattr(config, "langevin_train_steps_start", None)
    e = getattr(config, "langevin_train_steps_end", None)
    if s is None or e is None:
        # Fallback: use fixed langevin_train_steps if curriculum not configured
        return getattr(config, "langevin_train_steps", config.langevin_steps)
    if total_epochs <= 1:
        return int(s)
    t = min(epoch / (total_epochs - 1), 1.0)
    return int(round(s + t * (e - s)))


def _get_langevin_ratio(epoch: int, total_epochs: int, config: EBMv3Config) -> float:
    """Compute annealed Langevin mixing ratio for current epoch."""
    if total_epochs <= 1:
        return config.langevin_ratio_start
    t = epoch / (total_epochs - 1)
    return config.langevin_ratio_start + t * (
        config.langevin_ratio_end - config.langevin_ratio_start
    )


def train_epoch_gold(
    model: TrajectoryZonalEBM,
    sampler: NormalizedTemporalLangevinSampler,
    train_loader,
    optimizer: torch.optim.Optimizer,
    loss_fn: ContrastiveDivergenceLoss,
    config: EBMv3Config,
    scaler=None,
    epoch: int = 0,
) -> Dict[str, float]:
    """Run one epoch of gold pre-training with mixed negative sampling."""
    model.train()
    sampler.set_mode("train")

    use_amp = config.use_amp and config.device == "cuda"
    total_metrics = {}
    n_batches = 0
    n_langevin = 0

    # Annealing: early epochs use mostly random negatives (strong signal),
    # later epochs shift to Langevin negatives (sharp landscape).
    langevin_ratio = _get_langevin_ratio(
        epoch, config.gold_epochs, config,
    )

    for batch_idx, batch in enumerate(train_loader):
        u_pos = batch["u_zt"].to(config.device)       # [B, Z_max, T, F]
        h_zt = batch["h_zt"].to(config.device)         # [B, Z_max, T, D]
        zone_mask = batch["zone_mask"].to(config.device)  # [B, Z_max]

        optimizer.zero_grad(set_to_none=True)

        # -- Positive energy --
        with torch.amp.autocast("cuda", enabled=use_amp):
            E_pos = model(u_pos, h_zt, zone_mask)

        # -- Mixed negatives: Langevin / corruption / random --
        roll = torch.rand(1).item()
        if roll < langevin_ratio:
            u_neg = sampler.sample(h_zt, zone_mask)
            n_langevin += 1
            neg_type = "L"
        elif roll < langevin_ratio + (1 - langevin_ratio) * 0.5:
            # Corruption: flip bits in real positives (hard negatives)
            u_neg = _generate_corruption_negatives(
                u_pos, zone_mask, config.corruption_flip_rate,
            )
            neg_type = "C"
        else:
            # Random Bernoulli (easy negatives)
            u_neg = _generate_random_negatives(
                u_pos, zone_mask, config.random_neg_sparsity,
            )
            neg_type = "R"

        # Sampler diagnostic: how different are negatives from positives?
        with torch.no_grad():
            neg_sparsity = u_neg.mean().item()
            pos_sparsity = u_pos.mean().item()
            flip_rate = (u_neg - u_pos).abs().mean().item()

        # -- Negative energy + loss --
        with torch.amp.autocast("cuda", enabled=use_amp):
            E_neg = model(u_neg.detach(), h_zt, zone_mask)
            loss, metrics = loss_fn(E_pos, E_neg)

        if scaler is not None:
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip)
            optimizer.step()

        # Accumulate metrics
        for k, v in metrics.items():
            total_metrics[k] = total_metrics.get(k, 0.0) + v
        total_metrics["flip_rate"] = total_metrics.get("flip_rate", 0.0) + flip_rate
        total_metrics["neg_sparsity"] = total_metrics.get("neg_sparsity", 0.0) + neg_sparsity
        n_batches += 1

        if (batch_idx + 1) % config.log_every == 0:
            avg_gap = total_metrics.get("E_gap", 0) / n_batches
            print(
                f"  [Epoch {epoch}] Batch {batch_idx+1}/{len(train_loader)} | "
                f"CD={metrics['cd_loss']:.4f} | Gap={avg_gap:.4f} | "
                f"E+={metrics['E_pos_mean']:.3f} E-={metrics['E_neg_mean']:.3f} "
                f"flip={flip_rate:.3f} [{neg_type}]"
            )

    pct_l = 100 * n_langevin / max(n_batches, 1)
    total_metrics["langevin_pct"] = pct_l
    avg_flip = total_metrics.get("flip_rate", 0) / max(n_batches, 1)
    avg_neg_sp = total_metrics.get("neg_sparsity", 0) / max(n_batches, 1)
    print(f"  Langevin ratio: {langevin_ratio:.2f} (actual: {pct_l:.0f}%) | "
          f"flip_rate={avg_flip:.3f} neg_sparsity={avg_neg_sp:.3f} pos_sparsity={pos_sparsity:.3f}")

    # Average metrics
    return {k: v / max(n_batches, 1) for k, v in total_metrics.items()}


def validate_gold(
    model: TrajectoryZonalEBM,
    sampler: NormalizedTemporalLangevinSampler,
    val_loader,
    loss_fn: ContrastiveDivergenceLoss,
    config: EBMv3Config,
    langevin_val_batches: int = 3,
) -> Dict[str, float]:
    """
    Validate on gold data with dual gap measurement:
      - ValGap_R: random Bernoulli negatives (all batches)
      - ValGap_L: Langevin negatives (first N batches only, for cost)
    Also logs E+/E- mean, RMS, and sampler movement stats.
    """
    model.eval()

    # Random gap accumulators
    sum_E_pos = 0.0
    sum_E_neg_rand = 0.0
    sum_E_pos_sq = 0.0
    sum_E_neg_rand_sq = 0.0
    n_rand = 0

    # Langevin gap accumulators
    sum_E_neg_lang = 0.0
    sum_movement = 0.0
    n_lang = 0

    for batch_idx, batch in enumerate(val_loader):
        u_pos = batch["u_zt"].to(config.device)
        h_zt = batch["h_zt"].to(config.device)
        zone_mask = batch["zone_mask"].to(config.device)

        with torch.no_grad():
            E_pos = model(u_pos, h_zt, zone_mask)

        # ── Random negatives (all batches) ──
        with torch.no_grad():
            u_neg_rand = torch.bernoulli(torch.full_like(u_pos, config.random_neg_sparsity))
            if zone_mask is not None:
                mask_4d = zone_mask.unsqueeze(-1).unsqueeze(-1).float()
                u_neg_rand = u_neg_rand * mask_4d
            E_neg_rand = model(u_neg_rand, h_zt, zone_mask)

        sum_E_pos += E_pos.mean().item()
        sum_E_neg_rand += E_neg_rand.mean().item()
        sum_E_pos_sq += E_pos.pow(2).mean().item()
        sum_E_neg_rand_sq += E_neg_rand.pow(2).mean().item()
        n_rand += 1

        # ── Langevin negatives (first N batches only) ──
        if batch_idx < langevin_val_batches:
            u_neg_lang = sampler.sample(h_zt, zone_mask)
            with torch.no_grad():
                E_neg_lang = model(u_neg_lang, h_zt, zone_mask)
            sum_E_neg_lang += E_neg_lang.mean().item()
            movement = (u_neg_lang - 0.5).abs().mean().item()
            sum_movement += movement
            n_lang += 1

    model.train()

    E_pos_mean = sum_E_pos / max(n_rand, 1)
    E_neg_rand_mean = sum_E_neg_rand / max(n_rand, 1)
    gap_rand = E_neg_rand_mean - E_pos_mean

    results = {
        "E_gap": gap_rand,
        "E_gap_rand": gap_rand,
        "E_pos_mean": E_pos_mean,
        "E_neg_rand_mean": E_neg_rand_mean,
        "E_pos_rms": (sum_E_pos_sq / max(n_rand, 1)) ** 0.5,
        "E_neg_rand_rms": (sum_E_neg_rand_sq / max(n_rand, 1)) ** 0.5,
    }

    if n_lang > 0:
        E_neg_lang_mean = sum_E_neg_lang / n_lang
        gap_lang = E_neg_lang_mean - E_pos_mean
        results["E_gap_lang"] = gap_lang
        results["E_neg_lang_mean"] = E_neg_lang_mean
        results["sampler_movement"] = sum_movement / n_lang

    return results


def run_gold_pretraining(config: EBMv3Config) -> Tuple[TrajectoryZonalEBM, Dict]:
    """
    Step A: Full gold pre-training loop.

    Returns:
        model: trained TrajectoryZonalEBM
        history: training history dict
    """
    print("=" * 80)
    print("STEP A: GOLD PRE-TRAINING")
    print("=" * 80)

    # Build data loaders
    train_loader, val_loader, dataset = build_dataloaders(
        reports_dir=config.reports_dir,
        embeddings_dir=config.embeddings_dir,
        classification_index_path=config.classification_index_path,
        tier="gold",
        n_timesteps=config.n_timesteps,
        embed_dim=config.embed_dim,
        batch_size=config.batch_size,
        val_split=config.val_split,
        num_workers=config.num_workers,
        seed=config.seed,
    )

    # Initialize model
    energy_max = getattr(config, "energy_max", 10.0)
    model = TrajectoryZonalEBM(
        embed_dim=config.embed_dim,
        n_features=config.n_features,
        hidden_dim=config.hidden_dim,
        gru_layers=config.gru_layers,
        bidirectional=config.bidirectional,
        dropout=config.dropout,
        use_peak_term=config.use_peak_term,
        peak_tau=config.peak_tau,
        peak_weight=config.peak_weight,
        energy_max=energy_max,
    ).to(config.device)

    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {n_params:,}")

    # Initialize sampler — curriculum ramps steps from start to end
    steps_start = getattr(config, "langevin_train_steps_start", 10)
    steps_end = getattr(config, "langevin_train_steps_end", 50)
    sampler = NormalizedTemporalLangevinSampler(
        model=model,
        n_features=config.n_features,
        num_steps=steps_start,  # will be updated each epoch by curriculum
        step_size=config.langevin_step_size,
        noise_scale=config.langevin_noise,
        temp_max=config.langevin_temp_max,
        temp_min=config.langevin_temp_min,
        init_mode=config.langevin_init_mode,
        prior_p=config.langevin_prior_p,
        prior_strength=config.langevin_prior_strength,
        normalize_grad=config.langevin_normalize_grad,
        device=config.device,
        mode="train",
    )
    print(f"Training sampler: {steps_start}→{steps_end} Langevin steps (curriculum), "
          f"step_size={config.langevin_step_size}, noise={config.langevin_noise}")

    # Loss, optimizer, scheduler
    loss_fn = ContrastiveDivergenceLoss(alpha_reg=0.01)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=config.gold_lr, weight_decay=config.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=config.gold_epochs,
    )
    scaler = torch.amp.GradScaler("cuda") if config.use_amp and config.device == "cuda" else None

    # Training loop
    history = {"train": [], "val": []}
    best_val_gap = -float("inf")
    patience_counter = 0

    for epoch in range(config.gold_epochs):
        t0 = time.time()

        # Curriculum: ramp Langevin steps from start to end over training
        curr_steps = _get_langevin_train_steps(epoch, config.gold_epochs, config)
        sampler.set_num_steps(curr_steps)

        train_metrics = train_epoch_gold(
            model, sampler, train_loader, optimizer, loss_fn,
            config, scaler, epoch,
        )
        val_metrics = validate_gold(model, sampler, val_loader, loss_fn, config)
        scheduler.step()

        dt = time.time() - t0
        history["train"].append(train_metrics)
        history["val"].append(val_metrics)

        val_gap_rand = val_metrics.get("E_gap_rand", 0)
        val_gap_lang = val_metrics.get("E_gap_lang", float("nan"))
        print(
            f"Epoch {epoch+1}/{config.gold_epochs} ({dt:.1f}s) | "
            f"Train CD={train_metrics.get('cd_loss', 0):.4f} Gap={train_metrics.get('E_gap', 0):.4f} | "
            f"ValGap_R={val_gap_rand:.4f} ValGap_L={val_gap_lang:.4f} | "
            f"E+={val_metrics.get('E_pos_mean', 0):.2f} | "
            f"LR={scheduler.get_last_lr()[0]:.2e} | Lsteps={curr_steps}"
        )

        # Early stopping on ValGap_L (Langevin gap) — the meaningful metric.
        # ValGap_R is easy to inflate via density shortcut; ValGap_L tests
        # whether the model resists adversarial Langevin samples.
        val_gap = val_gap_lang if not (val_gap_lang != val_gap_lang) else val_gap_rand  # fallback if nan
        if val_gap > best_val_gap:
            best_val_gap = val_gap
            patience_counter = 0
            _save_checkpoint(
                model, optimizer, epoch, val_metrics,
                os.path.join(config.output_dir, "gold_best.pt"),
            )
            print(f"  >> New best val gap (L): {val_gap:.4f}")
        else:
            patience_counter += 1
            if patience_counter >= config.gold_patience:
                print(f"  Early stopping at epoch {epoch+1}")
                break

        # Periodic save
        if (epoch + 1) % config.save_every_epoch == 0:
            _save_checkpoint(
                model, optimizer, epoch, val_metrics,
                os.path.join(config.output_dir, f"gold_epoch_{epoch+1}.pt"),
            )

    # Load best model
    best_path = os.path.join(config.output_dir, "gold_best.pt")
    if os.path.exists(best_path):
        ckpt = torch.load(best_path, map_location=config.device, weights_only=False)
        model.load_state_dict(ckpt["model_state_dict"])
        print(f"Loaded best gold model (gap={ckpt.get('val_gap', 'N/A')})")

    # Save history
    _save_history(history, os.path.join(config.output_dir, "gold_history.json"))

    return model, history


# ==============================================================================
# STEP B: SILVER FINE-TUNING
# ==============================================================================

def _get_silver_langevin_steps(epoch: int, total_epochs: int, config: EBMv3Config) -> int:
    """Silver curriculum: linearly ramp Langevin steps from silver_langevin_start to silver_langevin_end."""
    s = getattr(config, "silver_langevin_start", 20)
    e = getattr(config, "silver_langevin_end", 35)
    if total_epochs <= 1:
        return int(s)
    t = min(epoch / (total_epochs - 1), 1.0)
    return int(round(s + t * (e - s)))


def _lp_evaluate_candidates(
    model: TrajectoryZonalEBM,
    sampler: NormalizedTemporalLangevinSampler,
    h_zt_batch: torch.Tensor,
    zone_mask_batch: torch.Tensor,
    scenario_ids: List[str],
    config: EBMv3Config,
    n_candidates: int = 2,
    lp_worker=None,
) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
    """
    Generate candidates, evaluate with LP worker directly (no feasibility decoder).

    By bypassing the HierarchicalFeasibilityDecoder, raw binary differences are
    preserved — the LP multi-stage repair produces different costs for different
    binary inputs, enabling real preference pairs.

    Returns:
        E_better: [N] energies of candidates with lower LP cost
        E_worse:  [N] energies of candidates with higher LP cost
        (or None, None if LP evaluation fails)
    """
    if lp_worker is None:
        try:
            from src.milp.lp_worker_two_stage import LPWorkerTwoStage
            lp_worker = LPWorkerTwoStage(
                scenarios_dir=config.scenarios_dir,
                solver_name=config.lp_solver,
                verbose=False,
            )
        except ImportError as e:
            print(f"  LP evaluation unavailable: {e}")
            return None, None

    B = min(config.silver_lp_scenarios_per_batch, h_zt_batch.shape[0])
    E_better_list = []
    E_worse_list = []

    model.eval()
    sampler.set_mode("infer")

    for i in range(B):
        sc_id = scenario_ids[i]
        h_i = h_zt_batch[i:i+1]
        mask_i = zone_mask_batch[i:i+1]
        n_zones_i = int(zone_mask_batch[i].sum().item())

        # Generate candidates with diversity
        # First candidate: direct Langevin sample
        # Additional candidates: perturb by random bit-flips (~20%)
        candidates = []
        energies = []
        lp_costs = []
        base_bin = sampler.sample_binary(h_i, mask_i)  # [1, Z_max, T, F]

        for c_idx in range(n_candidates):
            if c_idx == 0:
                u_bin = base_bin
            else:
                # Flip ~20% of bits for diversity
                flip_mask = (torch.rand_like(base_bin) < 0.20).float()
                u_bin = (base_bin + flip_mask) % 2  # XOR via modular arithmetic
                u_bin = sampler._apply_mask(u_bin, mask_i)
            u_bin_trimmed = u_bin[0, :n_zones_i]         # [Z_actual, T, F]

            with torch.no_grad():
                E = model(u_bin, h_i, mask_i).item()
            candidates.append(u_bin_trimmed)
            energies.append(E)

            # LP evaluation — pass raw binary directly (no feasibility decoder)
            try:
                result = lp_worker.solve(
                    scenario_id=sc_id,
                    decoder_output=u_bin_trimmed,
                )
                lp_cost = getattr(result, "objective_value", float("inf"))
            except Exception as e:
                print(f"    LP solve failed for {sc_id}: {e}")
                lp_cost = float("inf")

            lp_costs.append(lp_cost)

        # Create preference pairs
        if len(lp_costs) >= 2:
            # Sort by LP cost (lower is better)
            sorted_idx = sorted(range(len(lp_costs)), key=lambda k: lp_costs[k])
            best_idx = sorted_idx[0]
            worst_idx = sorted_idx[-1]

            if lp_costs[best_idx] < lp_costs[worst_idx] and \
               lp_costs[best_idx] < float("inf"):
                E_better_list.append(energies[best_idx])
                E_worse_list.append(energies[worst_idx])
                print(f"    LP pair {sc_id}: cost_best={lp_costs[best_idx]:.0f} "
                      f"cost_worst={lp_costs[worst_idx]:.0f} "
                      f"E_better={energies[best_idx]:.4f} E_worse={energies[worst_idx]:.4f}")
            else:
                inf_count = sum(1 for c in lp_costs if c == float("inf"))
                if inf_count == 0:
                    print(f"    LP no pair {sc_id}: costs equal "
                          f"({lp_costs[best_idx]:.0f} == {lp_costs[worst_idx]:.0f})")
                else:
                    print(f"    LP no pair {sc_id}: {inf_count}/{len(lp_costs)} inf costs")

    sampler.set_mode("train")
    model.train()

    if not E_better_list:
        return None, None

    device = config.device
    return (
        torch.tensor(E_better_list, device=device),
        torch.tensor(E_worse_list, device=device),
    )


def train_epoch_silver(
    model: TrajectoryZonalEBM,
    sampler: NormalizedTemporalLangevinSampler,
    train_loader,
    optimizer: torch.optim.Optimizer,
    loss_fn: CombinedLoss,
    config: EBMv3Config,
    scaler=None,
    epoch: int = 0,
    lp_worker=None,
) -> Dict[str, float]:
    """Run one epoch of silver fine-tuning with CD + preference loss."""
    model.train()
    sampler.set_mode("train")

    use_amp = config.use_amp and config.device == "cuda"
    total_metrics = {}
    n_batches = 0

    for batch_idx, batch in enumerate(train_loader):
        u_pos = batch["u_zt"].to(config.device)
        h_zt = batch["h_zt"].to(config.device)
        zone_mask = batch["zone_mask"].to(config.device)
        scenario_ids = batch["scenario_ids"]

        optimizer.zero_grad(set_to_none=True)

        # CD component
        with torch.amp.autocast("cuda", enabled=use_amp):
            E_pos = model(u_pos, h_zt, zone_mask)

        # 100% Langevin negatives (model is pre-trained, can handle full adversary)
        u_neg = sampler.sample(h_zt, zone_mask)

        with torch.amp.autocast("cuda", enabled=use_amp):
            E_neg = model(u_neg.detach(), h_zt, zone_mask)

        # Preference component (periodic LP evaluation)
        E_better, E_worse = None, None
        if (batch_idx + 1) % config.silver_lp_eval_every == 0:
            E_better, E_worse = _lp_evaluate_candidates(
                model, sampler, h_zt, zone_mask, scenario_ids, config,
                lp_worker=lp_worker,
            )

        with torch.amp.autocast("cuda", enabled=use_amp):
            loss, metrics = loss_fn(E_pos, E_neg, E_better, E_worse)

        if scaler is not None:
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip)
            optimizer.step()

        for k, v in metrics.items():
            total_metrics[k] = total_metrics.get(k, 0.0) + v
        n_batches += 1

        if (batch_idx + 1) % config.log_every == 0:
            print(
                f"  [Epoch {epoch}] Batch {batch_idx+1}/{len(train_loader)} | "
                f"Total={metrics.get('loss_total', 0):.4f} | "
                f"CD={metrics.get('cd/cd_loss', 0):.4f} | "
                f"Gap={metrics.get('cd/E_gap', 0):.4f} [L]"
            )

    return {k: v / max(n_batches, 1) for k, v in total_metrics.items()}


def run_silver_finetuning(
    model: TrajectoryZonalEBM,
    config: EBMv3Config,
) -> Tuple[TrajectoryZonalEBM, Dict]:
    """
    Step B: Silver fine-tuning with CD + preference learning.

    Args:
        model: pre-trained model from Step A
        config: EBMv3Config

    Returns:
        model: fine-tuned model
        history: training history
    """
    print("=" * 80)
    print("STEP B: SILVER FINE-TUNING")
    print("=" * 80)

    # Build silver data loaders
    train_loader, val_loader, dataset = build_dataloaders(
        reports_dir=config.reports_dir,
        embeddings_dir=config.embeddings_dir,
        classification_index_path=config.classification_index_path,
        tier="silver",
        n_timesteps=config.n_timesteps,
        embed_dim=config.embed_dim,
        batch_size=config.batch_size,
        val_split=config.val_split,
        num_workers=config.num_workers,
        seed=config.seed,
    )

    # Sampler — use curriculum: start gentle, ramp to gold's peak strength
    silver_start = getattr(config, "silver_langevin_start", 20)
    silver_end = getattr(config, "silver_langevin_end", 35)
    init_steps = _get_silver_langevin_steps(0, config.silver_epochs, config)
    sampler = NormalizedTemporalLangevinSampler(
        model=model,
        n_features=config.n_features,
        num_steps=init_steps,
        step_size=config.langevin_step_size,
        noise_scale=config.langevin_noise,
        temp_max=config.langevin_temp_max,
        temp_min=config.langevin_temp_min,
        init_mode=config.langevin_init_mode,
        prior_p=config.langevin_prior_p,
        prior_strength=config.langevin_prior_strength,
        normalize_grad=config.langevin_normalize_grad,
        device=config.device,
        mode="train",
    )
    print(f"Silver sampler: {silver_start}→{silver_end} Langevin steps (curriculum), "
          f"step_size={config.langevin_step_size}, noise={config.langevin_noise}")

    # Combined loss
    loss_fn = CombinedLoss(
        lambda_cd=config.silver_lambda_cd,
        lambda_pref=config.silver_lambda_pref,
        margin=config.silver_preference_margin,
        alpha_reg=0.01,
    )

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=config.silver_lr, weight_decay=config.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=config.silver_epochs,
    )
    scaler = torch.amp.GradScaler("cuda") if config.use_amp and config.device == "cuda" else None

    history = {"train": [], "val": []}
    best_val_gap = -float("inf")
    patience_counter = 0

    cd_loss_fn = ContrastiveDivergenceLoss(alpha_reg=0.01)

    # Cache LP worker (avoid re-init every batch)
    lp_worker = None
    try:
        from src.milp.lp_worker_two_stage import LPWorkerTwoStage
        lp_worker = LPWorkerTwoStage(
            scenarios_dir=config.scenarios_dir,
            solver_name=config.lp_solver,
            verbose=False,
        )
        print("LP worker cached for silver fine-tuning")
    except ImportError:
        print("LP worker unavailable — silver will use CD loss only")

    min_delta = getattr(config, "silver_min_delta", 0.01)

    for epoch in range(config.silver_epochs):
        t0 = time.time()

        # Silver curriculum: ramp Langevin steps
        curr_steps = _get_silver_langevin_steps(epoch, config.silver_epochs, config)
        sampler.set_num_steps(curr_steps)

        train_metrics = train_epoch_silver(
            model, sampler, train_loader, optimizer, loss_fn,
            config, scaler, epoch, lp_worker=lp_worker,
        )
        val_metrics = validate_gold(model, sampler, val_loader, cd_loss_fn, config)
        scheduler.step()

        dt = time.time() - t0
        history["train"].append(train_metrics)
        history["val"].append(val_metrics)

        val_gap_rand = val_metrics.get("E_gap_rand", 0)
        val_gap_lang = val_metrics.get("E_gap_lang", float("nan"))
        print(
            f"Epoch {epoch+1}/{config.silver_epochs} ({dt:.1f}s) | "
            f"Train Total={train_metrics.get('loss_total', 0):.4f} "
            f"CD={train_metrics.get('cd/cd_loss', 0):.4f} | "
            f"ValGap_R={val_gap_rand:.4f} ValGap_L={val_gap_lang:.4f} | "
            f"E+={val_metrics.get('E_pos_mean', 0):.2f} | "
            f"LR={scheduler.get_last_lr()[0]:.2e} | Lsteps={curr_steps}"
        )

        # Early stopping on ValGap_L with min_delta
        val_gap = val_gap_lang if not (val_gap_lang != val_gap_lang) else val_gap_rand
        if val_gap > best_val_gap + min_delta:
            best_val_gap = val_gap
            patience_counter = 0
            _save_checkpoint(
                model, optimizer, epoch, val_metrics,
                os.path.join(config.output_dir, "silver_best.pt"),
            )
            print(f"  >> New best val gap (L): {val_gap:.4f}")
        else:
            patience_counter += 1
            if patience_counter >= config.silver_patience:
                print(f"  Early stopping at epoch {epoch+1} (best ValGap_L={best_val_gap:.4f})")
                break

        if (epoch + 1) % config.save_every_epoch == 0:
            _save_checkpoint(
                model, optimizer, epoch, val_metrics,
                os.path.join(config.output_dir, f"silver_epoch_{epoch+1}.pt"),
            )

    best_path = os.path.join(config.output_dir, "silver_best.pt")
    if os.path.exists(best_path):
        ckpt = torch.load(best_path, map_location=config.device, weights_only=False)
        model.load_state_dict(ckpt["model_state_dict"])
        print(f"Loaded best silver model")

    _save_history(history, os.path.join(config.output_dir, "silver_history.json"))

    return model, history


# ==============================================================================
# FULL PIPELINE
# ==============================================================================

def run_full_pipeline(config: EBMv3Config) -> TrajectoryZonalEBM:
    """
    Run complete two-step training pipeline:
      Step A: Gold pre-training
      Step B: Silver fine-tuning

    Returns:
        Final trained model
    """
    print("=" * 80)
    print("EBM V3 FULL TRAINING PIPELINE")
    print(f"Output dir: {config.output_dir}")
    print(f"Device: {config.device}")
    print("=" * 80)

    # Step A
    model, gold_history = run_gold_pretraining(config)

    # Step B
    model, silver_history = run_silver_finetuning(model, config)

    # Save final model
    final_path = os.path.join(config.output_dir, "ebm_v3_final.pt")
    torch.save({
        "model_state_dict": model.state_dict(),
        "config": {
            "embed_dim": config.embed_dim,
            "n_features": config.n_features,
            "hidden_dim": config.hidden_dim,
            "gru_layers": config.gru_layers,
            "bidirectional": config.bidirectional,
            "dropout": config.dropout,
            "use_peak_term": config.use_peak_term,
            "peak_tau": config.peak_tau,
            "peak_weight": config.peak_weight,
        },
    }, final_path)
    print(f"\nFinal model saved to: {final_path}")

    return model


# ==============================================================================
# UTILITIES
# ==============================================================================

def _save_checkpoint(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    metrics: Dict,
    path: str,
):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save({
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "epoch": epoch,
        "val_gap": metrics.get("E_gap", 0),
        "metrics": metrics,
    }, path)


def _save_history(history: Dict, path: str):
    """Save training history as JSON (convert tensors to floats)."""
    def _convert(obj):
        if isinstance(obj, torch.Tensor):
            return obj.item()
        if isinstance(obj, dict):
            return {k: _convert(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [_convert(v) for v in obj]
        return obj

    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(_convert(history), f, indent=2)
