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
import math
import torch
import torch.nn as nn
import numpy as np
from typing import Any, Dict, Optional, Tuple, List
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


def _sample_binary_with_init_mode(
    sampler: NormalizedTemporalLangevinSampler,
    h_zt: torch.Tensor,
    zone_mask: Optional[torch.Tensor],
    init_mode: str,
    u_oracle: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Temporarily override sampler init mode for one binary sample."""
    prev_init_mode = sampler.init_mode
    try:
        sampler.init_mode = init_mode
        return sampler.sample_binary(h_zt, zone_mask, u_oracle=u_oracle)
    finally:
        sampler.init_mode = prev_init_mode


def _resolve_lp_candidate_sources(
    config: EBMv3Config,
    n_candidates: int,
    has_incumbent: bool,
) -> List[str]:
    """Build a prioritized source list for LP candidate generation."""
    desired_sources: List[Tuple[str, int]] = [
        ("incumbent", int(getattr(config, "silver_lp_incumbent_candidates", 0))),
        ("oracle_langevin", int(getattr(config, "silver_lp_oracle_langevin_candidates", 0))),
        ("corrupt", int(getattr(config, "silver_lp_corrupt_candidates", 0))),
        ("langevin", int(getattr(config, "silver_lp_langevin_candidates", 0))),
    ]

    plan: List[str] = []
    for source, count in desired_sources:
        if count <= 0:
            continue
        if source != "langevin" and not has_incumbent:
            continue
        plan.extend([source] * count)

    fallback_source = "oracle_langevin" if has_incumbent else "langevin"
    while len(plan) < n_candidates:
        plan.append(fallback_source)

    return plan[:n_candidates]


def _resolve_lp_scenario_path(scenarios_dir: str, scenario_id: str) -> Optional[Path]:
    """Resolve the raw scenario JSON path expected by the LP worker."""
    root = Path(scenarios_dir)
    for path in (
        root / f"{scenario_id}.json",
        root / "dispatch_batch" / f"{scenario_id}.json",
    ):
        if path.exists():
            return path
    return None


def _validate_lp_scenario_inputs(
    config: EBMv3Config,
    dataset,
    sample_size: int = 8,
) -> None:
    """Fail fast when silver LP ranking cannot access raw scenario JSONs."""
    needs_lp_inputs = (
        float(getattr(config, "silver_lambda_pref", 0.0)) > 0.0
        or int(getattr(config, "silver_val_ranking_scenarios", 0)) > 0
    )
    if not needs_lp_inputs:
        return

    valid_scenarios = getattr(dataset, "valid_scenarios", [])
    sample_ids = [
        str(item["scenario_id"])
        for item in valid_scenarios[:sample_size]
        if isinstance(item, dict) and "scenario_id" in item
    ]
    if not sample_ids:
        return

    found_paths = [
        _resolve_lp_scenario_path(config.scenarios_dir, scenario_id)
        for scenario_id in sample_ids
    ]
    if any(path is not None for path in found_paths):
        return

    scenarios_root = Path(config.scenarios_dir)
    top_level_entries: List[str] = []
    if scenarios_root.exists():
        try:
            top_level_entries = sorted(child.name for child in scenarios_root.iterdir())[:10]
        except OSError:
            top_level_entries = []

    expected_example = scenarios_root / f"{sample_ids[0]}.json"
    sample_preview = ", ".join(sample_ids[:3])
    entries_preview = ", ".join(top_level_entries) if top_level_entries else "<empty>"
    raise ValueError(
        "Silver LP ranking requires raw scenario JSON inputs, but none of the sampled "
        f"scenario files were found under {scenarios_root}. Expected files like "
        f"{expected_example}. Sample ids checked: {sample_preview}. Top-level entries: "
        f"{entries_preview}. This usually means only reports were synced, so the LP "
        "worker returns 'File not found' and every candidate is scored as non-finite."
    )


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

_LP_STAGE_NAMES: Tuple[str, ...] = (
    "hard_fix",
    "repair_20",
    "repair_100",
    "full_soft",
    "round_refix",
    "failed",
)


def _get_silver_langevin_steps(epoch: int, total_epochs: int, config: EBMv3Config) -> int:
    """Silver curriculum: linearly ramp Langevin steps from silver_langevin_start to silver_langevin_end."""
    s = getattr(config, "silver_langevin_start", 20)
    e = getattr(config, "silver_langevin_end", 35)
    if total_epochs <= 1:
        return int(s)
    t = min(epoch / (total_epochs - 1), 1.0)
    return int(round(s + t * (e - s)))


def _safe_float(value: Any, default: float = float("inf")) -> float:
    """Convert values to float while treating NaN/None as default."""
    try:
        out = float(value)
    except (TypeError, ValueError):
        return default
    return out if math.isfinite(out) else default


def _invalid_preference_score(config: EBMv3Config) -> float:
    """Finite fallback score used to rank invalid LP outcomes as very bad."""
    penalty = _safe_float(getattr(config, "silver_pref_invalid_score", 1e12), default=1e12)
    return penalty if penalty > 0.0 else 1e12


def _normalize_stage_name(stage_used: Any) -> str:
    """Normalize LP stage enum/string values into stable metric keys."""
    if stage_used is None:
        return "failed"
    value = getattr(stage_used, "value", stage_used)
    stage_name = str(value).strip().lower()
    return stage_name if stage_name in _LP_STAGE_NAMES else "failed"


def _resolve_repair_distance(
    decoder_deviation: float,
    rounded_flips: float,
    config: EBMv3Config,
) -> float:
    """Resolve the repair-distance scalar used in the ranking score."""
    metric = getattr(config, "silver_pref_repair_metric", "decoder_deviation")
    if metric == "rounded_flips":
        return rounded_flips
    return decoder_deviation


def _compute_preference_score_details(
    cost: float,
    slack: float,
    repair_distance: float,
    config: EBMv3Config,
) -> Tuple[float, bool]:
    """Return the preference score and whether LP outputs were non-finite."""
    penalty = _invalid_preference_score(config)
    slack_weight = float(getattr(config, "silver_pref_slack_weight", 0.0) or 0.0)
    repair_weight = float(getattr(config, "silver_pref_repair_weight", 0.0) or 0.0)

    if not math.isfinite(cost):
        return penalty, True

    if slack_weight > 0.0 and not math.isfinite(slack):
        return penalty, True

    if repair_weight > 0.0 and not math.isfinite(repair_distance):
        return penalty, True

    score = cost
    if slack_weight > 0.0:
        score += slack_weight * max(slack, 0.0)
    if repair_weight > 0.0:
        score += repair_weight * max(repair_distance, 0.0)
    if not math.isfinite(score):
        return penalty, True
    return score, False


def _compute_preference_score(
    cost: float,
    slack: float,
    repair_distance: float,
    config: EBMv3Config,
) -> float:
    """Preference score J(u) = cost + lambda_slack * slack + lambda_repair * repair."""
    score, _is_non_finite = _compute_preference_score_details(cost, slack, repair_distance, config)
    return score


def _relative_score_gap(score_better: float, score_worse: float) -> float:
    """Relative ranking gap used for informative-pair filtering and margins."""
    if not (math.isfinite(score_better) and math.isfinite(score_worse)):
        return 0.0
    denom = max(abs(score_better), 1.0)
    return max(0.0, (score_worse - score_better) / denom)


def _resolve_preference_margin(rel_gap: float, config: EBMv3Config) -> float:
    """Resolve per-pair ranking margin from config."""
    base_margin = float(getattr(config, "silver_preference_margin", 0.1))
    mode = getattr(config, "silver_pref_margin_mode", "constant")
    if mode == "scaled_gap":
        cap = max(float(getattr(config, "silver_pref_margin_rel_gap_cap", 1.0)), 0.0)
        return base_margin * min(max(rel_gap, 0.0), cap)
    return base_margin


def _safe_ratio(numerator: float, denominator: float, default: float = float("nan")) -> float:
    """Safely divide while returning a fallback when the denominator is empty."""
    if denominator <= 0:
        return default
    return numerator / denominator


def _init_ranking_aggregate() -> Dict[str, float]:
    """Initialize per-epoch ranking diagnostics."""
    aggregate = {
        "n_lp_scenarios_attempted": 0.0,
        "n_lp_scenarios_with_pairs": 0.0,
        "n_pairs_total": 0.0,
        "n_lp_candidates_total": 0.0,
        "n_non_finite_candidates_total": 0.0,
        "lp_slack_used_sum": 0.0,
        "lp_slack_used_count": 0.0,
        "lp_decoder_deviation_sum": 0.0,
        "lp_decoder_deviation_count": 0.0,
        "lp_rounded_flips_sum": 0.0,
        "lp_rounded_flips_count": 0.0,
        "lp_repair_distance_sum": 0.0,
        "lp_repair_distance_count": 0.0,
    }
    for stage_name in _LP_STAGE_NAMES:
        aggregate[f"lp_stage_{stage_name}_count"] = 0.0
    return aggregate


def _update_ranking_aggregate(
    aggregate: Dict[str, float],
    candidates: List[Dict[str, Any]],
    n_pairs: int,
) -> None:
    """Accumulate ranking diagnostics from one LP-ranked scenario."""
    aggregate["n_lp_scenarios_attempted"] += 1.0
    aggregate["n_lp_scenarios_with_pairs"] += 1.0 if n_pairs > 0 else 0.0
    aggregate["n_pairs_total"] += float(n_pairs)
    aggregate["n_lp_candidates_total"] += float(len(candidates))
    aggregate["n_non_finite_candidates_total"] += float(
        sum(1 for cand in candidates if cand.get("is_non_finite", False))
    )
    for cand in candidates:
        stage_name = _normalize_stage_name(cand.get("stage_used"))
        aggregate[f"lp_stage_{stage_name}_count"] += 1.0

        slack = _safe_float(cand.get("slack"), default=float("inf"))
        if math.isfinite(slack):
            aggregate["lp_slack_used_sum"] += slack
            aggregate["lp_slack_used_count"] += 1.0

        decoder_deviation = _safe_float(cand.get("decoder_deviation"), default=float("inf"))
        if math.isfinite(decoder_deviation):
            aggregate["lp_decoder_deviation_sum"] += decoder_deviation
            aggregate["lp_decoder_deviation_count"] += 1.0

        rounded_flips = _safe_float(cand.get("rounded_flips"), default=float("inf"))
        if math.isfinite(rounded_flips):
            aggregate["lp_rounded_flips_sum"] += rounded_flips
            aggregate["lp_rounded_flips_count"] += 1.0

        repair_distance = _safe_float(cand.get("repair_distance"), default=float("inf"))
        if math.isfinite(repair_distance):
            aggregate["lp_repair_distance_sum"] += repair_distance
            aggregate["lp_repair_distance_count"] += 1.0


def _finalize_ranking_aggregate(
    aggregate: Dict[str, float],
    prefix: str = "",
) -> Dict[str, float]:
    """Finalize ranking diagnostics with derived coverage metrics."""
    attempted = aggregate["n_lp_scenarios_attempted"]
    with_pairs = aggregate["n_lp_scenarios_with_pairs"]
    total_candidates = aggregate["n_lp_candidates_total"]
    non_finite = aggregate["n_non_finite_candidates_total"]

    metrics = {
        f"{prefix}n_lp_scenarios_attempted": attempted,
        f"{prefix}n_lp_scenarios_with_pairs": with_pairs,
        f"{prefix}n_pairs_total": aggregate["n_pairs_total"],
        f"{prefix}n_lp_candidates_total": total_candidates,
        f"{prefix}n_non_finite_candidates_total": non_finite,
        f"{prefix}pair_coverage": _safe_ratio(with_pairs, attempted),
        f"{prefix}pct_non_finite_candidates": _safe_ratio(non_finite, total_candidates),
        f"{prefix}lp_slack_used_mean": _safe_ratio(
            aggregate["lp_slack_used_sum"], aggregate["lp_slack_used_count"],
        ),
        f"{prefix}lp_decoder_deviation_mean": _safe_ratio(
            aggregate["lp_decoder_deviation_sum"], aggregate["lp_decoder_deviation_count"],
        ),
        f"{prefix}lp_rounded_flips_mean": _safe_ratio(
            aggregate["lp_rounded_flips_sum"], aggregate["lp_rounded_flips_count"],
        ),
        f"{prefix}lp_repair_distance_mean": _safe_ratio(
            aggregate["lp_repair_distance_sum"], aggregate["lp_repair_distance_count"],
        ),
    }
    for stage_name in _LP_STAGE_NAMES:
        count = aggregate[f"lp_stage_{stage_name}_count"]
        metrics[f"{prefix}lp_stage_{stage_name}_count"] = count
        metrics[f"{prefix}lp_stage_{stage_name}_share"] = _safe_ratio(count, total_candidates)
    return metrics


def _fmt_float(value: float, decimals: int = 4) -> str:
    """Format floats while keeping missing values readable."""
    return f"{value:.{decimals}f}" if math.isfinite(value) else "n/a"


def _fmt_pct(value: float, decimals: int = 0) -> str:
    """Format ratios as percentages while handling NaN cleanly."""
    return f"{100.0 * value:.{decimals}f}%" if math.isfinite(value) else "n/a"


def _rankdata(values: List[float]) -> np.ndarray:
    """Average-rank implementation for Spearman correlation without SciPy."""
    arr = np.asarray(values, dtype=float)
    sorter = np.argsort(arr, kind="mergesort")
    ranks = np.empty(len(arr), dtype=float)
    start = 0
    while start < len(arr):
        end = start
        while end + 1 < len(arr) and arr[sorter[end + 1]] == arr[sorter[start]]:
            end += 1
        rank = 0.5 * (start + end) + 1.0
        ranks[sorter[start:end + 1]] = rank
        start = end + 1
    return ranks


def _spearman_correlation(x_values: List[float], y_values: List[float]) -> float:
    """Compute Spearman rho for one scenario's energy/score ordering."""
    if len(x_values) < 2 or len(y_values) < 2:
        return float("nan")

    x_arr = np.asarray(x_values, dtype=float)
    y_arr = np.asarray(y_values, dtype=float)
    mask = np.isfinite(x_arr) & np.isfinite(y_arr)
    if mask.sum() < 2:
        return float("nan")

    x_rank = _rankdata(x_arr[mask].tolist())
    y_rank = _rankdata(y_arr[mask].tolist())
    if np.allclose(x_rank, x_rank[0]) or np.allclose(y_rank, y_rank[0]):
        return float("nan")
    return float(np.corrcoef(x_rank, y_rank)[0, 1])


def _lp_evaluate_candidates(
    model: TrajectoryZonalEBM,
    sampler: NormalizedTemporalLangevinSampler,
    h_zt_batch: torch.Tensor,
    zone_mask_batch: torch.Tensor,
    scenario_ids: List[str],
    config: EBMv3Config,
    n_candidates: int = 2,
    lp_worker=None,
) -> Optional[List[Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]]]:
    """
    Generate candidates, evaluate with LP worker directly (no feasibility decoder).

    By bypassing the HierarchicalFeasibilityDecoder, raw binary differences are
    preserved — the LP multi-stage repair produces different costs for different
    binary inputs, enabling real preference pairs.

    Returns:
        List of (u_better, u_worse, h, mask) tuples for gradient-connected
        energy re-computation in the training loop, or None if no valid pairs.
        Each u_better/u_worse has shape [1, Z_max, T, F].
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
            return None

    B = min(config.silver_lp_scenarios_per_batch, h_zt_batch.shape[0])
    pair_data = []  # list of (u_better, u_worse, h, mask) tuples

    model.eval()
    sampler.set_mode("infer")

    for i in range(B):
        sc_id = scenario_ids[i]
        h_i = h_zt_batch[i:i+1]
        mask_i = zone_mask_batch[i:i+1]
        n_zones_i = int(zone_mask_batch[i].sum().item())

        # Generate candidates: independent Langevin samples
        # Both candidates are in the low-energy region, so LP preference
        # signal doesn't conflict with CD loss (which pushes off-manifold UP)
        candidates_padded = []  # keep padded [1, Z_max, T, F] for model forward
        energies = []
        lp_costs = []

        for c_idx in range(n_candidates):
            u_bin = sampler.sample_binary(h_i, mask_i)  # independent sample
            u_bin_trimmed = u_bin[0, :n_zones_i]         # [Z_actual, T, F]

            with torch.no_grad():
                E = model(u_bin, h_i, mask_i).item()
            candidates_padded.append(u_bin.detach())
            energies.append(E)

            # LP evaluation — pass raw binary directly (no feasibility decoder)
            # max_stages=2: skip expensive stages 3-5 for fast preference pairs
            try:
                result = lp_worker.solve(
                    scenario_id=sc_id,
                    decoder_output=u_bin_trimmed,
                    max_stages=2,
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

            cost_gap = lp_costs[worst_idx] - lp_costs[best_idx]
            cost_ref = max(abs(lp_costs[best_idx]), 1.0)
            rel_gap = cost_gap / cost_ref

            if lp_costs[best_idx] < lp_costs[worst_idx] and \
               lp_costs[best_idx] < float("inf") and rel_gap >= 0.02:
                pair_data.append((
                    candidates_padded[best_idx],   # [1, Z_max, T, F]
                    candidates_padded[worst_idx],
                    h_i.detach(),                  # [1, Z_max, T, D]
                    mask_i.detach(),               # [1, Z_max]
                ))
                print(f"    LP pair {sc_id}: cost_best={lp_costs[best_idx]:.0f} "
                      f"cost_worst={lp_costs[worst_idx]:.0f} "
                      f"Δ={rel_gap:.1%} "
                      f"E_better={energies[best_idx]:.4f} E_worse={energies[worst_idx]:.4f}")
            else:
                inf_count = sum(1 for c in lp_costs if c == float("inf"))
                if inf_count == 0:
                    if rel_gap < 0.02:
                        print(f"    LP skip {sc_id}: gap too small "
                              f"({lp_costs[best_idx]:.0f} vs {lp_costs[worst_idx]:.0f}, "
                              f"Δ={rel_gap:.1%})")
                    else:
                        print(f"    LP no pair {sc_id}: costs equal "
                              f"({lp_costs[best_idx]:.0f} == {lp_costs[worst_idx]:.0f})")
                else:
                    print(f"    LP no pair {sc_id}: {inf_count}/{len(lp_costs)} inf costs")

    sampler.set_mode("train")
    model.train()

    return pair_data if pair_data else None


def _sample_lp_candidates_v2(
    model: TrajectoryZonalEBM,
    sampler: NormalizedTemporalLangevinSampler,
    h_i: torch.Tensor,
    mask_i: torch.Tensor,
    u_pos_i: Optional[torch.Tensor],
    scenario_id: str,
    config: EBMv3Config,
    lp_worker,
    n_candidates: int,
    max_stages: int,
) -> List[Dict[str, Any]]:
    """Sample binary candidates and score them with LP cost + slack."""
    n_zones_i = int(mask_i[0].sum().item())
    candidates: List[Dict[str, Any]] = []
    has_incumbent = u_pos_i is not None
    candidate_sources = _resolve_lp_candidate_sources(config, n_candidates, has_incumbent)
    corrupt_flip_rate = getattr(config, "silver_lp_corrupt_flip_rate", None)
    if corrupt_flip_rate is None:
        corrupt_flip_rate = getattr(config, "corruption_flip_rate", 0.05)
    corrupt_flip_rate = float(corrupt_flip_rate)

    for source in candidate_sources:
        if source == "incumbent" and u_pos_i is not None:
            u_bin = u_pos_i.detach().clone()
        elif source == "oracle_langevin" and u_pos_i is not None:
            u_bin = _sample_binary_with_init_mode(
                sampler, h_i, mask_i, init_mode="oracle", u_oracle=u_pos_i,
            )
        elif source == "corrupt" and u_pos_i is not None:
            u_bin = _generate_corruption_negatives(u_pos_i, mask_i, corrupt_flip_rate)
        else:
            u_bin = sampler.sample_binary(h_i, mask_i)

        u_bin_trimmed = u_bin[0, :n_zones_i]

        with torch.no_grad():
            energy = float(model(u_bin, h_i, mask_i).item())

        cost = float("inf")
        slack = float("inf")
        decoder_deviation = float("inf")
        rounded_flips = float("inf")
        repair_distance = float("inf")
        stage_used = "failed"
        try:
            result = lp_worker.solve(
                scenario_id=scenario_id,
                decoder_output=u_bin_trimmed,
                max_stages=max_stages,
            )
            cost = _safe_float(getattr(result, "objective_value", float("inf")))
            slack = _safe_float(getattr(result, "slack_used", float("inf")))
            decoder_deviation = _safe_float(getattr(result, "decoder_deviation", float("inf")))
            rounded_flips = _safe_float(getattr(result, "n_flips", float("inf")))
            repair_distance = _resolve_repair_distance(decoder_deviation, rounded_flips, config)
            stage_used = _normalize_stage_name(getattr(result, "stage_used", None))
        except Exception as exc:
            print(f"    LP solve failed for {scenario_id}: {exc}")

        score, is_non_finite = _compute_preference_score_details(
            cost, slack, repair_distance, config,
        )
        candidates.append(
            {
                "u_bin": u_bin.detach(),
                "energy": energy,
                "cost": cost,
                "slack": slack,
                "decoder_deviation": decoder_deviation,
                "rounded_flips": rounded_flips,
                "repair_distance": repair_distance,
                "score": score,
                "is_non_finite": is_non_finite,
                "source": source,
                "stage_used": stage_used,
            }
        )

    return candidates


def _select_preference_pairs_v2(
    candidates: List[Dict[str, Any]],
    h_i: torch.Tensor,
    mask_i: torch.Tensor,
    scenario_id: str,
    config: EBMv3Config,
    verbose: bool = True,
) -> List[Dict[str, Any]]:
    """Build informative ranking pairs from LP-evaluated candidates."""
    finite_candidates = [
        cand for cand in candidates
        if math.isfinite(cand["score"]) and math.isfinite(cand["energy"])
    ]
    if len(finite_candidates) < 2:
        inf_count = len(candidates) - len(finite_candidates)
        if verbose:
            print(f"    LP no pair {scenario_id}: {inf_count}/{len(candidates)} non-finite scores")
        return []

    min_rel_gap = float(getattr(config, "silver_pref_min_relative_gap", 0.02))
    max_pairs = int(getattr(config, "silver_pref_max_pairs_per_scenario", 1))
    use_all_pairs = bool(getattr(config, "silver_pref_all_informative_pairs", False))
    ordered = sorted(finite_candidates, key=lambda cand: cand["score"])

    raw_pairs: List[Dict[str, Any]] = []
    if use_all_pairs:
        for better_idx in range(len(ordered) - 1):
            for worse_idx in range(better_idx + 1, len(ordered)):
                better = ordered[better_idx]
                worse = ordered[worse_idx]
                rel_gap = _relative_score_gap(better["score"], worse["score"])
                if rel_gap < min_rel_gap:
                    continue
                raw_pairs.append(
                    {
                        "u_better": better["u_bin"],
                        "u_worse": worse["u_bin"],
                        "h": h_i.detach(),
                        "mask": mask_i.detach(),
                        "margin": _resolve_preference_margin(rel_gap, config),
                        "rel_gap": rel_gap,
                        "score_better": better["score"],
                        "score_worse": worse["score"],
                        "cost_better": better["cost"],
                        "cost_worse": worse["cost"],
                        "slack_better": better["slack"],
                        "slack_worse": worse["slack"],
                        "repair_better": better["repair_distance"],
                        "repair_worse": worse["repair_distance"],
                        "stage_better": better["stage_used"],
                        "stage_worse": worse["stage_used"],
                        "energy_better": better["energy"],
                        "energy_worse": worse["energy"],
                    }
                )
        raw_pairs.sort(key=lambda pair: pair["rel_gap"], reverse=True)
    else:
        better = ordered[0]
        worse = ordered[-1]
        rel_gap = _relative_score_gap(better["score"], worse["score"])
        if rel_gap >= min_rel_gap and better["score"] < worse["score"]:
            raw_pairs.append(
                {
                    "u_better": better["u_bin"],
                    "u_worse": worse["u_bin"],
                    "h": h_i.detach(),
                    "mask": mask_i.detach(),
                    "margin": _resolve_preference_margin(rel_gap, config),
                    "rel_gap": rel_gap,
                    "score_better": better["score"],
                    "score_worse": worse["score"],
                    "cost_better": better["cost"],
                    "cost_worse": worse["cost"],
                    "slack_better": better["slack"],
                    "slack_worse": worse["slack"],
                    "repair_better": better["repair_distance"],
                    "repair_worse": worse["repair_distance"],
                    "stage_better": better["stage_used"],
                    "stage_worse": worse["stage_used"],
                    "energy_better": better["energy"],
                    "energy_worse": worse["energy"],
                }
            )

    if max_pairs > 0:
        raw_pairs = raw_pairs[:max_pairs]

    if verbose and raw_pairs:
        for pair in raw_pairs:
            print(
                f"    LP pair {scenario_id}: "
                f"J_best={pair['score_better']:.0f} J_worst={pair['score_worse']:.0f} "
                f"(cost {pair['cost_better']:.0f}->{pair['cost_worse']:.0f}, "
                f"slack {pair['slack_better']:.1f}->{pair['slack_worse']:.1f}, "
                f"repair {pair['repair_better']:.2f}->{pair['repair_worse']:.2f}, "
                f"stage {pair['stage_better']}->{pair['stage_worse']}) "
                f"gap={pair['rel_gap']:.1%} m={pair['margin']:.3f} "
                f"E_better={pair['energy_better']:.4f} E_worse={pair['energy_worse']:.4f}"
            )
    elif verbose:
        best = ordered[0]
        worst = ordered[-1]
        rel_gap = _relative_score_gap(best["score"], worst["score"])
        print(
            f"    LP skip {scenario_id}: informative gap too small "
            f"(J_best={best['score']:.0f}, J_worst={worst['score']:.0f}, gap={rel_gap:.1%})"
        )

    return raw_pairs


def _lp_evaluate_candidates_v2(
    model: TrajectoryZonalEBM,
    sampler: NormalizedTemporalLangevinSampler,
    h_zt_batch: torch.Tensor,
    zone_mask_batch: torch.Tensor,
    u_zt_batch: Optional[torch.Tensor],
    scenario_ids: List[str],
    config: EBMv3Config,
    lp_worker=None,
) -> Tuple[List[Dict[str, Any]], Dict[str, float]]:
    """Generate LP-ranked preference pairs plus aggregate diagnostics."""
    if lp_worker is None:
        try:
            from src.milp.lp_worker_two_stage import LPWorkerTwoStage
            lp_worker = LPWorkerTwoStage(
                scenarios_dir=config.scenarios_dir,
                solver_name=config.lp_solver,
                verbose=False,
            )
        except ImportError as exc:
            print(f"  LP evaluation unavailable: {exc}")
            return [], _finalize_ranking_aggregate(_init_ranking_aggregate())

    batch_scenarios = min(config.silver_lp_scenarios_per_batch, h_zt_batch.shape[0])
    n_candidates = int(getattr(config, "silver_lp_candidates_per_scenario", 2))
    max_stages = int(getattr(config, "silver_lp_max_stages", 2))
    pair_data: List[Dict[str, Any]] = []
    ranking_agg = _init_ranking_aggregate()
    verbose_pairs = bool(getattr(config, "silver_log_individual_pairs", False))

    model.eval()
    sampler.set_mode("infer")

    for i in range(batch_scenarios):
        sc_id = scenario_ids[i]
        h_i = h_zt_batch[i:i + 1]
        mask_i = zone_mask_batch[i:i + 1]
        u_pos_i = u_zt_batch[i:i + 1] if u_zt_batch is not None else None
        candidates = _sample_lp_candidates_v2(
            model, sampler, h_i, mask_i, u_pos_i, sc_id, config,
            lp_worker=lp_worker,
            n_candidates=n_candidates,
            max_stages=max_stages,
        )
        scenario_pairs = _select_preference_pairs_v2(
            candidates, h_i, mask_i, sc_id, config, verbose=verbose_pairs,
        )
        _update_ranking_aggregate(ranking_agg, candidates, len(scenario_pairs))
        pair_data.extend(scenario_pairs)

    sampler.set_mode("train")
    model.train()

    return pair_data, _finalize_ranking_aggregate(ranking_agg)


def validate_silver_ranking(
    model: TrajectoryZonalEBM,
    sampler: NormalizedTemporalLangevinSampler,
    val_loader,
    config: EBMv3Config,
    lp_worker=None,
) -> Dict[str, float]:
    """Validate silver fine-tuning on ranking metrics aligned with LP score J."""
    max_scenarios = int(getattr(config, "silver_val_ranking_scenarios", 0))
    if max_scenarios <= 0 or lp_worker is None:
        return {}

    n_candidates = int(
        getattr(config, "silver_val_candidates_per_scenario", getattr(config, "silver_lp_candidates_per_scenario", 2))
    )
    max_stages = getattr(config, "silver_val_max_stages", None)
    if max_stages is None:
        max_stages = getattr(config, "silver_lp_max_stages", 2)
    max_stages = int(max_stages)
    min_rel_gap = float(getattr(config, "silver_pref_min_relative_gap", 0.02))

    model.eval()
    sampler.set_mode("infer")

    spearmans: List[float] = []
    bestofk_gaps: List[float] = []
    pref_correct = 0
    pref_total = 0
    ranked_scenarios = 0
    ranking_agg = _init_ranking_aggregate()

    for batch in val_loader:
        u_zt = batch["u_zt"].to(config.device)
        h_zt = batch["h_zt"].to(config.device)
        zone_mask = batch["zone_mask"].to(config.device)
        scenario_ids = batch["scenario_ids"]

        remaining = max_scenarios - ranked_scenarios
        if remaining <= 0:
            break

        batch_size = min(h_zt.shape[0], remaining)
        for i in range(batch_size):
            sc_id = scenario_ids[i]
            u_i = u_zt[i:i + 1]
            h_i = h_zt[i:i + 1]
            mask_i = zone_mask[i:i + 1]
            candidates = _sample_lp_candidates_v2(
                model, sampler, h_i, mask_i, u_i, sc_id, config,
                lp_worker=lp_worker,
                n_candidates=n_candidates,
                max_stages=max_stages,
            )
            finite_candidates = [
                cand for cand in candidates
                if math.isfinite(cand["score"]) and math.isfinite(cand["energy"])
            ]
            if len(finite_candidates) < 2:
                _update_ranking_aggregate(ranking_agg, candidates, 0)
                continue

            ranked_scenarios += 1
            ordered = sorted(finite_candidates, key=lambda cand: cand["score"])
            rho = _spearman_correlation(
                [cand["energy"] for cand in ordered],
                [cand["score"] for cand in ordered],
            )
            if math.isfinite(rho):
                spearmans.append(rho)

            energy_best = min(ordered, key=lambda cand: cand["energy"])
            score_best = ordered[0]
            bestofk_gaps.append(
                _relative_score_gap(score_best["score"], energy_best["score"])
            )

            scenario_pref_pairs = 0
            for better_idx in range(len(ordered) - 1):
                for worse_idx in range(better_idx + 1, len(ordered)):
                    better = ordered[better_idx]
                    worse = ordered[worse_idx]
                    rel_gap = _relative_score_gap(better["score"], worse["score"])
                    if rel_gap < min_rel_gap:
                        continue
                    scenario_pref_pairs += 1
                    pref_total += 1
                    if better["energy"] < worse["energy"]:
                        pref_correct += 1
            _update_ranking_aggregate(ranking_agg, candidates, scenario_pref_pairs)

        if ranked_scenarios >= max_scenarios:
            break

    sampler.set_mode("train")
    model.train()

    metrics: Dict[str, float] = _finalize_ranking_aggregate(ranking_agg, prefix="val_")
    metrics["val_rank_scenarios"] = float(ranked_scenarios)
    metrics["val_pref_pairs"] = float(pref_total)
    if pref_total > 0:
        metrics["val_pref_accuracy"] = pref_correct / pref_total
    if spearmans:
        metrics["val_spearman"] = float(np.mean(spearmans))
        metrics["val_spearman_energy_J"] = metrics["val_spearman"]
    if bestofk_gaps:
        metrics["val_bestofk_gap"] = float(np.mean(bestofk_gaps))
        metrics["val_best_of_K_gap"] = metrics["val_bestofk_gap"]
    return metrics


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
    n_pref_batches = 0
    ranking_agg = _init_ranking_aggregate()
    pref_pairs_correct = 0.0
    pref_pairs_total = 0.0

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
        E_better, E_worse, pair_margins = None, None, None
        pair_stats = None
        if (batch_idx + 1) % config.silver_lp_eval_every == 0:
            pair_data, ranking_stats = _lp_evaluate_candidates_v2(
                model, sampler, h_zt, zone_mask, u_pos, scenario_ids, config,
                lp_worker=lp_worker,
            )
            for key in ranking_agg:
                ranking_agg[key] += ranking_stats.get(key, 0.0)
            # Re-compute energies WITH gradients for preference loss
            # Temporarily zero dropout to get deterministic rankings
            # (can't use model.eval() — cuDNN RNN backward needs training mode)
            if pair_data:
                saved_dropout = {}
                for name, mod in model.named_modules():
                    if isinstance(mod, torch.nn.Dropout) and mod.p > 0:
                        saved_dropout[name] = mod.p
                        mod.p = 0.0
                    elif isinstance(mod, torch.nn.GRU):
                        saved_dropout[name] = mod.dropout
                        mod.dropout = 0.0
                E_better_list, E_worse_list = [], []
                margin_values, rel_gap_values, score_gap_values = [], [], []
                with torch.amp.autocast("cuda", enabled=use_amp):
                    for pair in pair_data:
                        E_better_list.append(model(pair["u_better"], pair["h"], pair["mask"]))
                        E_worse_list.append(model(pair["u_worse"], pair["h"], pair["mask"]))
                        margin_values.append(pair["margin"])
                        rel_gap_values.append(pair["rel_gap"])
                        score_gap_values.append(pair["score_worse"] - pair["score_better"])
                E_better = torch.cat(E_better_list)  # [N]
                E_worse = torch.cat(E_worse_list)    # [N]
                pref_pairs_correct += float((E_better.detach() < E_worse.detach()).sum().item())
                pref_pairs_total += float(E_better.numel())
                pair_margins = torch.tensor(
                    margin_values,
                    device=E_better.device,
                    dtype=E_better.dtype,
                )
                pair_stats = {
                    "pref/rel_gap_mean": float(np.mean(rel_gap_values)),
                    "pref/score_gap_mean": float(np.mean(score_gap_values)),
                    "pref/n_pairs": float(len(pair_data)),
                    "n_lp_scenarios_attempted": ranking_stats.get("n_lp_scenarios_attempted", 0.0),
                    "n_lp_scenarios_with_pairs": ranking_stats.get("n_lp_scenarios_with_pairs", 0.0),
                    "n_pairs_total": ranking_stats.get("n_pairs_total", 0.0),
                    "pair_coverage": ranking_stats.get("pair_coverage", float("nan")),
                    "pct_non_finite_candidates": ranking_stats.get("pct_non_finite_candidates", float("nan")),
                    "lp_stage_full_soft_share": ranking_stats.get("lp_stage_full_soft_share", float("nan")),
                    "lp_stage_round_refix_share": ranking_stats.get("lp_stage_round_refix_share", float("nan")),
                    "lp_stage_failed_share": ranking_stats.get("lp_stage_failed_share", float("nan")),
                    "lp_slack_used_mean": ranking_stats.get("lp_slack_used_mean", float("nan")),
                    "lp_decoder_deviation_mean": ranking_stats.get("lp_decoder_deviation_mean", float("nan")),
                    "lp_rounded_flips_mean": ranking_stats.get("lp_rounded_flips_mean", float("nan")),
                    "lp_repair_distance_mean": ranking_stats.get("lp_repair_distance_mean", float("nan")),
                }
                for name, mod in model.named_modules():
                    if name in saved_dropout:
                        if isinstance(mod, torch.nn.Dropout):
                            mod.p = saved_dropout[name]
                        elif isinstance(mod, torch.nn.GRU):
                            mod.dropout = saved_dropout[name]

        with torch.amp.autocast("cuda", enabled=use_amp):
            loss, metrics = loss_fn(
                E_pos, E_neg, E_better, E_worse, pair_margins=pair_margins,
            )
        if pair_stats is not None:
            metrics.update(pair_stats)

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
        if 'pref/E_better_mean' in metrics:
            n_pref_batches += 1

        if (batch_idx + 1) % config.log_every == 0:
            pref_str = ""
            if 'pref/E_better_mean' in metrics:
                pref_str = (
                    f" | Pref={_fmt_float(metrics.get('loss_pref_weighted', float('nan')), 4)} "
                    f"Acc={_fmt_pct(metrics.get('pref/pref_accuracy', float('nan')), 1)} "
                    f"Pairs={int(round(metrics.get('n_pairs_total', 0.0)))} "
                    f"Cov={_fmt_pct(metrics.get('pair_coverage', float('nan')))} "
                    f"NonFinite={_fmt_pct(metrics.get('pct_non_finite_candidates', float('nan')))} "
                    f"S4={_fmt_pct(metrics.get('lp_stage_full_soft_share', float('nan')))} "
                    f"Failed={_fmt_pct(metrics.get('lp_stage_failed_share', float('nan')))} "
                    f"Slack={_fmt_float(metrics.get('lp_slack_used_mean', float('nan')), 2)} "
                    f"Deviation={_fmt_float(metrics.get('lp_decoder_deviation_mean', float('nan')), 2)} "
                    f"Flips={_fmt_float(metrics.get('lp_rounded_flips_mean', float('nan')), 2)}"
                )
            print(
                f"  [Epoch {epoch}] Batch {batch_idx+1}/{len(train_loader)} | "
                f"Total={metrics.get('loss_total', 0):.4f} | "
                f"CD={metrics.get('cd/cd_loss', 0):.4f} | "
                f"Gap={metrics.get('cd/E_gap', 0):.4f} [L]{pref_str}"
            )

    avg = {}
    for k, v in total_metrics.items():
        if k.startswith('pref/') or k == 'loss_pref_weighted':
            avg[k] = v / max(n_pref_batches, 1)
        else:
            avg[k] = v / max(n_batches, 1)
    avg.update(_finalize_ranking_aggregate(ranking_agg))
    avg["train_pref_accuracy"] = _safe_ratio(pref_pairs_correct, pref_pairs_total)
    avg["train_pref_pairs_total"] = pref_pairs_total
    avg["train_pref_pairs_correct"] = pref_pairs_correct
    return avg


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
    _validate_lp_scenario_inputs(config, dataset)

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
    print(
        "Silver LP score: "
        f"J = cost + {float(getattr(config, 'silver_pref_slack_weight', 0.0)):.3g} * slack "
        f"+ {float(getattr(config, 'silver_pref_repair_weight', 0.0)):.3g} * "
        f"{getattr(config, 'silver_pref_repair_metric', 'decoder_deviation')}"
    )

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
    silver_prefix = getattr(config, "silver_output_prefix", "silver")
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

    warmup_epochs = getattr(config, "silver_pref_warmup_epochs", 5)

    for epoch in range(config.silver_epochs):
        t0 = time.time()

        # Silver curriculum: ramp Langevin steps
        curr_steps = _get_silver_langevin_steps(epoch, config.silver_epochs, config)
        sampler.set_num_steps(curr_steps)

        # Lambda_pref warmup: ramp from 0 → full over first N epochs
        if warmup_epochs > 0 and epoch < warmup_epochs:
            warmup_ratio = epoch / warmup_epochs
            loss_fn.lambda_pref = config.silver_lambda_pref * warmup_ratio
        else:
            loss_fn.lambda_pref = config.silver_lambda_pref

        train_metrics = train_epoch_silver(
            model, sampler, train_loader, optimizer, loss_fn,
            config, scaler, epoch, lp_worker=lp_worker,
        )
        val_metrics = validate_gold(model, sampler, val_loader, cd_loss_fn, config)
        val_metrics.update(
            validate_silver_ranking(model, sampler, val_loader, config, lp_worker=lp_worker)
        )
        scheduler.step()

        dt = time.time() - t0
        history["train"].append(train_metrics)
        history["val"].append(val_metrics)

        val_gap_rand = val_metrics.get("E_gap_rand", 0)
        val_gap_lang = val_metrics.get("E_gap_lang", float("nan"))
        val_pref_acc = val_metrics.get("val_pref_accuracy", float("nan"))
        val_spearman = val_metrics.get("val_spearman_energy_J", val_metrics.get("val_spearman", float("nan")))
        val_bestofk_gap = val_metrics.get("val_best_of_K_gap", val_metrics.get("val_bestofk_gap", float("nan")))
        train_pref_loss = train_metrics.get("loss_pref_weighted", float("nan"))
        train_pref_acc = train_metrics.get("train_pref_accuracy", float("nan"))
        n_pairs_total = train_metrics.get("n_pairs_total", 0.0)
        n_lp_scenarios_attempted = train_metrics.get("n_lp_scenarios_attempted", 0.0)
        n_lp_scenarios_with_pairs = train_metrics.get("n_lp_scenarios_with_pairs", 0.0)
        pair_coverage = train_metrics.get("pair_coverage", float("nan"))
        pct_non_finite = train_metrics.get("pct_non_finite_candidates", float("nan"))
        lp_stage4_share = train_metrics.get("lp_stage_full_soft_share", float("nan"))
        lp_stage5_share = train_metrics.get("lp_stage_round_refix_share", float("nan"))
        lp_failed_share = train_metrics.get("lp_stage_failed_share", float("nan"))
        lp_early_share = sum(
            train_metrics.get(metric_name, 0.0)
            for metric_name in (
                "lp_stage_hard_fix_share",
                "lp_stage_repair_20_share",
                "lp_stage_repair_100_share",
            )
        )
        lp_slack_mean = train_metrics.get("lp_slack_used_mean", float("nan"))
        lp_deviation_mean = train_metrics.get("lp_decoder_deviation_mean", float("nan"))
        lp_flips_mean = train_metrics.get("lp_rounded_flips_mean", float("nan"))
        print(
            f"Epoch {epoch+1}/{config.silver_epochs} ({dt:.1f}s) | "
            f"Total={train_metrics.get('loss_total', 0):.4f} "
            f"CD={train_metrics.get('cd/cd_loss', 0):.4f} "
            f"Pref={_fmt_float(train_pref_loss, 4)} | "
            f"Pairs={int(round(n_pairs_total))} "
            f"Attempted={int(round(n_lp_scenarios_attempted))} "
            f"WithPairs={int(round(n_lp_scenarios_with_pairs))} "
            f"PairCoverage={_fmt_pct(pair_coverage)} "
            f"NonFinite={_fmt_pct(pct_non_finite)} | "
            f"Early={_fmt_pct(lp_early_share)} "
            f"Stage4={_fmt_pct(lp_stage4_share)} "
            f"Stage5={_fmt_pct(lp_stage5_share)} "
            f"Failed={_fmt_pct(lp_failed_share)} "
            f"Slack={_fmt_float(lp_slack_mean, 2)} "
            f"Deviation={_fmt_float(lp_deviation_mean, 2)} "
            f"Flips={_fmt_float(lp_flips_mean, 2)} | "
            f"TrainPrefAcc={_fmt_pct(train_pref_acc)} "
            f"ValPrefAcc={_fmt_pct(val_pref_acc)} "
            f"Spearman(E,J)={_fmt_float(val_spearman, 3)} "
            f"BestOfKGap={_fmt_pct(val_bestofk_gap, 1)} | "
            f"ValGap_R={_fmt_float(val_gap_rand, 4)} "
            f"ValGap_L={_fmt_float(val_gap_lang, 4)} | "
            f"LR={scheduler.get_last_lr()[0]:.2e} | "
            f"Lsteps={curr_steps} | "
            f"LambdaPref={loss_fn.lambda_pref:.2f}"
        )
        coverage_floor = float(getattr(config, "silver_pair_coverage_floor", 0.20))
        if (
            n_lp_scenarios_attempted > 0
            and math.isfinite(pair_coverage)
            and pair_coverage < coverage_floor
        ):
            print(
                f"  !! PairCoverage below floor: {_fmt_pct(pair_coverage)} "
                f"< {_fmt_pct(coverage_floor)}. Preference signal is sparse."
            )

        # Early stopping on the configured validation metric.
        early_stop_metric = getattr(config, "silver_early_stop_metric", "val_gap_lang")
        metric_name = "ValGap_L"
        val_gap = val_gap_lang if not (val_gap_lang != val_gap_lang) else val_gap_rand
        if early_stop_metric == "val_pref_accuracy" and math.isfinite(val_pref_acc):
            val_gap = val_pref_acc
            metric_name = "ValPref"
        elif early_stop_metric == "val_spearman" and math.isfinite(val_spearman):
            val_gap = val_spearman
            metric_name = "ValRho"
        if val_gap > best_val_gap + min_delta:
            best_val_gap = val_gap
            patience_counter = 0
            _save_checkpoint(
                model, optimizer, epoch, val_metrics,
                os.path.join(config.output_dir, f"{silver_prefix}_best.pt"),
            )
            print(f"  >> New best {metric_name}: {val_gap:.4f}")
        else:
            patience_counter += 1
            if patience_counter >= config.silver_patience:
                print(f"  Early stopping at epoch {epoch+1} (best {metric_name}={best_val_gap:.4f})")
                break

        if (epoch + 1) % config.save_every_epoch == 0:
            _save_checkpoint(
                model, optimizer, epoch, val_metrics,
                os.path.join(config.output_dir, f"{silver_prefix}_epoch_{epoch+1}.pt"),
            )

    best_path = os.path.join(config.output_dir, f"{silver_prefix}_best.pt")
    if os.path.exists(best_path):
        ckpt = torch.load(best_path, map_location=config.device, weights_only=False)
        model.load_state_dict(ckpt["model_state_dict"])
        print(f"Loaded best {silver_prefix} model")

    _save_history(history, os.path.join(config.output_dir, f"{silver_prefix}_history.json"))

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
