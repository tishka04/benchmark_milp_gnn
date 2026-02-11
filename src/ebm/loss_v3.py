# ==============================================================================
# EBM V3 LOSS FUNCTIONS
# ==============================================================================
# Contrastive Divergence loss for Step A (gold pre-training)
# Combined CD + Preference loss for Step B (silver fine-tuning)
# ==============================================================================

from __future__ import annotations

import torch
import torch.nn as nn
from typing import Dict, Optional, Tuple


class ContrastiveDivergenceLoss(nn.Module):
    """
    Contrastive Divergence loss for EBM training.

    L_CD = E_data[E_θ(u_pos, h)] - E_model[E_θ(u_neg, h)]

    The model should assign LOW energy to real data (u_pos)
    and HIGH energy to samples (u_neg).

    Also includes optional L2 regularization on energy magnitudes
    to prevent energy collapse.
    """

    def __init__(
        self,
        alpha_reg: float = 0.01,
        energy_clamp: float = 100.0,
        cd_clamp: float = 10.0,
    ):
        super().__init__()
        self.alpha_reg = alpha_reg
        self.energy_clamp = energy_clamp
        self.cd_clamp = cd_clamp

    def forward(
        self,
        E_pos: torch.Tensor,
        E_neg: torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Compute CD loss.

        Args:
            E_pos: [B] energy of positive (real) samples
            E_neg: [B] energy of negative (sampled) samples

        Returns:
            loss: scalar
            metrics: dict of diagnostic values
        """
        # Clamp for stability
        E_pos = torch.clamp(E_pos, -self.energy_clamp, self.energy_clamp)
        E_neg = torch.clamp(E_neg, -self.energy_clamp, self.energy_clamp)

        # CD objective: minimize E_pos, maximize E_neg
        cd_loss_raw = E_pos.mean() - E_neg.mean()
        # Clamp CD loss to prevent extreme gradients from easy negatives
        cd_loss = torch.clamp(cd_loss_raw, -self.cd_clamp, self.cd_clamp)

        # Regularization: soft constraint to prevent energy magnitude explosion
        # Only penalize energies that exceed a threshold, don't collapse them to 0
        reg_loss = self.alpha_reg * (
            torch.clamp(E_pos.abs() - 30.0, min=0.0).pow(2).mean()
            + torch.clamp(E_neg.abs() - 30.0, min=0.0).pow(2).mean()
        )

        loss = cd_loss + reg_loss

        metrics = {
            "cd_loss": cd_loss.item(),
            "reg_loss": reg_loss.item(),
            "E_pos_mean": E_pos.mean().item(),
            "E_neg_mean": E_neg.mean().item(),
            "E_gap": (E_neg.mean() - E_pos.mean()).item(),
        }

        return loss, metrics


class PreferenceLoss(nn.Module):
    """
    Preference / margin ranking loss for Step B (silver fine-tuning).

    Given two candidates with LP-evaluated costs c_a < c_b,
    the EBM should assign lower energy to the better candidate:
        E_θ(u_a, h) < E_θ(u_b, h)

    L_pref = max(0, E_better - E_worse + margin)
    """

    def __init__(self, margin: float = 0.1):
        super().__init__()
        self.margin = margin

    def forward(
        self,
        E_better: torch.Tensor,
        E_worse: torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Args:
            E_better: [N] energy of candidates with lower LP cost
            E_worse:  [N] energy of candidates with higher LP cost

        Returns:
            loss: scalar margin ranking loss
            metrics: dict
        """
        # Margin ranking: E_better should be < E_worse by at least margin
        pref_loss = torch.clamp(
            E_better - E_worse + self.margin, min=0.0
        ).mean()

        # Fraction of correctly ordered pairs
        correct = (E_better < E_worse).float().mean().item()

        metrics = {
            "pref_loss": pref_loss.item(),
            "pref_accuracy": correct,
            "E_better_mean": E_better.mean().item(),
            "E_worse_mean": E_worse.mean().item(),
        }

        return pref_loss, metrics


class CombinedLoss(nn.Module):
    """
    Combined loss for Step B: λ_cd * L_CD + λ_pref * L_pref

    Used during silver fine-tuning where we have both:
    - Standard CD loss (sampled negatives)
    - Preference loss (LP-evaluated candidate pairs)
    """

    def __init__(
        self,
        lambda_cd: float = 1.0,
        lambda_pref: float = 0.5,
        margin: float = 0.1,
        alpha_reg: float = 0.01,
    ):
        super().__init__()
        self.lambda_cd = lambda_cd
        self.lambda_pref = lambda_pref
        self.cd_loss = ContrastiveDivergenceLoss(alpha_reg=alpha_reg)
        self.pref_loss = PreferenceLoss(margin=margin)

    def forward(
        self,
        E_pos: torch.Tensor,
        E_neg: torch.Tensor,
        E_better: Optional[torch.Tensor] = None,
        E_worse: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Compute combined loss.

        Args:
            E_pos, E_neg: for CD loss
            E_better, E_worse: for preference loss (optional)
        """
        cd_val, cd_metrics = self.cd_loss(E_pos, E_neg)

        total = self.lambda_cd * cd_val
        metrics = {f"cd/{k}": v for k, v in cd_metrics.items()}
        metrics["loss_cd_weighted"] = (self.lambda_cd * cd_val).item()

        if E_better is not None and E_worse is not None and len(E_better) > 0:
            pref_val, pref_metrics = self.pref_loss(E_better, E_worse)
            total = total + self.lambda_pref * pref_val
            metrics.update({f"pref/{k}": v for k, v in pref_metrics.items()})
            metrics["loss_pref_weighted"] = (self.lambda_pref * pref_val).item()

        metrics["loss_total"] = total.item()
        return total, metrics
