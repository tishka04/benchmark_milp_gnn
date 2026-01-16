# ==============================================================================
# PREFERENCE-BASED TRAINING LOSSES
# ==============================================================================
# Margin ranking loss for EBM energy shaping based on economic oracle feedback
# ==============================================================================

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple


class MarginRankingLoss(nn.Module):
    """
    Margin ranking loss for preference-based EBM training.
    
    L_rank(θ) = (1/|K|) Σ_k max(0, m + E_θ(u+ | h) - E_θ(u^(k) | h))
    
    The EBM should assign lower energy to the positive (MILP) decision
    than to negative (pipeline) candidates.
    """
    
    def __init__(
        self,
        margin: float = 1.0,
        reduction: str = "mean",  # "mean", "sum", "none"
    ):
        """
        Args:
            margin: Margin m > 0 for separation
            reduction: How to reduce the loss
        """
        super().__init__()
        self.margin = margin
        self.reduction = reduction
    
    def forward(
        self,
        energy_positive: torch.Tensor,  # [B] energies of positive examples
        energy_negatives: torch.Tensor,  # [B, K] energies of negative examples
    ) -> torch.Tensor:
        """
        Compute margin ranking loss.
        
        Args:
            energy_positive: [B] E_θ(u+ | h) for positive examples
            energy_negatives: [B, K] E_θ(u^(k) | h) for K negative examples
        
        Returns:
            loss: Scalar loss value
        """
        B, K = energy_negatives.shape
        
        # Expand positive energies: [B] -> [B, K]
        energy_pos_expanded = energy_positive.unsqueeze(1).expand(-1, K)
        
        # Margin loss: max(0, m + E(u+) - E(u-))
        # We want E(u+) < E(u-), so penalize when E(u+) + m > E(u-)
        losses = F.relu(self.margin + energy_pos_expanded - energy_negatives)
        
        if self.reduction == "mean":
            return losses.mean()
        elif self.reduction == "sum":
            return losses.sum()
        else:  # "none"
            return losses


class WeightedMarginRankingLoss(nn.Module):
    """
    Weighted margin ranking loss with cost-aware weighting.
    
    L_w-rank(θ) = (1/|K|) Σ_k (1 + α*w_k) * max(0, m + E_θ(u+ | h) - E_θ(u^(k) | h))
    
    where w_k = clip(log(1 + (C^(k) - C+)_+), 0, w_max)
    
    This emphasizes hard negatives with high cost gaps.
    """
    
    def __init__(
        self,
        margin: float = 1.0,
        alpha: float = 1.0,  # Weight scaling factor
        w_max: float = 5.0,  # Maximum weight
        reduction: str = "mean",
    ):
        """
        Args:
            margin: Margin m > 0 for separation
            alpha: Scaling factor for cost-aware weights
            w_max: Maximum weight to prevent extreme values
            reduction: How to reduce the loss
        """
        super().__init__()
        self.margin = margin
        self.alpha = alpha
        self.w_max = w_max
        self.reduction = reduction
    
    def forward(
        self,
        energy_positive: torch.Tensor,  # [B] energies of positive examples
        energy_negatives: torch.Tensor,  # [B, K] energies of negative examples
        cost_positive: torch.Tensor,     # [B] costs of positive examples
        costs_negative: torch.Tensor,    # [B, K] costs of negative examples
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute weighted margin ranking loss.
        
        Args:
            energy_positive: [B] E_θ(u+ | h)
            energy_negatives: [B, K] E_θ(u^(k) | h)
            cost_positive: [B] C+ (MILP cost)
            costs_negative: [B, K] C^(k) (candidate costs)
        
        Returns:
            loss: Scalar loss value
            weights: [B, K] computed weights for analysis
        """
        B, K = energy_negatives.shape
        
        # Compute cost gaps: (C^(k) - C+)_+
        cost_pos_expanded = cost_positive.unsqueeze(1).expand(-1, K)
        cost_gaps = F.relu(costs_negative - cost_pos_expanded)
        
        # Compute weights: clip(log(1 + gap), 0, w_max)
        weights = torch.log1p(cost_gaps).clamp(0, self.w_max)
        
        # Apply alpha scaling: (1 + α*w_k)
        scaled_weights = 1.0 + self.alpha * weights
        
        # Expand positive energies
        energy_pos_expanded = energy_positive.unsqueeze(1).expand(-1, K)
        
        # Margin losses
        margin_losses = F.relu(self.margin + energy_pos_expanded - energy_negatives)
        
        # Weighted losses
        weighted_losses = scaled_weights * margin_losses
        
        if self.reduction == "mean":
            loss = weighted_losses.mean()
        elif self.reduction == "sum":
            loss = weighted_losses.sum()
        else:
            loss = weighted_losses
        
        return loss, weights


class ContrastiveLoss(nn.Module):
    """
    Contrastive divergence loss for EBM training.
    
    L_CD = E_data[E_θ(u | h)] - E_model[E_θ(u | h)]
    
    This directly minimizes energy on positive examples and
    maximizes energy on samples from the model (Langevin samples).
    """
    
    def __init__(self, alpha: float = 1.0):
        """
        Args:
            alpha: Balance between positive and negative terms
        """
        super().__init__()
        self.alpha = alpha
    
    def forward(
        self,
        energy_positive: torch.Tensor,  # [B] energies on data
        energy_samples: torch.Tensor,   # [B, K] energies on Langevin samples
    ) -> torch.Tensor:
        """
        Compute contrastive divergence loss.
        
        Args:
            energy_positive: [B] E_θ(u+ | h) on data
            energy_samples: [B, K] E_θ(u_sampled | h) on Langevin samples
        
        Returns:
            loss: Scalar loss value
        """
        # Push down energy on data
        pos_term = energy_positive.mean()
        
        # Push up energy on samples
        neg_term = energy_samples.mean()
        
        return pos_term - self.alpha * neg_term


class CombinedPreferenceLoss(nn.Module):
    """
    Combined loss for preference-based EBM training.
    
    Combines:
    1. Weighted margin ranking (main objective)
    2. Energy regularization (prevent unbounded energies)
    3. Optional contrastive term
    """
    
    def __init__(
        self,
        margin: float = 1.0,
        alpha: float = 1.0,
        w_max: float = 5.0,
        energy_reg_weight: float = 0.01,
        contrastive_weight: float = 0.0,
    ):
        super().__init__()
        self.ranking_loss = WeightedMarginRankingLoss(
            margin=margin,
            alpha=alpha,
            w_max=w_max,
        )
        self.energy_reg_weight = energy_reg_weight
        self.contrastive_weight = contrastive_weight
        
        if contrastive_weight > 0:
            self.contrastive_loss = ContrastiveLoss()
    
    def forward(
        self,
        energy_positive: torch.Tensor,
        energy_negatives: torch.Tensor,
        cost_positive: torch.Tensor,
        costs_negative: torch.Tensor,
        energy_samples: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, dict]:
        """
        Compute combined loss.
        
        Returns:
            total_loss: Scalar loss
            components: Dict with individual loss components
        """
        # Main ranking loss
        ranking_loss, weights = self.ranking_loss(
            energy_positive, energy_negatives,
            cost_positive, costs_negative
        )
        
        # Energy regularization: prevent energies from growing unbounded
        all_energies = torch.cat([energy_positive, energy_negatives.flatten()])
        energy_reg = (all_energies ** 2).mean()
        
        total_loss = ranking_loss + self.energy_reg_weight * energy_reg
        
        components = {
            "ranking_loss": ranking_loss.item(),
            "energy_reg": energy_reg.item(),
            "mean_weight": weights.mean().item(),
        }
        
        # Optional contrastive term
        if self.contrastive_weight > 0 and energy_samples is not None:
            cd_loss = self.contrastive_loss(energy_positive, energy_samples)
            total_loss = total_loss + self.contrastive_weight * cd_loss
            components["contrastive_loss"] = cd_loss.item()
        
        return total_loss, components
