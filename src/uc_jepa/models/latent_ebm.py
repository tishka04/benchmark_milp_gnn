"""Latent-space EBM for UC-JEPA.

This module scores candidate UC latents ``z_uc`` conditioned on:
- the JEPA-predicted anchor latent for the scenario
- the per-(zone, time) scenario embedding ``h_scenario``

The model is intentionally lightweight so it can be trained on top of the
existing frozen AE + JEPA stack without introducing a second large encoder.
"""
from __future__ import annotations

from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


def _expand_context(
    z_context: torch.Tensor,
    z: torch.Tensor,
) -> torch.Tensor:
    """Expand ``z_context`` to match the sample axis of ``z`` when needed."""
    if z.dim() == 3:
        if z_context.dim() != 3:
            raise ValueError(
                f"Expected z_context [B,K,D] for z [B,K,D], got {tuple(z_context.shape)}",
            )
        return z_context

    if z.dim() != 4:
        raise ValueError(f"Expected z with ndim 3 or 4, got shape {tuple(z.shape)}")

    if z_context.dim() == 4:
        if z_context.shape[:2] != z.shape[:2]:
            raise ValueError(
                f"z_context sample dims {tuple(z_context.shape[:2])} do not match z "
                f"{tuple(z.shape[:2])}",
            )
        return z_context

    if z_context.dim() != 3:
        raise ValueError(
            f"Expected z_context [B,K,D] or [B,N,K,D], got {tuple(z_context.shape)}",
        )
    return z_context.unsqueeze(1).expand(-1, z.shape[1], -1, -1)


def _expand_scenario(
    h_scenario: Optional[torch.Tensor],
    z: torch.Tensor,
) -> Optional[torch.Tensor]:
    if h_scenario is None:
        return None
    if z.dim() == 3:
        return h_scenario
    return h_scenario.unsqueeze(1).expand(-1, z.shape[1], -1, -1, -1)


def _expand_zone_mask(
    zone_mask: Optional[torch.Tensor],
    z: torch.Tensor,
) -> Optional[torch.Tensor]:
    if zone_mask is None:
        return None
    if z.dim() == 3:
        return zone_mask
    return zone_mask.unsqueeze(1).expand(-1, z.shape[1], -1)


def _pool_scenario(
    h_scenario: Optional[torch.Tensor],
    zone_mask: Optional[torch.Tensor],
) -> Optional[torch.Tensor]:
    """Masked mean-pool over the scenario tensor [B, Z, T, D]."""
    if h_scenario is None:
        return None

    if zone_mask is None:
        return h_scenario.mean(dim=(1, 2))

    mask = zone_mask.to(h_scenario.dtype).unsqueeze(-1).unsqueeze(-1)
    mask = mask.expand(-1, -1, h_scenario.shape[2], 1)
    denom = mask.sum(dim=(1, 2)).clamp_min(1.0)
    return (h_scenario * mask).sum(dim=(1, 2)) / denom


class LatentEBM(nn.Module):
    """Energy model over UC latents conditioned on scenario context."""

    def __init__(
        self,
        latent_dim: int,
        n_tokens: int,
        scenario_dim: int = 128,
        hidden_dim: int = 256,
        token_hidden_dim: Optional[int] = None,
        dropout: float = 0.1,
        use_scenario_context: bool = True,
        energy_bound: float = 50.0,
    ):
        super().__init__()
        self.latent_dim = int(latent_dim)
        self.n_tokens = int(n_tokens)
        self.scenario_dim = int(scenario_dim)
        self.hidden_dim = int(hidden_dim)
        self.token_hidden_dim = int(token_hidden_dim or hidden_dim)
        self.use_scenario_context = bool(use_scenario_context)
        self.energy_bound = float(energy_bound)

        token_in_dim = 4 * self.latent_dim
        context_in_dim = 2 * self.latent_dim + (
            self.scenario_dim if self.use_scenario_context else 0
        )
        head_in_dim = 2 * self.token_hidden_dim + self.hidden_dim + 3

        self.token_mlp = nn.Sequential(
            nn.LayerNorm(token_in_dim),
            nn.Linear(token_in_dim, self.token_hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(self.token_hidden_dim, self.token_hidden_dim),
            nn.GELU(),
        )
        self.context_mlp = nn.Sequential(
            nn.LayerNorm(context_in_dim),
            nn.Linear(context_in_dim, self.hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.GELU(),
        )
        self.energy_head = nn.Sequential(
            nn.LayerNorm(head_in_dim),
            nn.Linear(head_in_dim, self.hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(self.hidden_dim, 1),
        )

    def _prepare_inputs(
        self,
        z: torch.Tensor,
        z_context: torch.Tensor,
        h_scenario: Optional[torch.Tensor] = None,
        zone_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor], Optional[Tuple[int, int]]]:
        """Flatten optional sample axis and align all conditioning tensors."""
        z_context = _expand_context(z_context, z)
        h_scenario = _expand_scenario(h_scenario, z)
        zone_mask = _expand_zone_mask(zone_mask, z)

        sample_shape: Optional[Tuple[int, int]] = None
        if z.dim() == 4:
            bsz, n_samples, n_tokens, latent_dim = z.shape
            sample_shape = (bsz, n_samples)
            z = z.reshape(bsz * n_samples, n_tokens, latent_dim)
            z_context = z_context.reshape(bsz * n_samples, n_tokens, latent_dim)
            if h_scenario is not None:
                h_scenario = h_scenario.reshape(
                    bsz * n_samples, h_scenario.shape[2], h_scenario.shape[3], h_scenario.shape[4],
                )
            if zone_mask is not None:
                zone_mask = zone_mask.reshape(bsz * n_samples, zone_mask.shape[2])

        return z, z_context, h_scenario, zone_mask, sample_shape

    def forward(
        self,
        z: torch.Tensor,
        z_context: torch.Tensor,
        h_scenario: Optional[torch.Tensor] = None,
        zone_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Compute scalar energy for one or many latent candidates.

        Args:
            z: candidate latent, shape ``[B, K, D]`` or ``[B, N, K, D]``
            z_context: JEPA anchor latent, shape ``[B, K, D]`` or broadcastable
            h_scenario: optional scenario tensor ``[B, Z, T, D_h]``
            zone_mask: optional zone mask ``[B, Z]``
        """
        z, z_context, h_scenario, zone_mask, sample_shape = self._prepare_inputs(
            z=z,
            z_context=z_context,
            h_scenario=h_scenario,
            zone_mask=zone_mask,
        )

        token_feat = torch.cat(
            [z, z_context, z - z_context, z * z_context],
            dim=-1,
        )
        token_hidden = self.token_mlp(token_feat)
        pooled_mean = token_hidden.mean(dim=1)
        pooled_max = token_hidden.amax(dim=1)

        ctx_stats = torch.cat(
            [
                z_context.mean(dim=1),
                z_context.std(dim=1, unbiased=False),
            ],
            dim=-1,
        )
        scenario_pool = _pool_scenario(h_scenario, zone_mask)
        if self.use_scenario_context and scenario_pool is not None:
            ctx_stats = torch.cat([ctx_stats, scenario_pool], dim=-1)
        context_hidden = self.context_mlp(ctx_stats)

        cand_norm = z.pow(2).mean(dim=(1, 2)).unsqueeze(-1)
        delta_norm = (z - z_context).pow(2).mean(dim=(1, 2)).unsqueeze(-1)
        align = F.cosine_similarity(
            z.reshape(z.shape[0], -1),
            z_context.reshape(z_context.shape[0], -1),
            dim=-1,
        ).unsqueeze(-1)

        head_in = torch.cat(
            [pooled_mean, pooled_max, context_hidden, cand_norm, delta_norm, align],
            dim=-1,
        )
        energy = self.energy_head(head_in).squeeze(-1)

        if self.energy_bound > 0:
            energy = self.energy_bound * torch.tanh(energy / self.energy_bound)

        if sample_shape is not None:
            energy = energy.view(sample_shape[0], sample_shape[1])
        return energy
