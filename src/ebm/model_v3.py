# ==============================================================================
# TRAJECTORY ZONAL EBM - v3
# ==============================================================================
# Energy-Based Model with temporal trajectory processing per zone.
# E = Σ_z E_θ(u_{z,1:T}, h_{z,1:T})  with masked zone aggregation.
#
# Architecture:
#   1. Concatenate u_zt and h_zt → project to hidden_dim
#   2. Bidirectional GRU for temporal encoding per zone
#   3. Energy head per (zone, time) → aggregate with mean + peak terms
#   4. Zone masking for variable-sized graphs
# ==============================================================================

from __future__ import annotations

import torch
import torch.nn as nn
from typing import Optional


class TrajectoryZonalEBM(nn.Module):
    """
    Energy-Based Model with temporal trajectory processing per zone.
    Robust to NaNs and padded zones.

    Input:
        u_zt: [B, Z, T, F] - binary/relaxed decisions
        h_zt: [B, Z, T, D] - zone-level temporal embeddings
        zone_mask: [B, Z]   - 1 for valid zones, 0 for padding

    Output:
        energy: [B] - scalar energy per sample
    """

    def __init__(
        self,
        embed_dim: int,
        n_features: int,
        hidden_dim: int = 128,
        gru_layers: int = 2,
        bidirectional: bool = True,
        dropout: float = 0.1,
        use_peak_term: bool = True,
        peak_tau: float = 0.5,
        peak_weight: float = 0.3,
        energy_max: float = 50.0,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.n_features = n_features
        self.hidden_dim = hidden_dim
        self.bidirectional = bidirectional

        self.use_peak_term = use_peak_term
        self.peak_tau = peak_tau
        self.peak_weight = peak_weight
        self.energy_max = energy_max

        # Input projection: [F + D] -> hidden_dim
        self.input_proj = nn.Sequential(
            nn.Linear(n_features + embed_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
        )

        # Temporal GRU
        self.gru = nn.GRU(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            num_layers=gru_layers,
            batch_first=True,
            bidirectional=bidirectional,
            dropout=dropout if gru_layers > 1 else 0,
        )

        # Energy head
        gru_out_dim = hidden_dim * 2 if bidirectional else hidden_dim
        self.energy_head = nn.Sequential(
            nn.Linear(gru_out_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.SiLU(),
            nn.Linear(hidden_dim // 2, 1),
        )

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight, gain=1.0)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def _masked_logsumexp(
        self, x: torch.Tensor, dim: int,
        mask: Optional[torch.Tensor], tau: float,
    ) -> torch.Tensor:
        if mask is None:
            return tau * torch.logsumexp(x / tau, dim=dim)
        neg_inf = torch.finfo(x.dtype).min
        x_masked = x.masked_fill(mask == 0, neg_inf)
        return tau * torch.logsumexp(x_masked / tau, dim=dim)

    def forward(
        self,
        u_zt: torch.Tensor,
        h_zt: torch.Tensor,
        zone_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Compute energy E_θ(u, h) with temporal understanding.

        Args:
            u_zt: [B, Z, T, F] decisions (relaxed or binary)
            h_zt: [B, Z, T, D] zone-level temporal embeddings
            zone_mask: [B, Z] valid zone indicator

        Returns:
            energy: [B] scalar energies
        """
        # Sanitize inputs
        if torch.isnan(u_zt).any():
            u_zt = torch.nan_to_num(u_zt, nan=0.0)
        if torch.isnan(h_zt).any():
            h_zt = torch.nan_to_num(h_zt, nan=0.0)

        B, Z, T, F = u_zt.shape
        D = h_zt.shape[-1]

        # Concatenate decisions + embeddings → project
        x = torch.cat([u_zt, h_zt], dim=-1)  # [B, Z, T, F+D]
        x = x.view(B * Z, T, F + D)

        x = self.input_proj(x)        # [B*Z, T, hidden]
        gru_out, _ = self.gru(x)      # [B*Z, T, gru_out_dim]
        e_zt = self.energy_head(gru_out)  # [B*Z, T, 1]
        e_zt = e_zt.view(B, Z, T)

        # ── Mean Energy Term ──
        if zone_mask is not None:
            m_e = zone_mask.view(B, Z, 1).float()
            denom = (m_e.sum(dim=1) * T).clamp_min(1.0)
            if denom.dim() == 1:
                denom = denom.unsqueeze(-1)
            E_mean = (e_zt * m_e).sum(dim=(1, 2)) / denom.squeeze()
        else:
            E_mean = e_zt.mean(dim=(1, 2))

        # ── Peak Energy Term ──
        if self.use_peak_term:
            tau = float(self.peak_tau)
            w = float(self.peak_weight)

            m = zone_mask.view(B, Z, 1).bool() if zone_mask is not None else None
            peak_z = self._masked_logsumexp(e_zt, dim=2, mask=m, tau=tau)  # [B, Z]

            if zone_mask is not None:
                mz = zone_mask.float()
                peak_z_safe = torch.nan_to_num(peak_z, neginf=0.0)
                denom_z = mz.sum(dim=1).clamp_min(1.0)
                E_peak = (peak_z_safe * mz).sum(dim=1) / denom_z
            else:
                E_peak = peak_z.mean(dim=1)

            energy = (1.0 - w) * E_mean + w * E_peak
        else:
            energy = E_mean

        # Bound energy to [-energy_max, energy_max] via tanh scaling
        if self.energy_max > 0:
            energy = self.energy_max * torch.tanh(energy / self.energy_max)

        return energy

    def energy_gradient(
        self,
        u_zt: torch.Tensor,
        h_zt: torch.Tensor,
        zone_mask: Optional[torch.Tensor] = None,
    ):
        """Compute energy and its gradient w.r.t. u_zt (for Langevin sampling)."""
        u_zt = u_zt.requires_grad_(True)
        energy = self.forward(u_zt, h_zt, zone_mask)
        grad_u = torch.autograd.grad(
            outputs=energy.sum(),
            inputs=u_zt,
            create_graph=True,
            retain_graph=True,
        )[0]
        return energy, grad_u
