"""UC autoencoder — variant B (strategy tokens).

Encoder:
    u_star [B, Z, T, F]  (+ zone_mask [B, Z])
        |- per-zone Linear(F -> H) + BiGRU over T
        |- temporal mean-pool -> zone embedding [B, Z, H]
        |- K learned query tokens cross-attend over masked zones
        |- LayerNorm + Linear(H -> d_latent)
        v
    z_uc [B, K, d_latent]

Decoder:
    z_uc [B, K, d_latent]
        |- (zone_emb + time_emb) build Z*T queries
        |- cross-attention onto z_uc keys/values
        |- residual FFN + Linear -> F logits per (z, t)
        v
    logits [B, Z, T, F]

The decoder is shape-agnostic at inference (Z and T are read from the call site).
Supported zones up to ``n_zones_max`` and timesteps up to ``n_timesteps_max`` thanks
to learned positional embeddings.

Losses (see :func:`uc_autoencoder_loss`):
    L = BCE(logits, u_star)
        + lambda_temporal * temporal_smoothness(prob)
        + lambda_consistency * startup_consistency(prob)
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.uc_jepa import (
    FEAT_THERMAL,
    FEAT_THERMAL_SU,
    N_UC_FEATURES,
)


# ---------------------------------------------------------------------------
# Encoder / decoder modules
# ---------------------------------------------------------------------------

class UCEncoder(nn.Module):
    """Encode u_star [B, Z, T, F] into K strategy tokens [B, K, d_latent]."""

    def __init__(
        self,
        n_features: int = N_UC_FEATURES,
        hidden_dim: int = 256,
        n_tokens: int = 16,
        latent_dim: int = 128,
        n_heads: int = 4,
        gru_layers: int = 1,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.n_tokens = n_tokens
        self.latent_dim = latent_dim
        self.hidden_dim = hidden_dim

        self.input_proj = nn.Linear(n_features, hidden_dim)
        self.gru = nn.GRU(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            num_layers=gru_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if gru_layers > 1 else 0.0,
        )
        self.zone_proj = nn.Linear(2 * hidden_dim, hidden_dim)
        self.queries = nn.Parameter(torch.randn(n_tokens, hidden_dim) * 0.02)
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=n_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.norm = nn.LayerNorm(hidden_dim)
        self.to_latent = nn.Linear(hidden_dim, latent_dim)

    def forward(
        self,
        u_zt: torch.Tensor,
        zone_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        B, Z, T, F_ = u_zt.shape
        x = self.input_proj(u_zt).reshape(B * Z, T, self.hidden_dim)
        gru_out, _ = self.gru(x)                       # [B*Z, T, 2H]
        zone_emb = gru_out.mean(dim=1).view(B, Z, -1)  # [B, Z, 2H]
        zone_emb = self.zone_proj(zone_emb)            # [B, Z, H]

        Q = self.queries.unsqueeze(0).expand(B, -1, -1)  # [B, K, H]
        # MultiheadAttention key_padding_mask: True positions are *ignored*.
        kpm = None
        if zone_mask is not None:
            kpm = zone_mask == 0
            # If a sample has zero valid zones, kpm is all-True and softmax NaNs.
            # Guard by leaving at least one slot active (the first), which is
            # itself zero anyway because zone_emb was padded with zeros.
            all_pad = kpm.all(dim=1)
            if all_pad.any():
                kpm = kpm.clone()
                kpm[all_pad, 0] = False

        attn_out, _ = self.cross_attn(Q, zone_emb, zone_emb, key_padding_mask=kpm)
        h = self.norm(Q + attn_out)
        z = self.to_latent(h)                           # [B, K, d_latent]
        return z


class UCDecoder(nn.Module):
    """Decode z_uc [B, K, d_latent] back into logits [B, Z, T, F]."""

    def __init__(
        self,
        n_features: int = N_UC_FEATURES,
        hidden_dim: int = 256,
        latent_dim: int = 128,
        n_zones_max: int = 64,
        n_timesteps_max: int = 48,
        n_heads: int = 4,
        ff_mult: int = 2,
        dropout: float = 0.1,
        scenario_dim: Optional[int] = None,
        n_layers: int = 1,
    ):
        super().__init__()
        self.n_zones_max = n_zones_max
        self.n_timesteps_max = n_timesteps_max
        self.scenario_dim = scenario_dim
        self.n_layers = n_layers

        self.from_latent = nn.Linear(latent_dim, hidden_dim)
        self.zone_emb = nn.Embedding(n_zones_max, hidden_dim)
        self.time_emb = nn.Embedding(n_timesteps_max, hidden_dim)
        # When ``scenario_dim`` is set the decoder gets per-(zone,time) context
        # from the upstream scenario encoder (h_scenario). Without it, zones
        # are only distinguished by their integer index, so the decoder can
        # only reconstruct marginal statistics — leading to over-prediction
        # in low-precision regimes.
        if scenario_dim is not None:
            # 2-layer MLP: linear features alone proved insufficient to map
            # the raw scenario embedding into a UC-discriminative space.
            self.scenario_proj = nn.Sequential(
                nn.Linear(scenario_dim, hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, hidden_dim),
            )
        else:
            self.scenario_proj = None

        # Stack of (cross-attention + FFN) decoder blocks. ``n_layers=1`` keeps
        # the original single-layer behaviour.
        self.cross_attns = nn.ModuleList([
            nn.MultiheadAttention(
                embed_dim=hidden_dim, num_heads=n_heads,
                dropout=dropout, batch_first=True,
            )
            for _ in range(n_layers)
        ])
        self.norms_a = nn.ModuleList([nn.LayerNorm(hidden_dim) for _ in range(n_layers)])
        self.ffs = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_dim, ff_mult * hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(ff_mult * hidden_dim, hidden_dim),
            )
            for _ in range(n_layers)
        ])
        self.norms_b = nn.ModuleList([nn.LayerNorm(hidden_dim) for _ in range(n_layers)])
        self.head = nn.Linear(hidden_dim, n_features)

    def forward(
        self,
        z_uc: torch.Tensor,
        n_zones: int,
        n_timesteps: int,
        h_scenario: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if n_zones > self.n_zones_max:
            raise ValueError(
                f"UCDecoder.n_zones_max={self.n_zones_max} but received n_zones={n_zones}"
            )
        if n_timesteps > self.n_timesteps_max:
            raise ValueError(
                f"UCDecoder.n_timesteps_max={self.n_timesteps_max} but received "
                f"n_timesteps={n_timesteps}"
            )
        B = z_uc.shape[0]
        Z = n_zones
        T = n_timesteps

        kv = self.from_latent(z_uc)                                 # [B, K, H]
        zi = torch.arange(Z, device=z_uc.device)
        ti = torch.arange(T, device=z_uc.device)
        q_zone = self.zone_emb(zi).view(1, Z, 1, -1).expand(B, Z, T, -1)
        q_time = self.time_emb(ti).view(1, 1, T, -1).expand(B, Z, T, -1)
        Q = q_zone + q_time                                         # [B, Z, T, H]
        if self.scenario_proj is not None and h_scenario is not None:
            # Trim/pad h_scenario to the requested (Z, T) window.
            h = h_scenario[:, :Z, :T, :]
            Q = Q + self.scenario_proj(h)
        Q = Q.reshape(B, Z * T, -1)                                 # [B, Z*T, H]

        x = Q
        for attn, norm_a, ff, norm_b in zip(self.cross_attns, self.norms_a, self.ffs, self.norms_b):
            attn_out, _ = attn(x, kv, kv)
            x = norm_a(x + attn_out)
            x = norm_b(x + ff(x))
        logits = self.head(x).view(B, Z, T, -1)                     # [B, Z, T, F]
        return logits


class UCAutoencoder(nn.Module):
    """Wraps :class:`UCEncoder` + :class:`UCDecoder`."""

    def __init__(
        self,
        n_features: int = N_UC_FEATURES,
        hidden_dim: int = 256,
        n_tokens: int = 16,
        latent_dim: int = 128,
        n_heads: int = 4,
        gru_layers: int = 1,
        dropout: float = 0.1,
        n_zones_max: int = 64,
        n_timesteps_max: int = 48,
        scenario_dim: Optional[int] = None,
        decoder_layers: int = 1,
    ):
        super().__init__()
        self.encoder = UCEncoder(
            n_features=n_features,
            hidden_dim=hidden_dim,
            n_tokens=n_tokens,
            latent_dim=latent_dim,
            n_heads=n_heads,
            gru_layers=gru_layers,
            dropout=dropout,
        )
        self.decoder = UCDecoder(
            n_features=n_features,
            hidden_dim=hidden_dim,
            latent_dim=latent_dim,
            n_zones_max=n_zones_max,
            n_timesteps_max=n_timesteps_max,
            n_heads=n_heads,
            dropout=dropout,
            scenario_dim=scenario_dim,
            n_layers=decoder_layers,
        )
        self.n_features = n_features
        self.latent_dim = latent_dim
        self.n_tokens = n_tokens
        self.scenario_dim = scenario_dim

    def encode(self, u_zt: torch.Tensor, zone_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        return self.encoder(u_zt, zone_mask)

    def decode(
        self,
        z_uc: torch.Tensor,
        n_zones: int,
        n_timesteps: int,
        h_scenario: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        return self.decoder(z_uc, n_zones, n_timesteps, h_scenario=h_scenario)

    def forward(
        self,
        u_zt: torch.Tensor,
        zone_mask: Optional[torch.Tensor] = None,
        h_scenario: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        B, Z, T, _ = u_zt.shape
        z = self.encode(u_zt, zone_mask)
        logits = self.decode(z, n_zones=Z, n_timesteps=T, h_scenario=h_scenario)
        return {"z_uc": z, "logits": logits}


# ---------------------------------------------------------------------------
# Losses
# ---------------------------------------------------------------------------

@dataclass
class UCAELossWeights:
    bce: float = 1.0
    temporal: float = 0.05
    consistency: float = 0.1


def _masked_mean(x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Mean over elements where ``mask`` is 1. Mask broadcasts over trailing dims.

    The denominator counts every element of ``x`` that is unmasked (mask
    broadcast to ``x.shape``), so the result is a true element-wise mean.
    """
    while mask.dim() < x.dim():
        mask = mask.unsqueeze(-1)
    mask = mask.to(x.dtype)
    mask_b = mask.expand_as(x)
    denom = mask_b.sum().clamp_min(1.0)
    return (x * mask_b).sum() / denom


def temporal_smoothness_loss(
    prob: torch.Tensor,
    zone_mask: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """L1 between consecutive timestep probabilities. ``prob`` is [B, Z, T, F]."""
    if prob.shape[2] < 2:
        return prob.new_zeros(())
    diff = (prob[:, :, 1:] - prob[:, :, :-1]).abs()  # [B, Z, T-1, F]
    if zone_mask is None:
        return diff.mean()
    return _masked_mean(diff, zone_mask)


def startup_consistency_loss(
    prob: torch.Tensor,
    zone_mask: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Encourage thermal_startup[t] >= thermal_on[t] - thermal_on[t-1].

    Uses MSE between predicted startup probability and ``ReLU(on[t] - on[t-1])``.
    Both feature columns are predictions, so this is a soft self-consistency term
    that pushes the decoder toward physical UC.
    """
    if prob.shape[2] < 2:
        return prob.new_zeros(())
    on = prob[..., FEAT_THERMAL]            # [B, Z, T]
    su = prob[..., FEAT_THERMAL_SU]         # [B, Z, T]
    target_su = F.relu(on[:, :, 1:] - on[:, :, :-1])
    err = (su[:, :, 1:] - target_su).pow(2)  # [B, Z, T-1]
    # Also require su[:, :, 0] ~ on[:, :, 0] (cold start).
    err0 = (su[:, :, :1] - on[:, :, :1]).pow(2)
    err = torch.cat([err0, err], dim=2)      # [B, Z, T]
    if zone_mask is None:
        return err.mean()
    return _masked_mean(err, zone_mask)


def uc_autoencoder_loss(
    logits: torch.Tensor,
    u_star: torch.Tensor,
    zone_mask: Optional[torch.Tensor] = None,
    weights: UCAELossWeights = UCAELossWeights(),
    pos_weight: Optional[torch.Tensor] = None,
) -> Dict[str, torch.Tensor]:
    """Compute the masked composite UC autoencoder loss.

    Args:
        logits: [B, Z, T, F] raw decoder output.
        u_star: [B, Z, T, F] ground-truth UC binaries (float in {0, 1}).
        zone_mask: [B, Z] with 1 for valid zones.
        weights: term weights.
        pos_weight: optional ``[F]`` tensor of per-feature positive-class weights
            for ``BCEWithLogits``. Use this to counter class imbalance: a value
            of ``(1 - p) / p`` rebalances feature ``f`` whose true positive rate
            is ``p``. When ``None``, ordinary BCE is used (equivalent to ``1``).

    Returns: dict with ``loss``, ``bce``, ``temporal``, ``consistency``,
        ``hamming`` (mean elementwise disagreement on the valid zones at 0.5
        threshold).
    """
    bce_elem = F.binary_cross_entropy_with_logits(
        logits, u_star.float(),
        pos_weight=pos_weight.to(logits.device) if pos_weight is not None else None,
        reduction="none",
    )  # [B, Z, T, F]
    if zone_mask is None:
        bce = bce_elem.mean()
    else:
        bce = _masked_mean(bce_elem, zone_mask)

    prob = torch.sigmoid(logits)
    l_temp = temporal_smoothness_loss(prob, zone_mask)
    l_cons = startup_consistency_loss(prob, zone_mask)

    total = (
        weights.bce * bce
        + weights.temporal * l_temp
        + weights.consistency * l_cons
    )

    # Hamming as a quick read-out (no gradient).
    with torch.no_grad():
        pred_bin = (prob >= 0.5).float()
        diff = (pred_bin - u_star.float()).abs()
        if zone_mask is None:
            hamming = diff.mean()
        else:
            hamming = _masked_mean(diff, zone_mask)

    return {
        "loss": total,
        "bce": bce.detach(),
        "temporal": l_temp.detach(),
        "consistency": l_cons.detach(),
        "hamming": hamming,
    }
