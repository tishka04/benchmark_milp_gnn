"""Phase 2 JEPA predictor: ``h_scenario [B, Z, T, D] -> latent + UC heads``.

Shared trunk:

    h_scenario [B, Z, T, D_scn]
        |- per-zone Linear(D_scn -> H) + BiGRU over T
        |- per-(zone,time) hidden [B, Z, T, H]
        |- zone summary [B, Z, H]
        |- system-time summary [B, T, H]

Heads:
    - latent head:      [B, Z, H] -> z_pred [B, K, d_latent]
    - storage head:     [B, Z, T, H] -> 4 logits
    - dr head:          [B, Z, T, H] -> 1 logit
    - thermal_on head:  [B, Z, T, H] -> 1 logit
    - startup head:     derived from thermal_on by default
    - import head:      [B, T, H]    -> 1 system-level logit, broadcast over Z

This lets JEPA learn a latent strategy while also receiving direct supervision
on the rare UC decisions that matter most for downstream MILP repair.
"""
from __future__ import annotations

from typing import Dict, Optional

import torch
import torch.nn as nn


class ScenarioToUCLatent(nn.Module):
    """Predict the AE latent ``z_uc`` and optional direct UC auxiliary heads."""

    def __init__(
        self,
        scenario_dim: int = 128,
        hidden_dim: int = 384,
        n_tokens: int = 32,
        latent_dim: int = 192,
        n_heads: int = 6,
        gru_layers: int = 1,
        dropout: float = 0.1,
        residual_latent: bool = False,
        aux_heads: bool = False,
        derive_startup_from_thermal: bool = False,
        import_head_global: bool = False,
    ):
        super().__init__()
        self.scenario_dim = scenario_dim
        self.hidden_dim = hidden_dim
        self.n_tokens = n_tokens
        self.latent_dim = latent_dim
        self.residual_latent = residual_latent
        self.aux_heads = aux_heads
        self.derive_startup_from_thermal = derive_startup_from_thermal
        self.import_head_global = import_head_global

        self.input_proj = nn.Linear(scenario_dim, hidden_dim)
        self.gru = nn.GRU(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            num_layers=gru_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if gru_layers > 1 else 0.0,
        )

        # Shared context projections.
        self.time_proj = nn.Linear(2 * hidden_dim, hidden_dim)
        self.zone_proj = nn.Linear(2 * hidden_dim, hidden_dim)
        self.time_norm = nn.LayerNorm(hidden_dim)
        self.time_dropout = nn.Dropout(dropout)

        # Latent strategy head.
        self.queries = nn.Parameter(torch.randn(n_tokens, hidden_dim) * 0.02)
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=n_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.norm = nn.LayerNorm(hidden_dim)
        self.to_latent = nn.Linear(hidden_dim, latent_dim)
        if residual_latent:
            self.base_latent = nn.Parameter(torch.randn(n_tokens, latent_dim) * 0.02)
        else:
            self.base_latent = None

        # Optional auxiliary heads.
        if aux_heads:
            self.storage_head = nn.Linear(hidden_dim, 4)
            self.dr_head = nn.Linear(hidden_dim, 1)
            self.thermal_on_head = nn.Linear(hidden_dim, 1)
            self.startup_head = None if derive_startup_from_thermal else nn.Linear(hidden_dim, 1)
            if import_head_global:
                self.import_head = nn.Linear(hidden_dim, 1)
            else:
                self.import_head = nn.Linear(hidden_dim, 1)
        else:
            self.storage_head = None
            self.dr_head = None
            self.thermal_on_head = None
            self.startup_head = None
            self.import_head = None

    def _key_padding_mask(self, zone_mask: Optional[torch.Tensor]) -> Optional[torch.Tensor]:
        if zone_mask is None:
            return None
        kpm = zone_mask == 0
        all_pad = kpm.all(dim=1)
        if all_pad.any():
            kpm = kpm.clone()
            kpm[all_pad, 0] = False
        return kpm

    def _masked_zone_mean(
        self,
        x: torch.Tensor,
        zone_mask: Optional[torch.Tensor],
    ) -> torch.Tensor:
        if zone_mask is None:
            return x.mean(dim=1)
        mask = zone_mask.to(x.dtype).unsqueeze(-1).unsqueeze(-1)
        denom = mask.sum(dim=1).clamp_min(1.0)
        return (x * mask).sum(dim=1) / denom

    def _derive_startup_logits(self, thermal_on_logits: torch.Tensor) -> torch.Tensor:
        prob_on = torch.sigmoid(thermal_on_logits)
        startup_prob = torch.zeros_like(prob_on)
        startup_prob[:, :, :1, :] = prob_on[:, :, :1, :]
        if prob_on.shape[2] > 1:
            startup_prob[:, :, 1:, :] = (prob_on[:, :, 1:, :] - prob_on[:, :, :-1, :]).clamp_min(0.0)
        startup_prob = startup_prob.clamp(1e-4, 1.0 - 1e-4)
        return torch.logit(startup_prob)

    def _encode_context(
        self,
        h_scenario: torch.Tensor,
        zone_mask: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        B, Z, T, _ = h_scenario.shape
        x = self.input_proj(h_scenario).reshape(B * Z, T, self.hidden_dim)
        gru_out, _ = self.gru(x)                                        # [B*Z, T, 2H]
        time_hidden = self.time_proj(gru_out).view(B, Z, T, self.hidden_dim)
        zone_hidden_raw = gru_out.mean(dim=1).view(B, Z, -1)            # [B, Z, 2H]
        zone_hidden = self.zone_proj(zone_hidden_raw)                   # [B, Z, H]
        time_hidden = self.time_norm(time_hidden + zone_hidden.unsqueeze(2))
        time_hidden = self.time_dropout(time_hidden)
        system_time_hidden = self._masked_zone_mean(time_hidden, zone_mask)  # [B, T, H]
        return {
            "time_hidden": time_hidden,
            "zone_hidden": zone_hidden,
            "system_time_hidden": system_time_hidden,
        }

    def _predict_latent(
        self,
        zone_hidden: torch.Tensor,
        zone_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        B = zone_hidden.shape[0]
        Q = self.queries.unsqueeze(0).expand(B, -1, -1)
        attn_out, _ = self.cross_attn(
            Q,
            zone_hidden,
            zone_hidden,
            key_padding_mask=self._key_padding_mask(zone_mask),
        )
        h = self.norm(Q + attn_out)
        z = self.to_latent(h)
        if self.base_latent is not None:
            z = z + self.base_latent.unsqueeze(0)
        return z

    def _predict_aux(
        self,
        time_hidden: torch.Tensor,
        system_time_hidden: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        B, Z, T, _ = time_hidden.shape
        storage_logits = self.storage_head(time_hidden)
        dr_logits = self.dr_head(time_hidden)
        thermal_on_logits = self.thermal_on_head(time_hidden)
        if self.startup_head is not None:
            startup_logits = self.startup_head(time_hidden)
        elif self.derive_startup_from_thermal:
            startup_logits = self._derive_startup_logits(thermal_on_logits)
        else:
            startup_logits = torch.zeros_like(thermal_on_logits)

        import_logits_t = self.import_head(system_time_hidden)          # [B, T, 1]
        import_logits = import_logits_t.unsqueeze(1).expand(B, Z, T, 1)

        aux_logits = torch.cat(
            [storage_logits, dr_logits, startup_logits, thermal_on_logits, import_logits],
            dim=-1,
        )
        return {
            "aux_logits": aux_logits,
            "storage_logits": storage_logits,
            "dr_logits": dr_logits,
            "thermal_on_logits": thermal_on_logits,
            "startup_logits": startup_logits,
            "import_logits_time": import_logits_t.squeeze(-1),
        }

    def forward(
        self,
        h_scenario: torch.Tensor,
        zone_mask: Optional[torch.Tensor] = None,
    ) -> Dict[str, Optional[torch.Tensor]]:
        """Return a dict with ``z_pred`` and optional direct UC auxiliary heads."""
        ctx = self._encode_context(h_scenario, zone_mask=zone_mask)
        z_pred = self._predict_latent(ctx["zone_hidden"], zone_mask=zone_mask)

        out: Dict[str, Optional[torch.Tensor]] = {
            "z_pred": z_pred,
            "aux_logits": None,
            "storage_logits": None,
            "dr_logits": None,
            "thermal_on_logits": None,
            "startup_logits": None,
            "import_logits_time": None,
        }
        if self.aux_heads:
            out.update(self._predict_aux(ctx["time_hidden"], ctx["system_time_hidden"]))
        return out
