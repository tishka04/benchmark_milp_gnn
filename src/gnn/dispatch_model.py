# ==============================================================================
# DISPATCH GNN - Predict Continuous Dispatch from EBM Binaries + HTE Embeddings
# ==============================================================================
# Graph Neural Network that replaces the Decoder + LP Worker stages.
#
# Architecture:
#   1. Input projection: concat(u_zt [Z,T,7], h_zt [Z,T,D]) → [Z,T,H]
#   2. Temporal GRU: bidirectional per-zone temporal encoding
#   3. Zone GAT layers: multi-head attention across zones (spatial GNN)
#   4. Output MLP: predicts 11 continuous dispatch channels per (zone, timestep)
#
# Dispatch channels (11):
#   0: thermal        5: battery_discharge   10: hydro_release
#   1: nuclear        6: pumped_charge
#   2: solar          7: pumped_discharge
#   3: wind           8: demand_response
#   4: battery_charge 9: unserved
# ==============================================================================

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F_act
from typing import Optional


DISPATCH_CHANNELS = [
    'thermal',           # 0
    'nuclear',           # 1
    'solar',             # 2
    'wind',              # 3
    'battery_charge',    # 4
    'battery_discharge', # 5
    'pumped_charge',     # 6
    'pumped_discharge',  # 7
    'demand_response',   # 8
    'unserved',          # 9
    'hydro_release',     # 10
]
N_DISPATCH = len(DISPATCH_CHANNELS)


class ZoneGATLayer(nn.Module):
    """
    Graph Attention layer over zones (spatial message passing).

    Uses multi-head attention where each zone attends to all other zones
    in the same scenario, with masking for padded zones. Equivalent to
    GAT on a fully-connected zone graph.
    """

    def __init__(self, hidden_dim: int, n_heads: int = 4, dropout: float = 0.1):
        super().__init__()
        self.mha = nn.MultiheadAttention(
            hidden_dim, n_heads, dropout=dropout, batch_first=True,
        )
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.Dropout(dropout),
        )

    def forward(
        self,
        x: torch.Tensor,
        key_padding_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            x: [B, Z, H] zone features
            key_padding_mask: [B, Z] True for PADDED zones (PyTorch convention)
        Returns:
            [B, Z, H]
        """
        attn_out, _ = self.mha(x, x, x, key_padding_mask=key_padding_mask)
        x = self.norm1(x + attn_out)
        x = self.norm2(x + self.ffn(x))
        return x


class DispatchGNN(nn.Module):
    """
    GNN for predicting continuous dispatch from EBM binaries + HTE embeddings.

    Input:
        u_zt: [B, Z, T, F] - EBM binary decisions (F=7)
        h_zt: [B, Z, T, D] - HTE zone-level temporal embeddings (D=128)
        zone_mask: [B, Z]   - 1 for valid zones, 0 for padding

    Output:
        dispatch: [B, Z, T, C] - continuous dispatch predictions (C=11)
    """

    def __init__(
        self,
        n_binary_features: int = 7,
        embed_dim: int = 128,
        hidden_dim: int = 256,
        n_dispatch: int = N_DISPATCH,
        gru_layers: int = 2,
        n_gat_layers: int = 3,
        n_heads: int = 8,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.n_dispatch = n_dispatch

        input_dim = n_binary_features + embed_dim

        # Input projection
        self.input_proj = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        # Temporal GRU (per zone)
        self.temporal_gru = nn.GRU(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            num_layers=gru_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if gru_layers > 1 else 0,
        )
        self.gru_proj = nn.Linear(hidden_dim * 2, hidden_dim)

        # Spatial GAT layers (zone interaction)
        self.gat_layers = nn.ModuleList([
            ZoneGATLayer(hidden_dim, n_heads, dropout)
            for _ in range(n_gat_layers)
        ])

        # Output head
        self.output_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Linear(hidden_dim // 2, n_dispatch),
        )

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight, gain=1.0)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(
        self,
        u_zt: torch.Tensor,
        h_zt: torch.Tensor,
        zone_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            u_zt: [B, Z, T, F] binary decisions
            h_zt: [B, Z, T, D] embeddings
            zone_mask: [B, Z] valid zone indicator (1=valid, 0=pad)

        Returns:
            dispatch: [B, Z, T, C] predicted continuous dispatch (non-negative)
        """
        B, Z, T, F = u_zt.shape
        D = h_zt.shape[-1]

        # Sanitize inputs
        u_zt = torch.nan_to_num(u_zt, nan=0.0)
        h_zt = torch.nan_to_num(h_zt, nan=0.0)

        # ── 1. Input projection ──
        x = torch.cat([u_zt, h_zt], dim=-1)        # [B, Z, T, F+D]
        x = x.view(B * Z, T, F + D)
        x = self.input_proj(x)                       # [B*Z, T, H]

        # ── 2. Temporal GRU ──
        gru_out, _ = self.temporal_gru(x)            # [B*Z, T, 2H]
        x = self.gru_proj(gru_out)                   # [B*Z, T, H]
        x = x.view(B, Z, T, self.hidden_dim)

        # ── 3. Spatial GAT: process zones at each timestep ──
        # Reshape: [B, Z, T, H] → [B*T, Z, H]
        x = x.permute(0, 2, 1, 3).reshape(B * T, Z, self.hidden_dim)

        # Build key_padding_mask: [B*T, Z], True for padded
        if zone_mask is not None:
            pad_mask = (zone_mask == 0)                          # [B, Z]
            pad_mask = pad_mask.unsqueeze(1).expand(B, T, Z)     # [B, T, Z]
            pad_mask = pad_mask.reshape(B * T, Z)
        else:
            pad_mask = None

        for gat in self.gat_layers:
            x = gat(x, key_padding_mask=pad_mask)

        # Reshape back: [B*T, Z, H] → [B, Z, T, H]
        x = x.view(B, T, Z, self.hidden_dim).permute(0, 2, 1, 3)

        # ── 4. Output head ──
        dispatch = self.output_head(x)               # [B, Z, T, C]

        # All dispatch values are non-negative (MW)
        dispatch = F_act.relu(dispatch)

        # Zero out padded zones
        if zone_mask is not None:
            mask = zone_mask.view(B, Z, 1, 1).float()
            dispatch = dispatch * mask

        return dispatch
