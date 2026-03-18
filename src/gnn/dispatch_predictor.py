# ==============================================================================
# GNN DISPATCH PREDICTOR - Drop-in replacement for Decoder + LP Worker
# ==============================================================================
# Loads a trained DispatchGNN and predicts continuous dispatch directly
# from EBM binary candidates + HTE embeddings, bypassing the feasibility
# decoder and LP worker stages entirely.
#
# Usage in pipeline:
#   predictor = GNNDispatchPredictor(model_path, device)
#   dispatch_dict = predictor.predict(u_bin, h_zt, zone_mask, zone_names)
# ==============================================================================

from __future__ import annotations

import torch
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional
from dataclasses import dataclass, field


@dataclass
class GNNDispatchResult:
    """Result from GNN dispatch prediction (mirrors TwoStageResult interface)."""
    scenario_id: str
    status: str = "optimal"
    stage_used: str = "gnn_dispatch"
    objective_value: float = 0.0
    solve_time: float = 0.0
    continuous_vars: Dict[str, np.ndarray] = field(default_factory=dict)

    # Compatibility fields
    slack_used: float = 0.0
    n_flips: int = 0
    warm_started: bool = False


class GNNDispatchPredictor:
    """
    Drop-in replacement for Decoder + LP Worker using trained DispatchGNN.

    Predicts continuous dispatch [Z, T, 11] from EBM binaries [Z, T, 7]
    and HTE embeddings [Z, T, D] in a single forward pass.
    """

    def __init__(
        self,
        model_path: str,
        device: str = 'cuda',
    ):
        from src.gnn.dispatch_model import DispatchGNN, DISPATCH_CHANNELS, N_DISPATCH

        self.device = device
        self.dispatch_channels = DISPATCH_CHANNELS
        self.n_dispatch = N_DISPATCH

        # Load checkpoint
        checkpoint = torch.load(model_path, map_location=device, weights_only=False)
        config = checkpoint.get("config", {})

        self.model = DispatchGNN(
            n_binary_features=config.get("n_binary_features", 7),
            embed_dim=config.get("embed_dim", 128),
            hidden_dim=config.get("hidden_dim", 256),
            n_dispatch=config.get("n_dispatch", N_DISPATCH),
            gru_layers=config.get("gru_layers", 2),
            n_gat_layers=config.get("n_gat_layers", 3),
            n_heads=config.get("n_heads", 8),
            dropout=0.0,  # no dropout at inference
        ).to(device)

        if "model_state_dict" in checkpoint:
            self.model.load_state_dict(checkpoint["model_state_dict"])
        else:
            self.model.load_state_dict(checkpoint)
        self.model.eval()

        n_params = sum(p.numel() for p in self.model.parameters())
        print(f"GNNDispatchPredictor loaded: {n_params:,} params from {model_path}")

    @torch.no_grad()
    def predict(
        self,
        u_bin: torch.Tensor,
        h_zt: torch.Tensor,
        zone_mask: Optional[torch.Tensor] = None,
        zone_names: Optional[List[str]] = None,
        scenario_id: str = "",
    ) -> GNNDispatchResult:
        """
        Predict continuous dispatch from EBM binaries + HTE embeddings.

        Args:
            u_bin:     [Z, T, 7] or [1, Z, T, 7] binary decisions
            h_zt:      [Z, T, D] or [1, Z, T, D] zone embeddings
            zone_mask: [Z] or [1, Z] valid zone indicator
            zone_names: list of zone name strings
            scenario_id: scenario identifier

        Returns:
            GNNDispatchResult with dispatch dict keyed by channel name
        """
        import time
        t0 = time.perf_counter()

        # Ensure batch dim
        if u_bin.dim() == 3:
            u_bin = u_bin.unsqueeze(0)
        if h_zt.dim() == 3:
            h_zt = h_zt.unsqueeze(0)

        u_bin = u_bin.to(self.device)
        h_zt = h_zt.to(self.device)

        B, Z, T, F = u_bin.shape

        if zone_mask is None:
            zone_mask = torch.ones(B, Z, device=self.device)
        elif zone_mask.dim() == 1:
            zone_mask = zone_mask.unsqueeze(0)
        zone_mask = zone_mask.to(self.device)

        # Forward pass
        dispatch = self.model(u_bin, h_zt, zone_mask)  # [1, Z, T, C]
        dispatch = dispatch.squeeze(0).cpu()             # [Z, T, C]

        elapsed = time.perf_counter() - t0

        # Build dispatch dict (zone_name -> list of T values per channel)
        if zone_names is None:
            zone_names = [f"Z{z}" for z in range(Z)]

        dispatch_dict = {}
        for c_idx, channel in enumerate(self.dispatch_channels):
            channel_dict = {}
            for z_idx, zname in enumerate(zone_names):
                channel_dict[zname] = dispatch[z_idx, :, c_idx].numpy().tolist()
            dispatch_dict[channel] = channel_dict

        # Build continuous_vars as numpy arrays [Z, T] for compatibility
        continuous_vars = {}
        for c_idx, channel in enumerate(self.dispatch_channels):
            continuous_vars[channel] = dispatch[:, :, c_idx].numpy()

        # Estimate objective (sum of dispatch costs - rough approximation)
        # In practice, the user should compute the true objective from the dispatch
        objective_approx = 0.0

        return GNNDispatchResult(
            scenario_id=scenario_id,
            status="optimal",
            stage_used="gnn_dispatch",
            objective_value=objective_approx,
            solve_time=elapsed,
            continuous_vars=continuous_vars,
        )

    @torch.no_grad()
    def predict_batch(
        self,
        u_bin: torch.Tensor,
        h_zt: torch.Tensor,
        zone_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Batch prediction (no result wrapping, just raw dispatch tensor).

        Args:
            u_bin:     [B, Z, T, 7]
            h_zt:      [B, Z, T, D]
            zone_mask: [B, Z]

        Returns:
            dispatch: [B, Z, T, C] continuous dispatch predictions
        """
        u_bin = u_bin.to(self.device)
        h_zt = h_zt.to(self.device)
        if zone_mask is not None:
            zone_mask = zone_mask.to(self.device)
        return self.model(u_bin, h_zt, zone_mask).cpu()
