# ==============================================================================
# CONDITIONING AND FEATURE EXTRACTION FOR PREFERENCE LEARNING
# ==============================================================================
# HConditioner: Attention-based pooling of h_raw (zone-level embeddings)
# DecisionFeatureExtractor: Aggregate features from decisions for cost proxy
# ==============================================================================

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, Dict
from dataclasses import dataclass


class HConditioner(nn.Module):
    """
    Attention-based conditioning from zone-level embeddings.
    
    Converts h_raw [B, Z, T, D] or h_z [B, Z, D] to h_global [B, D]
    using learned attention over zones with proper masking.
    """
    
    def __init__(
        self,
        embed_dim: int,
        hidden_dim: int = 64,
        use_temporal: bool = True,
        n_heads: int = 4,
    ):
        """
        Args:
            embed_dim: Dimension D of embeddings
            hidden_dim: Hidden dimension for attention MLP
            use_temporal: If True, also aggregate over time
            n_heads: Number of attention heads (if using multi-head)
        """
        super().__init__()
        self.embed_dim = embed_dim
        self.use_temporal = use_temporal
        
        # Temporal aggregation (if needed)
        if use_temporal:
            self.temporal_pool = nn.Sequential(
                nn.Linear(embed_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, embed_dim),
            )
        
        # Zone attention
        self.zone_attention = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )
        
        # Output projection
        self.output_proj = nn.Linear(embed_dim, embed_dim)
    
    def forward(
        self,
        h_raw: torch.Tensor,
        zone_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Aggregate zone-level embeddings to global embedding.
        
        Args:
            h_raw: [B, Z, T, D] or [B, Z, D] zone embeddings
            zone_mask: [B, Z] boolean mask (True = valid zone)
        
        Returns:
            h_global: [B, D] global embedding
        """
        if h_raw.dim() == 4:
            # [B, Z, T, D] -> [B, Z, D] via temporal pooling
            B, Z, T, D = h_raw.shape
            
            if self.use_temporal:
                # Learnable temporal aggregation
                h_temporal = self.temporal_pool(h_raw)  # [B, Z, T, D]
                h_z = h_temporal.mean(dim=2)  # [B, Z, D]
            else:
                h_z = h_raw.mean(dim=2)  # [B, Z, D]
        else:
            # Already [B, Z, D]
            h_z = h_raw
            B, Z, D = h_z.shape
        
        # Compute attention logits
        logits = self.zone_attention(h_z)  # [B, Z, 1]
        
        # Apply zone mask
        if zone_mask is not None:
            # zone_mask: [B, Z] -> [B, Z, 1]
            mask = zone_mask.unsqueeze(-1)
            logits = logits.masked_fill(~mask.bool(), float('-inf'))
        
        # Softmax attention
        att = F.softmax(logits, dim=1)  # [B, Z, 1]
        
        # Weighted sum
        h_global = (att * h_z).sum(dim=1)  # [B, D]
        
        # Output projection
        h_global = self.output_proj(h_global)
        
        return h_global
    
    def forward_with_attention(
        self,
        h_raw: torch.Tensor,
        zone_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward with attention weights for visualization."""
        if h_raw.dim() == 4:
            B, Z, T, D = h_raw.shape
            if self.use_temporal:
                h_temporal = self.temporal_pool(h_raw)
                h_z = h_temporal.mean(dim=2)
            else:
                h_z = h_raw.mean(dim=2)
        else:
            h_z = h_raw
        
        logits = self.zone_attention(h_z)
        
        if zone_mask is not None:
            mask = zone_mask.unsqueeze(-1)
            logits = logits.masked_fill(~mask.bool(), float('-inf'))
        
        att = F.softmax(logits, dim=1)
        h_global = (att * h_z).sum(dim=1)
        h_global = self.output_proj(h_global)
        
        return h_global, att.squeeze(-1)


@dataclass
class FeatureConfig:
    """Configuration for decision feature extraction."""
    # Feature indices in decision tensor [Z, T, F]
    idx_battery_charge: int = 0
    idx_battery_discharge: int = 1
    idx_pumped_charge: int = 2
    idx_pumped_discharge: int = 3
    idx_dr_active: int = 4
    idx_nuclear: int = 5
    idx_thermal: int = 6
    idx_import_mode: int = 7
    
    # Number of timesteps for late-hour features
    late_hours: int = 6
    
    # Whether to include cross-zone features
    use_cross_zone: bool = True


class DecisionFeatureExtractor(nn.Module):
    """
    Extract aggregated features from decision tensors for cost proxy.
    
    Produces ~30-50 informative features instead of flat Z×T×F vector.
    Features are designed to capture "cost-sensitive" patterns:
    - Toggle counts (startup costs)
    - Thermal usage (expensive generation)
    - DR activations
    - Simultaneous charge/discharge (infeasibility indicator)
    - Late-hour patterns (tension/ramping)
    """
    
    def __init__(self, config: Optional[FeatureConfig] = None):
        super().__init__()
        self.config = config or FeatureConfig()
        self._n_features = None
    
    @property
    def n_features(self) -> int:
        """Number of output features."""
        if self._n_features is None:
            # Compute from a dummy forward pass
            dummy_u = torch.zeros(1, 10, 24, 8)
            dummy_mask = torch.ones(1, 10)
            features = self._extract(dummy_u, dummy_mask)
            self._n_features = features.shape[-1]
        return self._n_features
    
    def forward(
        self,
        u: torch.Tensor,
        zone_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Extract features from decisions.
        
        Args:
            u: [B, Z, T, F] or [K, Z, T, F] decisions
            zone_mask: [B, Z] or [K, Z] valid zone mask
        
        Returns:
            features: [B, P] or [K, P] with P ~ 30-50
        """
        return self._extract(u, zone_mask)
    
    def _extract(
        self,
        u: torch.Tensor,
        zone_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Extract all features."""
        cfg = self.config
        
        if u.dim() == 3:
            # [Z, T, F] -> [1, Z, T, F]
            u = u.unsqueeze(0)
            if zone_mask is not None and zone_mask.dim() == 1:
                zone_mask = zone_mask.unsqueeze(0)
        
        B, Z, T, F = u.shape
        device = u.device
        
        # Create default mask if not provided
        if zone_mask is None:
            zone_mask = torch.ones(B, Z, device=device)
        
        # Expand mask for broadcasting
        mask_zt = zone_mask.unsqueeze(-1)  # [B, Z, 1]
        
        features = []
        
        # === TOGGLE FEATURES (startup/shutdown costs) ===
        # Toggles per feature
        u_shifted = torch.cat([u[:, :, :1, :], u[:, :, :-1, :]], dim=2)
        toggles = (u - u_shifted).abs()  # [B, Z, T, F]
        
        # Total toggles per feature (masked)
        for f_idx in range(min(F, 8)):
            toggle_f = (toggles[:, :, :, f_idx] * zone_mask.unsqueeze(-1)).sum(dim=(1, 2))
            features.append(toggle_f)
        
        # Total toggles overall
        total_toggles = (toggles * mask_zt.unsqueeze(-1)).sum(dim=(1, 2, 3))
        features.append(total_toggles)
        
        # === ON-STATE FEATURES (utilization) ===
        # Mean ON per feature
        valid_count = (zone_mask.sum(dim=1, keepdim=True) * T).clamp(min=1)
        for f_idx in range(min(F, 8)):
            on_f = (u[:, :, :, f_idx] * zone_mask.unsqueeze(-1)).sum(dim=(1, 2)) / valid_count.squeeze(-1)
            features.append(on_f)
        
        # === THERMAL FEATURES (expensive) ===
        thermal = u[:, :, :, cfg.idx_thermal]  # [B, Z, T]
        thermal_masked = thermal * zone_mask.unsqueeze(-1)
        
        # Total thermal ON
        thermal_total = thermal_masked.sum(dim=(1, 2))
        features.append(thermal_total)
        
        # Max thermal per timestep
        thermal_per_t = thermal_masked.sum(dim=1)  # [B, T]
        thermal_max = thermal_per_t.max(dim=1)[0]
        features.append(thermal_max)
        
        # Thermal in late hours
        late_h = cfg.late_hours
        thermal_late = thermal_masked[:, :, -late_h:].sum(dim=(1, 2))
        features.append(thermal_late)
        
        # === DR FEATURES ===
        dr = u[:, :, :, cfg.idx_dr_active]
        dr_masked = dr * zone_mask.unsqueeze(-1)
        
        # Total DR activations
        dr_total = dr_masked.sum(dim=(1, 2))
        features.append(dr_total)
        
        # DR in late hours
        dr_late = dr_masked[:, :, -late_h:].sum(dim=(1, 2))
        features.append(dr_late)
        
        # === STORAGE FEATURES ===
        # Battery
        batt_chg = u[:, :, :, cfg.idx_battery_charge]
        batt_dis = u[:, :, :, cfg.idx_battery_discharge]
        
        # Simultaneous charge/discharge (infeasibility indicator)
        simul_batt = (batt_chg * batt_dis * zone_mask.unsqueeze(-1)).sum(dim=(1, 2))
        features.append(simul_batt)
        
        # Net battery flow
        batt_net = ((batt_dis - batt_chg) * zone_mask.unsqueeze(-1)).sum(dim=(1, 2))
        features.append(batt_net)
        
        # Pumped storage
        pump_chg = u[:, :, :, cfg.idx_pumped_charge]
        pump_dis = u[:, :, :, cfg.idx_pumped_discharge]
        
        simul_pump = (pump_chg * pump_dis * zone_mask.unsqueeze(-1)).sum(dim=(1, 2))
        features.append(simul_pump)
        
        pump_net = ((pump_dis - pump_chg) * zone_mask.unsqueeze(-1)).sum(dim=(1, 2))
        features.append(pump_net)
        
        # === NUCLEAR FEATURES ===
        nuclear = u[:, :, :, cfg.idx_nuclear]
        nuclear_masked = nuclear * zone_mask.unsqueeze(-1)
        
        nuclear_total = nuclear_masked.sum(dim=(1, 2))
        features.append(nuclear_total)
        
        nuclear_toggles = (toggles[:, :, :, cfg.idx_nuclear] * zone_mask.unsqueeze(-1)).sum(dim=(1, 2))
        features.append(nuclear_toggles)
        
        # === IMPORT FEATURES ===
        imports = u[:, :, :, cfg.idx_import_mode]
        import_masked = imports * zone_mask.unsqueeze(-1)
        
        import_total = import_masked.sum(dim=(1, 2))
        features.append(import_total)
        
        # === TEMPORAL PROFILE FEATURES ===
        # Peak hour (max total ON across zones)
        total_on_per_t = (u.sum(dim=-1) * zone_mask.unsqueeze(-1)).sum(dim=1)  # [B, T]
        peak_t = total_on_per_t.max(dim=1)[0]
        features.append(peak_t)
        
        # Valley hour (min)
        valley_t = total_on_per_t.min(dim=1)[0]
        features.append(valley_t)
        
        # Variance (ramping indicator)
        var_t = total_on_per_t.var(dim=1)
        features.append(var_t)
        
        # Late hour tension (sum in last hours / total)
        late_ratio = total_on_per_t[:, -late_h:].sum(dim=1) / (total_on_per_t.sum(dim=1) + 1e-6)
        features.append(late_ratio)
        
        # === CROSS-ZONE FEATURES ===
        if cfg.use_cross_zone:
            # Zone diversity (how many zones are active)
            zone_active = (u.sum(dim=(2, 3)) > 0).float()  # [B, Z]
            n_active_zones = (zone_active * zone_mask).sum(dim=1)
            features.append(n_active_zones)
            
            # Correlation between zones (simplified)
            # High correlation = synchronized behavior
            if Z > 1:
                u_flat_z = u.sum(dim=-1)  # [B, Z, T]
                u_mean = u_flat_z.mean(dim=1, keepdim=True)  # [B, 1, T]
                zone_corr = ((u_flat_z - u_mean).abs() * zone_mask.unsqueeze(-1)).mean(dim=(1, 2))
                features.append(zone_corr)
            else:
                features.append(torch.zeros(B, device=device))
        
        # Stack all features
        features = torch.stack(features, dim=-1)  # [B, P]
        
        return features


class ConditionedEBMWrapper(nn.Module):
    """
    Wrapper that combines HConditioner with base EBM.
    
    Provides clean API: E(u, h_raw, zone_mask) instead of E(u_flat, h_flat).
    """
    
    def __init__(
        self,
        base_ebm: nn.Module,
        conditioner: HConditioner,
        max_zones: int,
        n_timesteps: int,
        n_features: int,
    ):
        super().__init__()
        self.base_ebm = base_ebm
        self.conditioner = conditioner
        self.max_zones = max_zones
        self.n_timesteps = n_timesteps
        self.n_features = n_features
        self.max_decision_dim = max_zones * n_timesteps * n_features
    
    def forward(
        self,
        u: torch.Tensor,
        h_raw: torch.Tensor,
        zone_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Compute energy E(u | h).
        
        Args:
            u: [B, Z, T, F] decisions or [B, D] flattened
            h_raw: [B, Z, T, D] or [B, Z, D] zone embeddings
            zone_mask: [B, Z] valid zone mask
        
        Returns:
            energy: [B] scalar energies
        """
        # Flatten u if needed
        if u.dim() == 4:
            B = u.shape[0]
            u_flat = u.view(B, -1)
            # Pad to max size
            if u_flat.shape[-1] < self.max_decision_dim:
                pad = self.max_decision_dim - u_flat.shape[-1]
                u_flat = F.pad(u_flat, (0, pad))
        else:
            u_flat = u
        
        # Get global conditioning
        h_global = self.conditioner(h_raw, zone_mask)
        
        # Compute energy
        return self.base_ebm(u_flat, h_global)
    
    def forward_with_attention(
        self,
        u: torch.Tensor,
        h_raw: torch.Tensor,
        zone_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward with attention weights for visualization."""
        if u.dim() == 4:
            B = u.shape[0]
            u_flat = u.view(B, -1)
            if u_flat.shape[-1] < self.max_decision_dim:
                pad = self.max_decision_dim - u_flat.shape[-1]
                u_flat = F.pad(u_flat, (0, pad))
        else:
            u_flat = u
        
        h_global, attention = self.conditioner.forward_with_attention(h_raw, zone_mask)
        energy = self.base_ebm(u_flat, h_global)
        
        return energy, attention


class FeatureBasedCostProxy(nn.Module):
    """
    Cost proxy using aggregated features instead of flat decision vector.
    
    Much smaller and more interpretable than the previous CostProxy.
    Input: ~50 features + h_global (~128 dims) = ~180 dims total
    Instead of: 129 * 24 * 8 + 128 = ~25000 dims
    """
    
    def __init__(
        self,
        feature_dim: int,
        embed_dim: int,
        hidden_dim: int = 128,
    ):
        super().__init__()
        self.feature_extractor = DecisionFeatureExtractor()
        self.feature_dim = feature_dim or self.feature_extractor.n_features
        
        input_dim = self.feature_dim + embed_dim
        
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),
        )
        
        # Running statistics for normalization
        self.register_buffer('cost_mean', torch.tensor(0.0))
        self.register_buffer('cost_std', torch.tensor(1.0))
        self.n_updates = 0
    
    def forward(
        self,
        u: torch.Tensor,
        h_global: torch.Tensor,
        zone_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Predict cost from decisions and embedding.
        
        Args:
            u: [B, Z, T, F] decisions (NOT flattened)
            h_global: [B, D] global embedding
            zone_mask: [B, Z] valid zone mask
        
        Returns:
            predicted_cost: [B]
        """
        # Extract features
        features = self.feature_extractor(u, zone_mask)  # [B, P]
        
        # Concat with embedding
        x = torch.cat([features, h_global], dim=-1)
        
        return self.net(x).squeeze(-1)
    
    def update_statistics(self, costs: torch.Tensor):
        """Update running mean/std from observed LP costs."""
        alpha = 0.1
        batch_mean = costs.mean().item()
        batch_std = costs.std().item() + 1e-6
        
        self.cost_mean = (1 - alpha) * self.cost_mean + alpha * batch_mean
        self.cost_std = (1 - alpha) * self.cost_std + alpha * batch_std
        self.n_updates += 1
    
    def normalize(self, costs: torch.Tensor) -> torch.Tensor:
        """Normalize costs using running statistics."""
        return (costs - self.cost_mean.item()) / self.cost_std.item()
