# ==============================================================================
# CONDITIONAL ENERGY-BASED MODEL
# ==============================================================================
# E_θ(u | h) : Conditional energy function
# Low energy = high quality decisions for scenario context h
# ==============================================================================

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, Dict, Any


class ConditionalEBM(nn.Module):
    """
    Conditional Energy-Based Model for operational decisions.
    
    Computes E_θ(u | h) where:
        - h: HTE embedding of scenario [B, d_h]
        - u: Binary decision vector [B, M] or [B, Z, T, 7]
    
    Architecture:
        1. Project decision u to d_h dimension
        2. Concatenate or multiply with h
        3. MLP to scalar energy
    
    The model defines an implicit distribution:
        p_θ(u | h) ∝ exp(-E_θ(u | h))
    """
    
    def __init__(
        self,
        h_dim: int,
        decision_dim: int,
        hidden_dim: int = 256,
        num_layers: int = 4,
        dropout: float = 0.1,
        use_layer_norm: bool = True,
        conditioning: str = "concat",  # "concat", "film", "cross_attention"
        spectral_norm: bool = True,  # For stability
    ):
        """
        Args:
            h_dim: Dimension of scenario embedding h
            decision_dim: Dimension of flattened decision u (Z * T * 7)
            hidden_dim: Hidden layer dimension
            num_layers: Number of MLP layers
            dropout: Dropout rate
            use_layer_norm: Whether to use layer normalization
            conditioning: How to condition on h ("concat", "film", "cross_attention")
            spectral_norm: Whether to use spectral normalization for stability
        """
        super().__init__()
        
        self.h_dim = h_dim
        self.decision_dim = decision_dim
        self.hidden_dim = hidden_dim
        self.conditioning = conditioning
        
        # Decision encoder
        self.decision_encoder = nn.Sequential(
            nn.Linear(decision_dim, hidden_dim),
            nn.LayerNorm(hidden_dim) if use_layer_norm else nn.Identity(),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
        )
        
        # Context encoder
        self.context_encoder = nn.Sequential(
            nn.Linear(h_dim, hidden_dim),
            nn.LayerNorm(hidden_dim) if use_layer_norm else nn.Identity(),
            nn.GELU(),
        )
        
        # Conditioning mechanism
        if conditioning == "concat":
            joint_dim = hidden_dim * 2
        elif conditioning == "film":
            # Feature-wise Linear Modulation
            self.film_gamma = nn.Linear(hidden_dim, hidden_dim)
            self.film_beta = nn.Linear(hidden_dim, hidden_dim)
            joint_dim = hidden_dim
        elif conditioning == "cross_attention":
            self.cross_attn = nn.MultiheadAttention(
                embed_dim=hidden_dim,
                num_heads=4,
                dropout=dropout,
                batch_first=True,
            )
            joint_dim = hidden_dim
        else:
            raise ValueError(f"Unknown conditioning: {conditioning}")
        
        # Energy MLP
        layers = []
        in_dim = joint_dim
        for i in range(num_layers - 1):
            linear = nn.Linear(in_dim, hidden_dim)
            if spectral_norm:
                linear = nn.utils.spectral_norm(linear)
            layers.extend([
                linear,
                nn.LayerNorm(hidden_dim) if use_layer_norm else nn.Identity(),
                nn.GELU(),
                nn.Dropout(dropout),
            ])
            in_dim = hidden_dim
        
        # Final layer to scalar energy
        final_linear = nn.Linear(hidden_dim, 1)
        if spectral_norm:
            final_linear = nn.utils.spectral_norm(final_linear)
        layers.append(final_linear)
        
        self.energy_mlp = nn.Sequential(*layers)
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights for stable training."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight, gain=0.1)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
    
    def forward(
        self,
        u: torch.Tensor,
        h: torch.Tensor,
        return_features: bool = False,
    ) -> torch.Tensor:
        """
        Compute energy E_θ(u | h).
        
        Args:
            u: Decision vector [B, M] or [B, Z, T, 7]
            h: Scenario embedding [B, d_h]
            return_features: If True, also return intermediate features
        
        Returns:
            energy: [B] scalar energies
            features: (optional) intermediate features for analysis
        """
        B = h.shape[0]
        
        # Flatten decision if needed
        if u.dim() > 2:
            u = u.view(B, -1)
        
        # Encode decision and context
        u_enc = self.decision_encoder(u)  # [B, hidden_dim]
        h_enc = self.context_encoder(h)   # [B, hidden_dim]
        
        # Apply conditioning
        if self.conditioning == "concat":
            joint = torch.cat([u_enc, h_enc], dim=-1)  # [B, 2*hidden_dim]
        
        elif self.conditioning == "film":
            gamma = self.film_gamma(h_enc)  # [B, hidden_dim]
            beta = self.film_beta(h_enc)    # [B, hidden_dim]
            joint = gamma * u_enc + beta    # [B, hidden_dim]
        
        elif self.conditioning == "cross_attention":
            # u_enc attends to h_enc
            u_enc = u_enc.unsqueeze(1)  # [B, 1, hidden_dim]
            h_enc = h_enc.unsqueeze(1)  # [B, 1, hidden_dim]
            joint, _ = self.cross_attn(u_enc, h_enc, h_enc)
            joint = joint.squeeze(1)  # [B, hidden_dim]
        
        # Compute energy
        energy = self.energy_mlp(joint).squeeze(-1)  # [B]
        
        if return_features:
            return energy, {"u_enc": u_enc, "h_enc": h_enc, "joint": joint}
        return energy
    
    def energy(self, u: torch.Tensor, h: torch.Tensor) -> torch.Tensor:
        """Alias for forward()."""
        return self.forward(u, h)
    
    def energy_gradient(
        self,
        u: torch.Tensor,
        h: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute energy and its gradient w.r.t. u.
        Used for Langevin sampling.
        
        Args:
            u: Decision vector [B, M] (requires_grad should be True)
            h: Scenario embedding [B, d_h]
        
        Returns:
            energy: [B] energies
            grad_u: [B, M] gradients
        """
        u = u.requires_grad_(True)
        energy = self.forward(u, h)
        
        grad_u = torch.autograd.grad(
            outputs=energy.sum(),
            inputs=u,
            create_graph=True,
            retain_graph=True,
        )[0]
        
        return energy, grad_u


class ConditionalEBMWithZoneAttention(nn.Module):
    """
    Extended EBM that uses zone-level attention for structured decisions.
    
    Instead of flattening u to [B, Z*T*7], this model:
    1. Encodes each zone's decisions separately
    2. Uses attention to aggregate zone information
    3. Conditions on both zone-level and global context
    
    This is more parameter-efficient for large Z.
    """
    
    def __init__(
        self,
        h_dim: int,
        n_timesteps: int,
        n_features: int = 8,  # Number of binary decisions per (zone, timestep)
        hidden_dim: int = 128,
        num_heads: int = 4,
        num_layers: int = 3,
        dropout: float = 0.1,
    ):
        super().__init__()
        
        self.h_dim = h_dim
        self.n_timesteps = n_timesteps
        self.n_features = n_features
        self.hidden_dim = hidden_dim
        
        # Per-zone temporal encoder
        self.zone_encoder = nn.Sequential(
            nn.Linear(n_timesteps * n_features, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        
        # Context projection
        self.context_proj = nn.Linear(h_dim, hidden_dim)
        
        # Zone attention (zones attend to each other conditioned on context)
        self.zone_attention = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        
        # Cross-attention: zones attend to context
        self.cross_attention = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        
        # Energy head
        self.energy_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1),
        )
        
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)
    
    def forward(
        self,
        u: torch.Tensor,
        h: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute energy E_θ(u | h).
        
        Args:
            u: Decision tensor [B, Z, T, 7]
            h: Scenario embedding [B, d_h] or [B, Z, d_h] for zone-level
        
        Returns:
            energy: [B] scalar energies
        """
        B, Z, T, F = u.shape
        
        # Encode each zone's decisions: [B, Z, T*F] -> [B, Z, hidden_dim]
        u_flat = u.view(B, Z, T * F)
        zone_enc = self.zone_encoder(u_flat)  # [B, Z, hidden_dim]
        
        # Project context
        if h.dim() == 2:
            # Global context [B, d_h] -> [B, 1, hidden_dim]
            h_proj = self.context_proj(h).unsqueeze(1)
        else:
            # Zone-level context [B, Z, d_h] -> [B, Z, hidden_dim]
            h_proj = self.context_proj(h)
        
        # Zone self-attention
        zone_enc = self.norm1(zone_enc + self.zone_attention(zone_enc, zone_enc, zone_enc)[0])
        
        # Cross-attention with context
        zone_enc = self.norm2(zone_enc + self.cross_attention(zone_enc, h_proj, h_proj)[0])
        
        # Global pooling and energy
        global_enc = zone_enc.mean(dim=1)  # [B, hidden_dim]
        energy = self.energy_head(global_enc).squeeze(-1)  # [B]
        
        return energy


class ConditionalEBMWithGRU(nn.Module):
    """
    Conditional EBM with GRU for temporal understanding of decisions.
    
    Instead of flattening u to [B, Z*T*8], this model:
    1. Processes each zone's temporal sequence with a GRU
    2. Aggregates zone-level temporal representations
    3. Conditions on scenario context h
    
    This captures temporal dependencies in commitment decisions
    (e.g., thermal startup/shutdown patterns, storage cycling).
    """
    
    def __init__(
        self,
        h_dim: int,
        n_zones: int,
        n_timesteps: int,
        n_features: int = 8,
        hidden_dim: int = 128,
        gru_layers: int = 2,
        num_heads: int = 4,
        dropout: float = 0.1,
        bidirectional: bool = True,
        use_zone_attention: bool = True,
    ):
        """
        Args:
            h_dim: Dimension of scenario embedding h
            n_zones: Number of zones Z
            n_timesteps: Number of timesteps T
            n_features: Number of features per (z,t) = 8
            hidden_dim: Hidden dimension for GRU and MLP
            gru_layers: Number of GRU layers
            num_heads: Number of attention heads for zone aggregation
            dropout: Dropout rate
            bidirectional: Whether to use bidirectional GRU
            use_zone_attention: Whether to use attention for zone aggregation
        """
        super().__init__()
        
        self.h_dim = h_dim
        self.n_zones = n_zones
        self.n_timesteps = n_timesteps
        self.n_features = n_features
        self.hidden_dim = hidden_dim
        self.bidirectional = bidirectional
        self.use_zone_attention = use_zone_attention
        
        # Input projection per timestep
        self.input_proj = nn.Linear(n_features, hidden_dim)
        
        # GRU for temporal encoding (processes each zone's temporal sequence)
        self.gru = nn.GRU(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            num_layers=gru_layers,
            batch_first=True,
            dropout=dropout if gru_layers > 1 else 0.0,
            bidirectional=bidirectional,
        )
        
        gru_output_dim = hidden_dim * 2 if bidirectional else hidden_dim
        
        # Project GRU output
        self.gru_proj = nn.Sequential(
            nn.Linear(gru_output_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
        )
        
        # Zone aggregation
        if use_zone_attention:
            self.zone_attention = nn.MultiheadAttention(
                embed_dim=hidden_dim,
                num_heads=num_heads,
                dropout=dropout,
                batch_first=True,
            )
            self.zone_norm = nn.LayerNorm(hidden_dim)
        
        # Context encoder
        self.context_encoder = nn.Sequential(
            nn.Linear(h_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
        )
        
        # FiLM conditioning (context modulates decision encoding)
        self.film_gamma = nn.Linear(hidden_dim, hidden_dim)
        self.film_beta = nn.Linear(hidden_dim, hidden_dim)
        
        # Energy head
        self.energy_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Linear(hidden_dim // 2, 1),
        )
        
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights for stable training."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight, gain=0.1)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.GRU):
                for name, param in m.named_parameters():
                    if 'weight' in name:
                        nn.init.orthogonal_(param, gain=0.1)
                    elif 'bias' in name:
                        nn.init.zeros_(param)
    
    def forward(
        self,
        u: torch.Tensor,
        h: torch.Tensor,
        return_features: bool = False,
    ) -> torch.Tensor:
        """
        Compute energy E_θ(u | h) with temporal understanding.
        
        Args:
            u: Decision tensor [B, Z, T, 8] or [B, M] (will be reshaped)
            h: Scenario embedding [B, d_h]
            return_features: If True, also return intermediate features
        
        Returns:
            energy: [B] scalar energies
        """
        B = h.shape[0]
        
        # Reshape if flattened - infer dimensions dynamically
        if u.dim() == 2:
            M = u.shape[1]
            # Infer n_zones from flattened size: M = Z * T * F
            T = self.n_timesteps
            F = self.n_features
            Z = M // (T * F)
            u = u.view(B, Z, T, F)
        
        Z, T, F = u.shape[1], u.shape[2], u.shape[3]
        
        # Project input: [B, Z, T, F] -> [B, Z, T, hidden_dim]
        u_proj = self.input_proj(u)
        
        # Process each zone's temporal sequence with GRU
        # Reshape: [B, Z, T, D] -> [B*Z, T, D]
        u_flat = u_proj.view(B * Z, T, self.hidden_dim)
        
        # GRU forward pass
        gru_out, _ = self.gru(u_flat)  # [B*Z, T, D*2] if bidirectional
        
        # Take last timestep (or mean pool)
        # Using mean pooling over time for robustness
        gru_pooled = gru_out.mean(dim=1)  # [B*Z, D*2]
        
        # Project GRU output
        zone_enc = self.gru_proj(gru_pooled)  # [B*Z, D]
        zone_enc = zone_enc.view(B, Z, self.hidden_dim)  # [B, Z, D]
        
        # Zone aggregation
        if self.use_zone_attention:
            # Self-attention over zones
            zone_attn, _ = self.zone_attention(zone_enc, zone_enc, zone_enc)
            zone_enc = self.zone_norm(zone_enc + zone_attn)
        
        # Global pooling over zones
        global_enc = zone_enc.mean(dim=1)  # [B, D]
        
        # Encode context
        h_enc = self.context_encoder(h)  # [B, D]
        
        # FiLM conditioning: modulate decision encoding with context
        gamma = self.film_gamma(h_enc)  # [B, D]
        beta = self.film_beta(h_enc)    # [B, D]
        conditioned = gamma * global_enc + beta  # [B, D]
        
        # Compute energy
        energy = self.energy_head(conditioned).squeeze(-1)  # [B]
        
        if return_features:
            return energy, {
                "zone_enc": zone_enc,
                "global_enc": global_enc,
                "h_enc": h_enc,
                "conditioned": conditioned,
            }
        return energy
    
    def energy(self, u: torch.Tensor, h: torch.Tensor) -> torch.Tensor:
        """Alias for forward()."""
        return self.forward(u, h)
    
    def energy_gradient(
        self,
        u: torch.Tensor,
        h: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute energy and its gradient w.r.t. u.
        Used for Langevin sampling.
        """
        u = u.requires_grad_(True)
        energy = self.forward(u, h)
        
        grad_u = torch.autograd.grad(
            outputs=energy.sum(),
            inputs=u,
            create_graph=True,
            retain_graph=True,
        )[0]
        
        return energy, grad_u


class ConditionalEBMWithTemporalTransformer(nn.Module):
    """
    Conditional EBM with Temporal Transformer for decision understanding.
    
    Combines:
    1. Per-zone GRU for local temporal patterns
    2. Transformer for global temporal attention across all zones
    3. Cross-attention conditioning on scenario context
    
    This is the most expressive architecture for capturing complex
    spatio-temporal dependencies in operational decisions.
    """
    
    def __init__(
        self,
        h_dim: int,
        n_zones: int,
        n_timesteps: int,
        n_features: int = 8,
        hidden_dim: int = 128,
        gru_layers: int = 1,
        transformer_layers: int = 2,
        num_heads: int = 4,
        dropout: float = 0.1,
    ):
        super().__init__()
        
        self.h_dim = h_dim
        self.n_zones = n_zones
        self.n_timesteps = n_timesteps
        self.n_features = n_features
        self.hidden_dim = hidden_dim
        
        # Input projection
        self.input_proj = nn.Linear(n_features, hidden_dim)
        
        # Local temporal GRU (per zone)
        self.local_gru = nn.GRU(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            num_layers=gru_layers,
            batch_first=True,
            bidirectional=True,
        )
        self.gru_proj = nn.Linear(hidden_dim * 2, hidden_dim)
        
        # Positional encoding for zones and timesteps
        self.zone_pos = nn.Parameter(torch.randn(1, n_zones, 1, hidden_dim) * 0.02)
        self.time_pos = nn.Parameter(torch.randn(1, 1, n_timesteps, hidden_dim) * 0.02)
        
        # Global Transformer (attends over all zone-time pairs)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout,
            activation='gelu',
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=transformer_layers,
        )
        
        # Context encoder
        self.context_encoder = nn.Sequential(
            nn.Linear(h_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
        )
        
        # Cross-attention: decision representation attends to context
        self.cross_attention = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.cross_norm = nn.LayerNorm(hidden_dim)
        
        # Energy head
        self.energy_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )
    
    def forward(
        self,
        u: torch.Tensor,
        h: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute energy E_θ(u | h).
        
        Args:
            u: Decision tensor [B, Z, T, 8] or [B, M]
            h: Scenario embedding [B, d_h]
        
        Returns:
            energy: [B] scalar energies
        """
        B = h.shape[0]
        
        # Reshape if flattened - infer dimensions dynamically
        if u.dim() == 2:
            M = u.shape[1]
            T = self.n_timesteps
            F = self.n_features
            Z = M // (T * F)
            u = u.view(B, Z, T, F)
        
        Z, T = u.shape[1], u.shape[2]
        
        # Project input: [B, Z, T, F] -> [B, Z, T, D]
        u_proj = self.input_proj(u)
        
        # Add positional encodings
        u_proj = u_proj + self.zone_pos[:, :Z, :, :] + self.time_pos[:, :, :T, :]
        
        # Local GRU per zone: [B, Z, T, D] -> [B*Z, T, D]
        u_flat = u_proj.view(B * Z, T, self.hidden_dim)
        gru_out, _ = self.local_gru(u_flat)  # [B*Z, T, D*2]
        gru_out = self.gru_proj(gru_out)  # [B*Z, T, D]
        gru_out = gru_out.view(B, Z, T, self.hidden_dim)  # [B, Z, T, D]
        
        # Flatten for global transformer: [B, Z*T, D]
        global_seq = gru_out.view(B, Z * T, self.hidden_dim)
        
        # Global transformer attention
        global_enc = self.transformer(global_seq)  # [B, Z*T, D]
        
        # Pool to single vector
        global_pooled = global_enc.mean(dim=1)  # [B, D]
        
        # Encode context
        h_enc = self.context_encoder(h).unsqueeze(1)  # [B, 1, D]
        
        # Cross-attention: decision attends to context
        global_pooled_seq = global_pooled.unsqueeze(1)  # [B, 1, D]
        cross_out, _ = self.cross_attention(global_pooled_seq, h_enc, h_enc)
        conditioned = self.cross_norm(global_pooled_seq + cross_out).squeeze(1)  # [B, D]
        
        # Compute energy
        energy = self.energy_head(conditioned).squeeze(-1)  # [B]
        
        return energy


def build_ebm(
    config: Dict[str, Any],
    h_dim: int,
    decision_dim: int,
    n_zones: Optional[int] = None,
    n_timesteps: Optional[int] = None,
    n_features: int = 8,
) -> nn.Module:
    """
    Factory function to build EBM from config.
    
    Args:
        config: Dictionary with EBM configuration
            - architecture: "mlp", "gru", "transformer", "zone_attention"
        h_dim: Dimension of HTE embedding
        decision_dim: Dimension of flattened decision vector
        n_zones: Number of zones (required for GRU/Transformer)
        n_timesteps: Number of timesteps (required for GRU/Transformer)
        n_features: Number of binary features per (zone, timestep)
    
    Returns:
        EBM instance (ConditionalEBM, ConditionalEBMWithGRU, etc.)
    """
    architecture = config.get("architecture", "mlp")
    
    if architecture == "gru":
        if n_zones is None or n_timesteps is None:
            raise ValueError("n_zones and n_timesteps required for GRU architecture")
        return ConditionalEBMWithGRU(
            h_dim=h_dim,
            n_zones=n_zones,
            n_timesteps=n_timesteps,
            n_features=n_features,
            hidden_dim=config.get("hidden_dim", 128),
            gru_layers=config.get("gru_layers", 2),
            num_heads=config.get("num_heads", 4),
            dropout=config.get("dropout", 0.1),
            bidirectional=config.get("bidirectional", True),
            use_zone_attention=config.get("use_zone_attention", True),
        )
    
    elif architecture == "transformer":
        if n_zones is None or n_timesteps is None:
            raise ValueError("n_zones and n_timesteps required for Transformer architecture")
        return ConditionalEBMWithTemporalTransformer(
            h_dim=h_dim,
            n_zones=n_zones,
            n_timesteps=n_timesteps,
            n_features=n_features,
            hidden_dim=config.get("hidden_dim", 128),
            gru_layers=config.get("gru_layers", 1),
            transformer_layers=config.get("transformer_layers", 2),
            num_heads=config.get("num_heads", 4),
            dropout=config.get("dropout", 0.1),
        )
    
    elif architecture == "zone_attention":
        if n_timesteps is None:
            raise ValueError("n_timesteps required for zone_attention architecture")
        return ConditionalEBMWithZoneAttention(
            h_dim=h_dim,
            n_timesteps=n_timesteps,
            n_features=n_features,
            hidden_dim=config.get("hidden_dim", 128),
            num_heads=config.get("num_heads", 4),
            num_layers=config.get("num_layers", 3),
            dropout=config.get("dropout", 0.1),
        )
    
    else:  # "mlp" or default
        return ConditionalEBM(
            h_dim=h_dim,
            decision_dim=decision_dim,
            hidden_dim=config.get("hidden_dim", 256),
            num_layers=config.get("num_layers", 4),
            dropout=config.get("dropout", 0.1),
            use_layer_norm=config.get("use_layer_norm", True),
            conditioning=config.get("conditioning", "concat"),
            spectral_norm=config.get("spectral_norm", True),
        )
