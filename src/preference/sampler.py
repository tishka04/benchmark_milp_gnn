# ==============================================================================
# LANGEVIN SAMPLER FOR CANDIDATE GENERATION
# ==============================================================================
# Generates K candidate decisions by running gradient-based sampling
# targeting p_θ(u | h) ∝ exp(-E_θ(u | h))
# ==============================================================================

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, List, Callable
from dataclasses import dataclass

from .data_models import DecisionVector


@dataclass
class SamplerConfig:
    """Configuration for Langevin sampler."""
    num_steps: int = 50  # Number of Langevin steps
    step_size: float = 0.01  # Step size (epsilon)
    noise_scale: float = 0.01  # Noise multiplier
    temperature: float = 1.0  # Sampling temperature
    clip_grad: float = 1.0  # Gradient clipping
    use_normalized: bool = True  # Use normalized gradients
    init_noise_scale: float = 0.5  # Initial noise for warm start
    anneal_schedule: str = "constant"  # "constant", "linear", "cosine"


class LangevinSampler(nn.Module):
    """
    Langevin dynamics sampler for EBM-based decision generation.
    
    Implements:
        u_{t+1} = u_t - ε * ∇_u E_θ(u_t | h) + √(2ε) * z
    
    where z ~ N(0, 1) and ε is the step size.
    
    For normalized Langevin:
        u_{t+1} = u_t - ε * ∇_u E_θ(u_t | h) / ||∇_u E_θ(u_t | h)|| + √(2ε) * z
    """
    
    def __init__(
        self,
        ebm: nn.Module,
        config: Optional[SamplerConfig] = None,
    ):
        """
        Args:
            ebm: Energy-based model with forward(u, h) -> energy
            config: Sampler configuration
        """
        super().__init__()
        self.ebm = ebm
        self.config = config or SamplerConfig()
    
    def sample(
        self,
        h: torch.Tensor,
        n_samples: int,
        n_zones: int,
        n_timesteps: int,
        n_features: int = 8,
        init_u: Optional[torch.Tensor] = None,
        return_trajectory: bool = False,
    ) -> Tuple[torch.Tensor, Optional[List[torch.Tensor]]]:
        """
        Generate samples from p_θ(u | h) using Langevin dynamics.
        
        Args:
            h: Scenario embedding [B, d_h]
            n_samples: Number of samples K to generate per scenario
            n_zones: Number of zones Z
            n_timesteps: Number of timesteps T
            n_features: Number of features per (z,t) = 8
            init_u: Optional initial u [B, K, Z, T, F]
            return_trajectory: Whether to return sampling trajectory
        
        Returns:
            u_samples: [B, K, Z, T, F] sampled decisions (in relaxed space)
            trajectory: (optional) list of intermediate u values
        """
        B = h.shape[0]
        device = h.device
        decision_shape = (B, n_samples, n_zones, n_timesteps, n_features)
        
        # Initialize u (logit space for unconstrained optimization)
        if init_u is not None:
            # Convert from (0,1) to logit space
            u = torch.logit(init_u.clamp(1e-6, 1-1e-6))
        else:
            # Random initialization in logit space
            u = torch.randn(decision_shape, device=device) * self.config.init_noise_scale
        
        u.requires_grad_(True)
        
        trajectory = [torch.sigmoid(u.detach().clone())] if return_trajectory else None
        
        # Expand h for K samples: [B, d_h] -> [B*K, d_h]
        h_expanded = h.unsqueeze(1).expand(-1, n_samples, -1).reshape(B * n_samples, -1)
        
        for step in range(self.config.num_steps):
            # Get step size (potentially annealed)
            eps = self._get_step_size(step)
            noise_scale = self._get_noise_scale(step)
            
            # Flatten u for EBM: [B, K, Z, T, F] -> [B*K, Z*T*F]
            u_flat = u.view(B * n_samples, -1)
            
            # Compute energy and gradient
            energy = self.ebm(u_flat, h_expanded)  # [B*K]
            
            grad = torch.autograd.grad(
                outputs=energy.sum(),
                inputs=u,
                create_graph=False,
                retain_graph=False,
            )[0]
            
            # Clip gradients
            if self.config.clip_grad > 0:
                grad = torch.clamp(grad, -self.config.clip_grad, self.config.clip_grad)
            
            # Normalize gradients if requested
            if self.config.use_normalized:
                grad_norm = grad.view(B * n_samples, -1).norm(dim=-1, keepdim=True)
                grad_norm = grad_norm.view(B, n_samples, 1, 1, 1) + 1e-8
                grad = grad / grad_norm
            
            # Langevin update
            noise = torch.randn_like(u) * noise_scale
            
            with torch.no_grad():
                u = u - eps * grad + (2 * eps) ** 0.5 * noise
            
            u.requires_grad_(True)
            
            if return_trajectory and (step + 1) % 10 == 0:
                trajectory.append(torch.sigmoid(u.detach().clone()))
        
        # Convert from logit to probability space
        u_relaxed = torch.sigmoid(u.detach())
        
        if return_trajectory:
            trajectory.append(u_relaxed.clone())
        
        return u_relaxed, trajectory
    
    def sample_single(
        self,
        h: torch.Tensor,
        n_zones: int,
        n_timesteps: int,
        n_features: int = 8,
        init_u: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Generate a single sample per scenario.
        
        Args:
            h: Scenario embedding [B, d_h]
            n_zones, n_timesteps, n_features: Decision dimensions
            init_u: Optional initial u [B, Z, T, F]
        
        Returns:
            u_sample: [B, Z, T, F] sampled decision
        """
        if init_u is not None:
            init_u = init_u.unsqueeze(1)  # [B, 1, Z, T, F]
        
        u_samples, _ = self.sample(
            h=h,
            n_samples=1,
            n_zones=n_zones,
            n_timesteps=n_timesteps,
            n_features=n_features,
            init_u=init_u,
        )
        
        return u_samples.squeeze(1)  # [B, Z, T, F]
    
    def _get_step_size(self, step: int) -> float:
        """Get step size for current step (with optional annealing)."""
        base_eps = self.config.step_size
        
        if self.config.anneal_schedule == "constant":
            return base_eps
        
        progress = step / max(1, self.config.num_steps - 1)
        
        if self.config.anneal_schedule == "linear":
            return base_eps * (1 - 0.5 * progress)
        
        elif self.config.anneal_schedule == "cosine":
            import math
            return base_eps * (0.5 + 0.5 * math.cos(math.pi * progress))
        
        return base_eps
    
    def _get_noise_scale(self, step: int) -> float:
        """Get noise scale for current step."""
        base_noise = self.config.noise_scale
        
        if self.config.anneal_schedule == "constant":
            return base_noise
        
        progress = step / max(1, self.config.num_steps - 1)
        
        if self.config.anneal_schedule == "linear":
            return base_noise * (1 - 0.5 * progress)
        
        elif self.config.anneal_schedule == "cosine":
            import math
            return base_noise * (0.5 + 0.5 * math.cos(math.pi * progress))
        
        return base_noise
    
    @torch.no_grad()
    def refine(
        self,
        u: torch.Tensor,
        h: torch.Tensor,
        num_steps: int = 10,
    ) -> torch.Tensor:
        """
        Refine existing decisions with additional Langevin steps (no gradient tracking).
        
        Args:
            u: Current decisions [B, Z, T, F] in (0,1)
            h: Scenario embedding [B, d_h]
            num_steps: Number of refinement steps
        
        Returns:
            u_refined: [B, Z, T, F] refined decisions
        """
        original_steps = self.config.num_steps
        self.config.num_steps = num_steps
        
        u_refined = self.sample_single(
            h=h,
            n_zones=u.shape[1],
            n_timesteps=u.shape[2],
            n_features=u.shape[3],
            init_u=u,
        )
        
        self.config.num_steps = original_steps
        return u_refined


class ReplayBuffer:
    """
    Experience replay buffer for persistent contrastive divergence (PCD).
    Stores samples from previous iterations to warm-start Langevin chains.
    """
    
    def __init__(
        self,
        max_size: int = 10000,
        reinit_prob: float = 0.05,
    ):
        """
        Args:
            max_size: Maximum buffer size
            reinit_prob: Probability of reinitializing from noise vs buffer
        """
        self.max_size = max_size
        self.reinit_prob = reinit_prob
        self.buffer = []
        self.scenario_ids = []
    
    def get(
        self,
        scenario_id: str,
        shape: Tuple[int, ...],
        device: torch.device,
    ) -> torch.Tensor:
        """
        Get initialization from buffer or random noise.
        
        Args:
            scenario_id: Scenario identifier for lookup
            shape: Expected shape [Z, T, F]
            device: Target device
        
        Returns:
            u_init: [Z, T, F] initialization in (0,1)
        """
        # With probability reinit_prob, start from noise
        if torch.rand(1).item() < self.reinit_prob:
            return torch.sigmoid(torch.randn(shape, device=device) * 0.5)
        
        # Try to find existing sample for this scenario
        if scenario_id in self.scenario_ids:
            idx = self.scenario_ids.index(scenario_id)
            u = self.buffer[idx].to(device)
            if u.shape == shape:
                return u
        
        # Fall back to random noise
        return torch.sigmoid(torch.randn(shape, device=device) * 0.5)
    
    def add(self, scenario_id: str, u: torch.Tensor):
        """Add sample to buffer."""
        u = u.detach().cpu()
        
        if scenario_id in self.scenario_ids:
            idx = self.scenario_ids.index(scenario_id)
            self.buffer[idx] = u
        else:
            if len(self.buffer) >= self.max_size:
                # Remove oldest
                self.buffer.pop(0)
                self.scenario_ids.pop(0)
            self.buffer.append(u)
            self.scenario_ids.append(scenario_id)


def binarize_decisions(
    u_relaxed: torch.Tensor,
    method: str = "threshold",
    threshold: float = 0.5,
) -> torch.Tensor:
    """
    Binarize relaxed decisions.
    
    Args:
        u_relaxed: [B, ...] decisions in (0,1)
        method: "threshold" or "sample" (Bernoulli)
        threshold: Threshold for deterministic binarization
    
    Returns:
        u_binary: [B, ...] binary decisions in {0,1}
    """
    if method == "threshold":
        return (u_relaxed > threshold).float()
    elif method == "sample":
        return torch.bernoulli(u_relaxed)
    else:
        raise ValueError(f"Unknown binarization method: {method}")
