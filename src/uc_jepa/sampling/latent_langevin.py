"""Latent-space Langevin sampler for UC-JEPA."""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import List, Literal, Optional, Tuple

import torch


@dataclass
class LatentLangevinConfig:
    """Configuration for latent-space Langevin dynamics."""

    num_steps: int = 30
    step_size: float = 0.05
    noise_scale: float = 0.01
    temperature: float = 1.0
    clip_grad: float = 0.0
    normalize_grad: bool = True
    init_mode: Literal["anchor", "anchor_noise", "prior", "mixed"] = "anchor_noise"
    init_noise_std: float = 0.25
    drift_weight: float = 0.10
    prior_mix_prob: float = 0.25
    anneal_schedule: Literal["constant", "linear", "cosine"] = "cosine"
    latent_clip: Optional[float] = 6.0


class LatentLangevinSampler:
    """Sample latent candidates by minimizing the EBM energy with noise."""

    def __init__(self, config: Optional[LatentLangevinConfig] = None):
        self.config = config or LatentLangevinConfig()

    def _get_annealed_scale(self, base: float, step: int) -> float:
        if self.config.anneal_schedule == "constant":
            return base

        progress = step / max(1, self.config.num_steps - 1)
        if self.config.anneal_schedule == "linear":
            return base * (1.0 - 0.5 * progress)
        if self.config.anneal_schedule == "cosine":
            return base * (0.5 + 0.5 * math.cos(math.pi * progress))
        return base

    @staticmethod
    def _as_prior_tensor(
        prior: Optional[torch.Tensor],
        reference: torch.Tensor,
        fill_value: float,
    ) -> torch.Tensor:
        if prior is None:
            return torch.full_like(reference, fill_value)
        if prior.dim() == 2:
            return prior.unsqueeze(0).unsqueeze(0).to(reference.device, dtype=reference.dtype)
        if prior.dim() == 3:
            return prior.unsqueeze(1).to(reference.device, dtype=reference.dtype)
        if prior.dim() == 4:
            return prior.to(reference.device, dtype=reference.dtype)
        raise ValueError(f"Unsupported prior shape: {tuple(prior.shape)}")

    def _init_latents(
        self,
        z_anchor: torch.Tensor,
        n_samples: int,
        init_z: Optional[torch.Tensor] = None,
        prior_mean: Optional[torch.Tensor] = None,
        prior_std: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Build the initial latent tensor [B, N, K, D]."""
        if init_z is not None:
            if init_z.dim() == 3:
                return init_z.unsqueeze(1).clone()
            if init_z.dim() == 4:
                return init_z.clone()
            raise ValueError(f"Unsupported init_z shape: {tuple(init_z.shape)}")

        anchor = z_anchor.unsqueeze(1).expand(-1, n_samples, -1, -1)
        noise = torch.randn_like(anchor) * self.config.init_noise_std
        prior_loc = self._as_prior_tensor(prior_mean, anchor, 0.0).expand_as(anchor)
        prior_scale = self._as_prior_tensor(prior_std, anchor, 1.0).expand_as(anchor).clamp_min(1e-4)
        prior_sample = prior_loc + prior_scale * torch.randn_like(anchor)

        if self.config.init_mode == "anchor":
            return anchor.clone()
        if self.config.init_mode == "anchor_noise":
            return anchor + noise
        if self.config.init_mode == "prior":
            return prior_sample
        if self.config.init_mode == "mixed":
            choose_prior = (
                torch.rand(anchor.shape[0], anchor.shape[1], 1, 1, device=anchor.device)
                < self.config.prior_mix_prob
            )
            return torch.where(choose_prior, prior_sample, anchor + noise)
        raise ValueError(f"Unknown init_mode: {self.config.init_mode}")

    def sample(
        self,
        ebm,
        z_anchor: torch.Tensor,
        h_scenario: Optional[torch.Tensor] = None,
        zone_mask: Optional[torch.Tensor] = None,
        n_samples: int = 1,
        init_z: Optional[torch.Tensor] = None,
        prior_mean: Optional[torch.Tensor] = None,
        prior_std: Optional[torch.Tensor] = None,
        num_steps: Optional[int] = None,
        return_trajectory: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[List[torch.Tensor]]]:
        """Run latent Langevin dynamics.

        Returns:
            z_samples: [B, N, K, D]
            energies:  [B, N]
            trajectory: optional list of intermediate latent tensors
        """
        original_steps = self.config.num_steps
        try:
            if num_steps is not None:
                self.config.num_steps = int(num_steps)

            z_anchor = z_anchor.detach()
            if h_scenario is not None:
                h_scenario = h_scenario.detach()
            if zone_mask is not None:
                zone_mask = zone_mask.detach()

            z = self._init_latents(
                z_anchor=z_anchor,
                n_samples=n_samples,
                init_z=init_z,
                prior_mean=prior_mean,
                prior_std=prior_std,
            ).detach().requires_grad_(True)

            traj: Optional[List[torch.Tensor]] = [z.detach().clone()] if return_trajectory else None

            bsz, _, n_tokens, latent_dim = z.shape
            z_anchor_flat = z_anchor.unsqueeze(1).expand(-1, n_samples, -1, -1).reshape(
                bsz * n_samples, n_tokens, latent_dim,
            )
            h_flat = None
            if h_scenario is not None:
                h_flat = h_scenario.unsqueeze(1).expand(-1, n_samples, -1, -1, -1).reshape(
                    bsz * n_samples, h_scenario.shape[1], h_scenario.shape[2], h_scenario.shape[3],
                )
            mask_flat = None
            if zone_mask is not None:
                mask_flat = zone_mask.unsqueeze(1).expand(-1, n_samples, -1).reshape(
                    bsz * n_samples, zone_mask.shape[1],
                )

            for step in range(self.config.num_steps):
                eps = self._get_annealed_scale(self.config.step_size, step)
                noise_scale = self._get_annealed_scale(self.config.noise_scale, step)

                energy = ebm(
                    z.reshape(bsz * n_samples, n_tokens, latent_dim),
                    z_anchor_flat,
                    h_scenario=h_flat,
                    zone_mask=mask_flat,
                ).view(bsz, n_samples)
                if not energy.requires_grad:
                    raise RuntimeError(
                        "Latent Langevin energy is detached from z; sampler cannot take gradients.",
                    )
                if not torch.isfinite(energy).all():
                    energy = torch.nan_to_num(
                        energy,
                        nan=0.0,
                        posinf=float(self.config.latent_clip or 10.0),
                        neginf=-float(self.config.latent_clip or 10.0),
                    )

                grad = torch.autograd.grad(
                    energy.sum(),
                    z,
                    create_graph=False,
                    retain_graph=False,
                    allow_unused=False,
                )[0]
                grad = torch.nan_to_num(grad, nan=0.0, posinf=0.0, neginf=0.0)

                if self.config.clip_grad and self.config.clip_grad > 0:
                    grad = torch.clamp(grad, -self.config.clip_grad, self.config.clip_grad)
                if self.config.normalize_grad:
                    grad_norm = grad.reshape(bsz * n_samples, -1).norm(dim=-1, keepdim=True)
                    grad = grad / grad_norm.view(bsz, n_samples, 1, 1).clamp_min(1e-6)

                anchor = z_anchor.unsqueeze(1).expand_as(z)
                drift = self.config.drift_weight * (z - anchor)
                noise = torch.randn_like(z) * (2.0 * eps * self.config.temperature) ** 0.5 * noise_scale

                with torch.no_grad():
                    z = z - eps * grad - eps * drift + noise
                    z = torch.nan_to_num(z, nan=0.0, posinf=0.0, neginf=0.0)
                    if self.config.latent_clip is not None:
                        z = z.clamp(-self.config.latent_clip, self.config.latent_clip)
                z = z.detach().requires_grad_(True)

                if return_trajectory and (step + 1) % max(1, self.config.num_steps // 5) == 0:
                    traj.append(z.detach().clone())

            z_final = z.detach()
            with torch.no_grad():
                energies = ebm(
                    z_final.reshape(bsz * n_samples, n_tokens, latent_dim),
                    z_anchor_flat,
                    h_scenario=h_flat,
                    zone_mask=mask_flat,
                ).view(bsz, n_samples)
                energies = torch.nan_to_num(
                    energies,
                    nan=0.0,
                    posinf=float(self.config.latent_clip or 10.0),
                    neginf=-float(self.config.latent_clip or 10.0),
                )

            if traj is not None:
                traj.append(z_final.clone())
            return z_final, energies, traj
        finally:
            self.config.num_steps = original_steps

    @staticmethod
    def select_best(
        z_samples: torch.Tensor,
        energies: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Pick the lowest-energy latent per batch item."""
        best_idx = energies.argmin(dim=1)
        gather_idx = best_idx.view(-1, 1, 1, 1).expand(
            -1, 1, z_samples.shape[2], z_samples.shape[3],
        )
        best_z = z_samples.gather(dim=1, index=gather_idx).squeeze(1)
        best_energy = energies.gather(dim=1, index=best_idx.view(-1, 1)).squeeze(1)
        return best_z, best_energy, best_idx
