# ==============================================================================
# NORMALIZED TEMPORAL LANGEVIN SAMPLER - v3
# ==============================================================================
# Langevin dynamics in logit space for temporal zonal EBM sampling.
#
# Train mode: returns RELAXED u in (0,1) for stable EBM gradients
# Infer mode: returns BINARY u in {0,1} for LP/MILP plug-in
#
# Update rule (logit space):
#   z ← z - η·∇_z E(σ(z)) - η·λ·(z - logit(p)) + σ·√η·ε
# ==============================================================================

from __future__ import annotations

import math
import numpy as np
import torch
import torch.nn as nn
from typing import Optional, Tuple, List, Literal


class NormalizedTemporalLangevinSampler:
    """
    Normalized Langevin sampler with optional sparse prior drift
    for temporal zonal EBM.

    Public API:
      - set_mode("train" | "infer")
      - sample(...)           → mode-aware (relaxed or binary)
      - sample_relaxed(...)   → always relaxed u ∈ (0,1)
      - sample_binary(...)    → always binary u ∈ {0,1}
    """

    def __init__(
        self,
        model: nn.Module,
        n_features: int,
        # Langevin hyperparams
        num_steps: int = 100,
        step_size: float = 0.02,
        noise_scale: float = 0.3,
        temp_min: float = 0.03,
        temp_max: float = 0.3,
        # Init
        init_mode: Literal["soft", "prior", "bernoulli", "oracle"] = "soft",
        init_p: float = 0.5,
        # Sparse prior
        prior_p: float = 0.025,
        prior_strength: float = 0.0,
        # Misc
        normalize_grad: bool = True,
        device: str = "cuda",
        # Mode
        mode: Literal["train", "infer"] = "train",
        # Inference binarization
        infer_binarize: Literal["bernoulli", "threshold"] = "bernoulli",
        infer_threshold: float = 0.5,
        # cuDNN GRU backward requires model.train()
        require_train_mode_for_sampling: bool = True,
    ):
        self.model = model
        self.n_features = int(n_features)

        self.num_steps = int(num_steps)
        self.step_size = float(step_size)
        self.noise_scale = float(noise_scale)
        self.temp_min = float(temp_min)
        self.temp_max = float(temp_max)

        self.init_mode = init_mode
        self.init_p = float(init_p)

        self.prior_p = float(prior_p)
        self.prior_strength = float(prior_strength)
        self.prior_logit = float(np.log(self.prior_p / (1.0 - self.prior_p + 1e-9)))

        self.normalize_grad = bool(normalize_grad)
        self.device = device

        self.mode = mode
        self.infer_binarize = infer_binarize
        self.infer_threshold = float(infer_threshold)

        self.require_train_mode_for_sampling = bool(require_train_mode_for_sampling)

    # ── Mode helpers ──

    def set_mode(self, mode: Literal["train", "infer"]) -> None:
        if mode not in ("train", "infer"):
            raise ValueError(f"mode must be 'train' or 'infer', got: {mode}")
        self.mode = mode

    def set_num_steps(self, num_steps: int) -> None:
        """Dynamically change the number of Langevin steps (curriculum)."""
        self.num_steps = int(num_steps)

    def set_infer_policy(
        self,
        binarize: Literal["bernoulli", "threshold"] = "bernoulli",
        threshold: float = 0.5,
    ) -> None:
        self.infer_binarize = binarize
        self.infer_threshold = float(threshold)

    # ── Internal utilities ──

    def _get_temperature(self, k: int) -> float:
        if self.num_steps <= 1:
            return self.temp_max
        t = k / (self.num_steps - 1)
        return self.temp_max + t * (self.temp_min - self.temp_max)

    def _init_logits(
        self,
        shape: Tuple[int, int, int, int],
        u_oracle: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Initialize logits z0 ∈ R^(B,Z,T,F)."""
        B, Z, T, F = shape
        eps = 1e-4

        if self.init_mode == "oracle":
            if u_oracle is None:
                raise ValueError("init_mode='oracle' requires u_oracle")
            u0 = u_oracle.to(self.device).clamp(eps, 1 - eps)
            return torch.log(u0) - torch.log(1 - u0)

        if self.init_mode == "bernoulli":
            u0 = torch.bernoulli(
                torch.full(shape, self.init_p, device=self.device)
            ).clamp(eps, 1 - eps)
            return torch.log(u0) - torch.log(1 - u0)

        if self.init_mode == "prior":
            return torch.full(shape, self.prior_logit, device=self.device)

        if self.init_mode == "soft":
            return (
                torch.randn(shape, device=self.device) * 0.1 + self.prior_logit
            )

        raise ValueError(f"Unknown init_mode: {self.init_mode}")

    def _apply_mask(
        self, x: torch.Tensor, zone_mask: Optional[torch.Tensor],
    ) -> torch.Tensor:
        if zone_mask is None:
            return x
        B, Z = zone_mask.shape
        mask = zone_mask.to(x.device).view(B, Z, 1, 1).float()
        return x * mask

    # ── Core Langevin (logit space) ──

    @torch.enable_grad()
    def _langevin_relaxed(
        self,
        h_zt: torch.Tensor,
        zone_mask: Optional[torch.Tensor] = None,
        u_oracle: Optional[torch.Tensor] = None,
        return_trajectory: bool = False,
        verbose: bool = False,
    ):
        """Run Langevin dynamics, return relaxed u ∈ (0,1)."""
        h_zt = h_zt.to(self.device)
        if zone_mask is not None:
            zone_mask = zone_mask.to(self.device)

        B, Z, T, D = h_zt.shape
        F = self.n_features

        prev_train_state = self.model.training
        if self.require_train_mode_for_sampling:
            self.model.train(True)

        z = self._init_logits((B, Z, T, F), u_oracle=u_oracle).requires_grad_(True)

        traj: List[torch.Tensor] = []
        if return_trajectory:
            traj.append(torch.sigmoid(z).detach().clone())

        for k in range(self.num_steps):
            Tk = self._get_temperature(k)

            u = torch.sigmoid(z)
            u = self._apply_mask(u, zone_mask)

            energy = self.model(u, h_zt, zone_mask)
            E = energy.sum()

            grad_z = torch.autograd.grad(
                E, z, create_graph=False, retain_graph=False,
            )[0]

            if self.normalize_grad:
                g_std = grad_z.std()
                if g_std > 1e-9:
                    grad_z = grad_z / g_std

            if self.prior_strength != 0.0:
                prior_drift = self.prior_strength * (z.detach() - self.prior_logit)
            else:
                prior_drift = 0.0

            noise = torch.randn_like(z)
            step = self.step_size
            noise_term = self.noise_scale * Tk * math.sqrt(step) * noise

            z = z - step * grad_z - step * prior_drift + noise_term

            if zone_mask is not None:
                mask = zone_mask.view(B, Z, 1, 1).float()
                z = z * mask

            z = z.detach().requires_grad_(True)

            if return_trajectory and (k + 1) % max(1, self.num_steps // 10) == 0:
                traj.append(torch.sigmoid(z).detach().clone())

            if verbose and (k % max(1, self.num_steps // 10) == 0):
                with torch.no_grad():
                    u_cur = torch.sigmoid(z)
                    u_cur = self._apply_mask(u_cur, zone_mask)
                    e_mean = self.model(u_cur, h_zt, zone_mask).mean().item()
                print(f"  step {k:03d}/{self.num_steps} | T={Tk:.4f} | E_mean={e_mean:.4f}")

        u_relaxed = torch.sigmoid(z).detach()
        u_relaxed = self._apply_mask(u_relaxed, zone_mask)

        if self.require_train_mode_for_sampling:
            self.model.train(prev_train_state)

        if return_trajectory:
            traj.append(u_relaxed.clone())
            return u_relaxed, traj
        return u_relaxed

    # ── Public sampling methods ──

    def sample_relaxed(
        self,
        h_zt: torch.Tensor,
        zone_mask: Optional[torch.Tensor] = None,
        u_oracle: Optional[torch.Tensor] = None,
        return_trajectory: bool = False,
        verbose: bool = False,
    ):
        """Always returns relaxed u ∈ (0,1)."""
        return self._langevin_relaxed(
            h_zt=h_zt, zone_mask=zone_mask, u_oracle=u_oracle,
            return_trajectory=return_trajectory, verbose=verbose,
        )

    def sample_binary(
        self,
        h_zt: torch.Tensor,
        zone_mask: Optional[torch.Tensor] = None,
        u_oracle: Optional[torch.Tensor] = None,
        binarize: Optional[Literal["bernoulli", "threshold"]] = None,
        threshold: Optional[float] = None,
        return_trajectory: bool = False,
        verbose: bool = False,
    ):
        """Always returns binary u ∈ {0,1}."""
        if binarize is None:
            binarize = self.infer_binarize
        if threshold is None:
            threshold = self.infer_threshold

        out = self._langevin_relaxed(
            h_zt=h_zt, zone_mask=zone_mask, u_oracle=u_oracle,
            return_trajectory=return_trajectory, verbose=verbose,
        )

        if return_trajectory:
            u_relaxed, traj = out
        else:
            u_relaxed = out

        if binarize == "threshold":
            u_bin = (u_relaxed > float(threshold)).float()
        elif binarize == "bernoulli":
            u_bin = torch.bernoulli(u_relaxed)
        else:
            raise ValueError(f"Unknown binarize mode: {binarize}")

        u_bin = self._apply_mask(u_bin, zone_mask)

        if return_trajectory:
            return u_bin, traj
        return u_bin

    def sample(
        self,
        h_zt: torch.Tensor,
        zone_mask: Optional[torch.Tensor] = None,
        u_oracle: Optional[torch.Tensor] = None,
        return_trajectory: bool = False,
        verbose: bool = False,
    ):
        """Mode-aware entry point: train→relaxed, infer→binary."""
        if self.mode == "train":
            return self.sample_relaxed(
                h_zt=h_zt, zone_mask=zone_mask, u_oracle=u_oracle,
                return_trajectory=return_trajectory, verbose=verbose,
            )
        if self.mode == "infer":
            return self.sample_binary(
                h_zt=h_zt, zone_mask=zone_mask, u_oracle=u_oracle,
                return_trajectory=return_trajectory, verbose=verbose,
            )
        raise ValueError(f"Unknown mode: {self.mode}")
