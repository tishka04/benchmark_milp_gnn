"""Batched K16 Langevin with an independent random stream for each candidate."""
from __future__ import annotations

from dataclasses import fields
import time
from .common import PHYSICS, seed_for
from .embedding import OnlineHTE, require_gpu


class Generator:
    def __init__(self, settings):
        import torch
        from src.uc_jepa.latent_utils import load_frozen_ae, load_frozen_jepa, load_latent_ebm, threshold_vector
        from src.uc_jepa.sampling.latent_langevin import LatentLangevinSampler, LatentLangevinConfig
        self.device = require_gpu()
        self.hte = OnlineHTE(settings["hte"])
        for name in ("ae", "jepa") + (("ebm",) if settings["method"] == "pipeline" else ()):
            ck = torch.load(settings[name], map_location="cpu", weights_only=False)
            if ck.get("physics") != PHYSICS or ck.get("training_manifest") != settings["manifest_hash"]:
                raise ValueError(f"Uncertified/new-label checkpoint required: {name}")
        self.ae = load_frozen_ae(settings["ae"], self.device)
        self.predictor, jc = load_frozen_jepa(settings["jepa"], self.device)
        self.ebm, self.prior_mean, self.prior_std = None, None, None
        ec = jc.get("eval", {})
        if settings["method"] == "pipeline":
            self.ebm, ck = load_latent_ebm(settings["ebm"], self.device)
            for p in self.ebm.parameters():
                p.requires_grad_(False)
            cfg = ck["config"]["sampler"]
            self.sampler = LatentLangevinSampler(LatentLangevinConfig(**{f.name: cfg[f.name] for f in fields(LatentLangevinConfig) if f.name in cfg}))
            self.prior_mean, self.prior_std = ck.get("latent_prior_mean"), ck.get("latent_prior_std")
            ec = ck["config"].get("eval", ec)
        self.thresholds = threshold_vector(ec.get("threshold", .5), ec.get("feature_thresholds", {}), self.ae.n_features).to(self.device)
        torch.cuda.synchronize()

    def candidates(self, data, request, deadline):
        import torch
        from src.uc_jepa.latent_utils import apply_thresholds
        h = self.hte.encode(data)
        mask = torch.ones(1, len(data.zones), device=self.device)
        with torch.no_grad():
            anchor = self.predictor(h, zone_mask=mask)["z_pred"]
        k = 16 if self.ebm is not None else 1
        seeds = [seed_for(request["key"], request["repetition"], i) for i in range(k)]
        if self.ebm is None:
            z = anchor
        else:
            cfg = self.sampler.config
            streams = [torch.Generator(device=self.device).manual_seed(s) for s in seeds]
            def noise(shape, uniform=False):
                fn = torch.rand if uniform else torch.randn
                return torch.stack([fn(shape, generator=g, device=self.device) for g in streams])
            a = anchor.expand(k, -1, -1)
            n = noise(a.shape[1:]) * cfg.init_noise_std
            mean = self.sampler._as_prior_tensor(self.prior_mean, a.unsqueeze(0), 0)[0].expand_as(a)
            std = self.sampler._as_prior_tensor(self.prior_std, a.unsqueeze(0), 1)[0].expand_as(a).clamp_min(1e-4)
            prior = mean + std * noise(a.shape[1:])
            if cfg.init_mode == "anchor":
                z = a.clone()
            elif cfg.init_mode == "anchor_noise":
                z = a + n
            elif cfg.init_mode == "prior":
                z = prior
            elif cfg.init_mode == "mixed":
                z = torch.where(noise((1, 1), True) < cfg.prior_mix_prob, prior, a + n)
            else:
                raise ValueError("Unknown Langevin initialization")
            h_batch, m_batch = h.expand(k, -1, -1, -1), mask.expand(k, -1)
            for step in range(cfg.num_steps):
                if time.perf_counter() >= deadline:
                    raise TimeoutError("generation deadline")
                z = z.detach().requires_grad_(True)
                energy = self.ebm(z, a, h_scenario=h_batch, zone_mask=m_batch)
                if not torch.isfinite(energy).all():
                    raise ValueError("Nonfinite EBM energy")
                grad = torch.autograd.grad(energy.sum(), z)[0]
                if not torch.isfinite(grad).all():
                    raise ValueError("Nonfinite EBM gradient")
                if cfg.clip_grad > 0:
                    grad = grad.clamp(-cfg.clip_grad, cfg.clip_grad)
                if cfg.normalize_grad:
                    grad = grad / grad.flatten(1).norm(dim=1).view(k, 1, 1).clamp_min(1e-6)
                eps = self.sampler._get_annealed_scale(cfg.step_size, step)
                scale = self.sampler._get_annealed_scale(cfg.noise_scale, step)
                with torch.no_grad():
                    z = z - eps * grad - eps * cfg.drift_weight * (z - a) + noise(z.shape[1:]) * (2 * eps * cfg.temperature)**.5 * scale
                    if cfg.latent_clip is not None:
                        z = z.clamp(-cfg.latent_clip, cfg.latent_clip)
        with torch.no_grad():
            logits = self.ae.decode(z.detach(), n_zones=len(data.zones), n_timesteps=len(data.periods), h_scenario=h.expand(k, -1, -1, -1))
            if not torch.isfinite(logits).all():
                raise ValueError("Nonfinite candidate logits")
            candidates = apply_thresholds(torch.sigmoid(logits), self.thresholds).cpu().numpy()
        torch.cuda.synchronize()
        return candidates, seeds
