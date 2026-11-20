"""Timing probe: real worker/generator, synthetic CPU models and repair responses.

No checkpoint, CUDA calculation, or optimization is used. A controlled delay
inside HTE.encode must consume the supervisor's decision budget.
"""
from __future__ import annotations

import time
from types import SimpleNamespace
from unittest.mock import patch

import pandas  # noqa: F401 -- Windows DLL import order.
import torch

from src.operational_exit.common import atomic_json
from src.operational_exit.generation import Generator
from src.operational_exit.worker import main
from src.uc_jepa import latent_utils  # noqa: F401 -- normally imported by Generator.__init__.
from src.uc_jepa.sampling.latent_langevin import LatentLangevinConfig, LatentLangevinSampler


class DelayedHTEGenerator(Generator):
    def __init__(self, settings):
        self.settings = settings
        self.device = torch.device("cpu")
        self.hte = SimpleNamespace(encode=self.encode)
        self.predictor = lambda h, zone_mask: {"z_pred": torch.zeros(1, 2, 8)}
        self.ebm = (lambda z, anchor, **kw: z.square().mean(dim=(1, 2))) if settings["method"] == "pipeline" else None
        self.ae = SimpleNamespace(decode=lambda z, **kw: torch.zeros(len(z), kw["n_zones"], kw["n_timesteps"], 8))
        self.sampler = LatentLangevinSampler(LatentLangevinConfig(num_steps=1, init_mode="anchor"))
        self.prior_mean = self.prior_std = None
        self.thresholds = torch.full((8,), .5)

    def candidates(self, data, request, deadline):
        self.request, self.compute_deadline = request, deadline
        return super().candidates(data, request, deadline)

    def encode(self, data):
        trace = dict(start=self.request["start"], deadline=self.request["deadline"],
                     compute_deadline=self.compute_deadline, encode_start=time.perf_counter())
        atomic_json(self.settings["trace_file"], trace)
        time.sleep(self.settings["embedding_delay_s"])
        trace["encode_end"] = time.perf_counter()
        atomic_json(self.settings["trace_file"], trace)
        return torch.zeros(1, len(data.zones), len(data.periods), 4)


class SyntheticConnection:
    def __init__(self):
        self.messages = [{"ready": True}]

    def send(self, job):
        if job is not None:
            self.messages.append(dict(status="solution", candidate=job["index"],
                                      validation={"valid": True, "cost": 1.},
                                      dispatch={"synthetic": True}))

    def recv(self):
        return self.messages.pop(0)

    def poll(self, timeout=None):
        return bool(self.messages)

    def close(self):
        pass


class SyntheticProcess:
    def __init__(self, **kwargs):
        pass

    def start(self):
        pass

    def is_alive(self):
        return False

    def join(self):
        pass


def run():
    context = SimpleNamespace(Pipe=lambda: (SyntheticConnection(), SyntheticConnection()), Process=SyntheticProcess)
    with patch("src.operational_exit.generation.Generator", DelayedHTEGenerator), \
         patch("multiprocessing.get_context", return_value=context), \
         patch("src.operational_exit.physics.validate_dispatch", return_value={"valid": True, "cost": 1.}), \
         patch("torch.cuda.reset_peak_memory_stats"), patch("torch.cuda.synchronize"), \
         patch("torch.cuda.get_device_name", return_value="synthetic CPU probe"), \
         patch("torch.cuda.max_memory_allocated", return_value=0):
        main()


if __name__ == "__main__":
    run()
