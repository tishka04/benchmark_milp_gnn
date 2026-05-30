"""Torch dataset over UC label .npz files produced by ``extract_uc_labels``."""
from __future__ import annotations

import glob
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np
import torch
from torch.utils.data import Dataset

from src.uc_jepa import N_UC_FEATURES


@dataclass
class UCSample:
    u_star: torch.Tensor       # [Z, T, F] float32
    h_scenario: torch.Tensor   # [Z, T, D] float32
    n_zones: int
    n_timesteps: int
    scenario_id: str
    family: str
    criticality: float
    objective: float


class UCLabelDataset(Dataset):
    """Loads ``scenario_*_uc.npz`` files (output of ``extract_uc_labels``)."""

    def __init__(
        self,
        roots: Sequence[str],
        families: Optional[Sequence[str]] = None,
        max_samples: Optional[int] = None,
    ):
        files: List[str] = []
        for root in roots:
            if not os.path.isdir(root):
                continue
            for path in sorted(glob.glob(os.path.join(root, "**", "*_uc.npz"), recursive=True)):
                family = Path(path).parent.name
                if families is not None and family not in families:
                    continue
                files.append(path)
        if max_samples is not None:
            files = files[:max_samples]
        self.files = files
        if not files:
            raise FileNotFoundError(
                f"No UC label files found under roots={list(roots)}, families={families}."
            )

    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(self, idx: int) -> UCSample:
        path = self.files[idx]
        data = np.load(path, allow_pickle=True)

        u_star = torch.from_numpy(np.asarray(data["u_star"], dtype=np.float32))
        h_scenario = torch.from_numpy(np.asarray(data["h_scenario"], dtype=np.float32))

        if u_star.ndim != 3 or u_star.shape[-1] != N_UC_FEATURES:
            raise ValueError(
                f"{path}: expected u_star [Z, T, {N_UC_FEATURES}], got {tuple(u_star.shape)}"
            )
        Z, T, _ = u_star.shape
        return UCSample(
            u_star=u_star,
            h_scenario=h_scenario,
            n_zones=int(Z),
            n_timesteps=int(T),
            scenario_id=str(data["scenario_id"].item()) if data["scenario_id"].ndim == 0 else str(data["scenario_id"]),
            family=str(data["family"].item()) if data["family"].ndim == 0 else str(data["family"]),
            criticality=float(data["criticality"]),
            objective=float(data["milp_objective"]),
        )


def uc_collate(batch: List[UCSample]) -> Dict[str, torch.Tensor]:
    """Pad to Z_max within the batch and return tensors + zone_mask."""
    B = len(batch)
    Z_max = max(s.n_zones for s in batch)
    T = batch[0].n_timesteps
    F_ = batch[0].u_star.shape[-1]
    D = batch[0].h_scenario.shape[-1]

    u = torch.zeros(B, Z_max, T, F_, dtype=torch.float32)
    h = torch.zeros(B, Z_max, T, D, dtype=torch.float32)
    mask = torch.zeros(B, Z_max, dtype=torch.float32)

    scenario_ids: List[str] = []
    families: List[str] = []
    criticalities = torch.zeros(B, dtype=torch.float32)
    objectives = torch.zeros(B, dtype=torch.float32)

    for i, s in enumerate(batch):
        Z = s.n_zones
        u[i, :Z] = s.u_star[:, :T]
        # Align h zone count to Z (it should already match after extraction).
        h_i = s.h_scenario
        if h_i.shape[0] < Z:
            pad = torch.zeros(Z - h_i.shape[0], h_i.shape[1], h_i.shape[2])
            h_i = torch.cat([h_i, pad], dim=0)
        h[i, :Z] = h_i[:Z, :T, : h.shape[-1]]
        mask[i, :Z] = 1.0
        scenario_ids.append(s.scenario_id)
        families.append(s.family)
        criticalities[i] = s.criticality
        objectives[i] = s.objective

    return {
        "u_star": u,
        "h_scenario": h,
        "zone_mask": mask,
        "scenario_id": scenario_ids,
        "family": families,
        "criticality": criticalities,
        "milp_objective": objectives,
    }
