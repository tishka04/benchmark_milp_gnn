# ==============================================================================
# DISPATCH DATASET - Load (binaries, embeddings, dispatch_targets) triplets
# ==============================================================================
# Loads MILP report dispatch data as training targets for DispatchGNN.
# Pairs ground-truth binary decisions and HTE embeddings as inputs,
# with continuous dispatch values as regression targets.
#
# Compatible with variable zone counts via padding + zone masks.
# ==============================================================================

from __future__ import annotations

import os
import json
import glob
import numpy as np
import torch
import torch.nn.functional as F_nn
from torch.utils.data import Dataset, DataLoader, random_split
from typing import Dict, List, Optional, Tuple

from src.gnn.dispatch_model import DISPATCH_CHANNELS, N_DISPATCH


# ── Binary feature indices (same as EBM v3) ──
FEAT_BATT_CHARGE = 0
FEAT_BATT_DISCHARGE = 1
FEAT_PUMP_CHARGE = 2
FEAT_PUMP_DISCHARGE = 3
FEAT_DR = 4
FEAT_THERMAL_SU = 5
FEAT_THERMAL = 6
N_BINARY_FEATURES = 7


def extract_binaries_from_report(
    report: dict,
    n_timesteps: int = 24,
) -> Optional[torch.Tensor]:
    """
    Extract binary decision tensor [Z, T, 7] from a MILP report JSON.

    Reuses the same logic as src.ebm.dataset_v3._extract_binaries_from_report.
    """
    detail = report.get("detail", {})
    pd_section = report.get("pre_dispatch", {})
    src = detail if detail else pd_section
    if not src:
        return None

    zone_names = report.get("zones")
    if not zone_names:
        for probe_key in ["thermal_commitment", "thermal", "demand"]:
            probe = src.get(probe_key, {})
            if isinstance(probe, dict) and probe:
                zone_names = sorted(probe.keys())
                break
    if not zone_names:
        return None

    Z = len(zone_names)
    T = n_timesteps
    u = torch.zeros(Z, T, N_BINARY_FEATURES)

    def _get(key: str, zone: str) -> list:
        d = src.get(key, {})
        return d.get(zone, []) if isinstance(d, dict) else []

    for z_idx, zone in enumerate(zone_names):
        # Thermal commitment
        tc = _get("thermal_commitment", zone)
        for t in range(min(T, len(tc))):
            u[z_idx, t, FEAT_THERMAL] = 1.0 if abs(tc[t]) > 0.5 else 0.0

        # Thermal start-up
        tsu = _get("thermal_startup", zone)
        for t in range(min(T, len(tsu))):
            u[z_idx, t, FEAT_THERMAL_SU] = 1.0 if abs(tsu[t]) > 0.5 else 0.0
        if not tsu and tc:
            for t in range(min(T, len(tc))):
                on_now = abs(tc[t]) > 0.5
                on_prev = abs(tc[t - 1]) > 0.5 if t > 0 else False
                u[z_idx, t, FEAT_THERMAL_SU] = 1.0 if (on_now and not on_prev) else 0.0

        # Battery charge/discharge
        bch = _get("battery_charge", zone)
        bdis = _get("battery_discharge", zone)
        for t in range(T):
            ch = bch[t] if t < len(bch) else 0.0
            dis = bdis[t] if t < len(bdis) else 0.0
            u[z_idx, t, FEAT_BATT_CHARGE] = 1.0 if ch > 1e-3 else 0.0
            u[z_idx, t, FEAT_BATT_DISCHARGE] = 1.0 if dis > 1e-3 else 0.0

        # Pumped charge/discharge
        pch = _get("pumped_charge", zone)
        pdis = _get("pumped_discharge", zone)
        for t in range(T):
            ch = pch[t] if t < len(pch) else 0.0
            dis = pdis[t] if t < len(pdis) else 0.0
            u[z_idx, t, FEAT_PUMP_CHARGE] = 1.0 if ch > 1e-3 else 0.0
            u[z_idx, t, FEAT_PUMP_DISCHARGE] = 1.0 if dis > 1e-3 else 0.0

        # DR active
        dra = _get("dr_active", zone)
        for t in range(min(T, len(dra))):
            u[z_idx, t, FEAT_DR] = 1.0 if abs(dra[t]) > 0.5 else 0.0
        if not dra:
            drs = _get("demand_response", zone)
            if not drs:
                drs = _get("dr_shed", zone)
            for t in range(min(T, len(drs))):
                u[z_idx, t, FEAT_DR] = 1.0 if abs(drs[t]) > 1e-3 else 0.0

    return u


def extract_dispatch_from_report(
    report: dict,
    n_timesteps: int = 24,
) -> Optional[Tuple[torch.Tensor, List[str]]]:
    """
    Extract continuous dispatch tensor [Z, T, 11] from a MILP report JSON.

    Returns:
        dispatch: [Z, T, 11] tensor of dispatch values (MW)
        zone_names: list of zone name strings
    """
    detail = report.get("detail", {})
    if not detail:
        return None

    zone_names = report.get("zones")
    if not zone_names:
        for key in ["thermal", "demand", "nuclear"]:
            probe = detail.get(key, {})
            if isinstance(probe, dict) and probe:
                zone_names = sorted(probe.keys())
                break
    if not zone_names:
        return None

    Z = len(zone_names)
    T = n_timesteps
    dispatch = torch.zeros(Z, T, N_DISPATCH)

    for z_idx, zone in enumerate(zone_names):
        for c_idx, channel in enumerate(DISPATCH_CHANNELS):
            values = detail.get(channel, {})
            if isinstance(values, dict):
                values = values.get(zone, [])
            else:
                values = []
            for t in range(min(T, len(values))):
                dispatch[z_idx, t, c_idx] = max(0.0, float(values[t]))

    return dispatch, zone_names


class DispatchDataset(Dataset):
    """
    Dataset that loads (binaries, embeddings, dispatch_targets) triplets
    for training DispatchGNN.

    Each sample:
        u_zt: [Z, T, 7]  binary decisions (from MILP ground truth)
        h_zt: [Z, T, D]  HTE zone embeddings
        y_zt: [Z, T, 11] continuous dispatch targets
        n_zones: int
        scenario_id: str
        objective: float (MILP objective value)
    """

    def __init__(
        self,
        reports_dir: str,
        embeddings_dir: str,
        n_timesteps: int = 24,
        embed_dim: int = 128,
        max_scenarios: Optional[int] = None,
    ):
        self.reports_dir = reports_dir
        self.embeddings_dir = embeddings_dir
        self.n_timesteps = n_timesteps
        self.embed_dim = embed_dim

        # Build embedding lookup
        self._embedding_lookup: Dict[str, str] = {}
        self._build_embedding_lookup()

        # Scan reports and build index
        self.valid_scenarios: List[Dict] = []
        self._build_index(max_scenarios)

    def _build_embedding_lookup(self):
        """Recursively scan embeddings_dir for .npz/.pt files."""
        if not os.path.isdir(self.embeddings_dir):
            print(f"  WARNING: embeddings dir not found: {self.embeddings_dir}")
            return

        n_found = 0
        for root, _dirs, files in os.walk(self.embeddings_dir):
            for fname in files:
                ext = os.path.splitext(fname)[1]
                if ext not in (".npz", ".pt", ".npy"):
                    continue
                stem = os.path.splitext(fname)[0]
                parts = stem.replace("\\", "/").split("/")
                scenario_part = parts[-1]
                if scenario_part.startswith("scenario_"):
                    full_path = os.path.join(root, fname)
                    self._embedding_lookup[scenario_part] = full_path
                    n_found += 1

        if n_found > 0:
            print(f"  Found {n_found} embedding files")
        else:
            print(f"  WARNING: no embedding files found in {self.embeddings_dir}")

    def _build_index(self, max_scenarios: Optional[int] = None):
        """Scan reports directory and filter to scenarios with embeddings."""
        if not os.path.isdir(self.reports_dir):
            print(f"  WARNING: reports dir not found: {self.reports_dir}")
            return

        report_files = sorted(glob.glob(os.path.join(self.reports_dir, "scenario_*.json")))
        print(f"  Found {len(report_files)} report files")

        n_valid = 0
        for rpath in report_files:
            scenario_id = os.path.splitext(os.path.basename(rpath))[0]
            if scenario_id not in self._embedding_lookup:
                continue
            self.valid_scenarios.append({
                "scenario_id": scenario_id,
                "report_path": rpath,
            })
            n_valid += 1
            if max_scenarios and n_valid >= max_scenarios:
                break

        print(f"  Valid scenarios (report + embedding): {len(self.valid_scenarios)}")

    def _load_embedding(self, scenario_id: str) -> Optional[torch.Tensor]:
        """Load zone-level embedding [Z, T, D] from file."""
        path = self._embedding_lookup.get(scenario_id)
        if path is None:
            return None

        ext = os.path.splitext(path)[1]
        if ext == ".npz":
            data = np.load(path, allow_pickle=True)
            for key in ["zones", "zone_embeddings", "embeddings", "zone", "arr_0"]:
                if key in data:
                    arr = data[key]
                    if isinstance(arr, np.ndarray):
                        return torch.from_numpy(arr).float()
            for key in data.keys():
                arr = data[key]
                if isinstance(arr, np.ndarray) and arr.ndim >= 2:
                    return torch.from_numpy(arr).float()
        elif ext == ".pt":
            obj = torch.load(path, map_location="cpu", weights_only=False)
            if isinstance(obj, torch.Tensor):
                return obj.float()
            if isinstance(obj, dict):
                for key in ["zones", "zone_embeddings", "embeddings"]:
                    if key in obj and isinstance(obj[key], torch.Tensor):
                        return obj[key].float()
        elif ext == ".npy":
            return torch.from_numpy(np.load(path)).float()

        return None

    def __len__(self) -> int:
        return len(self.valid_scenarios)

    def __getitem__(self, idx: int) -> Dict:
        info = self.valid_scenarios[idx]
        scenario_id = info["scenario_id"]

        # Load report
        with open(info["report_path"], "r") as f:
            report = json.load(f)

        # Extract binaries [Z, T, 7]
        u_zt = extract_binaries_from_report(report, self.n_timesteps)
        if u_zt is None:
            u_zt = torch.zeros(1, self.n_timesteps, N_BINARY_FEATURES)

        # Extract dispatch targets [Z, T, 11]
        result = extract_dispatch_from_report(report, self.n_timesteps)
        if result is not None:
            y_zt, zone_names = result
        else:
            y_zt = torch.zeros(u_zt.shape[0], self.n_timesteps, N_DISPATCH)

        Z = u_zt.shape[0]

        # Load HTE embeddings [Z, T, D]
        h_zt = self._load_embedding(scenario_id)
        if h_zt is None:
            h_zt = torch.zeros(Z, self.n_timesteps, self.embed_dim)

        # Align zone dimensions between u_zt, h_zt, y_zt
        Z_u = u_zt.shape[0]
        Z_h = h_zt.shape[0]
        Z_y = y_zt.shape[0]
        Z_max = max(Z_u, Z_y)

        if Z_h < Z_max:
            h_zt = F_nn.pad(h_zt, (0, 0, 0, 0, 0, Z_max - Z_h))
        elif Z_h > Z_max:
            h_zt = h_zt[:Z_max]

        if Z_u < Z_max:
            u_zt = F_nn.pad(u_zt, (0, 0, 0, 0, 0, Z_max - Z_u))
        if Z_y < Z_max:
            y_zt = F_nn.pad(y_zt, (0, 0, 0, 0, 0, Z_max - Z_y))

        # Ensure temporal dim matches
        for_tensors = [u_zt, h_zt, y_zt]
        fixed = []
        for t in for_tensors:
            if t.shape[1] > self.n_timesteps:
                fixed.append(t[:, :self.n_timesteps, :])
            elif t.shape[1] < self.n_timesteps:
                pad_t = self.n_timesteps - t.shape[1]
                fixed.append(F_nn.pad(t, (0, 0, 0, pad_t)))
            else:
                fixed.append(t)
        u_zt, h_zt, y_zt = fixed

        # Ensure embed_dim
        if h_zt.shape[-1] != self.embed_dim:
            if h_zt.shape[-1] > self.embed_dim:
                h_zt = h_zt[:, :, :self.embed_dim]
            else:
                h_zt = F_nn.pad(h_zt, (0, self.embed_dim - h_zt.shape[-1]))

        n_zones = Z_max
        objective = float(report.get("mip", {}).get("objective", 0.0))

        return {
            "u_zt": u_zt.float(),
            "h_zt": h_zt.float(),
            "y_zt": y_zt.float(),
            "n_zones": n_zones,
            "n_timesteps": self.n_timesteps,
            "scenario_id": scenario_id,
            "objective": objective,
        }


def dispatch_collate_fn(batch: List[Dict]) -> Dict:
    """
    Collate function for variable-sized zone data.
    Pads zones to max in batch and creates zone_mask.

    Returns:
        u_zt:       [B, Z_max, T, 7]
        h_zt:       [B, Z_max, T, D]
        y_zt:       [B, Z_max, T, 11]
        zone_mask:  [B, Z_max]
        n_zones:    [B]
        objectives: [B]
        scenario_ids: List[str]
    """
    max_zones = max(d["n_zones"] for d in batch)
    n_timesteps = batch[0]["n_timesteps"]

    u_batch, h_batch, y_batch = [], [], []
    zone_masks, n_zones_list = [], []
    objectives, scenario_ids = [], []

    for d in batch:
        Z = d["n_zones"]
        pad_z = max_zones - Z

        u_batch.append(F_nn.pad(d["u_zt"], (0, 0, 0, 0, 0, pad_z)))
        h_batch.append(F_nn.pad(d["h_zt"], (0, 0, 0, 0, 0, pad_z)))
        y_batch.append(F_nn.pad(d["y_zt"], (0, 0, 0, 0, 0, pad_z)))

        mask = torch.cat([torch.ones(Z), torch.zeros(pad_z)])
        zone_masks.append(mask)

        n_zones_list.append(Z)
        objectives.append(d.get("objective", 0.0))
        scenario_ids.append(d["scenario_id"])

    return {
        "u_zt": torch.stack(u_batch),
        "h_zt": torch.stack(h_batch),
        "y_zt": torch.stack(y_batch),
        "zone_mask": torch.stack(zone_masks),
        "n_zones": torch.tensor(n_zones_list),
        "objectives": torch.tensor(objectives, dtype=torch.float32),
        "scenario_ids": scenario_ids,
    }


def compute_channel_stats(
    dataloader: DataLoader,
    n_dispatch: int = N_DISPATCH,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Compute per-channel mean and std from a dataloader for normalization.

    Returns:
        mean: [C]
        std:  [C] (clamped to min 1.0 to avoid division by zero)
    """
    sum_vals = torch.zeros(n_dispatch, dtype=torch.float64)
    sum_sq = torch.zeros(n_dispatch, dtype=torch.float64)
    count = torch.zeros(n_dispatch, dtype=torch.float64)

    for batch in dataloader:
        y = batch["y_zt"]                    # [B, Z, T, C]
        mask = batch["zone_mask"]            # [B, Z]
        valid = mask.unsqueeze(-1).unsqueeze(-1).float()  # [B, Z, 1, 1]

        # Only count valid zones
        y_valid = y * valid                  # [B, Z, T, C]
        n_valid = valid.sum() * y.shape[2]   # total valid (zone, time) pairs

        for c in range(n_dispatch):
            vals = y_valid[:, :, :, c]       # [B, Z, T]
            mask_flat = valid.squeeze(-1).squeeze(-1).unsqueeze(-1).expand_as(vals)
            sum_vals[c] += (vals * mask_flat).sum().item()
            sum_sq[c] += ((vals * mask_flat) ** 2).sum().item()
            count[c] += mask_flat.sum().item()

    mean = (sum_vals / count.clamp(min=1)).float()
    std = ((sum_sq / count.clamp(min=1) - mean.double() ** 2).clamp(min=0).sqrt()).float()
    std = std.clamp(min=1.0)  # avoid division by zero

    return mean, std


def build_dispatch_dataloaders(
    reports_dir: str,
    embeddings_dir: str,
    n_timesteps: int = 24,
    embed_dim: int = 128,
    batch_size: int = 16,
    val_split: float = 0.1,
    num_workers: int = 2,
    seed: int = 42,
    max_scenarios: Optional[int] = None,
) -> Tuple[DataLoader, DataLoader, DispatchDataset]:
    """
    Build train/val DataLoaders for dispatch prediction.

    Returns:
        train_loader, val_loader, full_dataset
    """
    print("Building DispatchDataset...")
    dataset = DispatchDataset(
        reports_dir=reports_dir,
        embeddings_dir=embeddings_dir,
        n_timesteps=n_timesteps,
        embed_dim=embed_dim,
        max_scenarios=max_scenarios,
    )

    if len(dataset) == 0:
        raise ValueError("No valid scenarios found!")

    val_size = max(1, int(len(dataset) * val_split))
    train_size = len(dataset) - val_size

    generator = torch.Generator().manual_seed(seed)
    train_ds, val_ds = random_split(dataset, [train_size, val_size], generator=generator)

    pin_memory = torch.cuda.is_available()

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        collate_fn=dispatch_collate_fn,
        drop_last=True,
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        collate_fn=dispatch_collate_fn,
    )

    print(f"  Train: {train_size} samples, {len(train_loader)} batches")
    print(f"  Val:   {val_size} samples, {len(val_loader)} batches")

    return train_loader, val_loader, dataset
