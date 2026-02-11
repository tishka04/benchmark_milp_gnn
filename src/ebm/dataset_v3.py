# ==============================================================================
# DATASET V3 - Scenario Report Dataset for Graph EBM
# ==============================================================================
# Loads binary decisions from MILP reports and pairs with HTE embeddings.
# Handles variable zone counts via padding in temporal_collate_fn.
#
# Binary features [Z, T, 7]:
#   0: battery_charge_mode   (b_charge > 0)
#   1: battery_discharge_mode (b_discharge > 0)
#   2: pumped_charge_mode
#   3: pumped_discharge_mode
#   4: dr_active
#   5: thermal_su            (thermal start-up: off→on transition)
#   6: thermal_on            (u_thermal or p_thermal > 0)
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


# ── Feature indices ──
FEAT_BATT_CHARGE = 0
FEAT_BATT_DISCHARGE = 1
FEAT_PUMP_CHARGE = 2
FEAT_PUMP_DISCHARGE = 3
FEAT_DR = 4
FEAT_THERMAL_SU = 5
FEAT_THERMAL = 6
N_FEATURES = 7


def load_classification_index(path: str) -> Dict[str, List[str]]:
    """Load gold/silver scenario classification from index JSON."""
    with open(path, "r") as f:
        data = json.load(f)
    return {
        "gold": data.get("gold", []),
        "silver": data.get("silver", []),
        "summary": data.get("summary", {}),
    }


def _extract_binaries_from_report(
    report: dict,
    n_timesteps: int = 24,
) -> Optional[torch.Tensor]:
    """
    Extract binary decision tensor [Z, T, 7] from a MILP report JSON.

    The report stores dispatch results under 'detail' with explicit binary
    keys: thermal_commitment, thermal_startup, battery_charge_mode,
    pumped_charge_mode, dr_active.  Battery discharge mode is inferred
    as (1 - battery_charge_mode) when battery is active.

    Falls back to 'pre_dispatch' if 'detail' is not available.
    """
    # Locate the dispatch data section
    detail = report.get("detail", {})
    pd = report.get("pre_dispatch", {})

    # Prefer 'detail' (v3 report format), fallback to 'pre_dispatch'
    src = detail if detail else pd
    if not src:
        return None

    # Get zone names from report top-level or from thermal data
    zone_names = report.get("zones")
    if not zone_names:
        # Fallback: infer from any per-zone dict in src
        for probe_key in ["thermal_commitment", "thermal", "demand"]:
            probe = src.get(probe_key, {})
            if isinstance(probe, dict) and probe:
                zone_names = sorted(probe.keys())
                break
    if not zone_names:
        return None

    Z = len(zone_names)
    T = n_timesteps

    u = torch.zeros(Z, T, N_FEATURES)

    # Helper to read a per-zone time series from src
    def _get(key: str, zone: str) -> list:
        d = src.get(key, {})
        if isinstance(d, dict):
            return d.get(zone, [])
        return []

    for z_idx, zone in enumerate(zone_names):
        # ── Thermal commitment (binary) ──
        tc = _get("thermal_commitment", zone)
        for t in range(min(T, len(tc))):
            u[z_idx, t, FEAT_THERMAL] = 1.0 if abs(tc[t]) > 0.5 else 0.0

        # ── Thermal start-up (binary, directly from report) ──
        tsu = _get("thermal_startup", zone)
        for t in range(min(T, len(tsu))):
            u[z_idx, t, FEAT_THERMAL_SU] = 1.0 if abs(tsu[t]) > 0.5 else 0.0

        # If thermal_startup not in report, derive from commitment transitions
        if not tsu and tc:
            for t in range(min(T, len(tc))):
                on_now = abs(tc[t]) > 0.5
                on_prev = abs(tc[t - 1]) > 0.5 if t > 0 else False
                u[z_idx, t, FEAT_THERMAL_SU] = 1.0 if (on_now and not on_prev) else 0.0

        # ── Battery charge / discharge (from actual dispatch MW) ──
        # Use continuous values, not charge_mode (which is a mutual-exclusion
        # helper and reads 0 even for zones without batteries).
        bch = _get("battery_charge", zone)
        bdis = _get("battery_discharge", zone)
        for t in range(T):
            ch = bch[t] if t < len(bch) else 0.0
            dis = bdis[t] if t < len(bdis) else 0.0
            u[z_idx, t, FEAT_BATT_CHARGE] = 1.0 if ch > 1e-3 else 0.0
            u[z_idx, t, FEAT_BATT_DISCHARGE] = 1.0 if dis > 1e-3 else 0.0

        # ── Pumped charge / discharge (from actual dispatch MW) ──
        pch = _get("pumped_charge", zone)
        pdis = _get("pumped_discharge", zone)
        for t in range(T):
            ch = pch[t] if t < len(pch) else 0.0
            dis = pdis[t] if t < len(pdis) else 0.0
            u[z_idx, t, FEAT_PUMP_CHARGE] = 1.0 if ch > 1e-3 else 0.0
            u[z_idx, t, FEAT_PUMP_DISCHARGE] = 1.0 if dis > 1e-3 else 0.0

        # ── Demand response active (binary) ──
        dra = _get("dr_active", zone)
        for t in range(min(T, len(dra))):
            u[z_idx, t, FEAT_DR] = 1.0 if abs(dra[t]) > 0.5 else 0.0

        # Fallback: infer from continuous DR shed
        if not dra:
            drs = _get("demand_response", zone)
            if not drs:
                drs = _get("dr_shed", zone)
            for t in range(min(T, len(drs))):
                u[z_idx, t, FEAT_DR] = 1.0 if abs(drs[t]) > 1e-3 else 0.0

    return u


class ScenarioReportDataset(Dataset):
    """
    Dataset that loads binary decisions from MILP reports and
    pairs them with zone-level HTE embeddings.

    Returns:
        u_zt: [Z, T, F] binary decisions
        h_zt: [Z, T, D] zone-level embeddings
        n_zones: int
        n_timesteps: int
        n_features: int
        scenario_id: str
        objective: float (MILP objective for preference learning)
    """

    def __init__(
        self,
        reports_dir: str,
        embeddings_dir: str,
        scenario_files: List[str],
        n_timesteps: int = 24,
        embed_dim: int = 128,
        device: str = "cpu",
    ):
        self.reports_dir = reports_dir
        self.embeddings_dir = embeddings_dir
        self.n_timesteps = n_timesteps
        self.embed_dim = embed_dim
        self.device = device

        # Build scenario_id -> embedding file path lookup
        # Handles nested paths like embeddings_v3/outputs/scenarios_v3/scenario_00001.npz
        self._embedding_lookup: Dict[str, str] = {}
        self._consolidated_embeddings: Optional[Dict[str, torch.Tensor]] = None
        self._build_embedding_lookup()

        # Filter to scenarios that have both report and embedding
        self.valid_scenarios: List[Dict] = []
        self._build_index(scenario_files)

    def _build_embedding_lookup(self):
        """
        Scan embeddings_dir recursively and build a map from
        scenario_id -> absolute file path.

        Handles nested structures like:
          embeddings_v3/outputs/scenarios_v3/scenario_00001.npz
        """
        if not os.path.isdir(self.embeddings_dir):
            print(f"  WARNING: embeddings dir does not exist: {self.embeddings_dir}")
            return

        n_found = 0
        for root, _dirs, files in os.walk(self.embeddings_dir):
            for fname in files:
                ext = os.path.splitext(fname)[1]
                if ext not in (".npz", ".pt", ".npy"):
                    continue
                stem = os.path.splitext(fname)[0]
                # Extract scenario_id from stem which may contain path
                # separators, e.g. "outputs\scenarios_v3\scenario_00001"
                # Split on both / and \ then take the last part
                parts = stem.replace("\\", "/").split("/")
                scenario_part = parts[-1]
                if scenario_part.startswith("scenario_"):
                    full_path = os.path.join(root, fname)
                    self._embedding_lookup[scenario_part] = full_path
                    n_found += 1

        # Also try loading a consolidated .pt dict
        if n_found == 0:
            self._try_load_consolidated()

        if n_found > 0:
            sample = next(iter(self._embedding_lookup.values()))
            print(f"  Found {n_found} embedding files (sample: {sample})")
        elif self._consolidated_embeddings:
            pass  # already printed
        else:
            print(f"  WARNING: no embedding files found in {self.embeddings_dir}")

    def _try_load_consolidated(self):
        """Load a consolidated .pt embeddings dict if it exists."""
        candidates = []
        if os.path.isdir(self.embeddings_dir):
            candidates += glob.glob(os.path.join(self.embeddings_dir, "*.pt"))
        parent = os.path.dirname(self.embeddings_dir)
        if os.path.isdir(parent):
            candidates += glob.glob(os.path.join(parent, "*.pt"))

        for path in candidates:
            try:
                data = torch.load(path, map_location="cpu", weights_only=False)
                if isinstance(data, dict) and len(data) > 10:
                    sample_key = next(iter(data))
                    if "scenario" in sample_key:
                        self._consolidated_embeddings = data
                        print(f"  Loaded consolidated embeddings: {path} "
                              f"({len(data)} scenarios)")
                        return
            except Exception:
                continue

    def _build_index(self, scenario_files: List[str]):
        """Build index of valid (report, embedding) pairs."""
        for fname in scenario_files:
            scenario_id = fname.replace(".json", "")
            report_path = os.path.join(self.reports_dir, fname)

            if not os.path.exists(report_path):
                continue

            if not self._has_embedding(scenario_id):
                continue

            self.valid_scenarios.append({
                "scenario_id": scenario_id,
                "report_path": report_path,
            })

    def _has_embedding(self, scenario_id: str) -> bool:
        """Check if an embedding exists for this scenario."""
        if scenario_id in self._embedding_lookup:
            return True
        if self._consolidated_embeddings is not None:
            return scenario_id in self._consolidated_embeddings
        return False

    def _load_embedding_file(self, path: str) -> Optional[torch.Tensor]:
        """Load zone-level embedding [Z, T, D] from a single file."""
        ext = os.path.splitext(path)[1]

        if ext == ".npz":
            data = np.load(path, allow_pickle=True)
            # Keys from generate.py: zones [Z,T,D], assets, regions, nation
            for key in ["zones", "zone_embeddings", "embeddings", "zone", "arr_0"]:
                if key in data:
                    arr = data[key]
                    if isinstance(arr, np.ndarray):
                        return torch.from_numpy(arr).float()
            # Fallback: first array-like key
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

    def _load_embedding(self, scenario_id: str) -> Optional[torch.Tensor]:
        """Load zone-level embedding [Z, T, D] for a scenario."""
        # 1. Per-scenario file (from recursive lookup)
        path = self._embedding_lookup.get(scenario_id)
        if path is not None:
            return self._load_embedding_file(path)

        # 2. Consolidated dict
        if self._consolidated_embeddings is not None:
            tensor = self._consolidated_embeddings.get(scenario_id)
            if tensor is not None:
                if isinstance(tensor, torch.Tensor):
                    return tensor.float()
                return torch.tensor(tensor).float()

        return None

    def __len__(self) -> int:
        return len(self.valid_scenarios)

    def __getitem__(self, idx: int) -> Dict:
        info = self.valid_scenarios[idx]
        scenario_id = info["scenario_id"]

        # Load report and extract binaries
        with open(info["report_path"], "r") as f:
            report = json.load(f)

        u_zt = _extract_binaries_from_report(report, self.n_timesteps)
        if u_zt is None:
            # Fallback: zeros
            u_zt = torch.zeros(1, self.n_timesteps, N_FEATURES)

        Z = u_zt.shape[0]

        # Load embedding
        h_zt = self._load_embedding(scenario_id)
        if h_zt is None:
            h_zt = torch.zeros(Z, self.n_timesteps, self.embed_dim)

        # Handle dimension mismatches between u_zt and h_zt
        Z_u = u_zt.shape[0]
        Z_h = h_zt.shape[0]

        if Z_h < Z_u:
            # Pad embedding zones
            pad_z = Z_u - Z_h
            h_zt = F_nn.pad(h_zt, (0, 0, 0, 0, 0, pad_z))
        elif Z_h > Z_u:
            # Truncate embedding to match
            h_zt = h_zt[:Z_u]

        # Ensure temporal dim matches
        if h_zt.shape[1] != self.n_timesteps:
            if h_zt.shape[1] > self.n_timesteps:
                h_zt = h_zt[:, :self.n_timesteps, :]
            else:
                pad_t = self.n_timesteps - h_zt.shape[1]
                h_zt = F_nn.pad(h_zt, (0, 0, 0, pad_t))

        # Ensure embed_dim matches
        if h_zt.shape[-1] != self.embed_dim:
            if h_zt.shape[-1] > self.embed_dim:
                h_zt = h_zt[:, :, :self.embed_dim]
            else:
                pad_d = self.embed_dim - h_zt.shape[-1]
                h_zt = F_nn.pad(h_zt, (0, pad_d))

        n_zones = u_zt.shape[0]

        # Extract MILP objective for preference learning
        objective = float(report.get("mip", {}).get("objective", 0.0))

        return {
            "u_zt": u_zt.float(),
            "h_zt": h_zt.float(),
            "n_zones": n_zones,
            "n_timesteps": self.n_timesteps,
            "n_features": N_FEATURES,
            "scenario_id": scenario_id,
            "objective": objective,
        }


def temporal_collate_fn(batch: List[Dict]) -> Dict:
    """
    Collate function for variable-sized temporal zonal data.
    Pads zones to max in batch and creates masks.

    Returns:
        u_zt: [B, Z_max, T, F]
        h_zt: [B, Z_max, T, D]
        zone_mask: [B, Z_max]
        n_zones: [B]
        objectives: [B]
        scenario_ids: List[str]
    """
    max_zones = max(d["n_zones"] for d in batch)
    n_timesteps = batch[0]["n_timesteps"]
    n_features = batch[0]["n_features"]
    embed_dim = batch[0]["h_zt"].shape[-1]

    u_batch = []
    h_batch = []
    zone_masks = []
    n_zones_list = []
    objectives = []
    scenario_ids = []

    for d in batch:
        Z = d["n_zones"]
        pad_z = max_zones - Z

        u_padded = F_nn.pad(d["u_zt"], (0, 0, 0, 0, 0, pad_z))
        u_batch.append(u_padded)

        h_padded = F_nn.pad(d["h_zt"], (0, 0, 0, 0, 0, pad_z))
        h_batch.append(h_padded)

        mask = torch.cat([torch.ones(Z), torch.zeros(pad_z)])
        zone_masks.append(mask)

        n_zones_list.append(Z)
        objectives.append(d.get("objective", 0.0))
        scenario_ids.append(d["scenario_id"])

    return {
        "u_zt": torch.stack(u_batch),
        "h_zt": torch.stack(h_batch),
        "zone_mask": torch.stack(zone_masks),
        "n_zones": torch.tensor(n_zones_list),
        "n_timesteps": n_timesteps,
        "n_features": n_features,
        "objectives": torch.tensor(objectives, dtype=torch.float32),
        "scenario_ids": scenario_ids,
    }


def build_dataloaders(
    reports_dir: str,
    embeddings_dir: str,
    classification_index_path: str,
    tier: str = "gold",
    n_timesteps: int = 24,
    embed_dim: int = 128,
    batch_size: int = 32,
    val_split: float = 0.1,
    num_workers: int = 0,
    seed: int = 42,
) -> Tuple[DataLoader, DataLoader, ScenarioReportDataset]:
    """
    Build train/val DataLoaders for a specific tier (gold or silver).

    Returns:
        train_loader, val_loader, full_dataset
    """
    index = load_classification_index(classification_index_path)
    scenario_files = index.get(tier, [])

    print(f"Building {tier} dataset: {len(scenario_files)} scenarios in index")

    dataset = ScenarioReportDataset(
        reports_dir=reports_dir,
        embeddings_dir=embeddings_dir,
        scenario_files=scenario_files,
        n_timesteps=n_timesteps,
        embed_dim=embed_dim,
    )

    print(f"  Valid scenarios with reports + embeddings: {len(dataset)}")

    if len(dataset) == 0:
        raise ValueError(f"No valid {tier} scenarios found!")

    # Split
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
        collate_fn=temporal_collate_fn,
        drop_last=True,
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        collate_fn=temporal_collate_fn,
    )

    print(f"  Train: {train_size} samples, {len(train_loader)} batches")
    print(f"  Val:   {val_size} samples, {len(val_loader)} batches")

    return train_loader, val_loader, dataset
