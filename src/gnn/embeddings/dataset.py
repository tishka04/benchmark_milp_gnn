"""
Dataset and data-loading utilities for v3 temporal heterogeneous graphs.

Key difference vs v1: N_base varies per scenario, so we use batch_size=1
and extract hierarchy mappings per-graph.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import random_split
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class TemporalGraphDatasetV3(torch.utils.data.Dataset):
    """
    Dataset loader for v3 temporal supra-graphs with variable N_base.

    Each .npz file contains:
        node_features, edge_index, edge_types, node_types, meta,
        zone_region_index, (optional) node_labels
    """

    def __init__(self, graph_files: List[Path], target_indices=None):
        self.graph_files = graph_files
        self.target_indices = target_indices

    def __len__(self):
        return len(self.graph_files)

    def __getitem__(self, idx):
        data_dict = np.load(self.graph_files[idx], allow_pickle=True)

        x = torch.from_numpy(data_dict["node_features"]).float()
        edge_index = torch.from_numpy(data_dict["edge_index"]).long()
        edge_type = torch.from_numpy(data_dict["edge_types"]).long()
        node_types = torch.from_numpy(data_dict["node_types"]).long()

        meta = data_dict["meta"].item()
        N_base = meta["N_base"]
        T = meta["T"]

        # ---- hierarchy: zone_to_region ----
        if "zone_region_index" in data_dict:
            zone_to_region = torch.from_numpy(data_dict["zone_region_index"]).long()
        else:
            base_node_types = node_types[:N_base]
            num_zones = (base_node_types == 2).sum().item()
            num_regions = max(1, (base_node_types == 1).sum().item())
            zone_to_region = torch.arange(num_zones) % num_regions

        # ---- hierarchy: asset_to_zone ----
        base_node_types = node_types[:N_base]
        asset_mask = base_node_types == 3
        zone_mask = base_node_types == 2
        zone_indices = torch.where(zone_mask)[0]

        asset_to_zone = torch.zeros(N_base, dtype=torch.long)

        spatial_mask = edge_type < 7
        spatial_edges = edge_index[:, spatial_mask]
        base_edges = spatial_edges % N_base

        for asset_idx in torch.where(asset_mask)[0]:
            outgoing = base_edges[0] == asset_idx
            if outgoing.any():
                targets = base_edges[1, outgoing]
                zone_targets = targets[zone_mask[targets]]
                if len(zone_targets) > 0:
                    zone_node_id = zone_targets[0].item()
                    zone_list_idx = (zone_indices == zone_node_id).nonzero(as_tuple=True)[0]
                    if len(zone_list_idx) > 0:
                        asset_to_zone[asset_idx] = zone_list_idx[0]

        asset_to_zone[~asset_mask] = 0

        # ---- labels (optional) ----
        y, label_mask = None, None
        if "node_labels" in data_dict and data_dict["node_labels"] is not None:
            zone_labels = torch.from_numpy(data_dict["node_labels"]).float()
            if self.target_indices is not None:
                zone_labels = zone_labels[:, :, self.target_indices]
            T_labels, N_zones, target_dim = zone_labels.shape
            zone_mask_base = base_node_types == 2

            y = torch.zeros(N_base * T, target_dim)
            label_mask = torch.zeros(N_base * T, dtype=torch.bool)
            for t in range(T):
                t_offset = t * N_base
                zone_indices_t = torch.where(zone_mask_base)[0] + t_offset
                y[zone_indices_t] = zone_labels[t]
                label_mask[zone_indices_t] = True

        data = Data(
            x=x,
            edge_index=edge_index,
            edge_type=edge_type,
            node_type=node_types,
            y=y,
            label_mask=label_mask,
            N_base=N_base,
            T=T,
            asset_to_zone=asset_to_zone,
            zone_to_region=zone_to_region,
        )
        return data


# ---------------------------------------------------------------------------
# Helper: load dataset + create train/val splits and loaders
# ---------------------------------------------------------------------------

def load_dataset_v3(
    config,
) -> Tuple[TemporalGraphDatasetV3, DataLoader, DataLoader, dict]:
    """
    Load the v3 dataset, split into train/val, and return loaders.

    Returns:
        (full_dataset, train_loader, val_loader, split_info)
    """
    data_dir = Path(config.data_dir)
    repo_root = Path(config.repo_path)
    index_path = data_dir / "dataset_index.json"

    with open(index_path, "r") as f:
        index_data = json.load(f)

    graph_files = []
    for entry in index_data["entries"]:
        rel_path = entry["graph_file"].replace("\\", "/")
        graph_files.append(repo_root / rel_path)

    dataset = TemporalGraphDatasetV3(graph_files)

    g = torch.Generator().manual_seed(config.seed)
    train_size = int(config.train_split * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(
        dataset, [train_size, val_size], generator=g,
    )

    # Save split for reproducibility
    split_info = {
        "seed": config.seed,
        "train_size": train_size,
        "val_size": val_size,
        "train_indices": train_dataset.indices,
        "val_indices": val_dataset.indices,
    }
    split_path = data_dir / f"split_seed{config.seed}_train{train_size}_val{val_size}.json"
    split_path.write_text(json.dumps(split_info, indent=2))

    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
        pin_memory=False,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        pin_memory=False,
    )
    return dataset, train_loader, val_loader, split_info


# ---------------------------------------------------------------------------
# Helper: compute normalization statistics from training set
# ---------------------------------------------------------------------------

def compute_norm_stats(
    train_dataset, device: str = "cpu",
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Compute per-feature mean and std over the training set.

    Returns:
        (mean, std) tensors on *device*, with std clamped >= 1e-6.
    """
    all_features = []
    indices = torch.randperm(len(train_dataset))
    for idx in indices:
        sample = train_dataset[idx]
        if hasattr(sample, "x") and sample.x is not None:
            all_features.append(sample.x)

    cat = torch.cat(all_features, dim=0)
    mean = cat.mean(dim=0).to(device)
    std = cat.std(dim=0).to(device)
    std = torch.where(std < 1e-6, torch.ones_like(std), std)
    return mean, std


# ---------------------------------------------------------------------------
# Helper: extract hierarchy from a batch
# ---------------------------------------------------------------------------

def get_hierarchy_from_batch(batch, device: str) -> dict:
    """
    Extract asset_to_zone and zone_to_region from a PyG batch object.
    """
    if hasattr(batch, "asset_to_zone") and hasattr(batch, "zone_to_region"):
        return {
            "asset_to_zone": batch.asset_to_zone.to(device),
            "zone_to_region": batch.zone_to_region.to(device),
        }
    raise RuntimeError(
        "No hierarchy mapping found in batch. "
        "Ensure asset_to_zone and zone_to_region are stored in each Data object."
    )
