# ==============================================================================
# EMBEDDING PROCESSOR FOR PREFERENCE LEARNING
# ==============================================================================
# Converts full-scale HTE embeddings to zonal format for EBM conditioning
# Based on train_ebm_evaluate_langevin_V1.2_zonal_temporal.ipynb
# ==============================================================================

from __future__ import annotations

import os
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass


@dataclass
class EmbeddingConfig:
    """Configuration for embedding processing."""
    embed_dim: int = 128
    n_timesteps: int = 24
    n_features: int = 8  # Binary decision features per (zone, timestep)
    temporal_aggregation: str = "none"  # "none", "mean", "max", "last"


class ZonalEmbeddingProcessor:
    """
    Processes HTE embeddings from multiscale format to zone-level format.
    
    Converts from:
        - Full multiscale: {'assets': [N_a, T, D], 'zones': [N_z, T, D], 'regions': [...], 'nation': [...]}
        - Or concatenated: [total_nodes, T, D]
    
    To zone-level format:
        - {scenario_id: tensor [Z, T, D]}
    
    This is required because:
        1. HTE produces embeddings at multiple spatial scales
        2. EBM conditions on zone-level decisions u ∈ [Z, T, 8]
        3. We need h ∈ [Z, T, D] to match the decision structure
    """
    
    def __init__(
        self,
        config: Optional[EmbeddingConfig] = None,
    ):
        self.config = config or EmbeddingConfig()
    
    def load_and_process(
        self,
        embedding_path: str,
        dataset_index_path: Optional[str] = None,
        output_path: Optional[str] = None,
        level: str = "zones",
    ) -> Dict[str, torch.Tensor]:
        """
        Load multiscale embeddings and extract zone-level.
        
        Args:
            embedding_path: Path to embeddings_multiscale_normalized.pt
            dataset_index_path: Path to dataset_index.json (for zone counts)
            output_path: Optional path to save processed embeddings
            level: Which level to extract ("zones", "assets", "regions", "nation")
        
        Returns:
            Dict {scenario_id: tensor [Z, T, D]}
        """
        print(f"📂 Loading embeddings from {embedding_path}...")
        raw_data = torch.load(embedding_path, map_location='cpu', weights_only=False)
        
        # Check format
        if isinstance(raw_data, dict):
            if 'embeddings' in raw_data:
                # Format: {'embeddings': {'zones': [...], 'assets': [...], ...}}
                all_embeddings = raw_data['embeddings']
            elif level in raw_data:
                # Format: {'zones': [...], 'assets': [...], ...}
                all_embeddings = raw_data
            else:
                # Already scenario-keyed: {scenario_id: tensor}
                print(f"✓ Embeddings already in scenario format: {len(raw_data)} scenarios")
                return raw_data
        else:
            raise ValueError(f"Unknown embedding format: {type(raw_data)}")
        
        # Extract the requested level
        if level not in all_embeddings:
            available = list(all_embeddings.keys())
            raise KeyError(f"Level '{level}' not found. Available: {available}")
        
        emb_list = all_embeddings[level]
        
        # Concatenate batches if list
        if isinstance(emb_list, list):
            full_tensor = torch.cat(emb_list, dim=0)
        else:
            full_tensor = emb_list
        
        print(f"   Raw tensor shape: {full_tensor.shape}")
        
        # Need dataset index to split by scenario
        if dataset_index_path is None:
            print("⚠️ No dataset_index_path provided, returning raw tensor")
            return {"_raw_": full_tensor}
        
        # Load zone counts per scenario
        zone_counts, scenario_ids = self._load_zone_counts(dataset_index_path)
        
        # Verify total matches
        total_zones = sum(zone_counts)
        if total_zones != full_tensor.shape[0]:
            print(f"⚠️ Zone count mismatch: {total_zones} vs {full_tensor.shape[0]}")
            # Try to handle gracefully
            if full_tensor.shape[0] > total_zones:
                full_tensor = full_tensor[:total_zones]
            else:
                # Pad with zeros
                pad_size = total_zones - full_tensor.shape[0]
                full_tensor = F.pad(full_tensor, (0, 0, 0, 0, 0, pad_size))
        
        # Split into per-scenario tensors
        formatted_dict = {}
        current_idx = 0
        
        for i, sc_id in enumerate(scenario_ids):
            n_zones = zone_counts[i]
            sc_tensor = full_tensor[current_idx : current_idx + n_zones].clone()
            current_idx += n_zones
            
            # Clean scenario ID
            clean_id = os.path.splitext(os.path.basename(sc_id.replace('\\', '/')))[0]
            formatted_dict[clean_id] = sc_tensor
        
        print(f"✓ Processed {len(formatted_dict)} scenarios")
        
        # Optionally save
        if output_path:
            print(f"💾 Saving to {output_path}...")
            torch.save(formatted_dict, output_path)
        
        return formatted_dict
    
    def _load_zone_counts(
        self,
        dataset_index_path: str,
    ) -> Tuple[List[int], List[str]]:
        """Load zone counts per scenario from dataset index."""
        print(f"📖 Loading dataset index from {dataset_index_path}...")
        
        with open(dataset_index_path, 'r') as f:
            index_data = json.load(f)
        
        scenario_ids = []
        zone_counts = []
        
        base_dir = Path(dataset_index_path).parent.parent  # Go up from index file
        
        for entry in index_data.get('entries', []):
            fname = os.path.basename(entry['graph_file'])
            scenario_ids.append(fname)
            
            # Load graph file to get zone count
            graph_file = entry['graph_file'].replace('\\', '/')
            graph_path = base_dir / graph_file
            
            if graph_path.exists():
                graph_data = np.load(graph_path, allow_pickle=True)
                n_zones = len(graph_data.get('zone_region_index', []))
            else:
                n_zones = 50  # Default fallback
            
            zone_counts.append(n_zones)
        
        print(f"   Found {len(scenario_ids)} scenarios")
        print(f"   Zone counts - min: {min(zone_counts)}, max: {max(zone_counts)}")
        
        return zone_counts, scenario_ids
    
    def aggregate_temporal(
        self,
        h_zt: torch.Tensor,
        method: str = "mean",
    ) -> torch.Tensor:
        """
        Aggregate temporal embeddings to a single vector per zone.
        
        Args:
            h_zt: [Z, T, D] zone-temporal embeddings
            method: "mean", "max", "last", "first"
        
        Returns:
            h_z: [Z, D] aggregated embeddings
        """
        if method == "mean":
            return h_zt.mean(dim=1)
        elif method == "max":
            return h_zt.max(dim=1)[0]
        elif method == "last":
            return h_zt[:, -1, :]
        elif method == "first":
            return h_zt[:, 0, :]
        else:
            raise ValueError(f"Unknown aggregation method: {method}")
    
    def get_global_embedding(
        self,
        h_zt: torch.Tensor,
        method: str = "mean",
    ) -> torch.Tensor:
        """
        Get a single global embedding from zone-temporal embeddings.
        
        Args:
            h_zt: [Z, T, D] zone-temporal embeddings
            method: "mean", "max"
        
        Returns:
            h: [D] global embedding
        """
        if method == "mean":
            return h_zt.mean(dim=(0, 1))
        elif method == "max":
            return h_zt.max(dim=0)[0].max(dim=0)[0]
        else:
            return h_zt.mean(dim=(0, 1))


class TemporalZonalDataset(torch.utils.data.Dataset):
    """
    Dataset that exposes temporal structure for zone-level EBM.
    
    Returns data with shape:
        u_zt: [Z, T, F] - binary decisions per (zone, time, feature)
        h_zt: [Z, T, D] - embeddings per (zone, time)
    
    This allows the model to learn intertemporal dependencies.
    """
    
    def __init__(
        self,
        scenarios_dir: str,
        zone_embeddings: Dict[str, torch.Tensor],
        milp_reports_dir: Optional[str] = None,
        n_features: int = 8,
        n_timesteps: int = 24,
    ):
        """
        Args:
            scenarios_dir: Directory with scenario JSON files
            zone_embeddings: Dict {scenario_id: tensor [Z, T, D]}
            milp_reports_dir: Directory with MILP report JSON files
            n_features: Number of binary features per (zone, time)
            n_timesteps: Number of timesteps
        """
        self.scenarios_dir = Path(scenarios_dir)
        self.zone_embeddings = zone_embeddings
        self.milp_reports_dir = Path(milp_reports_dir) if milp_reports_dir else None
        self.n_features = n_features
        self.n_timesteps = n_timesteps
        
        # Get scenario IDs that have embeddings
        self.scenario_ids = [
            sid for sid in zone_embeddings.keys()
            if (self.scenarios_dir / f"{sid}.json").exists() or
               (self.milp_reports_dir and (self.milp_reports_dir / f"{sid}.json").exists())
        ]
        
        print(f"✓ TemporalZonalDataset: {len(self.scenario_ids)} scenarios")
    
    def __len__(self) -> int:
        return len(self.scenario_ids)
    
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        scenario_id = self.scenario_ids[idx]
        
        # Get zone embeddings: [Z, T, D]
        h_zt = self.zone_embeddings[scenario_id].clone()
        n_zones = h_zt.shape[0]
        n_timesteps = h_zt.shape[1] if h_zt.dim() > 2 else self.n_timesteps
        embed_dim = h_zt.shape[-1]
        
        # Load MILP reference if available
        milp_objective = float("inf")
        if self.milp_reports_dir:
            report_path = self.milp_reports_dir / f"{scenario_id}.json"
            if report_path.exists():
                with open(report_path, 'r') as f:
                    report = json.load(f)
                milp_objective = report.get("mip", {}).get("objective", float("inf"))
        
        # Create placeholder decision tensor
        # In real training, this would be loaded from MILP solution
        u_zt = torch.zeros(n_zones, n_timesteps, self.n_features)
        
        return {
            'u_zt': u_zt.float(),           # [Z, T, F]
            'h_zt': h_zt.float(),           # [Z, T, D]
            'n_zones': n_zones,
            'n_timesteps': n_timesteps,
            'n_features': self.n_features,
            'embed_dim': embed_dim,
            'milp_objective': milp_objective,
            'scenario_id': scenario_id,
        }


def temporal_collate_fn(batch: List[Dict]) -> Dict[str, torch.Tensor]:
    """
    Collate function for variable-sized temporal zonal data.
    
    Pads zones to max in batch and creates masks.
    
    Args:
        batch: List of dicts from TemporalZonalDataset
    
    Returns:
        Dictionary with batched tensors:
            u_zt: [B, Z_max, T, F]
            h_zt: [B, Z_max, T, D]
            zone_mask: [B, Z_max] - 1 for valid zones, 0 for padding
            n_zones: [B] - original zone counts
    """
    # Find max zones in batch
    max_zones = max(d['n_zones'] for d in batch)
    n_timesteps = batch[0]['n_timesteps']
    n_features = batch[0]['n_features']
    embed_dim = batch[0]['embed_dim']
    
    u_batch = []
    h_batch = []
    zone_masks = []
    n_zones_list = []
    objectives = []
    scenario_ids = []
    
    for d in batch:
        Z = d['n_zones']
        pad_z = max_zones - Z
        
        # Pad u_zt: [Z, T, F] -> [Z_max, T, F]
        u_padded = F.pad(d['u_zt'], (0, 0, 0, 0, 0, pad_z))
        u_batch.append(u_padded)
        
        # Pad h_zt: [Z, T, D] -> [Z_max, T, D]
        h_padded = F.pad(d['h_zt'], (0, 0, 0, 0, 0, pad_z))
        h_batch.append(h_padded)
        
        # Zone mask: [Z_max]
        mask = torch.cat([torch.ones(Z), torch.zeros(pad_z)])
        zone_masks.append(mask)
        
        n_zones_list.append(Z)
        objectives.append(d['milp_objective'])
        scenario_ids.append(d['scenario_id'])
    
    return {
        'u_zt': torch.stack(u_batch),           # [B, Z_max, T, F]
        'h_zt': torch.stack(h_batch),           # [B, Z_max, T, D]
        'zone_mask': torch.stack(zone_masks),   # [B, Z_max]
        'n_zones': torch.tensor(n_zones_list),  # [B]
        'milp_objectives': torch.tensor(objectives),  # [B]
        'n_timesteps': n_timesteps,
        'n_features': n_features,
        'embed_dim': embed_dim,
        'scenario_ids': scenario_ids,
    }


def load_zone_embeddings(
    embedding_path: str,
    device: torch.device = torch.device("cpu"),
) -> Dict[str, torch.Tensor]:
    """
    Load zone-level embeddings from file.
    
    Args:
        embedding_path: Path to embeddings file
        device: Target device
    
    Returns:
        Dict {scenario_id: tensor [Z, T, D]}
    """
    print(f"📂 Loading zone embeddings from {embedding_path}...")
    
    data = torch.load(embedding_path, map_location=device, weights_only=False)
    
    # Handle different formats
    if isinstance(data, dict):
        if 'embeddings' in data:
            # Nested format
            embeddings = data['embeddings']
            if 'zones' in embeddings:
                return embeddings['zones']
        
        # Check if already scenario-keyed
        sample_key = next(iter(data.keys()))
        sample_val = data[sample_key]
        if isinstance(sample_val, torch.Tensor):
            print(f"✓ Loaded {len(data)} scenario embeddings")
            return data
    
    raise ValueError(f"Unknown embedding format in {embedding_path}")
