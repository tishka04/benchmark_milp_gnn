# ==============================================================================
# DATASET AND DATALOADER FOR PREFERENCE-BASED LEARNING
# ==============================================================================
# Loads scenarios, MILP references, and HTE embeddings for training
# ==============================================================================

from __future__ import annotations

import os
import json
import pickle
import torch
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any, Union
from dataclasses import dataclass
from torch.utils.data import Dataset, DataLoader

from .data_models import (
    ScenarioData, 
    DecisionVector, 
    MILPReference,
    PreferencePair,
    TrainingBatch,
)


@dataclass
class DatasetConfig:
    """Configuration for preference dataset."""
    scenarios_dir: str
    milp_reports_dir: str
    embeddings_path: Optional[str] = None  # Pre-computed HTE embeddings
    dispatch_dir: Optional[str] = None  # MILP dispatch solutions
    
    # Filtering
    max_scenarios: Optional[int] = None
    min_zones: int = 1
    max_zones: int = 200
    
    # Caching
    cache_dir: Optional[str] = None
    use_cache: bool = True


class PreferenceDataset(Dataset):
    """
    Dataset for preference-based EBM training.
    
    Each item contains:
    - Scenario data (for decoder and LP worker)
    - HTE embedding (for conditioning)
    - MILP reference decision (positive example)
    - Placeholder for negative examples (generated during training)
    """
    
    def __init__(
        self,
        config: DatasetConfig,
        scenario_ids: Optional[List[str]] = None,
        device: torch.device = torch.device("cpu"),
    ):
        """
        Args:
            config: Dataset configuration
            scenario_ids: Optional list of scenario IDs to use
            device: Target device for tensors
        """
        self.config = config
        self.device = device
        
        # Load scenario metadata
        self.scenario_ids = scenario_ids or self._discover_scenarios()
        
        # Load MILP references
        self.milp_references = self._load_milp_references()
        
        # Load HTE embeddings if available
        self.embeddings = self._load_embeddings()
        
        # Filter to scenarios with MILP references
        self.scenario_ids = [
            sid for sid in self.scenario_ids
            if sid in self.milp_references
        ]
        
        if config.max_scenarios:
            self.scenario_ids = self.scenario_ids[:config.max_scenarios]
        
        print(f"✓ PreferenceDataset initialized with {len(self.scenario_ids)} scenarios")
    
    def _discover_scenarios(self) -> List[str]:
        """Discover available scenario IDs."""
        scenarios_dir = Path(self.config.scenarios_dir)
        scenario_ids = []
        
        for f in scenarios_dir.glob("scenario_*.json"):
            scenario_ids.append(f.stem)
        
        scenario_ids.sort()
        return scenario_ids
    
    def _load_milp_references(self) -> Dict[str, Dict]:
        """Load MILP reports."""
        reports_dir = Path(self.config.milp_reports_dir)
        references = {}
        
        for f in reports_dir.glob("scenario_*.json"):
            scenario_id = f.stem
            try:
                with open(f, 'r') as fp:
                    report = json.load(fp)
                references[scenario_id] = report
            except Exception as e:
                print(f"Warning: Failed to load {f}: {e}")
        
        print(f"  Loaded {len(references)} MILP reports")
        return references
    
    def _load_embeddings(self) -> Optional[Dict[str, torch.Tensor]]:
        """Load pre-computed HTE embeddings."""
        if not self.config.embeddings_path:
            return None
        
        emb_path = Path(self.config.embeddings_path)
        if not emb_path.exists():
            print(f"  Warning: Embeddings not found at {emb_path}")
            return None
        
        try:
            embeddings = torch.load(emb_path, map_location=self.device)
            print(f"  Loaded embeddings for {len(embeddings)} scenarios")
            return embeddings
        except Exception as e:
            print(f"  Warning: Failed to load embeddings: {e}")
            return None
    
    def __len__(self) -> int:
        return len(self.scenario_ids)
    
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """
        Get a single training item.
        
        Returns dict with:
            - scenario_id: str
            - scenario_path: str
            - embedding: torch.Tensor [d] (if available)
            - milp_objective: float
            - milp_decision: torch.Tensor [Z, T, 8] (if dispatch available)
            - n_zones: int
            - n_timesteps: int
        """
        scenario_id = self.scenario_ids[idx]
        scenario_path = Path(self.config.scenarios_dir) / f"{scenario_id}.json"
        
        # Load basic scenario info
        with open(scenario_path, 'r') as f:
            scenario_data = json.load(f)
        
        n_zones = len(scenario_data.get("zones", []))
        n_timesteps = scenario_data.get("horizon", 24)
        
        # Get MILP reference
        milp_report = self.milp_references.get(scenario_id, {})
        milp_objective = milp_report.get("mip", {}).get("objective", float("inf"))
        
        # Get embedding
        if self.embeddings and scenario_id in self.embeddings:
            embedding = self.embeddings[scenario_id]
        else:
            # Placeholder - will be computed by HTE during training
            embedding = torch.zeros(128)
        
        # Load MILP decision if available
        milp_decision = self._load_milp_decision(scenario_id, n_zones, n_timesteps)
        
        return {
            "scenario_id": scenario_id,
            "scenario_path": str(scenario_path),
            "embedding": embedding,
            "milp_objective": milp_objective,
            "milp_decision": milp_decision,
            "n_zones": n_zones,
            "n_timesteps": n_timesteps,
            "raw_scenario": scenario_data,
        }
    
    def _load_milp_decision(
        self,
        scenario_id: str,
        n_zones: int,
        n_timesteps: int,
    ) -> Optional[torch.Tensor]:
        """Load MILP binary decision from dispatch directory."""
        if not self.config.dispatch_dir:
            return None
        
        dispatch_path = Path(self.config.dispatch_dir) / f"{scenario_id}.json"
        if not dispatch_path.exists():
            return None
        
        try:
            with open(dispatch_path, 'r') as f:
                dispatch = json.load(f)
            
            # Extract binary decisions
            u = torch.zeros(n_zones, n_timesteps, 8)
            
            detail = dispatch.get("detail", {})
            
            # Map dispatch to decision tensor
            # This depends on your dispatch format
            # Here's a generic approach:
            
            zones = list(dispatch.get("zones", {}).keys()) or \
                    [f"zone_{i}" for i in range(n_zones)]
            
            for z_idx, zone in enumerate(zones[:n_zones]):
                for t in range(n_timesteps):
                    # Battery mode
                    b_ch = self._get_value(detail, "battery_charge", zone, t)
                    b_dis = self._get_value(detail, "battery_discharge", zone, t)
                    u[z_idx, t, 0] = 1.0 if b_ch > b_dis else 0.0
                    u[z_idx, t, 1] = 1.0 if b_dis > b_ch else 0.0
                    
                    # Pumped mode
                    p_ch = self._get_value(detail, "pumped_charge", zone, t)
                    p_dis = self._get_value(detail, "pumped_discharge", zone, t)
                    u[z_idx, t, 2] = 1.0 if p_ch > p_dis else 0.0
                    u[z_idx, t, 3] = 1.0 if p_dis > p_ch else 0.0
                    
                    # DR active
                    dr = self._get_value(detail, "demand_response", zone, t)
                    u[z_idx, t, 4] = 1.0 if dr > 0 else 0.0
                    
                    # Nuclear on
                    nuc = self._get_value(detail, "nuclear", zone, t)
                    u[z_idx, t, 5] = 1.0 if nuc > 0 else 0.0
                    
                    # Thermal on
                    th = self._get_value(detail, "thermal", zone, t)
                    u[z_idx, t, 6] = 1.0 if th > 0 else 0.0
                    
                    # Import mode (net_import > 0 means importing)
                    net_import = self._get_value(detail, "net_import", zone, t)
                    u[z_idx, t, 7] = 1.0 if net_import > 0 else 0.0
            
            return u
            
        except Exception as e:
            return None
    
    def _get_value(
        self,
        detail: Dict,
        key: str,
        zone: str,
        t: int,
    ) -> float:
        """Safely get a value from dispatch detail."""
        if key not in detail:
            return 0.0
        zone_data = detail[key]
        if isinstance(zone_data, dict) and zone in zone_data:
            arr = zone_data[zone]
            if isinstance(arr, list) and t < len(arr):
                return float(arr[t])
        return 0.0
    
    def get_scenario_data(self, scenario_id: str) -> ScenarioData:
        """Load full scenario data for LP worker."""
        scenario_path = Path(self.config.scenarios_dir) / f"{scenario_id}.json"
        
        with open(scenario_path, 'r') as f:
            raw = json.load(f)
        
        zones = raw.get("zones", [])
        n_zones = len(zones)
        n_timesteps = raw.get("horizon", 24)
        
        # Extract time series
        demand = torch.zeros(n_zones, n_timesteps)
        solar_available = torch.zeros(n_zones, n_timesteps)
        wind_available = torch.zeros(n_zones, n_timesteps)
        
        for z_idx, zone in enumerate(zones):
            zone_data = zone.get("data", zone)
            
            if "demand" in zone_data:
                demand[z_idx] = torch.tensor(zone_data["demand"][:n_timesteps])
            if "solar_available" in zone_data:
                solar_available[z_idx] = torch.tensor(zone_data["solar_available"][:n_timesteps])
            if "wind_available" in zone_data:
                wind_available[z_idx] = torch.tensor(zone_data["wind_available"][:n_timesteps])
        
        # Extract capacities
        thermal_capacity = torch.zeros(n_zones)
        nuclear_capacity = torch.zeros(n_zones)
        battery_power = torch.zeros(n_zones)
        battery_capacity = torch.zeros(n_zones)
        pumped_power = torch.zeros(n_zones)
        dr_capacity = torch.zeros(n_zones)
        
        for z_idx, zone in enumerate(zones):
            zone_data = zone.get("data", zone)
            thermal_capacity[z_idx] = zone_data.get("thermal_capacity", 0)
            nuclear_capacity[z_idx] = zone_data.get("nuclear_capacity", 0)
            battery_power[z_idx] = zone_data.get("battery_power_mw", 0)
            battery_capacity[z_idx] = zone_data.get("battery_capacity_mwh", 0)
            pumped_power[z_idx] = zone_data.get("pumped_power_mw", 0)
            dr_capacity[z_idx] = zone_data.get("dr_capacity_mw", 0)
        
        return ScenarioData(
            scenario_id=scenario_id,
            n_zones=n_zones,
            n_timesteps=n_timesteps,
            dt_hours=raw.get("dt_hours", 1.0),
            demand=demand,
            solar_available=solar_available,
            wind_available=wind_available,
            thermal_capacity=thermal_capacity,
            nuclear_capacity=nuclear_capacity,
            battery_power=battery_power,
            battery_capacity=battery_capacity,
            pumped_power=pumped_power,
            dr_capacity=dr_capacity,
            zone_names=[z.get("name", f"zone_{i}") for i, z in enumerate(zones)],
            raw_data=raw,
        )


def preference_collate_fn(batch: List[Dict]) -> Dict[str, Any]:
    """
    Custom collate function for preference dataset.
    
    Handles variable-sized scenarios by padding.
    """
    batch_size = len(batch)
    
    # Find max dimensions
    max_zones = max(item["n_zones"] for item in batch)
    max_timesteps = max(item["n_timesteps"] for item in batch)
    
    # Stack embeddings
    embeddings = torch.stack([item["embedding"] for item in batch])
    
    # Pad decisions if available
    milp_decisions = []
    for item in batch:
        if item["milp_decision"] is not None:
            u = item["milp_decision"]
            # Pad to max size
            padded = torch.zeros(max_zones, max_timesteps, 8)
            padded[:u.shape[0], :u.shape[1], :] = u
            milp_decisions.append(padded)
        else:
            milp_decisions.append(torch.zeros(max_zones, max_timesteps, 8))
    
    milp_decisions = torch.stack(milp_decisions)
    
    # Stack objectives
    milp_objectives = torch.tensor([item["milp_objective"] for item in batch])
    
    return {
        "scenario_ids": [item["scenario_id"] for item in batch],
        "scenario_paths": [item["scenario_path"] for item in batch],
        "embeddings": embeddings,
        "milp_decisions": milp_decisions,
        "milp_objectives": milp_objectives,
        "n_zones": [item["n_zones"] for item in batch],
        "n_timesteps": [item["n_timesteps"] for item in batch],
        "raw_scenarios": [item["raw_scenario"] for item in batch],
    }


class PreferenceDataLoader:
    """
    DataLoader wrapper for preference-based training.
    
    Handles batching and provides utilities for generating negative examples.
    """
    
    def __init__(
        self,
        dataset: PreferenceDataset,
        batch_size: int = 4,
        shuffle: bool = True,
        num_workers: int = 0,
        pin_memory: bool = False,
    ):
        self.dataset = dataset
        self.batch_size = batch_size
        
        self.dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            pin_memory=pin_memory,
            collate_fn=preference_collate_fn,
        )
    
    def __iter__(self):
        return iter(self.dataloader)
    
    def __len__(self):
        return len(self.dataloader)


def create_training_batch(
    batch: Dict[str, Any],
    negatives: torch.Tensor,
    negative_costs: torch.Tensor,
    weights: Optional[torch.Tensor] = None,
) -> TrainingBatch:
    """
    Create a TrainingBatch from dataloader batch and generated negatives.
    
    Args:
        batch: Dict from dataloader
        negatives: [B, K, Z, T, 8] negative decisions
        negative_costs: [B, K] costs from LP worker
        weights: [B, K] optional cost-aware weights
    
    Returns:
        TrainingBatch ready for loss computation
    """
    return TrainingBatch(
        h=batch["embeddings"],
        u_positive=batch["milp_decisions"],
        costs_positive=batch["milp_objectives"],
        u_negatives=negatives,
        costs_negative=negative_costs,
        weights=weights,
        scenario_ids=batch["scenario_ids"],
    )
