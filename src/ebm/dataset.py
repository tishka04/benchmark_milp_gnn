"""
Dataset loader for MILP binary variables and GNN embeddings.

Extracts UC/DR/Storage binary decisions from MILP solutions and pairs them
with embeddings from the Hierarchical Temporal Encoder.
"""

import json
import torch
import numpy as np
from pathlib import Path
from torch.utils.data import Dataset
from typing import Dict, List, Optional, Tuple
import pandas as pd


from .embedding_loader import EmbeddingLoader, GoogleDriveEmbeddingLoader


from torch_geometric.data import Data
from torch.utils.data import Dataset

class GraphBinaryDataset(Dataset):
    """
    Wrapper around MILPBinaryDataset to return PyTorch Geometric Data objects.
    This allows handling variable-sized instances (different number of binary variables)
    using PyG's collation method.
    """
    def __init__(self, base_dataset):
        self.base_dataset = base_dataset

    def __len__(self):
        return len(self.base_dataset)

    def __getitem__(self, idx):
        # Get dictionary sample from base dataset
        sample = self.base_dataset[idx]
        u = sample['u']
        h = sample['h']

        # Create node features x from binary variables u
        # Shape: [num_nodes, 1]
        x = u.view(-1, 1).float()

        # Reshape embedding h to match batching expectations
        # Shape: [1, embedding_dim]
        # This ensures that when batched, h becomes [batch_size, embedding_dim]
        h_reshaped = h.view(1, -1)

        # Create PyG Data object
        # We treat each binary variable as a node (set structure, no edges required for now)
        data = Data(x=x, h=h_reshaped)

        # Attach scenario_id for tracking
        if 'scenario_id' in sample:
            data.scenario_id = sample['scenario_id']

        return data


class MILPBinaryDataset(Dataset):
    """
    Dataset for EBM training that loads:
    1. Binary variables from MILP solutions (UC/DR/Storage) - VECTORIZED LOADING
    2. Graph embeddings from Hierarchical Temporal Encoder
    """
    
    def __init__(
        self,
        scenarios_dir: str,
        embedding_cache_dir: Optional[str] = None,
        embedding_file: Optional[str] = None,
        scenario_ids: Optional[List[str]] = None,
        extract_from_dispatch: bool = True,
        temporal: bool = False,
        temporal_aggregation: str = 'mean',
        device: str = 'cpu',
        embedding_loading_mode: str = 'lazy',
    ):
        self.scenarios_dir = Path(scenarios_dir)
        self.embedding_cache_dir = Path(embedding_cache_dir) if embedding_cache_dir else None
        self.temporal = temporal
        self.temporal_aggregation = temporal_aggregation
        self.device = device
        
        # 1. Initialize embedding loader
        self.embedding_loader = None
        if embedding_file:
            print(f"\nInitializing embedding loader from: {embedding_file}")
            try:
                # On utilise directement ta classe GoogleDriveEmbeddingLoader existante
                self.embedding_loader = GoogleDriveEmbeddingLoader(
                    drive_path=embedding_file,
                    loading_mode=embedding_loading_mode,
                    device='cpu', 
                )
            except NameError:
                # Fallback si la classe n'est pas définie dans cette cellule
                print("⚠️ GoogleDriveEmbeddingLoader not defined, assuming manual dict load.")
                self.embedding_loader = None # A gérer manuellement si besoin

        # 2. Find scenario files
        if extract_from_dispatch:
            # Gestion flexible du dossier dispatch
            if (self.scenarios_dir / "dispatch_batch").exists():
                self.dispatch_dir = self.scenarios_dir / "dispatch_batch"
            else:
                self.dispatch_dir = self.scenarios_dir
            
            # Recherche des fichiers
            scenario_files = sorted(list(self.dispatch_dir.glob("*_zone.csv")))
            if len(scenario_files) == 0:
                 print("⚠️ No *_zone.csv files found. Trying generic *.csv...")
                 scenario_files = sorted(list(self.dispatch_dir.glob("scenario_*.csv")))

            # Extract IDs
            self.scenario_ids = []
            for f in scenario_files:
                # Nettoyage robuste de l'ID
                sid = f.stem.replace("_zone", "")
                if scenario_ids is None or sid in scenario_ids:
                    self.scenario_ids.append(sid)
        else:
            # Load from reports
            report_dir = self.scenarios_dir / "reports"
            scenario_files = sorted(list(report_dir.glob("scenario_*.json")))
            self.scenario_ids = [f.stem for f in scenario_files]
        
        print(f"Loaded {len(self.scenario_ids)} scenarios from {self.dispatch_dir}")
        
        # Metadata defaults
        self.n_timesteps = 96 # Default
        self.n_binary_vars = 0
        self._load_metadata()
    
    def _load_metadata(self):
        if len(self.scenario_ids) > 0:
            # On essaye de déduire les dimensions du premier fichier CSV
            try:
                u = self._load_binary_variables(self.scenario_ids[0])
                if self.temporal:
                    self.n_timesteps = u.shape[0]
                    self.n_binary_vars = u.shape[0] * u.shape[1]
                else:
                    self.n_binary_vars = u.shape[0]
            except:
                pass
            print(f"Binary variables per scenario (est): {self.n_binary_vars}")

    def __len__(self):
        return len(self.scenario_ids)
    
    def __getitem__(self, idx) -> Dict[str, torch.Tensor]:
        scenario_id = self.scenario_ids[idx]
        u = self._load_binary_variables(scenario_id)
        h = self._load_embedding(scenario_id)
        
        return {
            'u': torch.tensor(u, dtype=torch.float32, device=self.device),
            'h': torch.tensor(h, dtype=torch.float32, device=self.device),
            'scenario_id': scenario_id,
        }
    
    def _load_binary_variables(self, scenario_id: str) -> np.ndarray:
        """🚀 VERSION OPTIMISÉE (Vectorisée)"""
        # Patterns de fichiers possibles
        f1 = self.dispatch_dir / f"{scenario_id}_zone.csv"
        f2 = self.dispatch_dir / f"{scenario_id}.csv"
        
        dispatch_file = f1 if f1.exists() else f2
        
        if not dispatch_file.exists():
            # Dummy fallback
            dims = (self.n_timesteps, 100) if self.temporal else (self.n_timesteps * 100,)
            return np.random.randint(0, 2, dims).astype(np.float32)
        
        # 1. Load CSV (Fast)
        df = pd.read_csv(dispatch_file)
        
        # 2. Sort to guarantee order (Time -> Zone)
        if 'zone_id' in df.columns:
            df = df.sort_values(by=['time_step', 'zone_id'])
        else:
            df = df.sort_values(by=['time_step'])

        # 3. Vectorized Extraction
        target_cols = ['battery_charge', 'battery_discharge', 'pumped_charge', 'pumped_discharge', 'demand_response', 'nuclear', 'thermal']
        cols_to_use = [c for c in target_cols if c in df.columns]
        
        # Binarization vectorielle
        data = (df[cols_to_use].values > 0.1).astype(np.float32)
        
        # 4. Reshape
        num_timesteps = df['time_step'].nunique()
        if num_timesteps == 0: num_timesteps = 96 # Safety
        
        try:
            # [T * Z, F] -> [T, Z * F]
            u_temporal = data.reshape(num_timesteps, -1)
        except ValueError:
            # Fallback si dimensions irrégulières
            u_temporal = np.resize(data, (num_timesteps, data.size // num_timesteps))

        if not self.temporal:
            return u_temporal.flatten()
            
        return u_temporal

    def _load_embedding(self, scenario_id: str) -> np.ndarray:
        if self.embedding_loader:
            try:
                h = self.embedding_loader.get_embedding(scenario_id)
                # Aggregate if needed
                if h.ndim == 2 and not self.temporal:
                     # Simple aggregation if model is not temporal
                     # But typically for EBM we keep temporal or flatten
                     h = h.flatten() # Or mean(0)
                return h.astype(np.float32)
            except KeyError:
                pass
        
        # Dummy fallback
        dim = 128
        return np.zeros((self.n_timesteps, dim) if self.temporal else dim, dtype=np.float32)

class MILPPairDataset(Dataset):
    """
    Dataset that creates pairs of (feasible, infeasible) configurations
    for contrastive learning.
    """
    
    def __init__(
        self,
        base_dataset: MILPBinaryDataset,
        corruption_rate: float = 0.1,
        corruption_strategy: str = 'random_flip',
    ):
        """
        Args:
            base_dataset: Base dataset with feasible configurations
            corruption_rate: Fraction of variables to corrupt
            corruption_strategy: How to generate negative samples
                - 'random_flip': Randomly flip bits
                - 'temporal_shuffle': Shuffle temporal order
                - 'constraint_violate': Explicitly violate constraints
        """
        self.base_dataset = base_dataset
        self.corruption_rate = corruption_rate
        self.corruption_strategy = corruption_strategy
    
    def __len__(self):
        return len(self.base_dataset)
    
    def __getitem__(self, idx) -> Dict[str, torch.Tensor]:
        """
        Get (positive, negative) pair.
        
        Returns:
            Dictionary with:
            - 'u_pos': Feasible configuration
            - 'u_neg': Corrupted configuration
            - 'h': Graph embedding
            - 'scenario_id': Scenario identifier
        """
        # Get base sample
        sample = self.base_dataset[idx]
        u_pos = sample['u']
        h = sample['h']
        
        # Generate negative sample
        u_neg = self._corrupt(u_pos)
        
        return {
            'u_pos': u_pos,
            'u_neg': u_neg,
            'h': h,
            'scenario_id': sample['scenario_id'],
        }
    
    def _corrupt(self, u: torch.Tensor) -> torch.Tensor:
        """
        Corrupt a binary configuration to create a negative sample.
        
        Args:
            u: Feasible binary configuration
            
        Returns:
            u_neg: Corrupted configuration
        """
        u_neg = u.clone()
        
        if self.corruption_strategy == 'random_flip':
            # Randomly flip bits
            num_flips = int(u.numel() * self.corruption_rate)
            flip_indices = torch.randperm(u.numel())[:num_flips]
            
            u_flat = u_neg.flatten()
            u_flat[flip_indices] = 1 - u_flat[flip_indices]
            u_neg = u_flat.reshape(u.shape)
        
        elif self.corruption_strategy == 'temporal_shuffle':
            # Shuffle temporal order (if temporal structure exists)
            if u.dim() > 1:
                perm = torch.randperm(u.shape[0])
                u_neg = u_neg[perm]
        
        elif self.corruption_strategy == 'constraint_violate':
            # Explicitly violate known constraints
            # For example: activate both charge and discharge simultaneously
            if u.dim() > 1:  # Temporal
                # Find charge/discharge pairs and activate both
                for t in range(u.shape[0]):
                    if torch.rand(1).item() < self.corruption_rate:
                        # Assuming first two dims are charge/discharge
                        u_neg[t, 0] = 1  # Charge
                        u_neg[t, 1] = 1  # Discharge (violates constraint)
        
        return u_neg


def collate_ebm_batch(batch: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
    """
    Custom collate function for EBM batches.
    
    Args:
        batch: List of samples from dataset
        
    Returns:
        Batched dictionary
    """
    # Stack tensors
    u_batch = torch.stack([sample['u'] for sample in batch])
    h_batch = torch.stack([sample['h'] for sample in batch])
    
    # Collect scenario IDs
    scenario_ids = [sample['scenario_id'] for sample in batch]
    
    result = {
        'u': u_batch,
        'h': h_batch,
        'scenario_id': scenario_ids,
    }
    
    # Handle pair datasets
    if 'u_pos' in batch[0]:
        result['u_pos'] = torch.stack([sample['u_pos'] for sample in batch])
        result['u_neg'] = torch.stack([sample['u_neg'] for sample in batch])
    
    return result
