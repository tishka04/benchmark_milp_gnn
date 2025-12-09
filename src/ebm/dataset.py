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
    1. Binary variables from MILP solutions (UC/DR/Storage)
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
        """
        Args:
            scenarios_dir: Directory containing MILP solutions
            embedding_cache_dir: Directory with cached GNN embeddings (individual .npy files)
            embedding_file: Path to single .pt file with all embeddings (e.g., from Google Drive)
            scenario_ids: List of scenario IDs to load (if None, load all)
            extract_from_dispatch: Extract binary vars from dispatch CSV files
            temporal: Whether to preserve temporal structure (T timesteps)
            temporal_aggregation: How to aggregate temporal embeddings ('mean', 'max', 'last', 'first')
            device: Device for tensors
            embedding_loading_mode: How to load embedding_file ('full' or 'lazy')
        """
        self.scenarios_dir = Path(scenarios_dir)
        self.embedding_cache_dir = Path(embedding_cache_dir) if embedding_cache_dir else None
        self.temporal = temporal
        self.temporal_aggregation = temporal_aggregation
        self.device = device
        
        # Initialize embedding loader if .pt file provided
        self.embedding_loader = None
        if embedding_file:
            print(f"\nInitializing embedding loader from: {embedding_file}")
            try:
                # Try Google Drive loader first
                self.embedding_loader = GoogleDriveEmbeddingLoader(
                    drive_path=embedding_file,
                    loading_mode=embedding_loading_mode,
                    device='cpu',  # Load to CPU first to avoid OOM
                )
            except FileNotFoundError:
                # Try regular loader
                self.embedding_loader = EmbeddingLoader(
                    embedding_file=embedding_file,
                    loading_mode=embedding_loading_mode,
                    device='cpu',
                )
            print(f"Embedding loader initialized with {len(self.embedding_loader)} scenarios")
        
        # Find all scenario files
        if extract_from_dispatch:
            self.dispatch_dir = self.scenarios_dir / "dispatch_batch"
            scenario_files = sorted(list(self.dispatch_dir.glob("scenario_*_zone.csv")))
            
            # Extract scenario IDs
            self.scenario_ids = []
            for f in scenario_files:
                # Extract ID from filename: scenario_00001_zone.csv -> scenario_00001
                scenario_id = f.stem.replace("_zone", "")
                if scenario_ids is None or scenario_id in scenario_ids:
                    self.scenario_ids.append(scenario_id)
        else:
            # Load from reports (JSON files)
            report_dir = self.scenarios_dir / "reports"
            scenario_files = sorted(list(report_dir.glob("scenario_*.json")))
            
            self.scenario_ids = []
            for f in scenario_files:
                scenario_id = f.stem
                if scenario_ids is None or scenario_id in scenario_ids:
                    self.scenario_ids.append(scenario_id)
        
        print(f"Loaded {len(self.scenario_ids)} scenarios from {self.scenarios_dir}")
        
        # Load scenario metadata to determine binary variable structure
        self._load_metadata()
    
    def _load_metadata(self):
        """
        Load metadata about binary variables from first scenario.
        """
        # Load a sample scenario to determine structure
        if len(self.scenario_ids) > 0:
            sample_id = self.scenario_ids[0]
            
            # Try loading from JSON report first
            report_file = self.scenarios_dir / "reports" / f"{sample_id}.json"
            if report_file.exists():
                with open(report_file, 'r') as f:
                    data = json.load(f)
                
                # Check for standard MILP output structure
                if 'difficulty_indicators' in data:
                    self.n_binary_vars = data['difficulty_indicators'].get('n_binary_variables', 0)
                    self.n_timesteps = data['difficulty_indicators'].get('n_timesteps', 96)
                else:
                    # Estimate from detail section if available
                    self.n_timesteps = len(data.get('detail', {}).get('time_steps', []))
                    self.n_binary_vars = self.n_timesteps * 100  # Rough estimate
            
            print(f"Binary variables per scenario: {self.n_binary_vars}")
            print(f"Timesteps: {self.n_timesteps}")
    
    def __len__(self):
        return len(self.scenario_ids)
    
    def __getitem__(self, idx) -> Dict[str, torch.Tensor]:
        """
        Get a single scenario's binary variables and embeddings.
        
        Returns:
            Dictionary with:
            - 'u': Binary decisions [dim_u] or [T, dim_u_per_t]
            - 'h': Graph embedding [dim_h] or [T, dim_h]
            - 'scenario_id': Scenario identifier
        """
        scenario_id = self.scenario_ids[idx]
        
        # Load binary variables
        u = self._load_binary_variables(scenario_id)
        
        # Load or compute embeddings
        h = self._load_embedding(scenario_id)
        
        return {
            'u': torch.tensor(u, dtype=torch.float32, device=self.device),
            'h': torch.tensor(h, dtype=torch.float32, device=self.device),
            'scenario_id': scenario_id,
        }
    
    def _load_binary_variables(self, scenario_id: str) -> np.ndarray:
        """
        Extract binary variables from MILP solution.
        
        For now, we extract from dispatch CSV:
        - Battery charge/discharge (0/1 indicators)
        - Pumped storage charge/discharge
        - Demand response activation
        - (Optional) Unit commitment if available
        
        Returns:
            u: Binary array [dim_u] or [T, dim_u_per_t]
        """
        # Load dispatch data
        dispatch_file = self.scenarios_dir / "dispatch_batch" / f"{scenario_id}_zone.csv"
        
        if not dispatch_file.exists():
            # Fallback: create dummy binary variables
            if self.temporal:
                return np.random.randint(0, 2, (self.n_timesteps, self.n_binary_vars // self.n_timesteps))
            else:
                return np.random.randint(0, 2, self.n_binary_vars)
        
        # Load dispatch CSV
        df = pd.read_csv(dispatch_file)
        
        # Extract binary indicators
        binary_vars = []
        
        for t in range(self.n_timesteps):
            df_t = df[df['time_step'] == t]
            
            # Create binary features for each timestep
            timestep_binary = []
            
            for _, row in df_t.iterrows():
                # Battery binary indicators
                battery_charging = 1 if row.get('battery_charge', 0) > 0.1 else 0
                battery_discharging = 1 if row.get('battery_discharge', 0) > 0.1 else 0
                
                # Pumped storage binary indicators
                pumped_charging = 1 if row.get('pumped_charge', 0) > 0.1 else 0
                pumped_discharging = 1 if row.get('pumped_discharge', 0) > 0.1 else 0
                
                # Demand response activation
                dr_active = 1 if row.get('demand_response', 0) > 0.1 else 0
                
                # Nuclear on/off (if nuclear > 0, it's on)
                nuclear_on = 1 if row.get('nuclear', 0) > 0.1 else 0
                
                # Thermal on/off
                thermal_on = 1 if row.get('thermal', 0) > 0.1 else 0
                
                timestep_binary.extend([
                    battery_charging, battery_discharging,
                    pumped_charging, pumped_discharging,
                    dr_active, nuclear_on, thermal_on
                ])
            
            binary_vars.append(timestep_binary)
        
        binary_vars = np.array(binary_vars, dtype=np.float32)  # [T, dim_u_per_t]
        
        if not self.temporal:
            # Flatten temporal dimension
            binary_vars = binary_vars.flatten()
        
        return binary_vars
    
    def _load_embedding(self, scenario_id: str) -> np.ndarray:
        """
        Load graph embedding from Hierarchical Temporal Encoder.
        
        Priority:
        1. If embedding_loader available (from .pt file), use it
        2. If embedding_cache_dir exists, load from .npy file
        3. Otherwise, return dummy embeddings
        
        Returns:
            h: Embedding array [dim_h] or [T, dim_h]
        """
        # Option 1: Load from embedding loader (.pt file)
        if self.embedding_loader is not None:
            try:
                h = self.embedding_loader.get_embedding(scenario_id)
                
                # Aggregate temporal dimension if needed
                if h.ndim == 2 and not self.temporal:
                    # [T, 128] -> [128]
                    h = self.embedding_loader.aggregate_temporal(h, method=self.temporal_aggregation)
                
                return h.astype(np.float32)
            
            except KeyError:
                print(f"Warning: Scenario {scenario_id} not found in embedding loader")
        
        # Option 2: Load from cache directory (.npy files)
        if self.embedding_cache_dir:
            embedding_file = self.embedding_cache_dir / f"{scenario_id}_embedding.npy"
            
            if embedding_file.exists():
                h = np.load(embedding_file)
                
                # Aggregate temporal dimension if needed
                if h.ndim == 2 and not self.temporal:
                    h = h.mean(axis=0)
                
                return h.astype(np.float32)
        
        # Option 3: Return dummy embeddings if not available
        dim_h = 128
        
        if self.temporal:
            # Return temporal embeddings [T, dim_h]
            h = np.random.randn(self.n_timesteps, dim_h).astype(np.float32)
        else:
            # Return global embedding [dim_h]
            h = np.random.randn(dim_h).astype(np.float32)
        
        return h


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
