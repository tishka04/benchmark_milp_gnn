"""
Efficient loader for pre-computed GNN embeddings.

Handles large .pt files from Hierarchical Temporal Encoder.
"""

import torch
import numpy as np
from pathlib import Path
from typing import Dict, Optional, Union
import gc


class EmbeddingLoader:
    """
    Loads pre-computed embeddings from Hierarchical Temporal Encoder.
    
    Supports:
    - Full in-memory loading (if RAM available)
    - On-demand loading per scenario
    - Memory-mapped access
    """
    
    def __init__(
        self,
        embedding_file: str,
        loading_mode: str = 'lazy',
        device: str = 'cpu',
    ):
        """
        Args:
            embedding_file: Path to .pt file with embeddings
            loading_mode: How to load embeddings
                - 'full': Load entire file into memory
                - 'lazy': Load on-demand per scenario
                - 'mmap': Memory-mapped loading (not supported for .pt)
            device: Device to load embeddings to
        """
        self.embedding_file = Path(embedding_file)
        self.loading_mode = loading_mode
        self.device = device
        
        if not self.embedding_file.exists():
            raise FileNotFoundError(f"Embedding file not found: {embedding_file}")
        
        print(f"Initializing EmbeddingLoader for {self.embedding_file}")
        print(f"File size: {self.embedding_file.stat().st_size / 1e9:.2f} GB")
        print(f"Loading mode: {loading_mode}")
        
        # Storage for embeddings
        self.embeddings = None
        self.scenario_ids = None
        
        # Initialize based on loading mode
        if loading_mode == 'full':
            self._load_full()
        elif loading_mode == 'lazy':
            self._initialize_lazy()
        else:
            raise ValueError(f"Unknown loading mode: {loading_mode}")
    
    def _load_full(self):
        """
        Load entire embedding file into memory.
        
        Expected format:
        {
            'scenario_00001': tensor([T, 128]) or tensor([128]),
            'scenario_00002': tensor([T, 128]) or tensor([128]),
            ...
        }
        or
        {
            'embeddings': {scenario_id: tensor},
            'config': {...},
        }
        """
        print("Loading full embeddings file (this may take a while)...")
        
        try:
            data = torch.load(self.embedding_file, map_location=self.device)
        except Exception as e:
            print(f"Error loading .pt file: {e}")
            print("Trying with weights_only=False...")
            data = torch.load(self.embedding_file, map_location=self.device, weights_only=False)
        
        # Handle different file structures
        if isinstance(data, dict):
            if 'embeddings' in data:
                # Nested structure
                self.embeddings = data['embeddings']
                print(f"Loaded nested structure with {len(self.embeddings)} scenarios")
            elif all(isinstance(k, str) and k.startswith('scenario_') for k in list(data.keys())[:10]):
                # Direct scenario_id -> embedding mapping
                self.embeddings = data
                print(f"Loaded direct mapping with {len(self.embeddings)} scenarios")
            else:
                # Try to infer structure
                print(f"Unknown structure. Keys: {list(data.keys())[:10]}")
                self.embeddings = data
        else:
            raise ValueError(f"Expected dict, got {type(data)}")
        
        # Extract scenario IDs
        self.scenario_ids = list(self.embeddings.keys())
        
        # Print info about embeddings
        sample_key = self.scenario_ids[0]
        sample_emb = self.embeddings[sample_key]
        print(f"Sample embedding shape: {sample_emb.shape}")
        print(f"Sample embedding dtype: {sample_emb.dtype}")
        print(f"Total scenarios: {len(self.scenario_ids)}")
        
        # Move to CPU if needed to save GPU memory
        if self.device == 'cpu':
            print("Embeddings already on CPU")
        else:
            print(f"Embeddings loaded to {self.device}")
    
    def _initialize_lazy(self):
        """
        Initialize for lazy loading (don't load embeddings yet).
        
        Just peek at the file structure to get scenario IDs.
        """
        print("Initializing lazy loading (peeking at file structure)...")
        
        # Load with mmap if possible to avoid loading entire file
        try:
            # Just load keys to get scenario IDs
            data = torch.load(
                self.embedding_file,
                map_location='cpu',
                weights_only=False,
            )
            
            if isinstance(data, dict):
                if 'embeddings' in data:
                    self.scenario_ids = list(data['embeddings'].keys())
                else:
                    self.scenario_ids = list(data.keys())
                
                # Store file path for later loading
                self._lazy_data = None  # Will load on first access
                
                print(f"Found {len(self.scenario_ids)} scenarios")
                
                # Free memory
                del data
                gc.collect()
            else:
                raise ValueError(f"Expected dict, got {type(data)}")
                
        except Exception as e:
            print(f"Error peeking at file: {e}")
            print("Will attempt full load on first access")
            self.scenario_ids = None
    
    def get_embedding(self, scenario_id: str) -> np.ndarray:
        """
        Get embedding for a specific scenario.
        
        Args:
            scenario_id: Scenario identifier (e.g., 'scenario_00001')
            
        Returns:
            embedding: NumPy array [128] or [T, 128]
        """
        if self.loading_mode == 'full':
            # Direct lookup from loaded embeddings
            if scenario_id not in self.embeddings:
                raise KeyError(f"Scenario {scenario_id} not found in embeddings")
            
            emb = self.embeddings[scenario_id]
            
            # Convert to numpy
            if isinstance(emb, torch.Tensor):
                emb = emb.cpu().numpy()
            
            return emb
        
        elif self.loading_mode == 'lazy':
            # Load on-demand
            if self._lazy_data is None:
                # First access - load the file
                print(f"Loading embeddings file (first access)...")
                self._lazy_data = torch.load(
                    self.embedding_file,
                    map_location='cpu',
                    weights_only=False,
                )
                
                if 'embeddings' in self._lazy_data:
                    self._lazy_data = self._lazy_data['embeddings']
            
            if scenario_id not in self._lazy_data:
                raise KeyError(f"Scenario {scenario_id} not found in embeddings")
            
            emb = self._lazy_data[scenario_id]
            
            if isinstance(emb, torch.Tensor):
                emb = emb.cpu().numpy()
            
            return emb
    
    def get_batch_embeddings(self, scenario_ids: list) -> Dict[str, np.ndarray]:
        """
        Get embeddings for multiple scenarios.
        
        Args:
            scenario_ids: List of scenario identifiers
            
        Returns:
            Dictionary mapping scenario_id -> embedding
        """
        return {
            sid: self.get_embedding(sid)
            for sid in scenario_ids
        }
    
    def aggregate_temporal(self, embedding: np.ndarray, method: str = 'mean') -> np.ndarray:
        """
        Aggregate temporal embeddings to single vector.
        
        Args:
            embedding: Embedding array [T, 128] or [128]
            method: Aggregation method ('mean', 'max', 'last', 'first')
            
        Returns:
            Aggregated embedding [128]
        """
        if embedding.ndim == 1:
            # Already aggregated
            return embedding
        
        # [T, 128] -> [128]
        if method == 'mean':
            return embedding.mean(axis=0)
        elif method == 'max':
            return embedding.max(axis=0)
        elif method == 'last':
            return embedding[-1]
        elif method == 'first':
            return embedding[0]
        else:
            raise ValueError(f"Unknown aggregation method: {method}")
    
    def __len__(self):
        """Number of scenarios in embeddings."""
        if self.scenario_ids is not None:
            return len(self.scenario_ids)
        return 0
    
    def __contains__(self, scenario_id: str):
        """Check if scenario exists in embeddings."""
        if self.scenario_ids is not None:
            return scenario_id in self.scenario_ids
        return False


class GoogleDriveEmbeddingLoader(EmbeddingLoader):
    """
    Specialized loader for embeddings stored on Google Drive.
    
    Handles mounting and path resolution for Colab/Drive environments.
    """
    
    def __init__(
        self,
        drive_path: str,
        loading_mode: str = 'lazy',
        device: str = 'cpu',
        auto_mount: bool = False,
    ):
        """
        Args:
            drive_path: Path relative to Google Drive root
                e.g., 'benchmark/outputs/encoders/hierchical_temporal/embeddings_multiscale_full.pt'
            loading_mode: How to load embeddings
            device: Device to load to
            auto_mount: Whether to automatically mount Google Drive (Colab only)
        """
        # Try to resolve Google Drive path
        if auto_mount:
            self._mount_drive()
        
        # Common Google Drive mount points
        possible_roots = [
            '/content/drive/MyDrive',  # Colab
            '/content/gdrive/MyDrive',  # Alternative Colab
            '~/Google Drive',  # Local sync
            Path.home() / 'Google Drive',  # Local sync
        ]
        
        full_path = None
        for root in possible_roots:
            root_path = Path(root)
            if root_path.exists():
                candidate = root_path / drive_path
                if candidate.exists():
                    full_path = candidate
                    print(f"Found embeddings at: {full_path}")
                    break
        
        if full_path is None:
            # Try as-is (maybe it's already a full path)
            full_path = Path(drive_path)
            if not full_path.exists():
                raise FileNotFoundError(
                    f"Could not find embeddings file. Tried:\n"
                    f"  - {drive_path}\n"
                    + "\n".join(f"  - {root / drive_path}" for root in possible_roots)
                )
        
        # Initialize parent class
        super().__init__(
            embedding_file=str(full_path),
            loading_mode=loading_mode,
            device=device,
        )
    
    def _mount_drive(self):
        """Mount Google Drive (Colab only)."""
        try:
            from google.colab import drive
            drive.mount('/content/drive')
            print("Google Drive mounted successfully")
        except ImportError:
            print("Not in Colab environment, skipping auto-mount")
        except Exception as e:
            print(f"Failed to mount Google Drive: {e}")
