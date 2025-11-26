"""
Temporal Heterogeneous GNN Models for Multi-Layer Grid Graphs

Implements models that can learn from time-expanded supra-graphs with
spatial and temporal edge types.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import HGTConv, RGCNConv, Linear
from typing import Dict, Optional, Tuple


class TemporalHeteroGNN(nn.Module):
    """
    Base class for temporal heterogeneous GNN models.
    
    Handles time-expanded graphs where nodes are (node_id, timestep) pairs
    and edges include both spatial (replicated per t) and temporal (across t) types.
    
    Edge Types:
        0-6: Spatial edges (Nation→Region, Region→Zone, etc.)
        7: Temporal SOC (storage continuity)
        8: Temporal Ramp (generator ramping)
        9: Temporal DR (demand response cooldown)
    """
    
    def __init__(
        self,
        node_feature_dim: int,
        hidden_dim: int,
        output_dim: int,
        num_edge_types: int = 10,
        num_layers: int = 3,
    ):
        super().__init__()
        self.node_feature_dim = node_feature_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.num_edge_types = num_edge_types
        self.num_layers = num_layers
        
    def forward(self, x, edge_index, edge_type, batch=None):
        """
        Args:
            x: [N*T, F] node features (including time encoding)
            edge_index: [2, E] edge connectivity
            edge_type: [E] edge type indices (0-9)
            batch: [N*T] batch assignment (for batching multiple graphs)
        
        Returns:
            [N*T, output_dim] predictions per node-timestep
        """
        raise NotImplementedError


class TemporalRGCN(TemporalHeteroGNN):
    """
    Temporal R-GCN: Relational Graph Convolutional Network for temporal graphs.
    
    Uses relation-specific transformations for different edge types.
    Separates spatial and temporal message passing via edge types.
    """
    
    def __init__(
        self,
        node_feature_dim: int,
        hidden_dim: int,
        output_dim: int,
        num_edge_types: int = 10,
        num_layers: int = 3,
        dropout: float = 0.1,
    ):
        super().__init__(node_feature_dim, hidden_dim, output_dim, num_edge_types, num_layers)
        
        # Input projection
        self.input_proj = Linear(node_feature_dim, hidden_dim)
        
        # R-GCN layers
        self.convs = nn.ModuleList([
            RGCNConv(
                hidden_dim,
                hidden_dim,
                num_relations=num_edge_types,
                num_bases=min(num_edge_types, 4),  # Basis decomposition
            )
            for _ in range(num_layers)
        ])
        
        self.norms = nn.ModuleList([
            nn.LayerNorm(hidden_dim)
            for _ in range(num_layers)
        ])
        
        self.dropout = nn.Dropout(dropout)
        
        # Output head
        self.output_head = nn.Sequential(
            Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            Linear(hidden_dim, output_dim),
        )
    
    def forward(self, x, edge_index, edge_type, batch=None):
        # Project input features
        h = self.input_proj(x)
        
        # Message passing
        for i, (conv, norm) in enumerate(zip(self.convs, self.norms)):
            h_new = conv(h, edge_index, edge_type)
            h_new = norm(h_new)
            h_new = F.relu(h_new)
            h_new = self.dropout(h_new)
            
            # Residual connection
            if i > 0:
                h = h + h_new
            else:
                h = h_new
        
        # Output
        out = self.output_head(h)
        return out


class SeparatedTemporalGNN(TemporalHeteroGNN):
    """
    Separated Spatial-Temporal GNN.
    
    Explicitly separates spatial message passing (within timestep)
    from temporal message passing (across timesteps).
    """
    
    def __init__(
        self,
        node_feature_dim: int,
        hidden_dim: int,
        output_dim: int,
        num_spatial_edge_types: int = 7,
        num_temporal_edge_types: int = 3,
        num_layers: int = 3,
        dropout: float = 0.1,
    ):
        super().__init__(
            node_feature_dim,
            hidden_dim,
            output_dim,
            num_spatial_edge_types + num_temporal_edge_types,
            num_layers,
        )
        
        self.num_spatial_edge_types = num_spatial_edge_types
        self.num_temporal_edge_types = num_temporal_edge_types
        
        # Input projection
        self.input_proj = Linear(node_feature_dim, hidden_dim)
        
        # Spatial R-GCN (for edges within same timestep)
        self.spatial_convs = nn.ModuleList([
            RGCNConv(
                hidden_dim,
                hidden_dim,
                num_relations=num_spatial_edge_types,
                num_bases=min(num_spatial_edge_types, 4),
            )
            for _ in range(num_layers)
        ])
        
        # Temporal R-GCN (for edges across timesteps)
        self.temporal_convs = nn.ModuleList([
            RGCNConv(
                hidden_dim,
                hidden_dim,
                num_relations=num_temporal_edge_types,
                num_bases=min(num_temporal_edge_types, 3),
            )
            for _ in range(num_layers)
        ])
        
        # Fusion layers
        self.fusion_layers = nn.ModuleList([
            nn.Linear(hidden_dim * 2, hidden_dim)
            for _ in range(num_layers)
        ])
        
        self.norms = nn.ModuleList([
            nn.LayerNorm(hidden_dim)
            for _ in range(num_layers)
        ])
        
        self.dropout = nn.Dropout(dropout)
        
        # Output head
        self.output_head = nn.Sequential(
            Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            Linear(hidden_dim, output_dim),
        )
    
    def forward(self, x, edge_index, edge_type, batch=None):
        # Project input
        h = self.input_proj(x)
        
        # Separate spatial and temporal edges
        spatial_mask = edge_type < self.num_spatial_edge_types
        temporal_mask = edge_type >= self.num_spatial_edge_types
        
        spatial_edge_index = edge_index[:, spatial_mask]
        spatial_edge_type = edge_type[spatial_mask]
        
        temporal_edge_index = edge_index[:, temporal_mask]
        temporal_edge_type = edge_type[temporal_mask] - self.num_spatial_edge_types
        
        # Message passing
        for i, (spatial_conv, temporal_conv, fusion, norm) in enumerate(
            zip(self.spatial_convs, self.temporal_convs, self.fusion_layers, self.norms)
        ):
            # Spatial message passing
            h_spatial = spatial_conv(h, spatial_edge_index, spatial_edge_type)
            
            # Temporal message passing
            h_temporal = temporal_conv(h, temporal_edge_index, temporal_edge_type)
            
            # Fuse spatial and temporal information
            h_fused = torch.cat([h_spatial, h_temporal], dim=-1)
            h_new = fusion(h_fused)
            h_new = norm(h_new)
            h_new = F.relu(h_new)
            h_new = self.dropout(h_new)
            
            # Residual connection
            if i > 0:
                h = h + h_new
            else:
                h = h_new
        
        # Output
        out = self.output_head(h)
        return out


class TemporalGraphDataset(torch.utils.data.Dataset):
    """
    Dataset loader for temporal supra-graphs.
    """
    
    def __init__(self, graph_files, target_indices=None):
        """
        Args:
            graph_files: List of paths to .npz temporal graph files
            target_indices: Indices to extract from node labels (e.g., [0, 1] for thermal+nuclear)
        """
        self.graph_files = graph_files
        self.target_indices = target_indices
    
    def __len__(self):
        return len(self.graph_files)
    
    def __getitem__(self, idx):
        import numpy as np
        from torch_geometric.data import Data
        
        # Load graph
        data_dict = np.load(self.graph_files[idx], allow_pickle=True)
        
        # Extract components
        x = torch.from_numpy(data_dict["node_features"]).float()
        edge_index = torch.from_numpy(data_dict["edge_index"]).long()
        edge_type = torch.from_numpy(data_dict["edge_types"]).long()
        node_types = torch.from_numpy(data_dict["node_types"]).long()
        
        # Extract metadata
        meta = data_dict["meta"].item()
        N_base = meta["N_base"]
        T = meta["T"]
        
        # Labels (if available)
        # Note: node_labels is [T, N_zones, label_dim] but we have N_base total nodes
        # We need to create full labels and mask for zone nodes only
        if "node_labels" in data_dict and data_dict["node_labels"] is not None:
            zone_labels = torch.from_numpy(data_dict["node_labels"]).float()  # [T, N_zones, label_dim]
            
            # Extract target variables if specified
            if self.target_indices is not None:
                zone_labels = zone_labels[:, :, self.target_indices]  # [T, N_zones, target_dim]
            
            T_labels, N_zones, target_dim = zone_labels.shape
            
            # Find zone node indices in base graph (node_type == 2 for zones)
            # Base node types (first N_base nodes at t=0)
            base_node_types = node_types[:N_base]
            zone_mask_base = base_node_types == 2  # Boolean mask for zones in base graph
            
            # Create full label tensor [N_base*T, target_dim] with zeros (will be masked)
            y_full = torch.zeros(N_base * T, target_dim)
            
            # Create mask for valid labels (zone nodes only)
            label_mask = torch.zeros(N_base * T, dtype=torch.bool)
            
            # Fill in zone labels at correct positions
            for t in range(T):
                # Indices for timestep t in the flattened graph
                t_offset = t * N_base
                
                # Zone indices at this timestep
                zone_indices_t = torch.where(zone_mask_base)[0] + t_offset
                
                # Fill in labels for zones at this timestep
                y_full[zone_indices_t] = zone_labels[t]  # [N_zones, target_dim]
                label_mask[zone_indices_t] = True
            
            y = y_full
        else:
            y = None
            label_mask = None
        
        # Create PyG Data object
        data = Data(
            x=x,
            edge_index=edge_index,
            edge_type=edge_type,
            node_type=node_types,
            y=y,
            label_mask=label_mask,  # Mask indicating which nodes have valid labels
            N_base=N_base,
            T=T,
        )
        
        return data


def create_temporal_hetero_model(
    model_type: str,
    node_feature_dim: int,
    hidden_dim: int,
    output_dim: int,
    **kwargs
) -> TemporalHeteroGNN:
    """
    Factory function to create temporal hetero GNN models.
    
    Args:
        model_type: "rgcn" or "separated"
        node_feature_dim: Input feature dimension
        hidden_dim: Hidden dimension
        output_dim: Output dimension (e.g., 13 for all dispatch variables)
        **kwargs: Additional model-specific arguments
    
    Returns:
        Initialized model
    """
    if model_type == "rgcn":
        return TemporalRGCN(
            node_feature_dim=node_feature_dim,
            hidden_dim=hidden_dim,
            output_dim=output_dim,
            **kwargs
        )
    elif model_type == "separated":
        return SeparatedTemporalGNN(
            node_feature_dim=node_feature_dim,
            hidden_dim=hidden_dim,
            output_dim=output_dim,
            **kwargs
        )
    else:
        raise ValueError(f"Unknown model type: {model_type}")
