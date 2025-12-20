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
import math


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
    
    def forward(self, x, edge_index, edge_type, node_type=None, batch=None):
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
    
    def forward(self, x, edge_index, edge_type, node_type=None, batch=None):
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


class TemporalHGT(TemporalHeteroGNN):
    """
    Heterogeneous Graph Transformer for temporal multi-layer graphs.
    
    Uses node-type-aware attention to handle heterogeneous node types
    (Nation, Region, Zone, Asset, Weather) and edge types (spatial + temporal).
    
    Note: HGTConv requires node-type and edge-type metadata dictionaries.
    Since our graphs don't have explicit PyG metadata, we simulate it by
    using node_type integers directly.
    """
    
    def __init__(
        self,
        node_feature_dim: int,
        hidden_dim: int,
        output_dim: int,
        num_node_types: int = 5,  # Nation, Region, Zone, Asset, Weather
        num_edge_types: int = 10,  # Spatial (7) + Temporal (3)
        num_layers: int = 3,
        num_heads: int = 4,
        dropout: float = 0.1,
    ):
        super().__init__(
            node_feature_dim,
            hidden_dim,
            output_dim,
            num_edge_types,
            num_layers,
        )
        
        self.num_node_types = num_node_types
        self.num_heads = num_heads
        
        # Input projection for each node type
        self.input_projs = nn.ModuleList([
            Linear(node_feature_dim, hidden_dim)
            for _ in range(num_node_types)
        ])
        
        # HGT layers
        # Note: HGTConv expects metadata but we'll use a simplified approach
        # with direct edge_type encoding
        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()
        
        for _ in range(num_layers):
            # For each layer, create separate transformations per node type
            self.convs.append(nn.ModuleDict({
                'lin_src': nn.ModuleList([Linear(hidden_dim, hidden_dim) for _ in range(num_node_types)]),
                'lin_dst': nn.ModuleList([Linear(hidden_dim, hidden_dim) for _ in range(num_node_types)]),
                'attn': nn.ModuleList([
                    nn.MultiheadAttention(hidden_dim, num_heads, dropout=dropout, batch_first=True)
                    for _ in range(num_edge_types)
                ]),
            }))
            self.norms.append(nn.LayerNorm(hidden_dim))
        
        self.dropout = nn.Dropout(dropout)
        
        # Output head
        self.output_head = nn.Sequential(
            Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            Linear(hidden_dim, output_dim),
        )
    
    def forward(self, x, edge_index, edge_type, node_type=None, batch=None):
        """
        Args:
            x: Node features [N, node_feature_dim]
            edge_index: Edge indices [2, E]
            edge_type: Edge types [E]
            node_type: Node types [N] (required for HGT)
            batch: Batch assignment (optional)
        """
        # If node_type not provided, infer from features or use default
        if node_type is None:
            # Assume uniform node type (fallback)
            node_type = torch.zeros(x.size(0), dtype=torch.long, device=x.device)
        
        # Project input features based on node type
        h = torch.zeros(x.size(0), self.hidden_dim, device=x.device)
        for nt in range(self.num_node_types):
            mask = node_type == nt
            if mask.any():
                h[mask] = self.input_projs[nt](x[mask])
        
        # Message passing with heterogeneous attention
        for layer_idx, (conv_dict, norm) in enumerate(zip(self.convs, self.norms)):
            h_new = torch.zeros_like(h)
            
            # For each edge type, perform message passing
            for et in range(self.num_edge_types):
                edge_mask = edge_type == et
                if not edge_mask.any():
                    continue
                
                et_edge_index = edge_index[:, edge_mask]
                src_nodes = et_edge_index[0]
                dst_nodes = et_edge_index[1]
                
                # Get unique source and destination node types
                src_types = node_type[src_nodes]
                dst_types = node_type[dst_nodes]
                
                # For simplicity, aggregate across all node type combinations
                # In a full HGT, you'd have separate attention per (src_type, edge_type, dst_type) triple
                for dst_nt in range(self.num_node_types):
                    dst_mask = dst_types == dst_nt
                    if not dst_mask.any():
                        continue
                    
                    # Get destination nodes of this type
                    dst_idx = dst_nodes[dst_mask]
                    src_idx = src_nodes[dst_mask]
                    
                    # Transform source and destination features
                    h_src = h[src_idx]  # [num_edges_of_type, hidden_dim]
                    h_dst = h[dst_idx]  # [num_edges_of_type, hidden_dim]
                    
                    # Apply attention (using destination as query)
                    # MultiheadAttention expects [batch, seq_len, hidden_dim]
                    h_dst_expanded = h_dst.unsqueeze(0)  # [1, num_dst, hidden_dim]
                    h_src_expanded = h_src.unsqueeze(0)  # [1, num_src, hidden_dim]
                    
                    attn_out, _ = conv_dict['attn'][et](
                        h_dst_expanded, h_src_expanded, h_src_expanded
                    )
                    attn_out = attn_out.squeeze(0)  # [num_dst, hidden_dim]
                    
                    # Aggregate to destination nodes (scatter_add for multiple edges to same dst)
                    h_new.index_add_(0, dst_idx, attn_out)
            
            # Normalize and residual
            h_new = norm(h_new)
            h_new = F.relu(h_new)
            h_new = self.dropout(h_new)
            
            if layer_idx > 0:
                h = h + h_new
            else:
                h = h_new
        
        # Output
        out = self.output_head(h)
        return out


class HGTTemporalTransformer(nn.Module):
    """
    HGT + Temporal Transformer Encoder (NO DECODER).
    
    Architecture:
    1. HGT: Spatial encoding at each timestep → [N_base*T, hidden_dim]
    2. Unroll: Reshape to [N_base, T, hidden_dim]
    3. Temporal Transformer: Self-attention across time dimension
    4. Output: Spatio-temporal embeddings [N_base, T, hidden_dim]
    
    Use case: Generate embeddings for EBM training, not for direct prediction.
    """
    
    def __init__(
        self,
        node_feature_dim: int,
        hidden_dim: int,
        num_node_types: int = 5,
        num_edge_types: int = 10,
        num_spatial_layers: int = 3,
        num_temporal_layers: int = 2,
        num_heads: int = 4,
        dropout: float = 0.1,
        max_seq_len: int = 168,  # Max T (e.g., 7 days * 24 hours)
    ):
        super().__init__()
        
        self.hidden_dim = hidden_dim
        self.num_node_types = num_node_types
        self.num_heads = num_heads
        
        # ===== SPATIAL ENCODER (HGT) =====
        
        # Input projection for each node type
        self.input_projs = nn.ModuleList([
            Linear(node_feature_dim, hidden_dim)
            for _ in range(num_node_types)
        ])
        
        # HGT layers for spatial encoding
        self.spatial_convs = nn.ModuleList()
        self.spatial_norms = nn.ModuleList()
        
        for _ in range(num_spatial_layers):
            self.spatial_convs.append(nn.ModuleDict({
                'attn': nn.ModuleList([
                    nn.MultiheadAttention(hidden_dim, num_heads, dropout=dropout, batch_first=True)
                    for _ in range(num_edge_types)
                ]),
            }))
            self.spatial_norms.append(nn.LayerNorm(hidden_dim))
        
        self.spatial_dropout = nn.Dropout(dropout)
        
        # ===== TEMPORAL ENCODER (Transformer) =====
        
        # Positional encoding for time dimension
        self.pos_encoding = PositionalEncoding(hidden_dim, max_seq_len, dropout)
        
        # Temporal Transformer layers
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout,
            activation='gelu',
            batch_first=True,
        )
        self.temporal_transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_temporal_layers,
        )
        
        # Final layer norm
        self.final_norm = nn.LayerNorm(hidden_dim)
    
    def forward(
        self,
        x,
        edge_index,
        edge_type,
        node_type,
        N_base,
        T,
        batch=None,
        return_sequence=True,
    ):
        """
        Args:
            x: Node features [N_base*T, node_feature_dim]
            edge_index: Edge indices [2, E]
            edge_type: Edge types [E]
            node_type: Node types [N_base*T]
            N_base: Number of base nodes (before time expansion)
            T: Number of timesteps
            batch: Batch assignment (optional)
            return_sequence: If True, return [N_base, T, hidden_dim]
                            If False, return [N_base*T, hidden_dim]
        
        Returns:
            embeddings: Spatio-temporal embeddings
                - If return_sequence: [N_base, T, hidden_dim] or [batch_size, N_base, T, hidden_dim]
                - If not return_sequence: [N_base*T, hidden_dim]
        """
        # ===== PHASE 1: SPATIAL ENCODING (HGT) =====
        
        # Project input features based on node type
        h = torch.zeros(x.size(0), self.hidden_dim, device=x.device)
        for nt in range(self.num_node_types):
            mask = node_type == nt
            if mask.any():
                h[mask] = self.input_projs[nt](x[mask])
        
        # Spatial message passing (simplified HGT)
        for layer_idx, (conv_dict, norm) in enumerate(zip(self.spatial_convs, self.spatial_norms)):
            h_new = torch.zeros_like(h)
            
            # For each edge type, perform message passing
            for et in range(len(conv_dict['attn'])):
                edge_mask = edge_type == et
                if not edge_mask.any():
                    continue
                
                et_edge_index = edge_index[:, edge_mask]
                src_nodes = et_edge_index[0]
                dst_nodes = et_edge_index[1]
                
                # Get unique destination node types
                dst_types = node_type[dst_nodes]
                
                for dst_nt in range(self.num_node_types):
                    dst_mask = dst_types == dst_nt
                    if not dst_mask.any():
                        continue
                    
                    dst_idx = dst_nodes[dst_mask]
                    src_idx = src_nodes[dst_mask]
                    
                    h_src = h[src_idx].unsqueeze(0)
                    h_dst = h[dst_idx].unsqueeze(0)
                    
                    attn_out, _ = conv_dict['attn'][et](h_dst, h_src, h_src)
                    attn_out = attn_out.squeeze(0)
                    
                    h_new.index_add_(0, dst_idx, attn_out)
            
            h_new = norm(h_new)
            h_new = F.gelu(h_new)
            h_new = self.spatial_dropout(h_new)
            
            if layer_idx > 0:
                h = h + h_new
            else:
                h = h_new
        
        # ===== PHASE 2: TEMPORAL ENCODING (Transformer) =====
        
        # Reshape to [N_base, T, hidden_dim] or [batch_size, N_base, T, hidden_dim]
        if batch is None:
            # Single graph
            h_spatial = h.view(N_base, T, self.hidden_dim)  # [N_base, T, D]
            
            # Add positional encoding
            h_temporal = self.pos_encoding(h_spatial)  # [N_base, T, D]
            
            # Apply Transformer across time dimension
            # Process each base node's time series independently
            h_out = self.temporal_transformer(h_temporal)  # [N_base, T, D]
            
            # Final normalization
            embeddings = self.final_norm(h_out)  # [N_base, T, D]
            
            if not return_sequence:
                embeddings = embeddings.view(N_base * T, self.hidden_dim)
        
        else:
            # Batched graphs - need to handle variable N_base per graph
            # For simplicity, assume all graphs have same N_base (pad if needed)
            batch_size = batch.max().item() + 1
            
            # Group by batch
            embeddings_list = []
            for b in range(batch_size):
                mask = batch == b
                h_b = h[mask].view(N_base, T, self.hidden_dim)
                
                h_temporal = self.pos_encoding(h_b)
                h_out = self.temporal_transformer(h_temporal)
                embeddings_b = self.final_norm(h_out)
                
                embeddings_list.append(embeddings_b)
            
            if return_sequence:
                embeddings = torch.stack(embeddings_list, dim=0)  # [B, N_base, T, D]
            else:
                embeddings = torch.cat([e.view(-1, self.hidden_dim) for e in embeddings_list], dim=0)
        
        return embeddings
    
    def get_embeddings(self, *args, **kwargs):
        """Alias for forward() for clarity."""
        return self.forward(*args, **kwargs)


class PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding for time dimension."""
    
    def __init__(self, d_model: int, max_len: int = 5000, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        # Create positional encoding matrix
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        
        pe = torch.zeros(1, max_len, d_model)
        pe[0, :, 0::2] = torch.sin(position * div_term)
        pe[0, :, 1::2] = torch.cos(position * div_term)
        
        self.register_buffer('pe', pe)
    
    def forward(self, x):
        """
        Args:
            x: [N_base, T, d_model] or [batch_size, N_base, T, d_model]
        """
        if x.dim() == 3:
            # [N_base, T, d_model]
            x = x + self.pe[:, :x.size(1), :]
        elif x.dim() == 4:
            # [batch_size, N_base, T, d_model]
            x = x + self.pe[:, :x.size(2), :]
        
        return self.dropout(x)


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
        
        # ===== EXTRACT HIERARCHY MAPPINGS =====
        
        # 1. Zone → Region mapping (from zone_region_index in graph)
        if 'zone_region_index' in data_dict:
            zone_to_region = torch.from_numpy(data_dict['zone_region_index']).long()
        else:
            # Fallback: create dummy mapping if not available
            base_node_types = node_types[:N_base]
            num_zones = (base_node_types == 2).sum().item()
            num_regions = max(1, (base_node_types == 1).sum().item())
            zone_to_region = torch.arange(num_zones) % num_regions
        
        # 2. Asset → Zone mapping (from edge structure)
        base_node_types = node_types[:N_base]
        asset_mask_base = base_node_types == 3
        zone_mask_base = base_node_types == 2
        
        num_zones = zone_mask_base.sum().item()
        zone_indices = torch.where(zone_mask_base)[0]
        
        # Initialize mapping
        asset_to_zone = torch.zeros(N_base, dtype=torch.long)
        
        # Extract base graph edges (spatial edges, types 0-6)
        spatial_mask = edge_type < 7
        spatial_edges = edge_index[:, spatial_mask]
        
        # Map temporal edges to base node indices
        base_edges = spatial_edges % N_base
        
        # For each asset, find its parent zone via edges
        for asset_idx in torch.where(asset_mask_base)[0]:
            # Find edges where this asset is the source
            outgoing_mask = base_edges[0] == asset_idx
            if outgoing_mask.any():
                targets = base_edges[1, outgoing_mask]
                # Find which targets are zones
                zone_targets = targets[zone_mask_base[targets]]
                if len(zone_targets) > 0:
                    # Map to zone list index (0 to num_zones-1)
                    zone_node_id = zone_targets[0].item()
                    zone_list_idx = (zone_indices == zone_node_id).nonzero(as_tuple=True)[0]
                    if len(zone_list_idx) > 0:
                        asset_to_zone[asset_idx] = zone_list_idx[0]
        
        # For non-asset nodes, assign to first zone (fallback)
        asset_to_zone[~asset_mask_base] = 0
        
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
            asset_to_zone=asset_to_zone,        # Hierarchy mapping
            zone_to_region=zone_to_region,      # Hierarchy mapping
        )
        
        return data


def create_temporal_hetero_model(
    model_type: str,
    node_feature_dim: int,
    hidden_dim: int,
    output_dim: int = None,
    **kwargs
) -> TemporalHeteroGNN:
    """
    Factory function to create temporal hetero GNN models.
    
    Args:
        model_type: "rgcn", "separated", "hgt", or "hgt-transformer"
        node_feature_dim: Input feature dimension
        hidden_dim: Hidden dimension
        output_dim: Output dimension (e.g., 13 for all dispatch variables)
                   Not required for "hgt-transformer" (encoder-only)
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
    elif model_type == "hgt":
        return TemporalHGT(
            node_feature_dim=node_feature_dim,
            hidden_dim=hidden_dim,
            output_dim=output_dim,
            **kwargs
        )
    elif model_type == "hgt-transformer":
        return HGTTemporalTransformer(
            node_feature_dim=node_feature_dim,
            hidden_dim=hidden_dim,
            **kwargs
        )
    else:
        raise ValueError(f"Unknown model type: {model_type}")
