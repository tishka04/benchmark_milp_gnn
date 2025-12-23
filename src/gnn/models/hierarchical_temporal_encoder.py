"""
Hierarchical Temporal Encoder for Energy System Graphs

Architecture optimized for multi-scale energy system graphs with hierarchical structure:
- Assets (604 nodes) → Zones (100 nodes) → Regions (10 nodes) → Nation (1 node)

Key features:
1. Bottom-up spatial encoding with sparse GAT (memory efficient)
2. Dense temporal transformer at the top (global context)
3. Top-down information propagation with skip connections
4. Multi-scale embeddings for downstream tasks (EBM, sampling, LP)

Memory: ~3-5 GB (vs 40+ GB for dense HGT)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATv2Conv
from torch_scatter import scatter_mean
import math


class PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding for temporal sequences."""
    
    def __init__(self, d_model, max_len=200, dropout=0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        # Create positional encoding matrix
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # [1, max_len, d_model]
        
        self.register_buffer('pe', pe)
    
    def forward(self, x):
        """
        Args:
            x: [batch_size, seq_len, d_model] or [seq_len, d_model]
        """
        if x.dim() == 2:
            x = x + self.pe[0, :x.size(0), :]
        else:
            x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)


class HierarchicalTemporalEncoder(nn.Module):
    """
    Hierarchical Temporal Encoder for multi-scale energy system graphs.
    
    Architecture:
    1. Bottom-up spatial encoding: Assets → Zones → Regions → Nation
       - Uses sparse GATv2Conv for memory efficiency
       - Hierarchical pooling to aggregate information
    
    2. Temporal encoding at Nation level:
       - Dense Transformer on global context (only 1 node × T timesteps)
       - Captures all temporal dependencies efficiently
    
    3. Top-down propagation:
       - Broadcasts global context back to all levels
       - Skip connections preserve local information
    
    Output: Multi-scale embeddings at all hierarchical levels
    """
    
    def __init__(
        self,
        node_feature_dim: int,
        hidden_dim: int = 128,
        num_heads: int = 8,
        num_spatial_layers: int = 2,
        num_temporal_layers: int = 4,
        dropout: float = 0.1,
        max_seq_len: int = 168,
    ):
        super().__init__()
        
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        
        # Input projection
        self.input_proj = nn.Linear(node_feature_dim, hidden_dim)
        
        # ===== ENCODER: Bottom-Up Hierarchical Spatial Encoding =====
        
        # Asset level (sparse GAT) - add_self_loops=True to preserve all nodes
        self.asset_convs = nn.ModuleList([
            GATv2Conv(hidden_dim, hidden_dim, heads=num_heads, concat=False, dropout=dropout, add_self_loops=True)
            for _ in range(num_spatial_layers)
        ])
        self.asset_norms = nn.ModuleList([
            nn.LayerNorm(hidden_dim) for _ in range(num_spatial_layers)
        ])
        
        # Zone level (sparse GAT) - add_self_loops=True to preserve all nodes
        self.zone_convs = nn.ModuleList([
            GATv2Conv(hidden_dim, hidden_dim, heads=num_heads, concat=False, dropout=dropout, add_self_loops=True)
            for _ in range(num_spatial_layers)
        ])
        self.zone_norms = nn.ModuleList([
            nn.LayerNorm(hidden_dim) for _ in range(num_spatial_layers)
        ])
        
        # Region level (sparse GAT) - add_self_loops=True to preserve all nodes
        self.region_convs = nn.ModuleList([
            GATv2Conv(hidden_dim, hidden_dim, heads=num_heads, concat=False, dropout=dropout, add_self_loops=True)
            for _ in range(num_spatial_layers)
        ])
        self.region_norms = nn.ModuleList([
            nn.LayerNorm(hidden_dim) for _ in range(num_spatial_layers)
        ])
        
        # ===== TEMPORAL TRANSFORMER (at Nation level) =====
        
        self.pos_encoding = PositionalEncoding(hidden_dim, max_seq_len, dropout)
        
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
        
        # ===== DECODER: Top-Down Information Propagation =====
        
        self.region_decoder = GATv2Conv(hidden_dim, hidden_dim, heads=num_heads, concat=False, dropout=dropout)
        self.zone_decoder = GATv2Conv(hidden_dim, hidden_dim, heads=num_heads, concat=False, dropout=dropout)
        self.asset_decoder = GATv2Conv(hidden_dim, hidden_dim, heads=num_heads, concat=False, dropout=dropout)
        
        # Final layer norms
        self.final_norms = nn.ModuleDict({
            'asset': nn.LayerNorm(hidden_dim),
            'zone': nn.LayerNorm(hidden_dim),
            'region': nn.LayerNorm(hidden_dim),
            'nation': nn.LayerNorm(hidden_dim),
        })
        
        self.dropout = nn.Dropout(dropout)
    
    def forward(
        self,
        x,                    # [N_base*T, feature_dim]
        edge_index,           # [2, E] - asset-level edges
        node_type,            # [N_base*T] - node types
        N_base,               # Number of base nodes (e.g., 604)
        T,                    # Number of timesteps (e.g., 96)
        hierarchy_mapping,    # Dict with asset→zone→region mappings
        zone_edges=None,      # Optional: pre-computed zone edges
        region_edges=None,    # Optional: pre-computed region edges
        batch=None,           # Optional: batch assignment
        return_sequence=True, # If True, return [N, T, D], else [N*T, D]
    ):
        """
        Forward pass with hierarchical encoding.
        
        Args:
            x: Node features [N_base*T, feature_dim]
            edge_index: Edge connectivity [2, E]
            node_type: Node types [N_base*T]
            N_base: Number of base nodes
            T: Number of timesteps
            hierarchy_mapping: Dict with keys:
                - 'asset_to_zone': [N_base] tensor, asset_id → zone_id
                - 'zone_to_region': [num_zones] tensor, zone_id → region_id
            zone_edges: Optional edge_index for zone-level graph
            region_edges: Optional edge_index for region-level graph
            batch: Optional batch assignment for batched graphs
            return_sequence: Whether to return [N, T, D] or [N*T, D]
        
        Returns:
            If return_sequence=True:
                Dict with keys 'assets', 'zones', 'regions', 'nation'
                Each: [N, T, hidden_dim] tensors (or batched)
            If return_sequence=False:
                'assets': [N*T, hidden_dim]
        """
        device = x.device
        
        # Project input
        h = self.input_proj(x)  # [N_base*T, hidden_dim]
        h = h.view(N_base, T, self.hidden_dim)  # [N_base, T, D]
        
        # Get hierarchy info
        asset_to_zone = hierarchy_mapping['asset_to_zone'].to(device)
        zone_to_region = hierarchy_mapping['zone_to_region'].to(device)
        
        num_zones = asset_to_zone.max().item() + 1
        num_regions = zone_to_region.max().item() + 1
        
        # Create edges if not provided
        if zone_edges is None:
            zone_edges = self._create_zone_edges(asset_to_zone, zone_to_region, num_zones).to(device)
        
        if region_edges is None:
            region_edges = self._create_region_edges(num_regions).to(device)
        
        # ===== PHASE 1: Bottom-Up Spatial Encoding =====
        
        # Asset level processing
        h_assets = h
        for conv, norm in zip(self.asset_convs, self.asset_norms):
            h_flat = h_assets.view(-1, self.hidden_dim)
            h_residual = h_flat
            
            h_flat = conv(h_flat, edge_index)
            h_flat = norm(h_flat)
            h_flat = F.gelu(h_flat)
            h_flat = self.dropout(h_flat)
            
            # Residual connection
            h_flat = h_flat + h_residual
            h_assets = h_flat.view(N_base, T, self.hidden_dim)
        
        h_assets_skip = h_assets.clone()  # Skip connection for decoder
        
        # Pool: Assets → Zones
        asset_to_zone = hierarchy_mapping["asset_to_zone"]
        zone_to_region = hierarchy_mapping["zone_to_region"]

        num_zones, num_regions = self._infer_cluster_sizes(asset_to_zone, zone_to_region)
        
        h_zones = self._pool_by_assignment(h_assets, asset_to_zone, num_zones, T)
        
        # Zone level processing
        for conv, norm in zip(self.zone_convs, self.zone_norms):
            h_flat = h_zones.view(-1, self.hidden_dim)
            h_residual = h_flat
            
            h_flat = conv(h_flat, zone_edges)
            h_flat = norm(h_flat)
            h_flat = F.gelu(h_flat)
            h_flat = self.dropout(h_flat)
            
            h_flat = h_flat + h_residual
            h_zones = h_flat.view(num_zones, T, self.hidden_dim)
        
        h_zones_skip = h_zones.clone()
        
        # Pool: Zones → Regions
        h_regions = self._pool_by_assignment(h_zones, zone_to_region, num_regions, T)
        
        # Region level processing
        for conv, norm in zip(self.region_convs, self.region_norms):
            h_flat = h_regions.view(-1, self.hidden_dim)
            h_residual = h_flat
            
            h_flat = conv(h_flat, region_edges)
            h_flat = norm(h_flat)
            h_flat = F.gelu(h_flat)
            h_flat = self.dropout(h_flat)
            
            h_flat = h_flat + h_residual
            h_regions = h_flat.view(num_regions, T, self.hidden_dim)
        
        h_regions_skip = h_regions.clone()
        
        # Pool: Regions → Nation (global mean)
        h_nation = h_regions.mean(dim=0, keepdim=True)  # [1, T, D]
        
        # ===== PHASE 2: Temporal Transformer (Dense at Nation level) =====
        
        # Add positional encoding
        h_nation = self.pos_encoding(h_nation)  # [1, T, D]
        
        # Temporal attention (dense, but only on 1 node × T timesteps)
        h_nation = self.temporal_transformer(h_nation)  # [1, T, D]
        h_nation = self.final_norms['nation'](h_nation.squeeze(0)).unsqueeze(0)
        
        # ===== PHASE 3: Top-Down Information Propagation =====
        
        # Nation → Regions (broadcast)
        h_regions_up = h_nation.expand(num_regions, -1, -1)  # [num_regions, T, D]
        h_regions_up = h_regions_up + h_regions_skip  # Skip connection
        
        h_regions_up_flat = h_regions_up.view(-1, self.hidden_dim)
        h_regions_up_flat = self.region_decoder(h_regions_up_flat, region_edges)
        h_regions_up_flat = self.final_norms['region'](h_regions_up_flat)
        h_regions_up = h_regions_up_flat.view(num_regions, T, self.hidden_dim)
        
        # Regions → Zones (scatter)
        h_zones_up = self._unpool_by_assignment(h_regions_up, zone_to_region, num_zones, T)
        h_zones_up = h_zones_up + h_zones_skip  # Skip connection
        
        h_zones_up_flat = h_zones_up.view(-1, self.hidden_dim)
        h_zones_up_flat = self.zone_decoder(h_zones_up_flat, zone_edges)
        h_zones_up_flat = self.final_norms['zone'](h_zones_up_flat)
        h_zones_up = h_zones_up_flat.view(num_zones, T, self.hidden_dim)
        
        # Zones → Assets (scatter)
        h_assets_up = self._unpool_by_assignment(h_zones_up, asset_to_zone, N_base, T)
        h_assets_up = h_assets_up + h_assets_skip  # Skip connection
        
        h_assets_up_flat = h_assets_up.view(-1, self.hidden_dim)
        h_assets_up_flat = self.asset_decoder(h_assets_up_flat, edge_index)
        h_assets_up_flat = self.final_norms['asset'](h_assets_up_flat)
        h_assets_up = h_assets_up_flat.view(N_base, T, self.hidden_dim)
        
        # ===== OUTPUT =====
        
        if return_sequence:
            # Return multi-scale embeddings
            return {
                'assets': h_assets_up,         # [N_base, T, D]
                'zones': h_zones_up,           # [num_zones, T, D]
                'regions': h_regions_up,       # [num_regions, T, D]
                'nation': h_nation.squeeze(0), # [T, D]
            }
        else:
            # Return flattened asset embeddings only
            return h_assets_up.view(N_base * T, self.hidden_dim)

    def _infer_cluster_sizes(self, asset_to_zone, zone_to_region):
        # zone_to_region length defines how many zones exist
        num_zones = int(zone_to_region.numel())
        if num_zones == 0:
            raise RuntimeError("zone_to_region is empty")

        max_a2z = int(asset_to_zone.max().item()) if asset_to_zone.numel() > 0 else -1
        if max_a2z >= num_zones:
            raise RuntimeError(
                f"Inconsistent hierarchy: asset_to_zone max={max_a2z} but zone_to_region has num_zones={num_zones}. "
                "Zones must be indexed 0..num_zones-1 and zone_to_region must have length=num_zones."
            )

        num_regions = int(zone_to_region.max().item()) + 1
        if num_regions <= 0:
            raise RuntimeError("Invalid num_regions inferred from zone_to_region")

        return num_zones, num_regions
    
    def _pool_by_assignment(self, h, assignment, num_clusters, T):
        """
        Pool nodes by assignment (e.g., assets → zones).
        
        Args:
            h: [num_nodes, T, D]
            assignment: [num_nodes] mapping to cluster IDs
            num_clusters: Number of target clusters
            T: Number of timesteps
        
        Returns:
            [num_clusters, T, D]
        """
        h_pooled_list = []
        for t in range(T):
            h_t = h[:, t, :]  # [num_nodes, D]
            h_pooled_t = scatter_mean(h_t, assignment, dim=0, dim_size=num_clusters)
            h_pooled_list.append(h_pooled_t)
        
        return torch.stack(h_pooled_list, dim=1)  # [num_clusters, T, D]
    
    def _unpool_by_assignment(self, h_pooled, assignment, num_nodes, T):
        """
        Unpool (scatter) embeddings from clusters to nodes.
        
        Args:
            h_pooled: [num_clusters, T, D]
            assignment: [num_nodes] mapping to cluster IDs
            num_nodes: Number of target nodes
            T: Number of timesteps
        
        Returns:
            [num_nodes, T, D]
        """
        h_unpooled_list = []
        for t in range(T):
            h_t = h_pooled[:, t, :]  # [num_clusters, D]
            h_unpooled_t = h_t[assignment]  # [num_nodes, D]
            h_unpooled_list.append(h_unpooled_t)
        
        return torch.stack(h_unpooled_list, dim=1)  # [num_nodes, T, D]
    
    def _create_zone_edges(self, asset_to_zone, zone_to_region, num_zones):
        """
        Create edges between zones.
        Strategy: Connect zones within the same region (fully connected).
        """
        edges = []
        
        # Group zones by region
        for region_id in range(zone_to_region.max().item() + 1):
            zones_in_region = torch.where(zone_to_region == region_id)[0]
            
            # Fully connect zones within region
            if len(zones_in_region) > 1:
                for i in range(len(zones_in_region)):
                    for j in range(i + 1, len(zones_in_region)):
                        edges.append([zones_in_region[i].item(), zones_in_region[j].item()])
                        edges.append([zones_in_region[j].item(), zones_in_region[i].item()])
        
        if len(edges) == 0:
            # Fallback: fully connected
            edge_index = torch.combinations(torch.arange(num_zones), r=2).t()
            edge_index = torch.cat([edge_index, edge_index.flip(0)], dim=1)
            return edge_index
        
        return torch.tensor(edges, dtype=torch.long).t()
    
    def _create_region_edges(self, num_regions):
        """
        Create edges between regions.
        Strategy: Fully connected (small number of regions).
        """
        if num_regions <= 1:
            return torch.empty((2, 0), dtype=torch.long)
        
        edge_index = torch.combinations(torch.arange(num_regions), r=2).t()
        edge_index = torch.cat([edge_index, edge_index.flip(0)], dim=1)
        return edge_index
    
    def get_embeddings(self, *args, **kwargs):
        """Alias for forward() for compatibility."""
        return self.forward(*args, **kwargs)
