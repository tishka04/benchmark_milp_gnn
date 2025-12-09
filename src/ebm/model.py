"""
Energy-Based Models for UC/DR/Storage Configuration Learning

Implements the energy function architecture from Section 3 of the methodology:
- Basic MLP energy function with bilinear interactions
- Optional structured quadratic interaction term A_θ(h)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import global_add_pool

class GraphEnergyModel(nn.Module):
    """
    Energy-Based Model for variable-sized graphs (Deep Sets architecture).
    Treats the MILP instance as a set of variables (nodes) conditioned on a global context (embedding).
    """
    def __init__(self, dim_u=1, dim_h=128, hidden_dims=[256, 256, 64], activation='gelu', dropout=0.1):
        super().__init__()

        self.dim_u = dim_u
        self.dim_h = dim_h

        # Build MLP
        layers = []
        input_dim = dim_u + dim_h

        # Select activation function
        if activation.lower() == 'relu':
            act_fn = nn.ReLU()
        elif activation.lower() == 'gelu':
            act_fn = nn.GELU()
        elif activation.lower() == 'tanh':
            act_fn = nn.Tanh()
        elif activation.lower() == 'leaky_relu':
            act_fn = nn.LeakyReLU()
        else:
            raise ValueError(f"Unknown activation: {activation}")

        # Hidden layers
        prev_dim = input_dim
        for h_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, h_dim))
            layers.append(act_fn)
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            prev_dim = h_dim

        # Final projection to scalar energy
        layers.append(nn.Linear(prev_dim, 1))

        self.mlp = nn.Sequential(*layers)

    def forward(self, batch):
        """
        Args:
            batch: PyG Batch object containing:
                - x: Node features [num_nodes, dim_u] (binary values)
                - h: Graph features [batch_size, dim_h] (embeddings)
                - batch: Batch indices [num_nodes]
        Returns:
            Energy scalar per graph [batch_size, 1]
        """
        x = batch.x
        h = batch.h
        batch_idx = batch.batch

        # Broadcast global embedding h to each node belonging to that graph
        # h is [batch_size, dim_h], batch_idx is [num_nodes]
        # h[batch_idx] results in [num_nodes, dim_h]
        h_expanded = h[batch_idx]

        # Concatenate node features with expanded global context
        # Shape: [num_nodes, dim_u + dim_h]
        combined = torch.cat([x, h_expanded], dim=1)

        # Compute energy contribution per node
        # Shape: [num_nodes, 1]
        node_energies = self.mlp(combined)

        # Aggregate node energies to get total graph energy
        # We use add pooling (sum) because energy is extensive
        # Shape: [batch_size, 1]
        graph_energy = global_add_pool(node_energies, batch_idx)

        return graph_energy


class EnergyModel(nn.Module):
    """
    Basic Energy-Based Model with bilinear interactions.
    
    E_θ(u | h) = f_θ([u || h || (u ⊙ h)])
    
    where:
    - u ∈ {0,1}^N: binary decisions (UC/DR/Storage)
    - h ∈ R^128: graph embedding from Hierarchical Temporal Encoder
    - f_θ: MLP with GELU activations
    """
    
    def __init__(
        self,
        dim_u: int,
        dim_h: int = 128,
        hidden_dims: list = [256, 256, 64],
        activation: str = 'gelu',
        dropout: float = 0.1,
    ):
        """
        Args:
            dim_u: Dimension of binary decision vector (number of binary variables)
            dim_h: Dimension of graph embedding (default: 128)
            hidden_dims: List of hidden layer dimensions
            activation: Activation function ('gelu' or 'silu')
            dropout: Dropout probability
        """
        super().__init__()
        
        self.dim_u = dim_u
        self.dim_h = dim_h
        
        # Input dimension: u + h + (u ⊙ h)
        # Note: u ⊙ h broadcasts h to match u's dimension for elementwise product
        input_dim = dim_u + dim_h + dim_u
        
        # Build MLP layers
        layers = []
        prev_dim = input_dim
        
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.LayerNorm(hidden_dim))
            layers.append(nn.GELU() if activation == 'gelu' else nn.SiLU())
            layers.append(nn.Dropout(dropout))
            prev_dim = hidden_dim
        
        # Output layer (energy is scalar)
        layers.append(nn.Linear(prev_dim, 1))
        
        self.net = nn.Sequential(*layers)
        
    def forward(self, u, h):
        """
        Compute energy E_θ(u | h).
        
        Args:
            u: Binary decisions [batch_size, dim_u] or [batch_size, T, dim_u]
            h: Graph embeddings [batch_size, dim_h] or [batch_size, T, dim_h]
            
        Returns:
            energy: Scalar energy [batch_size] or [batch_size, T]
        """
        # Handle temporal dimension if present
        if u.dim() == 3:  # [batch_size, T, dim_u]
            batch_size, T, _ = u.shape
            u_flat = u.reshape(batch_size * T, -1)
            h_flat = h.reshape(batch_size * T, -1) if h.dim() == 3 else h.unsqueeze(1).repeat(1, T, 1).reshape(batch_size * T, -1)
            
            # Compute bilinear interaction: u ⊙ h (broadcast h to match u's dim)
            # h_flat: [batch_size * T, dim_h]
            # We need to expand h to [batch_size * T, dim_u] for elementwise product
            h_expanded = h_flat.mean(dim=-1, keepdim=True).expand(-1, self.dim_u)
            uh = u_flat * h_expanded
            
            # Concatenate [u || h || (u ⊙ h)]
            x = torch.cat([u_flat, h_flat, uh], dim=-1)
            
            # Compute energy
            energy = self.net(x).squeeze(-1)  # [batch_size * T]
            energy = energy.reshape(batch_size, T)
            
        else:  # [batch_size, dim_u]
            # Compute bilinear interaction
            h_expanded = h.mean(dim=-1, keepdim=True).expand(-1, self.dim_u)
            uh = u * h_expanded
            
            # Concatenate [u || h || (u ⊙ h)]
            x = torch.cat([u, h, uh], dim=-1)
            
            # Compute energy
            energy = self.net(x).squeeze(-1)  # [batch_size]
        
        return energy


class StructuredEnergyModel(nn.Module):
    """
    Energy model with optional quadratic interaction term (Section 3).
    
    E_θ(u | h) = f_θ([u || h || (u ⊙ h)]) + u^T A_θ(h) u
    
    where A_θ(h) ∈ R^{N×N} captures pairwise interactions between binary variables,
    useful for encoding constraints like:
    - Ramping constraints
    - Min-up/min-down times
    - Storage consistency
    """
    
    def __init__(
        self,
        dim_u: int,
        dim_h: int = 128,
        hidden_dims: list = [256, 256, 64],
        activation: str = 'gelu',
        dropout: float = 0.1,
        use_quadratic: bool = True,
        quadratic_rank: int = 16,
    ):
        """
        Args:
            dim_u: Dimension of binary decision vector
            dim_h: Dimension of graph embedding
            hidden_dims: Hidden layer dimensions for base MLP
            activation: Activation function
            dropout: Dropout probability
            use_quadratic: Whether to include quadratic interaction term
            quadratic_rank: Rank for low-rank factorization of A_θ(h)
        """
        super().__init__()
        
        self.dim_u = dim_u
        self.dim_h = dim_h
        self.use_quadratic = use_quadratic
        
        # Base energy function
        self.base_energy = EnergyModel(
            dim_u=dim_u,
            dim_h=dim_h,
            hidden_dims=hidden_dims,
            activation=activation,
            dropout=dropout,
        )
        
        # Quadratic interaction term: A_θ(h) = L_θ(h) @ L_θ(h)^T
        # where L_θ(h) ∈ R^{N×r} (low-rank factorization for efficiency)
        if use_quadratic:
            self.quadratic_net = nn.Sequential(
                nn.Linear(dim_h, hidden_dims[0]),
                nn.GELU(),
                nn.Linear(hidden_dims[0], dim_u * quadratic_rank),
            )
            self.quadratic_rank = quadratic_rank
    
    def compute_quadratic_matrix(self, h):
        """
        Compute quadratic interaction matrix A_θ(h) from embedding h.
        
        Args:
            h: Graph embedding [batch_size, dim_h] or [batch_size, T, dim_h]
            
        Returns:
            A: Quadratic matrix [batch_size, dim_u, dim_u] or [batch_size, T, dim_u, dim_u]
        """
        if h.dim() == 3:  # Temporal case
            batch_size, T, _ = h.shape
            h_flat = h.reshape(batch_size * T, -1)
            
            # Compute low-rank factors: [batch_size * T, dim_u * rank]
            L_flat = self.quadratic_net(h_flat)
            L = L_flat.reshape(batch_size * T, self.dim_u, self.quadratic_rank)
            
            # A = L @ L^T: [batch_size * T, dim_u, dim_u]
            A_flat = torch.bmm(L, L.transpose(1, 2))
            A = A_flat.reshape(batch_size, T, self.dim_u, self.dim_u)
            
        else:
            # Compute low-rank factors: [batch_size, dim_u * rank]
            L_flat = self.quadratic_net(h)
            L = L_flat.reshape(-1, self.dim_u, self.quadratic_rank)
            
            # A = L @ L^T: [batch_size, dim_u, dim_u]
            A = torch.bmm(L, L.transpose(1, 2))
        
        return A
    
    def forward(self, u, h):
        """
        Compute structured energy E_θ(u | h).
        
        Args:
            u: Binary decisions [batch_size, dim_u] or [batch_size, T, dim_u]
            h: Graph embeddings [batch_size, dim_h] or [batch_size, T, dim_h]
            
        Returns:
            energy: Scalar energy [batch_size] or [batch_size, T]
        """
        # Base energy from MLP
        energy = self.base_energy(u, h)
        
        # Add quadratic interaction if enabled
        if self.use_quadratic:
            A = self.compute_quadratic_matrix(h)
            
            if u.dim() == 3:  # Temporal case
                # u: [batch_size, T, dim_u]
                # A: [batch_size, T, dim_u, dim_u]
                # Compute u^T A u for each (batch, t)
                batch_size, T, _ = u.shape
                u_flat = u.reshape(batch_size * T, self.dim_u, 1)
                A_flat = A.reshape(batch_size * T, self.dim_u, self.dim_u)
                
                # u^T @ A @ u: [batch_size * T, 1, 1]
                quadratic_flat = torch.bmm(torch.bmm(u_flat.transpose(1, 2), A_flat), u_flat).squeeze(-1).squeeze(-1)
                quadratic = quadratic_flat.reshape(batch_size, T)
                
            else:
                # u: [batch_size, dim_u]
                # A: [batch_size, dim_u, dim_u]
                u_expanded = u.unsqueeze(-1)  # [batch_size, dim_u, 1]
                
                # u^T @ A @ u: [batch_size, 1, 1]
                quadratic = torch.bmm(torch.bmm(u_expanded.transpose(1, 2), A), u_expanded).squeeze(-1).squeeze(-1)
            
            energy = energy + quadratic
        
        return energy


class FactorizedEnergyModel(nn.Module):
    """
    Factorized energy model (alternative formulation from Section 3).
    
    E(u | h) = Σ_i φ_i(u_i, h) + Σ_{i,j} ψ_{ij}(u_i, u_j, h)
    
    Useful for encoding specific constraint structures.
    """
    
    def __init__(
        self,
        dim_u: int,
        dim_h: int = 128,
        hidden_dim: int = 64,
    ):
        """
        Args:
            dim_u: Number of binary variables
            dim_h: Embedding dimension
            hidden_dim: Hidden dimension for potential networks
        """
        super().__init__()
        
        self.dim_u = dim_u
        self.dim_h = dim_h
        
        # Unary potentials φ_i(u_i, h)
        self.unary_net = nn.Sequential(
            nn.Linear(1 + dim_h, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1),
        )
        
        # Pairwise potentials ψ_{ij}(u_i, u_j, h)
        # For efficiency, share parameters across all pairs
        self.pairwise_net = nn.Sequential(
            nn.Linear(2 + dim_h, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1),
        )
    
    def forward(self, u, h):
        """
        Compute factorized energy.
        
        Args:
            u: Binary decisions [batch_size, dim_u]
            h: Graph embeddings [batch_size, dim_h]
            
        Returns:
            energy: Scalar energy [batch_size]
        """
        batch_size = u.shape[0]
        
        # Unary terms: Σ_i φ_i(u_i, h)
        unary_energy = 0
        for i in range(self.dim_u):
            u_i = u[:, i:i+1]  # [batch_size, 1]
            x_i = torch.cat([u_i, h], dim=-1)  # [batch_size, 1 + dim_h]
            unary_energy = unary_energy + self.unary_net(x_i).squeeze(-1)
        
        # Pairwise terms: Σ_{i<j} ψ_{ij}(u_i, u_j, h)
        # For efficiency, only compute for neighboring variables (optional: full pairwise)
        pairwise_energy = 0
        # Temporal neighbors (i, i+1)
        for i in range(self.dim_u - 1):
            u_i = u[:, i:i+1]
            u_j = u[:, i+1:i+2]
            x_ij = torch.cat([u_i, u_j, h], dim=-1)
            pairwise_energy = pairwise_energy + self.pairwise_net(x_ij).squeeze(-1)
        
        energy = unary_energy + pairwise_energy
        return energy
