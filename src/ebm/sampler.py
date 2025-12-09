"""
Sampling methods for Energy-Based Models.

Implements:
- Gibbs sampling (Section 5 of methodology)
- Discrete SGLD (Section 6 of methodology)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class GraphSampler:
    """
    Sampler for GraphEnergyModel (Deep Sets).
    Since variables are independent given the context h (Deep Sets property),
    we can sample exactly in one step by computing P(x_i=1|h) for all nodes.
    """
    def __init__(self, energy_model, device='cuda'):
        self.model = energy_model
        self.device = device

    def sample(self, batch, num_steps=1):
        """
        Args:
            batch: PyG Batch object
            num_steps: Ignored for Deep Sets (exact sampling), kept for API compatibility
        Returns:
            u_sample: Binary tensor [total_nodes, 1]
        """
        self.model.eval()

        # Prepare inputs
        # batch.h is [batch_size, dim_h]
        # batch.batch is [total_nodes]
        h_expanded = batch.h[batch.batch]

        # We want to compute P(x=1 | h) = sigmoid(E(x=0) - E(x=1))
        # Construct inputs for x=0 and x=1 for ALL nodes in parallel
        num_nodes = h_expanded.size(0)

        zeros = torch.zeros((num_nodes, 1), device=self.device)
        ones = torch.ones((num_nodes, 1), device=self.device)

        input_0 = torch.cat([zeros, h_expanded], dim=1)
        input_1 = torch.cat([ones, h_expanded], dim=1)

        with torch.no_grad():
            # Compute node-level energies using the MLP directly
            # (bypassing the pooling in model.forward)
            e_0 = self.model.mlp(input_0)
            e_1 = self.model.mlp(input_1)

            # Logits for Bernoulli: P(x=1) = 1 / (1 + exp(E1 - E0)) = sigmoid(E0 - E1)
            logits = e_0 - e_1
            probs = torch.sigmoid(logits)

            # Sample
            u_sample = torch.bernoulli(probs)

        return u_sample


class GibbsSampler:
    """
    Gibbs sampler for binary EBM.
    
    Samples from P(u | h) ∝ exp(-E_θ(u | h)) by iteratively sampling each u_i
    from its conditional distribution:
    
    P(u_i = 1 | u_{-i}, h) = σ(-ΔE_i)
    
    where ΔE_i = E(u_i=1 | u_{-i}, h) - E(u_i=0 | u_{-i}, h)
    """
    
    def __init__(
        self,
        energy_model: nn.Module,
        num_steps: int = 100,
        temperature: float = 1.0,
        device: str = 'cpu',
    ):
        """
        Args:
            energy_model: EBM that computes E_θ(u | h)
            num_steps: Number of Gibbs sampling steps
            temperature: Sampling temperature (higher = more random)
            device: Device for computation
        """
        self.energy_model = energy_model
        self.num_steps = num_steps
        self.temperature = temperature
        self.device = device
    
    def sample(self, h, initial_u=None, block_size=1):
        """
        Sample from P(u | h) using Gibbs sampling.
        
        Args:
            h: Graph embeddings [batch_size, dim_h]
            initial_u: Initial binary configuration [batch_size, dim_u]
                      If None, initialize randomly
            block_size: Number of variables to sample jointly (1 = standard Gibbs)
            
        Returns:
            u: Sampled binary configuration [batch_size, dim_u]
        """
        batch_size = h.shape[0]
        dim_u = self.energy_model.dim_u
        
        # Initialize
        if initial_u is None:
            u = torch.randint(0, 2, (batch_size, dim_u), dtype=torch.float32, device=self.device)
        else:
            u = initial_u.clone()
        
        # Gibbs sampling loop
        for step in range(self.num_steps):
            # Sample variables in random order
            indices = torch.randperm(dim_u, device=self.device)
            
            for idx in range(0, dim_u, block_size):
                # Get indices to update
                block_indices = indices[idx:idx+block_size]
                
                if block_size == 1:
                    # Standard Gibbs: sample one variable at a time
                    i = block_indices[0].item()
                    
                    # Compute conditional probabilities
                    u_0 = u.clone()
                    u_0[:, i] = 0
                    
                    u_1 = u.clone()
                    u_1[:, i] = 1
                    
                    E_0 = self.energy_model(u_0, h)
                    E_1 = self.energy_model(u_1, h)
                    
                    # P(u_i = 1 | u_{-i}, h) = σ(-(E_1 - E_0) / T)
                    delta_E = E_1 - E_0
                    prob_1 = torch.sigmoid(-delta_E / self.temperature)
                    
                    # Sample
                    u[:, i] = torch.bernoulli(prob_1).to(u.dtype)
                
                else:
                    # Block Gibbs: sample multiple variables jointly
                    # This is more complex - enumerate all 2^block_size configurations
                    raise NotImplementedError("Block Gibbs sampling not yet implemented")
        
        return u
    
    def sample_chain(self, h, initial_u=None, num_samples=10, thin=10):
        """
        Generate a chain of samples using Gibbs sampling.
        
        Args:
            h: Graph embeddings [batch_size, dim_h]
            initial_u: Initial configuration
            num_samples: Number of samples to collect
            thin: Thinning interval (collect every 'thin' steps)
            
        Returns:
            samples: List of sampled configurations
        """
        samples = []
        u = initial_u
        
        for _ in range(num_samples):
            u = self.sample(h, initial_u=u, block_size=1)
            samples.append(u.clone())
            
            # Additional thinning steps
            for _ in range(thin - 1):
                u = self.sample(h, initial_u=u, block_size=1)
        
        return torch.stack(samples, dim=0)  # [num_samples, batch_size, dim_u]


class SGLDSampler:
    """
    Stochastic Gradient Langevin Dynamics for discrete binary variables.
    
    Uses continuous relaxation:
    u_t+1 = u_t - η ∇_u E_θ(u_t | h) + √(2η/T) ε_t
    
    with projection back to {0, 1} via rounding or Gumbel-softmax.
    """
    
    def __init__(
        self,
        energy_model: nn.Module,
        num_steps: int = 100,
        step_size: float = 0.01,
        temperature: float = 1.0,
        noise_scale: float = 0.01,
        device: str = 'cpu',
        use_gumbel: bool = False,
    ):
        """
        Args:
            energy_model: EBM that computes E_θ(u | h)
            num_steps: Number of SGLD steps
            step_size: Step size η
            temperature: Temperature T
            noise_scale: Scale of Langevin noise
            device: Device for computation
            use_gumbel: Whether to use Gumbel-softmax relaxation
        """
        self.energy_model = energy_model
        self.num_steps = num_steps
        self.step_size = step_size
        self.temperature = temperature
        self.noise_scale = noise_scale
        self.device = device
        self.use_gumbel = use_gumbel
    
    def sample(self, h, initial_u=None):
        """
        Sample using discrete SGLD.
        
        Args:
            h: Graph embeddings [batch_size, dim_h]
            initial_u: Initial configuration [batch_size, dim_u]
            
        Returns:
            u: Sampled binary configuration [batch_size, dim_u]
        """
        batch_size = h.shape[0]
        dim_u = self.energy_model.dim_u
        
        # Initialize with continuous relaxation
        if initial_u is None:
            u_continuous = torch.rand(batch_size, dim_u, device=self.device)
        else:
            u_continuous = initial_u.clone().float()
        
        u_continuous.requires_grad_(True)
        
        # SGLD iterations
        for step in range(self.num_steps):
            # Compute energy gradient
            if u_continuous.grad is not None:
                u_continuous.grad.zero_()
            
            energy = self.energy_model(u_continuous, h).sum()
            energy.backward()
            
            with torch.no_grad():
                # SGLD update
                grad = u_continuous.grad
                noise = torch.randn_like(u_continuous) * self.noise_scale
                
                u_continuous = u_continuous - self.step_size * grad + noise
                
                # Project to [0, 1]
                u_continuous = torch.clamp(u_continuous, 0, 1)
                
                u_continuous.requires_grad_(True)
        
        # Convert to binary
        u_binary = (u_continuous > 0.5).float()
        
        return u_binary.detach()


class AdaptiveGibbsSampler(GibbsSampler):
    """
    Adaptive Gibbs sampler that adjusts temperature based on acceptance rate.
    
    Helps prevent mode collapse by ensuring sufficient exploration.
    """
    
    def __init__(
        self,
        energy_model: nn.Module,
        num_steps: int = 100,
        initial_temperature: float = 1.0,
        min_temperature: float = 0.5,
        max_temperature: float = 2.0,
        target_acceptance_rate: float = 0.234,  # Optimal for Metropolis
        adaptation_rate: float = 0.01,
        device: str = 'cpu',
    ):
        """
        Args:
            energy_model: EBM model
            num_steps: Number of sampling steps
            initial_temperature: Starting temperature
            min_temperature: Minimum allowed temperature
            max_temperature: Maximum allowed temperature
            target_acceptance_rate: Target acceptance rate
            adaptation_rate: Rate of temperature adaptation
            device: Computation device
        """
        super().__init__(energy_model, num_steps, initial_temperature, device)
        
        self.min_temperature = min_temperature
        self.max_temperature = max_temperature
        self.target_acceptance_rate = target_acceptance_rate
        self.adaptation_rate = adaptation_rate
        self.acceptance_count = 0
        self.total_count = 0
    
    def sample(self, h, initial_u=None, block_size=1):
        """
        Sample with adaptive temperature.
        """
        # Call parent sampler
        u = super().sample(h, initial_u, block_size)
        
        # Update temperature based on acceptance rate
        if self.total_count > 0:
            current_rate = self.acceptance_count / self.total_count
            
            if current_rate < self.target_acceptance_rate:
                # Acceptance too low - increase temperature
                self.temperature = min(
                    self.max_temperature,
                    self.temperature * (1 + self.adaptation_rate)
                )
            else:
                # Acceptance too high - decrease temperature
                self.temperature = max(
                    self.min_temperature,
                    self.temperature * (1 - self.adaptation_rate)
                )
        
        return u


class PersistentContrastiveDivergence:
    """
    Persistent Contrastive Divergence (PCD) sampler.
    
    Maintains persistent chains across training iterations to improve mixing.
    """
    
    def __init__(
        self,
        energy_model: nn.Module,
        num_chains: int = 100,
        num_steps: int = 10,
        temperature: float = 1.0,
        device: str = 'cpu',
    ):
        """
        Args:
            energy_model: EBM model
            num_chains: Number of persistent chains to maintain
            num_steps: Number of Gibbs steps per iteration
            temperature: Sampling temperature
            device: Computation device
        """
        self.energy_model = energy_model
        self.num_chains = num_chains
        self.num_steps = num_steps
        self.temperature = temperature
        self.device = device
        
        # Initialize persistent chains
        self.chains = None
        
        self.gibbs_sampler = GibbsSampler(
            energy_model, num_steps, temperature, device
        )
    
    def initialize_chains(self, h):
        """
        Initialize persistent chains.
        
        Args:
            h: Graph embeddings [batch_size, dim_h]
        """
        if self.chains is None:
            dim_u = self.energy_model.dim_u
            self.chains = torch.randint(
                0, 2, (self.num_chains, dim_u),
                dtype=torch.float32, device=self.device
            )
    
    def sample(self, h):
        """
        Sample from persistent chains.
        
        Args:
            h: Graph embeddings [batch_size, dim_h]
            
        Returns:
            u: Sampled configurations [batch_size, dim_u]
        """
        batch_size = h.shape[0]
        
        # Initialize chains if needed
        self.initialize_chains(h)
        
        # Select random subset of chains
        indices = torch.randperm(self.num_chains, device=self.device)[:batch_size]
        u_init = self.chains[indices]
        
        # Run Gibbs sampling
        # Replicate h for each selected chain
        h_replicated = h[:batch_size]
        u_sampled = self.gibbs_sampler.sample(h_replicated, initial_u=u_init)
        
        # Update persistent chains
        self.chains[indices] = u_sampled.detach()
        
        return u_sampled
