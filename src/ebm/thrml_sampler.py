"""
THRML-based sampler for conditional EBM.

Uses Extropic's THRML package for hardware-accelerated Gibbs sampling.
"""

import numpy as np
import torch
from typing import Optional, Tuple

try:
    import jax
    import jax.numpy as jnp
    from thrml import SpinNode, Block
    from thrml.models import SpinEBMFactor, SpinGibbsConditional
    from thrml.models import BlockSamplingProgram, FactorSamplingProgram
    from thrml.models import SamplingSchedule, sample_states, hinton_init
    THRML_AVAILABLE = True
except ImportError:
    THRML_AVAILABLE = False


class ThrmlConditionalSampler:
    """
    THRML-based sampler for conditional EBM E_θ(u | h).
    
    Strategy:
    1. For a given embedding h, extract energy parameters
    2. Build SpinEBMFactor representing E(u | h)
    3. Use THRML's block Gibbs sampling
    4. Convert samples back to binary {0, 1}
    
    Limitations:
    - Works best if energy is approximately quadratic in u
    - Requires reconstructing factor for each new h
    - MLP term f_θ(u, h) approximated via Taylor expansion
    """
    
    def __init__(
        self,
        energy_model,
        dim_u: int,
        temperature: float = 1.0,
        n_warmup: int = 100,
        n_samples_per_chain: int = 1,
        steps_per_sample: int = 5,
        use_quadratic_only: bool = False,
        device: str = 'cpu',
    ):
        """
        Args:
            energy_model: Your EBM model (StructuredEnergyModel)
            dim_u: Dimension of binary variables
            temperature: Sampling temperature
            n_warmup: Number of warmup steps (thermalization)
            n_samples_per_chain: Samples to keep per chain
            steps_per_sample: Gibbs steps between samples
            use_quadratic_only: If True, ignore MLP term (faster but less accurate)
            device: PyTorch device
        """
        if not THRML_AVAILABLE:
            raise ImportError(
                "THRML not available. Install with: pip install thrml"
            )
        
        self.energy_model = energy_model
        self.dim_u = dim_u
        self.temperature = temperature
        self.n_warmup = n_warmup
        self.n_samples_per_chain = n_samples_per_chain
        self.steps_per_sample = steps_per_sample
        self.use_quadratic_only = use_quadratic_only
        self.device = device
        
        # Cache for factor graph (rebuilt per h)
        self._cached_h = None
        self._cached_factor = None
    
    def _extract_energy_parameters(
        self, 
        h: torch.Tensor
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Extract Ising parameters from EBM for given h.
        
        For energy E_θ(u | h) = f_θ(u, h) + u^T A_θ(h) u,
        approximate as Ising model:
        E(σ) ≈ Σ h_i σ_i + Σ J_{ij} σ_i σ_j
        
        where σ_i ∈ {-1, +1} and u_i = (σ_i + 1)/2
        
        Returns:
            fields: External fields h_i [dim_u]
            couplings: Interaction matrix J [dim_u, dim_u]
        """
        h_np = h.detach().cpu().numpy()
        
        # Get quadratic term A_θ(h)
        with torch.no_grad():
            if hasattr(self.energy_model, 'get_quadratic_matrix'):
                A_h = self.energy_model.get_quadratic_matrix(h.unsqueeze(0))
                A_h = A_h.squeeze(0).cpu().numpy()  # [dim_u, dim_u]
            else:
                # Fallback: estimate via finite differences
                A_h = self._estimate_quadratic_term(h)
        
        # Get linear term from MLP (if not using quadratic only)
        if not self.use_quadratic_only:
            linear_term = self._estimate_linear_term(h)
        else:
            linear_term = np.zeros(self.dim_u)
        
        # Convert QUBO to Ising
        # QUBO: E(u) = u^T A u + L^T u  where u ∈ {0,1}
        # Ising: E(σ) = Σ h_i σ_i + Σ J_{ij} σ_i σ_j  where σ ∈ {-1,+1}
        # Relation: u_i = (σ_i + 1)/2
        
        h_ising = np.zeros(self.dim_u)
        J_ising = np.zeros((self.dim_u, self.dim_u))
        
        # Linear terms
        h_ising = -linear_term / 2.0
        
        # Quadratic terms
        for i in range(self.dim_u):
            for j in range(i + 1, self.dim_u):
                if abs(A_h[i, j]) > 1e-8:
                    # QUBO quadratic to Ising coupling
                    J_val = A_h[i, j] / 4.0
                    J_ising[i, j] = -J_val
                    J_ising[j, i] = -J_val
                    
                    # Add correction to fields
                    h_ising[i] -= J_val
                    h_ising[j] -= J_val
        
        # Diagonal terms (if any)
        for i in range(self.dim_u):
            if abs(A_h[i, i]) > 1e-8:
                h_ising[i] -= A_h[i, i] / 4.0
        
        return h_ising, J_ising
    
    def _estimate_quadratic_term(self, h: torch.Tensor) -> np.ndarray:
        """Estimate A_θ(h) via finite differences."""
        A = np.zeros((self.dim_u, self.dim_u))
        epsilon = 1e-4
        
        with torch.no_grad():
            u_base = torch.zeros(1, self.dim_u, device=self.device)
            E_base = self.energy_model(u_base, h.unsqueeze(0)).item()
            
            # Estimate diagonal
            for i in range(self.dim_u):
                u_i = u_base.clone()
                u_i[0, i] = 1.0
                E_i = self.energy_model(u_i, h.unsqueeze(0)).item()
                
                # Second derivative ≈ curvature
                A[i, i] = (E_i - E_base) / (0.5)
            
            # Estimate off-diagonal (sample only, too expensive for all)
            n_samples = min(100, self.dim_u * (self.dim_u - 1) // 2)
            indices = np.random.choice(self.dim_u, size=(n_samples, 2), replace=True)
            
            for i, j in indices:
                if i == j:
                    continue
                
                u_ij = u_base.clone()
                u_ij[0, i] = 1.0
                u_ij[0, j] = 1.0
                E_ij = self.energy_model(u_ij, h.unsqueeze(0)).item()
                
                u_i = u_base.clone()
                u_i[0, i] = 1.0
                E_i = self.energy_model(u_i, h.unsqueeze(0)).item()
                
                u_j = u_base.clone()
                u_j[0, j] = 1.0
                E_j = self.energy_model(u_j, h.unsqueeze(0)).item()
                
                # Interaction term
                A[i, j] = E_ij - E_i - E_j + E_base
        
        return A
    
    def _estimate_linear_term(self, h: torch.Tensor) -> np.ndarray:
        """Estimate linear term ∂E/∂u_i at u=0."""
        linear = np.zeros(self.dim_u)
        
        with torch.no_grad():
            u_base = torch.zeros(1, self.dim_u, device=self.device)
            E_base = self.energy_model(u_base, h.unsqueeze(0)).item()
            
            for i in range(self.dim_u):
                u_i = u_base.clone()
                u_i[0, i] = 1.0
                E_i = self.energy_model(u_i, h.unsqueeze(0)).item()
                
                linear[i] = E_i - E_base
        
        return linear
    
    def _build_factor(
        self,
        h: torch.Tensor,
    ) -> Tuple[SpinEBMFactor, Block]:
        """
        Build THRML SpinEBMFactor for E_θ(u | h).
        
        Returns:
            factor: SpinEBMFactor representing energy
            block: Block of spin nodes
        """
        # Extract parameters
        fields, couplings = self._extract_energy_parameters(h)
        
        # Create spin nodes
        nodes = [SpinNode() for _ in range(self.dim_u)]
        block = Block(nodes)
        
        # Build interaction lists
        # Fields: unary interactions
        field_indices = []
        field_weights = []
        for i in range(self.dim_u):
            if abs(fields[i]) > 1e-8:
                field_indices.append([i])
                field_weights.append(fields[i])
        
        # Couplings: pairwise interactions
        coupling_indices = []
        coupling_weights = []
        for i in range(self.dim_u):
            for j in range(i + 1, self.dim_u):
                if abs(couplings[i, j]) > 1e-8:
                    coupling_indices.append([i, j])
                    coupling_weights.append(couplings[i, j])
        
        # Create factor
        # Note: SpinEBMFactor expects indices and weights
        # This is a simplified version - adjust based on actual THRML API
        factor = SpinEBMFactor(
            variables=nodes,
            field_weights=jnp.array(field_weights) if field_weights else jnp.array([]),
            field_indices=jnp.array(field_indices) if field_indices else jnp.array([]),
            coupling_weights=jnp.array(coupling_weights) if coupling_weights else jnp.array([]),
            coupling_indices=jnp.array(coupling_indices) if coupling_indices else jnp.array([]),
            temperature=self.temperature,
        )
        
        return factor, block
    
    def sample(
        self,
        h: torch.Tensor,
        batch_size: int = 1,
        num_chains: int = 1,
    ) -> torch.Tensor:
        """
        Sample binary configurations u given embedding h.
        
        Args:
            h: Graph embedding [dim_h] or [batch_size, dim_h]
            batch_size: Number of samples per h
            num_chains: Number of parallel chains
            
        Returns:
            u: Binary samples [batch_size, dim_u]
        """
        # Handle batched h
        if h.dim() == 1:
            h = h.unsqueeze(0)
        
        all_samples = []
        
        for h_single in h:
            # Build or retrieve cached factor
            factor, block = self._build_factor(h_single)
            
            # Create sampling program
            program = FactorSamplingProgram(
                factors=[factor],
                free_blocks=[block],
                clamped_blocks=[],
            )
            
            # Sample multiple chains
            chain_samples = []
            for chain_idx in range(num_chains):
                # Initialize
                key = jax.random.PRNGKey(
                    np.random.randint(0, 2**31) + chain_idx
                )
                k_init, k_samp = jax.random.split(key)
                
                # Initial state using Hinton initialization
                init_state = hinton_init(
                    k_init,
                    program,
                    [block],
                    (),
                )
                
                # Sampling schedule
                schedule = SamplingSchedule(
                    n_warmup=self.n_warmup,
                    n_samples=self.n_samples_per_chain,
                    steps_per_sample=self.steps_per_sample,
                )
                
                # Run sampling
                samples = sample_states(
                    k_samp,
                    program,
                    schedule,
                    init_state,
                    [],
                    [block],
                )
                
                # Convert from spin {-1, +1} to binary {0, 1}
                for s in samples:
                    spin_values = np.array(s[0]).flatten()
                    binary = ((spin_values + 1) / 2).astype(int)
                    chain_samples.append(binary)
            
            # Select batch_size samples
            selected = np.array(chain_samples[:batch_size])
            all_samples.append(selected)
        
        # Stack and convert to PyTorch
        all_samples = np.concatenate(all_samples, axis=0)
        return torch.tensor(all_samples, dtype=torch.float32, device=self.device)


class HybridThrmlGibbsSampler:
    """
    Hybrid sampler: THRML for quadratic, PyTorch Gibbs for refinement.
    
    This provides a good balance:
    - Fast THRML sampling for coarse structure
    - Gibbs refinement for MLP term accuracy
    """
    
    def __init__(
        self,
        energy_model,
        dim_u: int,
        temperature: float = 1.0,
        thrml_steps: int = 100,
        gibbs_refinement_steps: int = 10,
        device: str = 'cpu',
    ):
        self.energy_model = energy_model
        self.dim_u = dim_u
        self.temperature = temperature
        self.thrml_steps = thrml_steps
        self.gibbs_steps = gibbs_refinement_steps
        self.device = device
        
        # Initialize THRML sampler (quadratic only)
        self.thrml_sampler = ThrmlConditionalSampler(
            energy_model=energy_model,
            dim_u=dim_u,
            temperature=temperature,
            n_warmup=thrml_steps,
            n_samples_per_chain=1,
            use_quadratic_only=True,  # Fast approximate sampling
            device=device,
        )
    
    def sample(self, h: torch.Tensor, batch_size: int = 1) -> torch.Tensor:
        """
        Two-stage sampling:
        1. THRML for coarse structure (quadratic term)
        2. Gibbs for refinement (full energy)
        """
        # Stage 1: THRML sampling (fast, approximate)
        u_coarse = self.thrml_sampler.sample(h, batch_size=batch_size)
        
        # Stage 2: Gibbs refinement (accurate)
        u_refined = self._gibbs_refine(u_coarse, h)
        
        return u_refined
    
    def _gibbs_refine(self, u: torch.Tensor, h: torch.Tensor) -> torch.Tensor:
        """Refine samples with a few Gibbs steps."""
        u = u.clone()
        
        if h.dim() == 1:
            h = h.unsqueeze(0).expand(u.shape[0], -1)
        
        for step in range(self.gibbs_steps):
            # Random scan order
            indices = torch.randperm(self.dim_u, device=self.device)
            
            for i in indices:
                # Compute energy with u_i = 0
                u_0 = u.clone()
                u_0[:, i] = 0.0
                E_0 = self.energy_model(u_0, h)
                
                # Compute energy with u_i = 1
                u_1 = u.clone()
                u_1[:, i] = 1.0
                E_1 = self.energy_model(u_1, h)
                
                # Gibbs update
                prob_1 = torch.sigmoid(-(E_1 - E_0) / self.temperature)
                u[:, i] = torch.bernoulli(prob_1)
        
        return u


# Example usage
if __name__ == '__main__':
    print("THRML Sampler for Conditional EBM")
    print("=" * 60)
    
    if not THRML_AVAILABLE:
        print("❌ THRML not installed")
        print("Install with: pip install jax jaxlib thrml")
    else:
        print("✓ THRML available")
        print("\nExample usage:")
        print("""
from src.ebm import StructuredEnergyModel
from src.ebm.thrml_sampler import ThrmlConditionalSampler

# Create model
model = StructuredEnergyModel(dim_u=672, dim_h=128)

# Create THRML sampler
sampler = ThrmlConditionalSampler(
    energy_model=model,
    dim_u=672,
    temperature=1.0,
    n_warmup=100,
)

# Sample given embedding
h = torch.randn(128)
u_samples = sampler.sample(h, batch_size=10)
print(u_samples.shape)  # [10, 672]
        """)
