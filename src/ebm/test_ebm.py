"""
Test script for EBM implementation.

Verifies that all components work correctly with dummy data.
"""

import torch
import numpy as np

from .model import EnergyModel, StructuredEnergyModel, FactorizedEnergyModel
from .sampler import GibbsSampler, SGLDSampler, PersistentContrastiveDivergence
from .metrics import EBMMetrics, TemporalMetrics


def test_energy_models():
    """Test all energy model architectures."""
    print("Testing Energy Models...")
    
    batch_size = 8
    dim_u = 100
    dim_h = 128
    
    # Create dummy data
    u = torch.randint(0, 2, (batch_size, dim_u), dtype=torch.float32)
    h = torch.randn(batch_size, dim_h)
    
    # Test basic model
    print("\n1. EnergyModel (Basic)")
    model_basic = EnergyModel(dim_u=dim_u, dim_h=dim_h)
    E_basic = model_basic(u, h)
    print(f"   Output shape: {E_basic.shape}")
    print(f"   Energy range: [{E_basic.min():.4f}, {E_basic.max():.4f}]")
    assert E_basic.shape == (batch_size,), "Basic model output shape incorrect"
    
    # Test structured model
    print("\n2. StructuredEnergyModel (with quadratic)")
    model_structured = StructuredEnergyModel(
        dim_u=dim_u,
        dim_h=dim_h,
        use_quadratic=True,
        quadratic_rank=8,
    )
    E_structured = model_structured(u, h)
    print(f"   Output shape: {E_structured.shape}")
    print(f"   Energy range: [{E_structured.min():.4f}, {E_structured.max():.4f}]")
    assert E_structured.shape == (batch_size,), "Structured model output shape incorrect"
    
    # Test quadratic matrix computation
    A = model_structured.compute_quadratic_matrix(h)
    print(f"   Quadratic matrix shape: {A.shape}")
    assert A.shape == (batch_size, dim_u, dim_u), "Quadratic matrix shape incorrect"
    
    # Test factorized model
    print("\n3. FactorizedEnergyModel")
    model_factorized = FactorizedEnergyModel(dim_u=dim_u, dim_h=dim_h)
    E_factorized = model_factorized(u, h)
    print(f"   Output shape: {E_factorized.shape}")
    print(f"   Energy range: [{E_factorized.min():.4f}, {E_factorized.max():.4f}]")
    assert E_factorized.shape == (batch_size,), "Factorized model output shape incorrect"
    
    print("\n✓ All energy models passed!")


def test_samplers():
    """Test sampling methods."""
    print("\nTesting Samplers...")
    
    batch_size = 4
    dim_u = 50
    dim_h = 128
    
    # Create dummy model
    model = EnergyModel(dim_u=dim_u, dim_h=dim_h)
    h = torch.randn(batch_size, dim_h)
    
    # Test Gibbs sampler
    print("\n1. GibbsSampler")
    gibbs_sampler = GibbsSampler(
        energy_model=model,
        num_steps=20,
        temperature=1.0,
    )
    u_gibbs = gibbs_sampler.sample(h)
    print(f"   Output shape: {u_gibbs.shape}")
    print(f"   Binary check: {torch.all((u_gibbs == 0) | (u_gibbs == 1))}")
    assert u_gibbs.shape == (batch_size, dim_u), "Gibbs sampler output shape incorrect"
    assert torch.all((u_gibbs == 0) | (u_gibbs == 1)), "Gibbs sampler output not binary"
    
    # Test sample chain
    chain = gibbs_sampler.sample_chain(h, num_samples=5, thin=5)
    print(f"   Chain shape: {chain.shape}")
    assert chain.shape == (5, batch_size, dim_u), "Gibbs chain shape incorrect"
    
    # Test SGLD sampler
    print("\n2. SGLDSampler")
    sgld_sampler = SGLDSampler(
        energy_model=model,
        num_steps=20,
        step_size=0.01,
    )
    u_sgld = sgld_sampler.sample(h)
    print(f"   Output shape: {u_sgld.shape}")
    print(f"   Binary check: {torch.all((u_sgld == 0) | (u_sgld == 1))}")
    assert u_sgld.shape == (batch_size, dim_u), "SGLD sampler output shape incorrect"
    
    # Test PCD
    print("\n3. PersistentContrastiveDivergence")
    pcd_sampler = PersistentContrastiveDivergence(
        energy_model=model,
        num_chains=10,
        num_steps=10,
    )
    u_pcd = pcd_sampler.sample(h)
    print(f"   Output shape: {u_pcd.shape}")
    print(f"   Chain initialized: {pcd_sampler.chains is not None}")
    assert u_pcd.shape == (batch_size, dim_u), "PCD sampler output shape incorrect"
    
    print("\n✓ All samplers passed!")


def test_metrics():
    """Test evaluation metrics."""
    print("\nTesting Metrics...")
    
    batch_size = 10
    dim_u = 50
    dim_h = 128
    
    # Create dummy data
    model = EnergyModel(dim_u=dim_u, dim_h=dim_h)
    h = torch.randn(batch_size, dim_h)
    u_pos = torch.randint(0, 2, (batch_size, dim_u), dtype=torch.float32)
    u_neg = torch.randint(0, 2, (batch_size, dim_u), dtype=torch.float32)
    
    metrics = EBMMetrics()
    
    # Test energy gap
    print("\n1. Energy Gap")
    gap = metrics.compute_energy_gap(model, u_pos, u_neg, h)
    print(f"   Gap: {gap:.4f}")
    
    # Test sample diversity
    print("\n2. Sample Diversity")
    diversity = metrics.compute_sample_diversity(u_pos)
    print(f"   Diversity: {diversity:.4f}")
    assert 0 <= diversity <= 1, "Diversity out of range"
    
    # Test feasibility rate
    print("\n3. Feasibility Rate")
    feasibility = metrics.compute_feasibility_rate(u_pos)
    print(f"   Feasibility: {feasibility:.4f}")
    assert 0 <= feasibility <= 1, "Feasibility out of range"
    
    # Test constraint violations
    print("\n4. Constraint Violations")
    violations = metrics.compute_constraint_violations(u_pos)
    print(f"   Violations: {violations}")
    
    # Test classification accuracy
    print("\n5. Classification Accuracy")
    acc = metrics.compute_classification_accuracy(model, u_pos, u_neg, h)
    print(f"   Accuracy: {acc:.4f}")
    assert 0 <= acc <= 1, "Accuracy out of range"
    
    # Test temporal metrics
    print("\n6. Temporal Metrics")
    u_temporal = torch.randint(0, 2, (batch_size, 10, 5), dtype=torch.float32)
    consistency = TemporalMetrics.compute_temporal_consistency(u_temporal)
    print(f"   Temporal consistency: {consistency:.4f}")
    assert 0 <= consistency <= 1, "Consistency out of range"
    
    ramping = TemporalMetrics.compute_ramping_violations(u_temporal)
    print(f"   Ramping violations: {ramping:.4f}")
    assert 0 <= ramping <= 1, "Ramping violations out of range"
    
    print("\n✓ All metrics passed!")


def test_training_step():
    """Test a single training step."""
    print("\nTesting Training Step...")
    
    batch_size = 8
    dim_u = 50
    dim_h = 128
    
    # Create model and optimizer
    model = StructuredEnergyModel(dim_u=dim_u, dim_h=dim_h)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    
    # Create sampler
    sampler = GibbsSampler(model, num_steps=10)
    
    # Create dummy batch
    u_pos = torch.randint(0, 2, (batch_size, dim_u), dtype=torch.float32)
    h = torch.randn(batch_size, dim_h)
    
    # Compute positive energy
    E_pos = model(u_pos, h)
    print(f"   Positive energy: {E_pos.mean():.4f}")
    
    # Generate negative samples
    u_neg = sampler.sample(h)
    E_neg = model(u_neg, h)
    print(f"   Negative energy: {E_neg.mean():.4f}")
    
    # Compute loss
    loss = E_pos.mean() - E_neg.mean()
    print(f"   Loss: {loss:.4f}")
    
    # Backward pass
    optimizer.zero_grad()
    loss.backward()
    
    # Check gradients
    has_grad = any(p.grad is not None and p.grad.abs().sum() > 0 for p in model.parameters())
    print(f"   Gradients computed: {has_grad}")
    assert has_grad, "No gradients computed"
    
    # Optimization step
    optimizer.step()
    print(f"   Optimization step completed")
    
    print("\n✓ Training step passed!")


def test_forward_backward():
    """Test forward and backward passes."""
    print("\nTesting Forward/Backward...")
    
    dim_u = 30
    dim_h = 64
    batch_size = 4
    
    model = StructuredEnergyModel(dim_u=dim_u, dim_h=dim_h)
    
    u = torch.randint(0, 2, (batch_size, dim_u), dtype=torch.float32, requires_grad=False)
    h = torch.randn(batch_size, dim_h, requires_grad=False)
    
    # Forward pass
    E = model(u, h)
    print(f"   Forward pass output: {E.shape}")
    
    # Backward pass
    loss = E.sum()
    loss.backward()
    
    # Check parameter gradients
    params_with_grad = sum(1 for p in model.parameters() if p.grad is not None)
    total_params = sum(1 for _ in model.parameters())
    print(f"   Parameters with gradients: {params_with_grad}/{total_params}")
    
    assert params_with_grad == total_params, "Not all parameters have gradients"
    
    print("\n✓ Forward/backward passed!")


def run_all_tests():
    """Run all tests."""
    print("=" * 60)
    print("EBM Implementation Tests")
    print("=" * 60)
    
    try:
        test_energy_models()
        test_samplers()
        test_metrics()
        test_training_step()
        test_forward_backward()
        
        print("\n" + "=" * 60)
        print("✓ ALL TESTS PASSED!")
        print("=" * 60)
        
    except Exception as e:
        print("\n" + "=" * 60)
        print(f"✗ TEST FAILED: {e}")
        print("=" * 60)
        raise


if __name__ == '__main__':
    run_all_tests()
