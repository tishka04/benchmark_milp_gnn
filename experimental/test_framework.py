"""
Quick test script to verify the experimental framework.

This script runs a minimal test to ensure all components work together.
"""
import sys
from pathlib import Path
import numpy as np

# Add experimental to path
sys.path.insert(0, str(Path(__file__).parent))

from data_models import UCInstance, ExperimentConfig
from instance_generator import UCInstanceGenerator
from milp_solver import MILPSolver
from hybrid_solver import HybridSolver
from dispatch_solver import solve_dispatch_given_commitment, calculate_total_cost


def test_instance_generation():
    """Test instance generator."""
    print("\n" + "="*80)
    print("TEST 1: Instance Generation")
    print("="*80)
    
    generator = UCInstanceGenerator()
    instance = generator.generate(n_units=10, n_periods=24, seed=42, instance_id=0)
    
    print(f"✓ Generated instance: {instance.scale_name}")
    print(f"  Units: {instance.n_units}, Periods: {instance.n_periods}")
    print(f"  Peak demand: {instance.demand.max():.0f} MW")
    print(f"  Total capacity: {instance.p_max.sum():.0f} MW")
    print(f"  Margin: {(instance.p_max.sum() / instance.demand.max() - 1) * 100:.1f}%")
    
    # Save and load
    test_file = Path(__file__).parent / "test_instance.json"
    instance.save(str(test_file))
    loaded = UCInstance.load(str(test_file))
    test_file.unlink()
    
    assert np.allclose(instance.demand, loaded.demand)
    print("✓ Save/load works correctly")
    
    return instance


def test_dispatch_solver(instance):
    """Test dispatch solver."""
    print("\n" + "="*80)
    print("TEST 2: Dispatch Solver")
    print("="*80)
    
    # Test with all units on
    commitment = np.ones(instance.n_units, dtype=int)
    
    dispatch, cost, feasible = solve_dispatch_given_commitment(
        instance, commitment, period=0
    )
    
    print(f"✓ Dispatch solver works")
    print(f"  Commitment: all on")
    print(f"  Demand: {instance.demand[0]:.0f} MW")
    print(f"  Dispatch sum: {dispatch.sum():.0f} MW")
    print(f"  Cost: ${cost:,.2f}")
    print(f"  Feasible: {feasible}")
    
    assert feasible, "All-on commitment should be feasible"
    assert abs(dispatch.sum() - instance.demand[0]) < 1, "Demand balance violated"
    
    return dispatch, cost


def test_milp_solver(instance):
    """Test MILP solver."""
    print("\n" + "="*80)
    print("TEST 3: MILP Solver")
    print("="*80)
    
    config = ExperimentConfig(
        time_budget_small=30  # 30 seconds for test
    )
    
    solver = MILPSolver(config)
    
    print("Running MILP solver (30s time limit)...")
    solution = solver.solve(instance, time_limit=30)
    
    print(f"✓ MILP solver completed")
    print(f"  Feasible: {solution.feasible}")
    print(f"  Cost: ${solution.total_cost:,.2f}")
    print(f"  Time: {solution.solve_time:.2f}s")
    if solution.optimal:
        print(f"  Status: OPTIMAL")
    else:
        print(f"  Status: Time limit reached")
    
    return solution


def test_hybrid_solver(instance):
    """Test Hybrid solver."""
    print("\n" + "="*80)
    print("TEST 4: Hybrid Solver")
    print("="*80)
    
    config = ExperimentConfig(
        hybrid_n_samples_per_period=5,  # Reduced for testing
        hybrid_n_seeds=2,
        hybrid_warmup=50,
        time_budget_small=30
    )
    
    solver = HybridSolver(config)
    
    print("Running Hybrid solver (30s time limit)...")
    solution = solver.solve(instance, time_limit=30)
    
    print(f"✓ Hybrid solver completed")
    print(f"  Feasible: {solution.feasible}")
    print(f"  Cost: ${solution.total_cost:,.2f}")
    print(f"  Time: {solution.solve_time:.2f}s")
    print(f"  Iterations: {solution.iterations}")
    
    return solution


def test_cost_alignment(instance, milp_solution, hybrid_solution):
    """Test that both solvers use the same cost function."""
    print("\n" + "="*80)
    print("TEST 5: Cost Alignment")
    print("="*80)
    
    # Recalculate costs using canonical function
    milp_cost_recalc, milp_fuel, milp_startup = calculate_total_cost(
        instance, milp_solution.commitment, milp_solution.power
    )
    
    hybrid_cost_recalc, hybrid_fuel, hybrid_startup = calculate_total_cost(
        instance, hybrid_solution.commitment, hybrid_solution.power
    )
    
    print(f"MILP cost (reported): ${milp_solution.total_cost:,.2f}")
    print(f"MILP cost (recalculated): ${milp_cost_recalc:,.2f}")
    print(f"Difference: ${abs(milp_solution.total_cost - milp_cost_recalc):,.2f}")
    
    print(f"\nHybrid cost (reported): ${hybrid_solution.total_cost:,.2f}")
    print(f"Hybrid cost (recalculated): ${hybrid_cost_recalc:,.2f}")
    print(f"Difference: ${abs(hybrid_solution.total_cost - hybrid_cost_recalc):,.2f}")
    
    # Allow small numerical differences
    assert abs(milp_solution.total_cost - milp_cost_recalc) < 1.0, "MILP cost mismatch"
    assert abs(hybrid_solution.total_cost - hybrid_cost_recalc) < 1.0, "Hybrid cost mismatch"
    
    print("\n✓ Cost alignment verified - both methods use same cost function")


def test_comparison(milp_solution, hybrid_solution):
    """Compare solutions."""
    print("\n" + "="*80)
    print("TEST 6: Solution Comparison")
    print("="*80)
    
    if milp_solution.feasible and hybrid_solution.feasible:
        cost_ratio = hybrid_solution.total_cost / milp_solution.total_cost
        time_ratio = milp_solution.solve_time / hybrid_solution.solve_time
        
        print(f"MILP:   ${milp_solution.total_cost:>12,.2f} in {milp_solution.solve_time:>6.2f}s")
        print(f"Hybrid: ${hybrid_solution.total_cost:>12,.2f} in {hybrid_solution.solve_time:>6.2f}s")
        print(f"\nCost ratio (Hybrid/MILP): {cost_ratio:.3f}")
        print(f"Time ratio (MILP/Hybrid): {time_ratio:.3f}")
        
        if cost_ratio < 1.0:
            print(f"\n✓ Hybrid found cheaper solution!")
        elif cost_ratio < 1.05:
            print(f"\n✓ Hybrid cost within 5% of MILP")
        else:
            print(f"\n⚠ Hybrid cost {(cost_ratio - 1) * 100:.1f}% more expensive")
        
        if time_ratio > 1.0:
            print(f"✓ Hybrid was {time_ratio:.2f}x faster")
        else:
            print(f"⚠ MILP was {1/time_ratio:.2f}x faster (expected for small instance)")
    else:
        print("⚠ Cannot compare - one or both solutions infeasible")


def main():
    """Run all tests."""
    print("\n" + "="*80)
    print("EXPERIMENTAL FRAMEWORK TEST SUITE")
    print("="*80)
    print("\nThis will verify that all components work correctly.")
    print("Running with a small instance (N=10, T=24)...\n")
    
    try:
        # Test 1: Instance generation
        instance = test_instance_generation()
        
        # Test 2: Dispatch solver
        test_dispatch_solver(instance)
        
        # Test 3: MILP solver
        milp_solution = test_milp_solver(instance)
        
        # Test 4: Hybrid solver
        hybrid_solution = test_hybrid_solver(instance)
        
        # Test 5: Cost alignment
        if milp_solution.feasible and hybrid_solution.feasible:
            test_cost_alignment(instance, milp_solution, hybrid_solution)
        
        # Test 6: Comparison
        test_comparison(milp_solution, hybrid_solution)
        
        print("\n" + "="*80)
        print("ALL TESTS PASSED ✓")
        print("="*80)
        print("\nThe experimental framework is ready to use!")
        print("\nNext steps:")
        print("  1. Run full experiments: python run_experiments.py --scales small")
        print("  2. Review results in results/analysis/REPORT.md")
        print("  3. Scale up to medium or full when ready")
        
    except Exception as e:
        print("\n" + "="*80)
        print("TEST FAILED ✗")
        print("="*80)
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
