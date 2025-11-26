"""
Run both Hybrid and MILP on simplest scenario and compare
"""
import subprocess
import json
from pathlib import Path
import time

print("=" * 90)
print("BENCHMARK: SIMPLEST SCENARIO (10 thermal units)")
print("=" * 90)

hybrid_file = Path(__file__).parent / 'hybrid_result_simple.json'
milp_file = Path(__file__).parent / 'milp_result_simple.json'

# Clean old results
for f in [hybrid_file, milp_file]:
    if f.exists():
        f.unlink()

print("\n1. Running HYBRID solver...")
print("-" * 90)
hybrid_start = time.time()
subprocess.run(['python', 'hybrid_solver_simple.py'], cwd=Path(__file__).parent)
hybrid_time_wall = time.time() - hybrid_start

print("\n" + "=" * 90)
print("2. Running MILP solver (NO SLACK - Fair Comparison)...")
print("-" * 90)
milp_start = time.time()
subprocess.run(['python', 'milp_solver_simple_clean.py'], cwd=Path(__file__).parent)
milp_time_wall = time.time() - milp_start

# Load results
print("\n" + "=" * 90)
print("FINAL COMPARISON")
print("=" * 90)

try:
    with open(hybrid_file, 'r') as f:
        hybrid_data = json.load(f)
    with open(milp_file, 'r') as f:
        milp_data = json.load(f)
    
    print(f"\nScenario: scenario_00285.json (SIMPLEST)")
    print(f"Problem: 10 thermal units, 96 periods")
    
    print(f"\n{'Method':<15} {'Time':<15} {'Cost':<20} {'Status':<15}")
    print("-" * 90)
    
    if hybrid_data.get('success'):
        h_time = hybrid_data['solve_time_seconds']
        h_cost = hybrid_data['total_cost_eur']
        print(f"{'Hybrid':<15} {h_time:>10.1f}s     EUR {h_cost:>15,.2f}  {'SUCCESS':<15}")
    else:
        print(f"{'Hybrid':<15} {'N/A':<15} {'N/A':<20} {'FAILED':<15}")
    
    if milp_data.get('success'):
        m_time = milp_data['solve_time_seconds']
        m_cost = milp_data.get('total_cost_eur', milp_data.get('generation_cost_eur', 0))
        print(f"{'MILP':<15} {m_time:>10.1f}s     EUR {m_cost:>15,.2f}  {'SUCCESS':<15}")
    else:
        print(f"{'MILP':<15} {'N/A':<15} {'N/A':<20} {'FAILED':<15}")
    
    if hybrid_data.get('success') and milp_data.get('success'):
        print(f"\n{'=' * 90}")
        print("VERDICT")
        print("=" * 90)
        
        time_ratio = m_time / h_time
        cost_diff = abs(m_cost - h_cost) / h_cost * 100
        
        print(f"\nSpeed:")
        if time_ratio > 1:
            print(f"  Hybrid was {time_ratio:.1f}x FASTER ({m_time:.1f}s vs {h_time:.1f}s)")
        else:
            print(f"  MILP was {1/time_ratio:.1f}x FASTER ({h_time:.1f}s vs {m_time:.1f}s)")
        
        print(f"\nCost:")
        print(f"  Difference: {cost_diff:.2f}%")
        
        print(f"\nConclusion:")
        if time_ratio > 2:
            print(f"  Hybrid wins on this simple problem")
            print(f"  Shows decomposition helps even for small problems")
        elif time_ratio < 0.5:
            print(f"  MILP wins on this simple problem")
            print(f"  Expected: 10 units is small enough for monolithic MILP")
        else:
            print(f"  Comparable performance")
            print(f"  At N=10, both methods work well")

except Exception as e:
    print(f"\nError loading results: {e}")

print(f"\n{'=' * 90}")
print("Comparison complete!")
print("=" * 90)
