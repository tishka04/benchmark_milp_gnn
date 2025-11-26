"""
Run Scaling Benchmark: Test N=10, 20, 30, 40, 50

Shows how Hybrid vs MILP performance changes with problem size.
"""
import subprocess
import json
from pathlib import Path
import time

print("=" * 90)
print("SCALING BENCHMARK: Hybrid vs MILP")
print("=" * 90)

sizes = [10, 20, 30, 40, 50]
results = []

for n in sizes:
    print(f"\n{'=' * 90}")
    print(f"TESTING N={n}")
    print("=" * 90)
    
    scenario_file = Path(__file__).parent / f'synthetic_scenario_N{n}.json'
    
    if not scenario_file.exists():
        print(f"Scenario not found: {scenario_file}")
        continue
    
    # Run benchmark
    start = time.time()
    result = subprocess.run(
        ['python', 'benchmark_scenario.py', str(n)],
        cwd=Path(__file__).parent,
        capture_output=True,
        text=True
    )
    wall_time = time.time() - start
    
    print(result.stdout)
    if result.stderr:
        print("STDERR:", result.stderr)
    
    # Try to parse results from output
    lines = result.stdout.split('\n')
    hybrid_time = None
    milp_time = None
    hybrid_cost = None
    milp_cost = None
    
    for line in lines:
        if 'Hybrid' in line and 'SUCCESS' in line:
            parts = line.split()
            try:
                hybrid_time = float(parts[1].replace('s', ''))
                hybrid_cost = float(parts[3].replace('EUR', '').replace(',', ''))
            except:
                pass
        elif 'MILP' in line and 'SUCCESS' in line and 'Hybrid' not in line:
            parts = line.split()
            try:
                milp_time = float(parts[1].replace('s', ''))
                milp_cost = float(parts[3].replace('EUR', '').replace(',', ''))
            except:
                pass
    
    results.append({
        'n': n,
        'hybrid_time': hybrid_time,
        'milp_time': milp_time,
        'hybrid_cost': hybrid_cost,
        'milp_cost': milp_cost,
        'speedup': milp_time / hybrid_time if (hybrid_time and milp_time) else None
    })

# Summary
print("\n" + "=" * 90)
print("SCALING SUMMARY")
print("=" * 90)

print(f"\n{'N':<6} {'Hybrid Time':<15} {'MILP Time':<15} {'Speedup':<15} {'Winner':<15}")
print("-" * 90)

for r in results:
    if r['hybrid_time'] and r['milp_time']:
        speedup = r['speedup']
        winner = "Hybrid" if speedup > 1 else "MILP"
        print(f"{r['n']:<6} {r['hybrid_time']:>10.1f}s     {r['milp_time']:>10.1f}s     {speedup:>10.2f}x     {winner:<15}")
    else:
        print(f"{r['n']:<6} {'N/A':<15} {'N/A':<15} {'N/A':<15} {'N/A':<15}")

# Analysis
print(f"\n{'=' * 90}")
print("ANALYSIS")
print("=" * 90)

valid_results = [r for r in results if r['speedup'] is not None]

if len(valid_results) > 0:
    print("\nScaling Behavior:")
    for i, r in enumerate(valid_results):
        if r['speedup'] > 1:
            print(f"  N={r['n']}: Hybrid {r['speedup']:.1f}x faster")
        else:
            print(f"  N={r['n']}: MILP {1/r['speedup']:.1f}x faster")
    
    # Find crossover
    hybrid_wins = [r for r in valid_results if r['speedup'] > 1]
    milp_wins = [r for r in valid_results if r['speedup'] <= 1]
    
    if len(hybrid_wins) > 0 and len(milp_wins) > 0:
        print(f"\nCrossover Point:")
        print(f"  MILP wins: N <= {max(r['n'] for r in milp_wins)}")
        print(f"  Hybrid wins: N >= {min(r['n'] for r in hybrid_wins)}")
    elif len(hybrid_wins) > 0:
        print(f"\nHybrid wins across all tested sizes!")
        print(f"  Decomposition advantage evident even at N={min(r['n'] for r in hybrid_wins)}")
    elif len(milp_wins) > 0:
        print(f"\nMILP wins across all tested sizes!")
        print(f"  Problems small enough for monolithic optimization")

# Save results
output_file = Path(__file__).parent / 'scaling_benchmark_results.json'
with open(output_file, 'w') as f:
    json.dump(results, f, indent=2)

print(f"\n{'=' * 90}")
print(f"Results saved to: {output_file.name}")
print("=" * 90)
