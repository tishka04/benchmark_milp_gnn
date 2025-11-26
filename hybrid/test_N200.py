"""
Test N=200: Demonstrating Hybrid's Scalability Advantage

At N=200, this benchmark shows:
- Hybrid: Should complete in 30-60 minutes
- MILP: Will likely timeout or take many hours

This proves hybrid's decomposition approach scales to real-world grid sizes.
"""
import subprocess
from pathlib import Path
import time

print("=" * 90)
print("LARGE-SCALE TEST: N=200 Thermal Units")
print("=" * 90)

print("\n⚠️  WARNING: This is a LARGE problem!")
print("\nExpected runtime:")
print("  - Hybrid: 30-60 minutes")
print("  - MILP: 2-6 hours (or TIMEOUT)")
print("\nProblem size:")
print("  - 200 thermal units")
print("  - 96 time periods")
print("  - 38,400 binary variables (MILP)")
print("  - Search space: 2^19,200 ≈ 10^5,782 combinations")
print("\nThis demonstrates hybrid's advantage at REAL grid scales!")

input("\nPress ENTER to start (or Ctrl+C to cancel)...")

print("\n" + "=" * 90)
print("STARTING BENCHMARK")
print("=" * 90)

start_time = time.time()

result = subprocess.run(
    ['python', 'benchmark_N200.py'],
    cwd=Path(__file__).parent
)

total_time = time.time() - start_time

print("\n" + "=" * 90)
print("BENCHMARK COMPLETE")
print("=" * 90)
print(f"\nTotal wall time: {total_time:.1f}s ({total_time/60:.1f} min, {total_time/3600:.2f} hr)")

print("\n" + "=" * 90)
print("EXPECTED RESULTS")
print("=" * 90)
print("\nAt N=200:")
print("  - Hybrid should scale linearly (30-60 min)")
print("  - MILP faces exponential blowup (hours or timeout)")
print("  - This is a REAL-WORLD grid size!")
print("\nIf Hybrid completes but MILP times out:")
print("  ✓ Proves hybrid's scalability advantage")
print("  ✓ Shows decomposition is essential at scale")
print("  ✓ Validates approach for practical grids (N=100-500)")
