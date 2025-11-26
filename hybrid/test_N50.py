"""
Quick test: N=50 to demonstrate hybrid advantage
"""
import subprocess
from pathlib import Path

print("=" * 90)
print("QUICK TEST: N=50 (Demonstrating Hybrid Advantage)")
print("=" * 90)

print("\nThis will take approximately:")
print("  - Hybrid: 10-15 minutes")
print("  - MILP: 30-60 minutes (or timeout)")
print("\nStarting...\n")

result = subprocess.run(
    ['python', 'benchmark_scenario.py', '50'],
    cwd=Path(__file__).parent
)

print("\n" + "=" * 90)
print("TEST COMPLETE")
print("=" * 90)
print("\nExpected result: Hybrid significantly faster at N=50")
