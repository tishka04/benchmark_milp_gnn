"""
Diagnose why MILP is infeasible - Check capacity vs demand
"""
import json
import numpy as np
from pathlib import Path

print("=" * 90)
print("INFEASIBILITY DIAGNOSIS")
print("=" * 90)

# Load scenario
scenario_path = Path(__file__).parent.parent / "outputs" / "scenarios_v1" / "scenario_00286.json"
with open(scenario_path, 'r') as f:
    scenario = json.load(f)

n_periods = 96
n_thermal = 40

# Generate same parameters
np.random.seed(42)
thermal_capacity = np.random.uniform(100, 800, n_thermal)
thermal_min_gen = thermal_capacity * 0.3

# Generate demand
base_demand = 5000.0
demand_scale = scenario['exogenous']['demand_scale_factor']

hours = np.linspace(0, 24, n_periods)
demand_profile = np.zeros(n_periods)

for t, h in enumerate(hours):
    morning_peak = 1.2 * np.exp(-((h - 8)**2) / 2)
    evening_peak = 1.4 * np.exp(-((h - 19)**2) / 2)
    night_valley = -0.3 * np.exp(-((h - 3)**2) / 2)
    demand_profile[t] = base_demand * demand_scale * (1.0 + morning_peak + evening_peak + night_valley)

renewable_profile = np.zeros(n_periods)
assets = scenario['meta']['assets']
for t, h in enumerate(hours):
    solar = assets['solar'] * 50 * max(0, np.sin(np.pi * (h - 6) / 12))
    wind = assets['wind'] * 80 * (0.3 + 0.2 * np.sin(2 * np.pi * h / 24 + 1.5))
    renewable_profile[t] = solar + wind

net_demand = demand_profile - renewable_profile

print(f"\nUnit Characteristics:")
print(f"  Total capacity (all units): {thermal_capacity.sum():.1f} MW")
print(f"  Total min generation (all units): {thermal_min_gen.sum():.1f} MW")
print(f"  Largest unit: {thermal_capacity.max():.1f} MW")
print(f"  Smallest unit: {thermal_capacity.min():.1f} MW")

print(f"\nDemand Profile:")
print(f"  Min demand: {net_demand.min():.1f} MW (period {np.argmin(net_demand)})")
print(f"  Max demand: {net_demand.max():.1f} MW (period {np.argmax(net_demand)})")
print(f"  Mean demand: {net_demand.mean():.1f} MW")

# Check feasibility for each period
print(f"\n" + "=" * 90)
print("PERIOD-BY-PERIOD FEASIBILITY CHECK")
print("=" * 90)

infeasible_periods = []

for t in range(n_periods):
    demand = net_demand[t]
    
    # Can we meet this demand?
    max_possible = thermal_capacity.sum()
    
    if demand > max_possible:
        print(f"Period {t}: INFEASIBLE - Demand {demand:.1f} > Max capacity {max_possible:.1f}")
        infeasible_periods.append(t)
        continue
    
    # Can we meet it without violating minimum generation?
    # Strategy: Turn on units until we have enough capacity
    sorted_by_capacity = np.argsort(-thermal_capacity)  # Largest first
    
    # Greedy: turn on units until capacity >= demand
    units_on = []
    total_capacity = 0
    total_min_gen = 0
    
    for idx in sorted_by_capacity:
        if total_capacity >= demand:
            break
        units_on.append(idx)
        total_capacity += thermal_capacity[idx]
        total_min_gen += thermal_min_gen[idx]
    
    # Check if minimum generation of selected units <= demand <= their total capacity
    if total_min_gen > demand:
        print(f"Period {t}: INFEASIBLE - Min gen {total_min_gen:.1f} > Demand {demand:.1f}")
        print(f"          (Need {len(units_on)} units, but their min gen exceeds demand)")
        infeasible_periods.append(t)
    elif total_capacity < demand:
        print(f"Period {t}: INFEASIBLE - Max capacity {total_capacity:.1f} < Demand {demand:.1f}")
        infeasible_periods.append(t)

if len(infeasible_periods) == 0:
    print("\n[OK] All periods are individually feasible!")
    print("    Problem must be in MILP formulation or multi-period coupling")
else:
    print(f"\n[PROBLEM] {len(infeasible_periods)} periods are infeasible:")
    for t in infeasible_periods:
        print(f"  Period {t}: Demand = {net_demand[t]:.1f} MW")

# Detailed analysis of most problematic periods
print(f"\n" + "=" * 90)
print("DETAILED ANALYSIS: LOW DEMAND PERIODS")
print("=" * 90)

# Check the 5 lowest demand periods
low_demand_periods = np.argsort(net_demand)[:5]

for t in low_demand_periods:
    demand = net_demand[t]
    print(f"\nPeriod {t} (hour {t*0.25:.1f}): Demand = {demand:.1f} MW")
    
    # What's the minimum we can generate with smallest unit?
    min_with_one_unit = thermal_min_gen.min()
    print(f"  Smallest unit min gen: {min_with_one_unit:.1f} MW")
    
    if demand < min_with_one_unit:
        print(f"  [INFEASIBLE] Demand < min gen of smallest unit!")
    else:
        print(f"  [OK] Can use smallest unit")

print(f"\n" + "=" * 90)
print("DETAILED ANALYSIS: HIGH DEMAND PERIODS")
print("=" * 90)

high_demand_periods = np.argsort(net_demand)[-5:]

for t in high_demand_periods:
    demand = net_demand[t]
    print(f"\nPeriod {t} (hour {t*0.25:.1f}): Demand = {demand:.1f} MW")
    print(f"  Total capacity (all units): {thermal_capacity.sum():.1f} MW")
    
    if demand > thermal_capacity.sum():
        print(f"  [INFEASIBLE] Demand exceeds total capacity!")
    else:
        print(f"  [OK] Sufficient capacity")

print(f"\n" + "=" * 90)
print("RECOMMENDATION")
print("=" * 90)

if len(infeasible_periods) > 0:
    print("\nThe problem has periods that are individually infeasible.")
    print("MILP correctly identifies this as infeasible.")
    print("\nHybrid solver succeeded because:")
    print("  1. It may have implicit flexibility in constraints")
    print("  2. It uses approximate/relaxed demand satisfaction")
    print("  3. Implementation differences in constraint handling")
else:
    print("\nAll periods are individually feasible.")
    print("MILP infeasibility must be due to:")
    print("  1. Bug in MILP constraint formulation")
    print("  2. Numerical issues with solver")
    print("  3. Unexpected interaction between constraints")
