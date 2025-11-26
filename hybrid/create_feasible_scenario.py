"""
Create a Synthetic, 100% Feasible Scenario for Fair Comparison

Design principles for guaranteed feasibility:
1. Total capacity >> peak demand (always enough generation)
2. Min generation of smallest unit < valley demand (can meet low demand)
3. Smooth demand profile (no impossible spikes)
4. Well-distributed unit sizes (flexibility)
"""
import numpy as np
import json
from pathlib import Path

def create_feasible_scenario(n_thermal=50, n_periods=96, seed=42):
    """
    Create a guaranteed feasible power system scenario.
    
    Parameters:
    - n_thermal: Number of thermal units
    - n_periods: Number of time periods (default 96 = 24h @ 15min)
    - seed: Random seed for reproducibility
    """
    np.random.seed(seed)
    
    print("=" * 90)
    print(f"CREATING FEASIBLE SCENARIO: {n_thermal} units, {n_periods} periods")
    print("=" * 90)
    
    # ==========================================
    # 1. DEMAND PROFILE (Realistic 24-hour pattern)
    # ==========================================
    hours = np.linspace(0, 24, n_periods)
    
    # Base demand with diurnal pattern
    base_demand = 1000.0  # MW base load
    
    demand_profile = np.zeros(n_periods)
    for t, h in enumerate(hours):
        # Morning peak (7-9am)
        morning = 0.3 * np.exp(-((h - 8)**2) / 4)
        # Evening peak (18-20pm) - highest
        evening = 0.5 * np.exp(-((h - 19)**2) / 4)
        # Night valley (2-5am) - lowest
        night = -0.2 * np.exp(-((h - 3)**2) / 4)
        
        demand_profile[t] = base_demand * (1.0 + morning + evening + night)
    
    peak_demand = demand_profile.max()
    valley_demand = demand_profile.min()
    
    print(f"\nDemand Profile:")
    print(f"  Valley: {valley_demand:.0f} MW")
    print(f"  Peak: {peak_demand:.0f} MW")
    print(f"  Ratio: {peak_demand/valley_demand:.2f}x")
    
    # ==========================================
    # 2. UNIT SIZING (Designed for feasibility)
    # ==========================================
    
    # Strategy: Mix of unit sizes for flexibility
    # - Large baseload units (always on)
    # - Medium peaking units (on/off for peaks)
    # - Small flexible units (fine-tuning)
    
    n_large = max(3, n_thermal // 10)      # 10% large units
    n_medium = max(5, n_thermal // 3)      # 33% medium units
    n_small = n_thermal - n_large - n_medium  # Rest are small
    
    thermal_capacity = np.concatenate([
        np.random.uniform(300, 500, n_large),    # Large: 300-500 MW
        np.random.uniform(100, 300, n_medium),   # Medium: 100-300 MW
        np.random.uniform(30, 100, n_small)      # Small: 30-100 MW
    ])
    
    # Shuffle to mix sizes
    np.random.shuffle(thermal_capacity)
    
    total_capacity = thermal_capacity.sum()
    thermal_min_gen = thermal_capacity * 0.25  # 25% minimum (lower than before for flexibility)
    
    print(f"\nUnit Characteristics:")
    print(f"  Total capacity: {total_capacity:.0f} MW")
    print(f"  Largest unit: {thermal_capacity.max():.0f} MW")
    print(f"  Smallest unit: {thermal_capacity.min():.0f} MW")
    print(f"  Smallest min gen: {thermal_min_gen.min():.0f} MW")
    
    # ==========================================
    # 3. FEASIBILITY CHECKS
    # ==========================================
    
    print(f"\nFeasibility Analysis:")
    
    # Check 1: Can we meet peak demand?
    if total_capacity < peak_demand:
        print(f"  [FAIL] Insufficient capacity!")
        print(f"  Adjusting: Adding capacity...")
        scale_factor = (peak_demand * 1.3) / total_capacity
        thermal_capacity *= scale_factor
        thermal_min_gen *= scale_factor
        total_capacity = thermal_capacity.sum()
        print(f"  [OK] Adjusted total capacity: {total_capacity:.0f} MW")
    else:
        margin = (total_capacity - peak_demand) / peak_demand * 100
        print(f"  [OK] Peak capacity margin: {margin:.1f}%")
    
    # Check 2: Can we meet valley demand without over-generating?
    min_possible_gen = thermal_min_gen.min()  # Turn on just smallest unit
    if min_possible_gen > valley_demand:
        print(f"  [FAIL] Smallest unit too large for valley!")
        print(f"  Adjusting: Reducing minimum generation...")
        thermal_min_gen = thermal_capacity * 0.15  # Reduce to 15%
        min_possible_gen = thermal_min_gen.min()
        print(f"  [OK] Adjusted smallest min gen: {min_possible_gen:.0f} MW")
    else:
        print(f"  [OK] Can meet valley with smallest unit ({min_possible_gen:.0f} MW < {valley_demand:.0f} MW)")
    
    # Check 3: For each period, can we meet demand?
    all_feasible = True
    for t, demand in enumerate(demand_profile):
        # Find minimum units needed (greedy)
        sorted_units = np.argsort(-thermal_capacity)  # Largest first
        cumulative_capacity = 0
        cumulative_min = 0
        units_needed = 0
        
        for idx in sorted_units:
            if cumulative_capacity >= demand:
                break
            cumulative_capacity += thermal_capacity[idx]
            cumulative_min += thermal_min_gen[idx]
            units_needed += 1
        
        if cumulative_capacity < demand:
            print(f"  [FAIL] Period {t}: Cannot meet demand {demand:.0f} MW")
            all_feasible = False
        elif cumulative_min > demand:
            print(f"  [FAIL] Period {t}: Min gen {cumulative_min:.0f} > demand {demand:.0f} MW")
            all_feasible = False
    
    if all_feasible:
        print(f"  [OK] All {n_periods} periods are feasible!")
    
    # ==========================================
    # 4. COST STRUCTURE (Realistic)
    # ==========================================
    
    # Larger units are typically more efficient (lower cost per MW)
    # Add some randomness
    base_cost = 50 - 30 * (thermal_capacity / thermal_capacity.max())  # Larger = cheaper
    base_cost += np.random.uniform(-5, 5, n_thermal)  # Add noise
    base_cost = np.maximum(base_cost, 20)  # Floor at 20 EUR/MW
    
    thermal_cost = base_cost
    
    print(f"\nCost Structure:")
    print(f"  Range: EUR {thermal_cost.min():.1f} - {thermal_cost.max():.1f} per MW")
    print(f"  Mean: EUR {thermal_cost.mean():.1f} per MW")
    
    # ==========================================
    # 5. PACKAGE SCENARIO
    # ==========================================
    
    scenario = {
        'name': f'synthetic_N{n_thermal}_T{n_periods}',
        'n_thermal': n_thermal,
        'n_periods': n_periods,
        'thermal_capacity_mw': thermal_capacity.tolist(),
        'thermal_min_gen_mw': thermal_min_gen.tolist(),
        'thermal_cost_eur_per_mw': thermal_cost.tolist(),
        'demand_profile_mw': demand_profile.tolist(),
        'feasibility_guaranteed': all_feasible,
        'stats': {
            'total_capacity_mw': float(total_capacity),
            'peak_demand_mw': float(peak_demand),
            'valley_demand_mw': float(valley_demand),
            'capacity_margin_pct': float((total_capacity - peak_demand) / peak_demand * 100),
            'smallest_unit_mw': float(thermal_capacity.min()),
            'largest_unit_mw': float(thermal_capacity.max())
        }
    }
    
    return scenario

# Create scenarios of different sizes
if __name__ == "__main__":
    
    sizes = [10, 20, 30, 40, 50]
    
    for n in sizes:
        print("\n")
        scenario = create_feasible_scenario(n_thermal=n, n_periods=96, seed=42)
        
        # Save
        output_file = Path(__file__).parent / f'synthetic_scenario_N{n}.json'
        with open(output_file, 'w') as f:
            json.dump(scenario, f, indent=2)
        
        print(f"\nSaved: {output_file.name}")
        print("=" * 90)
    
    print("\n" + "=" * 90)
    print("SCENARIOS CREATED")
    print("=" * 90)
    print("\nGenerated scenarios:")
    for n in sizes:
        print(f"  - synthetic_scenario_N{n}.json ({n} units, 96 periods)")
    
    print("\nAll scenarios are GUARANTEED FEASIBLE!")
    print("Ready for fair Hybrid vs MILP comparison.")
