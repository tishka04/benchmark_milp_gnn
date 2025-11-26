"""
Find the simplest (least computationally expensive) scenario in scenarios_v1
"""
import json
from pathlib import Path

scenarios_dir = Path(__file__).parent.parent / "outputs" / "scenarios_v1"

print("=" * 80)
print("FINDING SIMPLEST SCENARIO")
print("=" * 80)

scenarios = []

for scenario_file in sorted(scenarios_dir.glob("scenario_*.json")):
    try:
        with open(scenario_file, 'r') as f:
            data = json.load(f)
        
        estimates = data.get('estimates', {})
        meta = data.get('meta', {})
        
        scenario_info = {
            'file': scenario_file.name,
            'id': data.get('id', 'unknown'),
            'vars_total': estimates.get('vars_total', 0),
            'cons_total': estimates.get('cons_total', 0),
            'est_cpu_hours': estimates.get('est_cpu_hours', 0),
            'total_assets': sum(meta.get('assets', {}).values()),
            'thermal': meta.get('assets', {}).get('thermal', 0),
            'regions': meta.get('regions', 0),
            'zones': meta.get('zones', 0)
        }
        
        scenarios.append(scenario_info)
        
    except Exception as e:
        print(f"Error reading {scenario_file.name}: {e}")

# Sort by estimated CPU hours (ascending - simplest first)
scenarios.sort(key=lambda x: x['est_cpu_hours'])

print(f"\nTotal scenarios analyzed: {len(scenarios)}")
print("\n" + "=" * 80)
print("TOP 10 SIMPLEST SCENARIOS")
print("=" * 80)
print(f"\n{'Rank':<6} {'Scenario':<18} {'CPU Hours':<12} {'Variables':<12} {'Thermal':<10} {'Assets':<8}")
print("-" * 80)

for i, s in enumerate(scenarios[:10], 1):
    print(f"{i:<6} {s['file']:<18} {s['est_cpu_hours']:>10.2f}h {s['vars_total']:>10} {s['thermal']:>8} {s['total_assets']:>6}")

# Simplest scenario
simplest = scenarios[0]

print("\n" + "=" * 80)
print("SIMPLEST SCENARIO DETAILS")
print("=" * 80)
print(f"\nFile: {simplest['file']}")
print(f"ID: {simplest['id']}")
print(f"Estimated CPU time: {simplest['est_cpu_hours']:.2f} hours")
print(f"Total variables: {simplest['vars_total']:,}")
print(f"Total constraints: {simplest['cons_total']:,}")
print(f"Total assets: {simplest['total_assets']}")
print(f"  - Thermal units: {simplest['thermal']}")
print(f"  - Regions: {simplest['regions']}")
print(f"  - Zones: {simplest['zones']}")

# Compare with hardest
hardest = scenarios[-1]
print(f"\n" + "=" * 80)
print("COMPARISON: SIMPLEST vs HARDEST")
print("=" * 80)
print(f"\nSimplest: {simplest['file']}")
print(f"  CPU: {simplest['est_cpu_hours']:.2f}h, Thermal: {simplest['thermal']}, Vars: {simplest['vars_total']:,}")
print(f"\nHardest: {hardest['file']}")
print(f"  CPU: {hardest['est_cpu_hours']:.2f}h, Thermal: {hardest['thermal']}, Vars: {hardest['vars_total']:,}")
print(f"\nRatio: {hardest['est_cpu_hours'] / simplest['est_cpu_hours']:.1f}x harder")

# Save result
result = {
    'simplest_scenario': simplest['file'],
    'top_10_simplest': [s['file'] for s in scenarios[:10]],
    'stats': simplest
}

output_file = Path(__file__).parent / 'simplest_scenario.json'
with open(output_file, 'w') as f:
    json.dump(result, f, indent=2)

print(f"\n" + "=" * 80)
print(f"Results saved to: {output_file}")
print("=" * 80)
