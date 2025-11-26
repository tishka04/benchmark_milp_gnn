"""
Find the most computationally expensive scenario in scenarios_v1
"""
import json
from pathlib import Path
import numpy as np

scenarios_dir = Path(__file__).parent.parent / "outputs" / "scenarios_v1"

print("Scanning all scenarios to find the most computationally expensive...")
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
            'regions': meta.get('regions', 0),
            'zones': meta.get('zones', 0),
            'thermal': meta.get('assets', {}).get('thermal', 0)
        }
        
        scenarios.append(scenario_info)
        
    except Exception as e:
        print(f"Error reading {scenario_file.name}: {e}")

# Sort by estimated CPU hours
scenarios.sort(key=lambda x: x['est_cpu_hours'], reverse=True)

print(f"\nTotal scenarios analyzed: {len(scenarios)}")
print("\n" + "=" * 80)
print("TOP 10 MOST COMPUTATIONALLY EXPENSIVE SCENARIOS")
print("=" * 80)
print(f"\n{'Rank':<6} {'Scenario':<18} {'CPU Hours':<12} {'Variables':<12} {'Constraints':<14} {'Assets':<8}")
print("-" * 80)

for i, s in enumerate(scenarios[:10], 1):
    print(f"{i:<6} {s['file']:<18} {s['est_cpu_hours']:>10.2f}h {s['vars_total']:>10} {s['cons_total']:>12} {s['total_assets']:>6}")

# Most expensive scenario
hardest = scenarios[0]

print("\n" + "=" * 80)
print("MOST EXPENSIVE SCENARIO DETAILS")
print("=" * 80)
print(f"\nFile: {hardest['file']}")
print(f"ID: {hardest['id']}")
print(f"Estimated CPU time: {hardest['est_cpu_hours']:.2f} hours")
print(f"Total variables: {hardest['vars_total']:,}")
print(f"Total constraints: {hardest['cons_total']:,}")
print(f"Total assets: {hardest['total_assets']}")
print(f"  - Thermal units: {hardest['thermal']}")
print(f"  - Regions: {hardest['regions']}")
print(f"  - Zones: {hardest['zones']}")

# Save result
result = {
    'hardest_scenario': hardest['file'],
    'top_10': [s['file'] for s in scenarios[:10]],
    'stats': hardest
}

output_file = Path(__file__).parent / 'hardest_scenario.json'
with open(output_file, 'w') as f:
    json.dump(result, f, indent=2)

print(f"\nResults saved to: {output_file}")
print("\n" + "=" * 80)
