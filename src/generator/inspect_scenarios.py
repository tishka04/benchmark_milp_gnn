from __future__ import annotations

import argparse
import json
from pathlib import Path
from statistics import mean
from typing import Any, Dict, List


def load_scenarios(directory: Path) -> List[Dict[str, Any]]:
    scenarios = []
    for path in sorted(directory.glob('scenario_*.json')):
        stem = path.stem
        if not stem.startswith('scenario_'):
            continue
        suffix = stem[len('scenario_'):]
        if not (suffix.isdigit() and len(suffix) == 5):
            continue
        with path.open('r', encoding='utf-8') as f:
            scenarios.append(json.load(f))
    return scenarios


def summarize(scenarios: List[Dict[str, Any]]) -> Dict[str, Any]:
    if not scenarios:
        return {}
    regions = [s['graph']['regions'] for s in scenarios]
    zones = [sum(s['graph']['zones_per_region']) for s in scenarios]
    est_hours = [s['estimates']['est_cpu_hours'] for s in scenarios]
    demand_scale = [s['exogenous']['demand_scale_factor'] for s in scenarios]
    profiles = {}
    for s in scenarios:
        w = s['exogenous']['weather_profile']
        profiles[w] = profiles.get(w, 0) + 1
    return {
        'count': len(scenarios),
        'avg_regions': round(mean(regions), 2),
        'avg_zones': round(mean(zones), 2),
        'avg_est_cpu_hours': round(mean(est_hours), 2),
        'avg_demand_scale': round(mean(demand_scale), 3),
        'weather_profile_counts': profiles,
    }

def print_meta_table(scenarios: List[Dict[str, Any]]) -> None:
    print("\nScenario overview (id, regions, zones, demand_scale_factor, weather_profile):")
    for s in scenarios:
        sid = s['id'][:8]
        regions = s['graph']['regions']
        zones = sum(s['graph']['zones_per_region'])
        demand_scale = s['exogenous']['demand_scale_factor']
        weather = s['exogenous']['weather_profile']
        print(f"  {sid} | R={regions:>2} | Z={zones:>2} | demand_scale={demand_scale:>5.2f} | weather={weather}")


def main():
    parser = argparse.ArgumentParser(description='Inspect generated scenario metadata.')
    parser.add_argument('directory', nargs='?', type=Path, default=Path('outputs/scenarios_v1'), help='Scenario output directory (e.g., outputs/scenarios_v1)')
    args = parser.parse_args()
    directory = args.directory
    if not directory.exists():
        print(f"No scenarios found. Directory {directory} does not exist.")
        return

    scenarios = load_scenarios(directory)
    summary = summarize(scenarios)
    if not summary:
        print('No scenarios found.')
        return

    print(json.dumps(summary, indent=2))
    print_meta_table(scenarios)


if __name__ == '__main__':
    main()

