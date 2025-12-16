#!/usr/bin/env python3
"""
Analyze scenario difficulty distribution and estimated compute time.

Scans all scenario JSON files in outputs/scenarios_v1 and provides:
- Distribution of difficulty levels (complexity_score)
- Total estimated MILP solve time
- Statistics on binary variables, zones, timesteps
"""

import json
from pathlib import Path
from collections import Counter
from typing import Dict, Any, List
import statistics


def load_scenario(path: Path) -> Dict[str, Any]:
    """Load a single scenario JSON file."""
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)


def analyze_scenarios(scenarios_dir: Path) -> None:
    """Analyze all scenarios in the given directory."""
    
    scenario_files = sorted(scenarios_dir.glob("scenario_*.json"))
    
    if not scenario_files:
        print(f"No scenario files found in {scenarios_dir}")
        return
    
    print(f"Found {len(scenario_files)} scenario files")
    print("=" * 70)
    
    # Collect metrics
    complexity_scores = []
    solve_times_sec = []
    solve_times_hours = []
    n_binary_vars = []
    n_zones = []
    n_timesteps = []
    vre_penetrations = []
    volatilities = []
    
    for scenario_file in scenario_files:
        try:
            scenario = load_scenario(scenario_file)
            
            # Difficulty indicators
            diff = scenario.get("difficulty_indicators", {})
            if diff:
                complexity_scores.append(diff.get("complexity_score", "unknown"))
                solve_times_sec.append(diff.get("estimated_milp_solve_time_seconds", 0))
                n_binary_vars.append(diff.get("n_binary_variables", 0))
                n_zones.append(diff.get("n_zones", 0))
                n_timesteps.append(diff.get("n_timesteps", 0))
                vre_penetrations.append(diff.get("vre_penetration_pct", 0))
                volatilities.append(diff.get("net_demand_volatility", 0))
            
            # Also check estimates section for est_cpu_hours
            est = scenario.get("estimates", {})
            if est and "est_cpu_hours" in est:
                solve_times_hours.append(est["est_cpu_hours"])
            elif solve_times_sec:
                solve_times_hours.append(solve_times_sec[-1] / 3600.0)
                
        except Exception as e:
            print(f"Error loading {scenario_file.name}: {e}")
            continue
    
    # Calculate statistics
    print("\n### DIFFICULTY DISTRIBUTION ###")
    print("-" * 70)
    complexity_counts = Counter(complexity_scores)
    for complexity, count in sorted(complexity_counts.items()):
        pct = 100 * count / len(complexity_scores)
        print(f"  {complexity:15s}: {count:4d} scenarios ({pct:5.1f}%)")
    
    print("\n### ESTIMATED COMPUTE TIME ###")
    print("-" * 70)
    total_seconds = sum(solve_times_sec)
    total_hours = total_seconds / 3600.0
    total_days = total_hours / 24.0
    
    print(f"  Total estimated time (sequential):")
    print(f"    {total_seconds:,.0f} seconds")
    print(f"    {total_hours:,.1f} hours")
    print(f"    {total_days:,.1f} days")
    
    if solve_times_hours:
        avg_hours = statistics.mean(solve_times_hours)
        median_hours = statistics.median(solve_times_hours)
        print(f"\n  Per-scenario statistics:")
        print(f"    Mean:   {avg_hours:.2f} hours")
        print(f"    Median: {median_hours:.2f} hours")
        print(f"    Min:    {min(solve_times_hours):.2f} hours")
        print(f"    Max:    {max(solve_times_hours):.2f} hours")
    
    # Parallelization estimates
    print(f"\n  Parallel execution estimates:")
    for n_cores in [1, 4, 8, 12, 16, 24, 32, 64]:
        parallel_hours = total_hours / n_cores
        parallel_days = parallel_hours / 24.0
        print(f"    {n_cores:2d} cores: {parallel_hours:7.1f} hours ({parallel_days:5.1f} days)")
    
    print("\n### PROBLEM SIZE STATISTICS ###")
    print("-" * 70)
    if n_binary_vars:
        print(f"  Binary variables:")
        print(f"    Mean:   {statistics.mean(n_binary_vars):,.0f}")
        print(f"    Median: {statistics.median(n_binary_vars):,.0f}")
        print(f"    Min:    {min(n_binary_vars):,}")
        print(f"    Max:    {max(n_binary_vars):,}")
    
    if n_zones:
        print(f"\n  Zones per scenario:")
        print(f"    Mean:   {statistics.mean(n_zones):.1f}")
        print(f"    Median: {statistics.median(n_zones):.1f}")
        print(f"    Min:    {min(n_zones)}")
        print(f"    Max:    {max(n_zones)}")
    
    if n_timesteps:
        print(f"\n  Timesteps (T):")
        print(f"    Mean:   {statistics.mean(n_timesteps):.1f}")
        print(f"    Median: {statistics.median(n_timesteps):.1f}")
        print(f"    Min:    {min(n_timesteps)}")
        print(f"    Max:    {max(n_timesteps)}")
    
    print("\n### OPERATIONAL CHARACTERISTICS ###")
    print("-" * 70)
    if vre_penetrations:
        print(f"  VRE penetration (%):")
        print(f"    Mean:   {statistics.mean(vre_penetrations):.1f}%")
        print(f"    Median: {statistics.median(vre_penetrations):.1f}%")
        print(f"    Min:    {min(vre_penetrations):.1f}%")
        print(f"    Max:    {max(vre_penetrations):.1f}%")
    
    if volatilities:
        print(f"\n  Net demand volatility:")
        print(f"    Mean:   {statistics.mean(volatilities):.3f}")
        print(f"    Median: {statistics.median(volatilities):.3f}")
        print(f"    Min:    {min(volatilities):.3f}")
        print(f"    Max:    {max(volatilities):.3f}")
    
    print("\n" + "=" * 70)


def main():
    # Default path relative to script location
    script_dir = Path(__file__).parent
    default_scenarios_dir = script_dir.parent / "outputs" / "scenarios_v1"
    
    import sys
    if len(sys.argv) > 1:
        scenarios_dir = Path(sys.argv[1])
    else:
        scenarios_dir = default_scenarios_dir
    
    if not scenarios_dir.exists():
        print(f"Error: Directory not found: {scenarios_dir}")
        print(f"\nUsage: {sys.argv[0]} [scenarios_directory]")
        sys.exit(1)
    
    analyze_scenarios(scenarios_dir)


if __name__ == "__main__":
    main()
