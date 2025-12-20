#!/usr/bin/env python3
"""
Analyze MILP scenario reports and generate comprehensive statistics.

Usage:
    python -m src.milp.analyze_reports outputs/scenarios_v1/reports
    python -m src.milp.analyze_reports outputs/scenarios_v1/reports --output stats.json --verbose
"""

import argparse
import json
import sys
from pathlib import Path
from dataclasses import dataclass, field, asdict
from typing import List, Dict, Any, Optional
from collections import defaultdict
import statistics


@dataclass
class SolveStats:
    """Statistics for solve times."""
    count: int = 0
    min_seconds: float = float('inf')
    max_seconds: float = 0.0
    mean_seconds: float = 0.0
    median_seconds: float = 0.0
    std_seconds: float = 0.0
    p90_seconds: float = 0.0
    p95_seconds: float = 0.0
    p99_seconds: float = 0.0
    total_seconds: float = 0.0


@dataclass 
class CostStats:
    """Statistics for cost components."""
    min_eur: float = float('inf')
    max_eur: float = 0.0
    mean_eur: float = 0.0
    median_eur: float = 0.0
    total_eur: float = 0.0
    scenarios_with_nonzero: int = 0


@dataclass
class AnalysisReport:
    """Complete analysis report."""
    # General stats
    total_reports: int = 0
    reports_analyzed: int = 0
    parse_errors: int = 0
    
    # Termination status
    optimal_count: int = 0
    infeasible_count: int = 0
    timeout_count: int = 0
    other_termination_count: int = 0
    termination_breakdown: Dict[str, int] = field(default_factory=dict)
    
    # Solve time stats
    mip_solve_stats: SolveStats = field(default_factory=SolveStats)
    lp_solve_stats: SolveStats = field(default_factory=SolveStats)
    
    # MIP-LP gap
    mip_lp_gap_percent: Dict[str, float] = field(default_factory=dict)
    
    # Generation mix analysis
    scenarios_with_nuclear: int = 0
    scenarios_without_nuclear: int = 0
    scenarios_with_thermal: int = 0
    scenarios_with_unserved: int = 0
    scenarios_with_dr: int = 0
    scenarios_with_solar_spill: int = 0
    scenarios_with_wind_spill: int = 0
    scenarios_with_imports: int = 0
    scenarios_with_exports: int = 0
    
    # Cost breakdown stats
    cost_stats: Dict[str, CostStats] = field(default_factory=dict)
    
    # Objective stats
    objective_min: float = float('inf')
    objective_max: float = 0.0
    objective_mean: float = 0.0
    objective_median: float = 0.0
    
    # Lists for detailed analysis
    infeasible_scenarios: List[str] = field(default_factory=list)
    timeout_scenarios: List[str] = field(default_factory=list)
    high_unserved_scenarios: List[str] = field(default_factory=list)
    slow_scenarios: List[str] = field(default_factory=list)


def compute_percentile(data: List[float], percentile: float) -> float:
    """Compute percentile of a list of values."""
    if not data:
        return 0.0
    sorted_data = sorted(data)
    idx = int(len(sorted_data) * percentile / 100)
    idx = min(idx, len(sorted_data) - 1)
    return sorted_data[idx]


def compute_solve_stats(times: List[float]) -> SolveStats:
    """Compute statistics for solve times."""
    if not times:
        return SolveStats()
    
    return SolveStats(
        count=len(times),
        min_seconds=min(times),
        max_seconds=max(times),
        mean_seconds=statistics.mean(times),
        median_seconds=statistics.median(times),
        std_seconds=statistics.stdev(times) if len(times) > 1 else 0.0,
        p90_seconds=compute_percentile(times, 90),
        p95_seconds=compute_percentile(times, 95),
        p99_seconds=compute_percentile(times, 99),
        total_seconds=sum(times),
    )


def compute_cost_stats(values: List[float]) -> CostStats:
    """Compute statistics for cost values."""
    nonzero = [v for v in values if v > 0.01]
    if not values:
        return CostStats()
    
    return CostStats(
        min_eur=min(values) if values else 0.0,
        max_eur=max(values) if values else 0.0,
        mean_eur=statistics.mean(values) if values else 0.0,
        median_eur=statistics.median(values) if values else 0.0,
        total_eur=sum(values),
        scenarios_with_nonzero=len(nonzero),
    )


def analyze_reports(reports_dir: Path, verbose: bool = False) -> AnalysisReport:
    """Analyze all MILP reports in a directory."""
    report = AnalysisReport()
    
    # Collect all report files
    report_files = sorted(reports_dir.glob("scenario_*.json"))
    report.total_reports = len(report_files)
    
    if verbose:
        print(f"Found {report.total_reports} report files")
    
    # Data collectors
    mip_times: List[float] = []
    lp_times: List[float] = []
    objectives: List[float] = []
    mip_lp_gaps: List[float] = []
    cost_data: Dict[str, List[float]] = defaultdict(list)
    
    for report_file in report_files:
        try:
            with open(report_file, 'r') as f:
                data = json.load(f)
            
            scenario_name = report_file.stem
            report.reports_analyzed += 1
            
            # === MIP results ===
            mip = data.get("mip", {})
            mip_termination = str(mip.get("termination", "unknown")).lower()
            mip_time = mip.get("solve_seconds", 0.0)
            mip_objective = mip.get("objective", 0.0)
            
            # Track termination status
            report.termination_breakdown[mip_termination] = \
                report.termination_breakdown.get(mip_termination, 0) + 1
            
            if "optimal" in mip_termination:
                report.optimal_count += 1
                mip_times.append(mip_time)
                objectives.append(mip_objective)
            elif "infeasible" in mip_termination:
                report.infeasible_count += 1
                report.infeasible_scenarios.append(scenario_name)
            elif "time" in mip_termination or "limit" in mip_termination:
                report.timeout_count += 1
                report.timeout_scenarios.append(scenario_name)
            else:
                report.other_termination_count += 1
            
            # Track slow scenarios (>60s)
            if mip_time > 60:
                report.slow_scenarios.append(f"{scenario_name} ({mip_time:.1f}s)")
            
            # === LP results ===
            lp = data.get("lp", {})
            lp_time = lp.get("solve_seconds", 0.0)
            lp_objective = lp.get("objective", 0.0)
            
            if lp_time > 0:
                lp_times.append(lp_time)
            
            # MIP-LP gap
            if mip_objective > 0 and lp_objective > 0:
                gap = (mip_objective - lp_objective) / mip_objective * 100
                mip_lp_gaps.append(gap)
            
            # === Cost components ===
            costs = data.get("cost_components", {})
            
            for cost_name, cost_value in costs.items():
                cost_data[cost_name].append(cost_value if cost_value else 0.0)
            
            # Track generation mix
            nuclear_cost = costs.get("nuclear_fuel", 0.0)
            thermal_cost = costs.get("thermal_fuel", 0.0)
            unserved_cost = costs.get("unserved_energy", 0.0)
            dr_cost = costs.get("demand_response", 0.0)
            solar_spill = costs.get("solar_spill", 0.0)
            wind_spill = costs.get("wind_spill", 0.0)
            imports_cost = costs.get("imports", 0.0)
            exports_cost = costs.get("exports", 0.0)
            
            if nuclear_cost > 0.01:
                report.scenarios_with_nuclear += 1
            else:
                report.scenarios_without_nuclear += 1
            
            if thermal_cost > 0.01:
                report.scenarios_with_thermal += 1
            
            if unserved_cost > 0.01:
                report.scenarios_with_unserved += 1
                # High unserved: > 10% of objective
                if mip_objective > 0 and unserved_cost / mip_objective > 0.1:
                    report.high_unserved_scenarios.append(
                        f"{scenario_name} ({unserved_cost/mip_objective*100:.1f}%)"
                    )
            
            if dr_cost > 0.01:
                report.scenarios_with_dr += 1
            
            if solar_spill > 0.01:
                report.scenarios_with_solar_spill += 1
            
            if wind_spill > 0.01:
                report.scenarios_with_wind_spill += 1
            
            if imports_cost > 0.01:
                report.scenarios_with_imports += 1
            
            if exports_cost > 0.01:
                report.scenarios_with_exports += 1
                
        except Exception as e:
            report.parse_errors += 1
            if verbose:
                print(f"  Error parsing {report_file.name}: {e}")
    
    # === Compute aggregate statistics ===
    report.mip_solve_stats = compute_solve_stats(mip_times)
    report.lp_solve_stats = compute_solve_stats(lp_times)
    
    if objectives:
        report.objective_min = min(objectives)
        report.objective_max = max(objectives)
        report.objective_mean = statistics.mean(objectives)
        report.objective_median = statistics.median(objectives)
    
    if mip_lp_gaps:
        report.mip_lp_gap_percent = {
            "min": min(mip_lp_gaps),
            "max": max(mip_lp_gaps),
            "mean": statistics.mean(mip_lp_gaps),
            "median": statistics.median(mip_lp_gaps),
        }
    
    for cost_name, values in cost_data.items():
        report.cost_stats[cost_name] = compute_cost_stats(values)
    
    return report


def print_report(report: AnalysisReport) -> None:
    """Print analysis report to console."""
    print("\n" + "="*70)
    print("                    MILP SCENARIO REPORTS ANALYSIS")
    print("="*70)
    
    # General stats
    print(f"\n📊 GENERAL STATISTICS")
    print(f"   Total reports found:    {report.total_reports}")
    print(f"   Successfully analyzed:  {report.reports_analyzed}")
    print(f"   Parse errors:           {report.parse_errors}")
    
    # Termination status
    print(f"\n🎯 TERMINATION STATUS")
    print(f"   ✅ Optimal:     {report.optimal_count:>6} ({report.optimal_count/max(report.reports_analyzed,1)*100:.1f}%)")
    print(f"   ❌ Infeasible:  {report.infeasible_count:>6} ({report.infeasible_count/max(report.reports_analyzed,1)*100:.1f}%)")
    print(f"   ⏱️  Timeout:     {report.timeout_count:>6} ({report.timeout_count/max(report.reports_analyzed,1)*100:.1f}%)")
    print(f"   ❓ Other:       {report.other_termination_count:>6}")
    
    if report.termination_breakdown:
        print(f"\n   Breakdown by termination condition:")
        for term, count in sorted(report.termination_breakdown.items(), key=lambda x: -x[1]):
            print(f"      {term}: {count}")
    
    # Solve times
    mip = report.mip_solve_stats
    lp = report.lp_solve_stats
    print(f"\n⏱️  SOLVE TIMES (seconds)")
    print(f"                    MIP            LP")
    print(f"   Count:       {mip.count:>8}     {lp.count:>8}")
    print(f"   Min:         {mip.min_seconds:>8.2f}     {lp.min_seconds:>8.2f}")
    print(f"   Max:         {mip.max_seconds:>8.2f}     {lp.max_seconds:>8.2f}")
    print(f"   Mean:        {mip.mean_seconds:>8.2f}     {lp.mean_seconds:>8.2f}")
    print(f"   Median:      {mip.median_seconds:>8.2f}     {lp.median_seconds:>8.2f}")
    print(f"   Std Dev:     {mip.std_seconds:>8.2f}     {lp.std_seconds:>8.2f}")
    print(f"   P90:         {mip.p90_seconds:>8.2f}     {lp.p90_seconds:>8.2f}")
    print(f"   P95:         {mip.p95_seconds:>8.2f}     {lp.p95_seconds:>8.2f}")
    print(f"   P99:         {mip.p99_seconds:>8.2f}     {lp.p99_seconds:>8.2f}")
    print(f"   Total:       {mip.total_seconds/3600:>8.2f}h    {lp.total_seconds/3600:>8.2f}h")
    
    # MIP-LP gap
    if report.mip_lp_gap_percent:
        gap = report.mip_lp_gap_percent
        print(f"\n📈 MIP-LP INTEGRALITY GAP (%)")
        print(f"   Min:    {gap.get('min', 0):>8.4f}%")
        print(f"   Max:    {gap.get('max', 0):>8.4f}%")
        print(f"   Mean:   {gap.get('mean', 0):>8.4f}%")
        print(f"   Median: {gap.get('median', 0):>8.4f}%")
    
    # Objective stats
    print(f"\n💰 OBJECTIVE VALUES (EUR)")
    print(f"   Min:    {report.objective_min:>15,.2f}")
    print(f"   Max:    {report.objective_max:>15,.2f}")
    print(f"   Mean:   {report.objective_mean:>15,.2f}")
    print(f"   Median: {report.objective_median:>15,.2f}")
    
    # Generation mix
    n = max(report.reports_analyzed, 1)
    print(f"\n⚡ GENERATION MIX (scenario counts)")
    print(f"   With nuclear:      {report.scenarios_with_nuclear:>6} ({report.scenarios_with_nuclear/n*100:.1f}%)")
    print(f"   Without nuclear:   {report.scenarios_without_nuclear:>6} ({report.scenarios_without_nuclear/n*100:.1f}%)")
    print(f"   With thermal:      {report.scenarios_with_thermal:>6} ({report.scenarios_with_thermal/n*100:.1f}%)")
    print(f"   With DR usage:     {report.scenarios_with_dr:>6} ({report.scenarios_with_dr/n*100:.1f}%)")
    print(f"   With unserved:     {report.scenarios_with_unserved:>6} ({report.scenarios_with_unserved/n*100:.1f}%)")
    print(f"   With solar spill:  {report.scenarios_with_solar_spill:>6} ({report.scenarios_with_solar_spill/n*100:.1f}%)")
    print(f"   With wind spill:   {report.scenarios_with_wind_spill:>6} ({report.scenarios_with_wind_spill/n*100:.1f}%)")
    print(f"   With imports:      {report.scenarios_with_imports:>6} ({report.scenarios_with_imports/n*100:.1f}%)")
    print(f"   With exports:      {report.scenarios_with_exports:>6} ({report.scenarios_with_exports/n*100:.1f}%)")
    
    # Cost breakdown
    print(f"\n💵 COST COMPONENTS (EUR, mean values)")
    for cost_name, stats in sorted(report.cost_stats.items(), key=lambda x: -x[1].mean_eur):
        if stats.mean_eur > 0.01:
            print(f"   {cost_name:25s}: {stats.mean_eur:>15,.2f} (nonzero in {stats.scenarios_with_nonzero} scenarios)")
    
    # Problem scenarios
    if report.infeasible_scenarios:
        print(f"\n❌ INFEASIBLE SCENARIOS ({len(report.infeasible_scenarios)}):")
        for s in report.infeasible_scenarios[:10]:
            print(f"   - {s}")
        if len(report.infeasible_scenarios) > 10:
            print(f"   ... and {len(report.infeasible_scenarios) - 10} more")
    
    if report.timeout_scenarios:
        print(f"\n⏱️  TIMEOUT SCENARIOS ({len(report.timeout_scenarios)}):")
        for s in report.timeout_scenarios[:10]:
            print(f"   - {s}")
        if len(report.timeout_scenarios) > 10:
            print(f"   ... and {len(report.timeout_scenarios) - 10} more")
    
    if report.high_unserved_scenarios:
        print(f"\n⚠️  HIGH UNSERVED ENERGY SCENARIOS ({len(report.high_unserved_scenarios)}):")
        for s in report.high_unserved_scenarios[:10]:
            print(f"   - {s}")
        if len(report.high_unserved_scenarios) > 10:
            print(f"   ... and {len(report.high_unserved_scenarios) - 10} more")
    
    if report.slow_scenarios:
        print(f"\n🐢 SLOW SCENARIOS (>60s) ({len(report.slow_scenarios)}):")
        for s in sorted(report.slow_scenarios, key=lambda x: -float(x.split('(')[1].rstrip('s)')))[:10]:
            print(f"   - {s}")
        if len(report.slow_scenarios) > 10:
            print(f"   ... and {len(report.slow_scenarios) - 10} more")
    
    print("\n" + "="*70)


def save_report_json(report: AnalysisReport, output_path: Path) -> None:
    """Save report to JSON file."""
    # Convert dataclasses to dicts
    report_dict = {
        "general": {
            "total_reports": report.total_reports,
            "reports_analyzed": report.reports_analyzed,
            "parse_errors": report.parse_errors,
        },
        "termination": {
            "optimal": report.optimal_count,
            "infeasible": report.infeasible_count,
            "timeout": report.timeout_count,
            "other": report.other_termination_count,
            "breakdown": report.termination_breakdown,
        },
        "solve_times": {
            "mip": asdict(report.mip_solve_stats),
            "lp": asdict(report.lp_solve_stats),
        },
        "mip_lp_gap_percent": report.mip_lp_gap_percent,
        "objectives": {
            "min": report.objective_min if report.objective_min != float('inf') else None,
            "max": report.objective_max,
            "mean": report.objective_mean,
            "median": report.objective_median,
        },
        "generation_mix": {
            "with_nuclear": report.scenarios_with_nuclear,
            "without_nuclear": report.scenarios_without_nuclear,
            "with_thermal": report.scenarios_with_thermal,
            "with_unserved": report.scenarios_with_unserved,
            "with_dr": report.scenarios_with_dr,
            "with_solar_spill": report.scenarios_with_solar_spill,
            "with_wind_spill": report.scenarios_with_wind_spill,
            "with_imports": report.scenarios_with_imports,
            "with_exports": report.scenarios_with_exports,
        },
        "cost_stats": {
            name: asdict(stats) for name, stats in report.cost_stats.items()
        },
        "problem_scenarios": {
            "infeasible": report.infeasible_scenarios,
            "timeout": report.timeout_scenarios,
            "high_unserved": report.high_unserved_scenarios,
            "slow": report.slow_scenarios,
        },
    }
    
    with open(output_path, 'w') as f:
        json.dump(report_dict, f, indent=2)
    
    print(f"\n📁 Report saved to: {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Analyze MILP scenario reports and generate statistics."
    )
    parser.add_argument(
        "reports_dir",
        type=Path,
        help="Directory containing scenario report JSON files"
    )
    parser.add_argument(
        "--output", "-o",
        type=Path,
        default=None,
        help="Output JSON file for the analysis report"
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Print verbose output during analysis"
    )
    
    args = parser.parse_args()
    
    if not args.reports_dir.is_dir():
        print(f"Error: {args.reports_dir} is not a directory", file=sys.stderr)
        sys.exit(1)
    
    # Run analysis
    report = analyze_reports(args.reports_dir, verbose=args.verbose)
    
    # Print results
    print_report(report)
    
    # Save JSON if requested
    if args.output:
        save_report_json(report, args.output)


if __name__ == "__main__":
    main()
