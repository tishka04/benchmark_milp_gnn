"""
Export complete MILP solutions including all binary variables.

This script solves MILP problems and exports:
- All binary commitment variables (u_thermal, u_nuclear)
- All binary startup variables (v_thermal_startup, v_nuclear_startup)
- Continuous dispatch variables
- Cost components and duals

Usage:
    python src/milp/export_binary_variables.py
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Dict, List

from pyomo.environ import SolverFactory, value

from .model import build_uc_model
from .scenario_loader import load_scenario_data, ScenarioData
from .solve import _compute_cost_components, _relax_integrality, SolveSummary


def extract_all_variables(model, data: ScenarioData) -> Dict[str, Any]:
    """
    Extract all variables from solved MILP model.
    
    Args:
        model: Solved Pyomo model
        data: Scenario data
        
    Returns:
        Dictionary with all variables organized by category
    """
    periods = list(model.T)
    zones = [str(z) for z in model.Z]
    
    # Binary variables (MAIN FOCUS)
    binary_vars = {
        "u_thermal": {
            zone: [int(round(value(model.u_thermal[zone, t]))) for t in periods]
            for zone in zones
        },
        "v_thermal_startup": {
            zone: [int(round(value(model.v_thermal_startup[zone, t]))) for t in periods]
            for zone in zones
        },
        "u_nuclear": {
            zone: [int(round(value(model.u_nuclear[zone, t]))) for t in periods]
            for zone in zones
        },
        "v_nuclear_startup": {
            zone: [int(round(value(model.v_nuclear_startup[zone, t]))) for t in periods]
            for zone in zones
        },
    }
    
    # Continuous dispatch variables
    continuous_vars = {
        "thermal": {
            zone: [float(value(model.p_thermal[zone, t])) for t in periods]
            for zone in zones
        },
        "nuclear": {
            zone: [float(value(model.p_nuclear[zone, t])) for t in periods]
            for zone in zones
        },
        "solar": {
            zone: [float(value(model.p_solar[zone, t])) for t in periods]
            for zone in zones
        },
        "wind": {
            zone: [float(value(model.p_wind[zone, t])) for t in periods]
            for zone in zones
        },
        "solar_spill": {
            zone: [float(value(model.spill_solar[zone, t])) for t in periods]
            for zone in zones
        },
        "wind_spill": {
            zone: [float(value(model.spill_wind[zone, t])) for t in periods]
            for zone in zones
        },
        "hydro_release": {
            zone: [float(value(model.h_release[zone, t])) for t in periods]
            for zone in zones
        },
        "hydro_ror": {
            zone: [float(value(model.hydro_ror[zone, t])) for t in periods]
            for zone in zones
        },
        "hydro_spill": {
            zone: [float(value(model.h_spill[zone, t])) for t in periods]
            for zone in zones
        },
        "battery_charge": {
            zone: [float(value(model.b_charge[zone, t])) for t in periods]
            for zone in zones
        },
        "battery_discharge": {
            zone: [float(value(model.b_discharge[zone, t])) for t in periods]
            for zone in zones
        },
        "battery_soc": {
            zone: [float(value(model.b_soc[zone, t])) for t in periods]
            for zone in zones
        },
        "pumped_charge": {
            zone: [float(value(model.pumped_charge[zone, t])) for t in periods]
            for zone in zones
        },
        "pumped_discharge": {
            zone: [float(value(model.pumped_discharge[zone, t])) for t in periods]
            for zone in zones
        },
        "pumped_level": {
            zone: [float(value(model.pumped_level[zone, t])) for t in periods]
            for zone in zones
        },
        "demand_response": {
            zone: [float(value(model.dr_shed[zone, t])) for t in periods]
            for zone in zones
        },
        "unserved": {
            zone: [float(value(model.unserved[zone, t])) for t in periods]
            for zone in zones
        },
        "overgen_spill": {
            zone: [float(value(model.overgen_spill[zone, t])) for t in periods]
            for zone in zones
        },
    }
    
    # System-level variables
    system_vars = {
        "net_import": [float(value(model.net_import[t])) for t in periods],
        "net_export": [float(value(model.net_export[t])) for t in periods],
        "flows": {
            str(line): [float(value(model.flow[line, t])) for t in periods]
            for line in model.L
        },
    }
    
    # Metadata
    metadata = {
        "time_steps": [int(t) for t in periods],
        "time_hours": [float(t * data.dt_hours) for t in periods],
        "dt_hours": float(data.dt_hours),
        "zones": zones,
        "lines": [str(lid) for lid in model.L],
        "n_timesteps": len(periods),
        "n_zones": len(zones),
    }
    
    # Demand (input data, not variables)
    demand = {
        zone: [float(data.demand[(zone, t)]) for t in periods]
        for zone in zones
    }
    
    return {
        "metadata": metadata,
        "binary_variables": binary_vars,
        "continuous_variables": continuous_vars,
        "system_variables": system_vars,
        "demand": demand,
    }


def solve_and_export_scenario(
    scenario_id: str,
    scenarios_dir: Path,
    output_dir: Path,
    solver_name: str = "highs",
    solve_lp: bool = True,
) -> Dict[str, Any]:
    """
    Solve MILP for a scenario and export complete solution.
    
    Args:
        scenario_id: Scenario identifier (e.g., "scenario_00001")
        scenarios_dir: Directory containing scenario .pkl files
        output_dir: Directory to save JSON output
        solver_name: Solver to use (default: "highs")
        solve_lp: Whether to also solve relaxed LP
        
    Returns:
        Dictionary with solve summary and file path
    """
    print(f"\n{'='*80}")
    print(f"Processing: {scenario_id}")
    print(f"{'='*80}")
    
    # Load scenario data
    scenario_path = scenarios_dir / f"{scenario_id}.pkl"
    if not scenario_path.exists():
        # Try .json extension
        scenario_path = scenarios_dir / f"{scenario_id}.json"
        if not scenario_path.exists():
            raise FileNotFoundError(f"Scenario file not found: {scenario_id}")
    
    print(f"Loading scenario from: {scenario_path}")
    data = load_scenario_data(scenario_path)
    
    # Initialize solver
    solver = SolverFactory(solver_name)
    
    # ========== SOLVE MILP ==========
    print(f"\n[1/2] Solving MILP...")
    mip_model = build_uc_model(data, enable_duals=False)
    
    mip_start = time.perf_counter()
    mip_results = solver.solve(mip_model, tee=False)
    mip_elapsed = time.perf_counter() - mip_start
    
    mip_objective = value(mip_model.obj)
    mip_termination = str(mip_results.solver.termination_condition)
    mip_status = str(mip_results.solver.status)
    
    print(f"  Status: {mip_status}")
    print(f"  Termination: {mip_termination}")
    print(f"  Objective: {mip_objective:,.2f}")
    print(f"  Solve time: {mip_elapsed:.2f}s")
    
    # Extract all variables from MIP solution
    print(f"\n[2/2] Extracting variables...")
    all_vars = extract_all_variables(mip_model, data)
    
    # Compute cost components
    cost_components = _compute_cost_components(mip_model)
    
    # Count binary variables
    n_bins_thermal = sum(
        sum(all_vars["binary_variables"]["u_thermal"][z]) 
        for z in all_vars["metadata"]["zones"]
    )
    n_bins_nuclear = sum(
        sum(all_vars["binary_variables"]["u_nuclear"][z]) 
        for z in all_vars["metadata"]["zones"]
    )
    n_startups_thermal = sum(
        sum(all_vars["binary_variables"]["v_thermal_startup"][z]) 
        for z in all_vars["metadata"]["zones"]
    )
    n_startups_nuclear = sum(
        sum(all_vars["binary_variables"]["v_nuclear_startup"][z]) 
        for z in all_vars["metadata"]["zones"]
    )
    
    print(f"  Thermal units ON: {n_bins_thermal} / {len(all_vars['metadata']['zones']) * len(all_vars['metadata']['time_steps'])}")
    print(f"  Nuclear units ON: {n_bins_nuclear} / {len(all_vars['metadata']['zones']) * len(all_vars['metadata']['time_steps'])}")
    print(f"  Thermal startups: {n_startups_thermal}")
    print(f"  Nuclear startups: {n_startups_nuclear}")
    
    # ========== SOLVE LP (optional) ==========
    lp_summary = None
    if solve_lp:
        print(f"\n[LP] Solving relaxed LP for bound...")
        lp_model = build_uc_model(data, enable_duals=False)
        _relax_integrality(lp_model)
        
        lp_start = time.perf_counter()
        lp_results = solver.solve(lp_model, tee=False)
        lp_elapsed = time.perf_counter() - lp_start
        
        lp_objective = value(lp_model.obj)
        lp_gap = ((mip_objective - lp_objective) / mip_objective * 100) if mip_objective > 0 else 0.0
        
        print(f"  LP Objective: {lp_objective:,.2f}")
        print(f"  MIP-LP Gap: {lp_gap:.2f}%")
        print(f"  LP Solve time: {lp_elapsed:.2f}s")
        
        lp_summary = {
            "objective": lp_objective,
            "termination": str(lp_results.solver.termination_condition),
            "status": str(lp_results.solver.status),
            "solve_seconds": lp_elapsed,
            "mip_lp_gap_percent": lp_gap,
        }
    
    # ========== PREPARE OUTPUT ==========
    output = {
        "scenario_id": scenario_id,
        "scenario_uuid": data.scenario_id,
        "mip_solution": {
            "objective": mip_objective,
            "termination": mip_termination,
            "status": mip_status,
            "solve_seconds": mip_elapsed,
        },
        "lp_solution": lp_summary,
        "cost_components": cost_components,
        "variables": all_vars,
        "binary_statistics": {
            "thermal_commitment_count": n_bins_thermal,
            "nuclear_commitment_count": n_bins_nuclear,
            "thermal_startup_count": n_startups_thermal,
            "nuclear_startup_count": n_startups_nuclear,
            "total_binary_decisions": (
                len(all_vars['metadata']['zones']) * 
                len(all_vars['metadata']['time_steps']) * 4  # 4 binaries per zone per timestep
            ),
        },
    }
    
    # ========== SAVE TO FILE ==========
    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir / f"{scenario_id}_complete.json"
    
    print(f"\nSaving to: {output_file}")
    with open(output_file, 'w') as f:
        json.dump(output, f, indent=2)
    
    file_size_mb = output_file.stat().st_size / (1024 * 1024)
    print(f"File size: {file_size_mb:.2f} MB")
    print(f"✓ Export complete!")
    
    return {
        "scenario_id": scenario_id,
        "output_file": str(output_file),
        "mip_objective": mip_objective,
        "mip_time": mip_elapsed,
        "lp_objective": lp_objective if solve_lp else None,
        "success": mip_termination == "optimal",
    }


def main():
    """Main execution function."""
    import sys
    from pathlib import Path
    
    # Configuration
    REPO_ROOT = Path(__file__).parent.parent.parent
    SCENARIOS_DIR = REPO_ROOT / "outputs" / "scenarios_v1"
    OUTPUT_DIR = REPO_ROOT / "outputs" / "scenarios_v1" / "eval"
    
    # Target scenarios
    TARGET_SCENARIOS = [
        'scenario_00990',
        'scenario_01627',
        'scenario_01998',
        'scenario_01058',
        'scenario_01888',
    ]
    
    print("="*80)
    print("MILP SOLVER WITH COMPLETE BINARY VARIABLE EXPORT")
    print("="*80)
    print(f"Scenarios directory: {SCENARIOS_DIR}")
    print(f"Output directory: {OUTPUT_DIR}")
    print(f"Number of scenarios: {len(TARGET_SCENARIOS)}")
    print(f"Scenarios: {TARGET_SCENARIOS}")
    print("="*80)
    
    # Check if scenarios exist
    print("\nChecking scenario files...")
    for scenario_id in TARGET_SCENARIOS:
        pkl_path = SCENARIOS_DIR / f"{scenario_id}.pkl"
        json_path = SCENARIOS_DIR / f"{scenario_id}.json"
        if pkl_path.exists():
            print(f"  ✓ {scenario_id}.pkl")
        elif json_path.exists():
            print(f"  ✓ {scenario_id}.json")
        else:
            print(f"  ✗ {scenario_id} NOT FOUND")
            sys.exit(1)
    
    # Solve all scenarios
    results = []
    total_start = time.time()
    
    for i, scenario_id in enumerate(TARGET_SCENARIOS, 1):
        print(f"\n{'#'*80}")
        print(f"# Scenario {i}/{len(TARGET_SCENARIOS)}: {scenario_id}")
        print(f"{'#'*80}")
        
        try:
            result = solve_and_export_scenario(
                scenario_id=scenario_id,
                scenarios_dir=SCENARIOS_DIR,
                output_dir=OUTPUT_DIR,
                solver_name="highs",
                solve_lp=True,
            )
            results.append(result)
        except Exception as e:
            print(f"\n✗ ERROR processing {scenario_id}: {e}")
            import traceback
            traceback.print_exc()
            results.append({
                "scenario_id": scenario_id,
                "success": False,
                "error": str(e),
            })
    
    total_elapsed = time.time() - total_start
    
    # Summary
    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)
    
    successful = sum(1 for r in results if r.get("success", False))
    print(f"Successful: {successful}/{len(TARGET_SCENARIOS)}")
    print(f"Total time: {total_elapsed:.2f}s")
    print(f"Average time per scenario: {total_elapsed/len(TARGET_SCENARIOS):.2f}s")
    
    print("\nPer-scenario results:")
    for result in results:
        if result.get("success"):
            print(f"  ✓ {result['scenario_id']}: "
                  f"MIP={result['mip_objective']:,.0f}, "
                  f"time={result['mip_time']:.2f}s")
        else:
            print(f"  ✗ {result['scenario_id']}: {result.get('error', 'Unknown error')}")
    
    # Save summary
    summary_file = OUTPUT_DIR / "export_summary.json"
    with open(summary_file, 'w') as f:
        json.dump({
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "total_scenarios": len(TARGET_SCENARIOS),
            "successful": successful,
            "total_time_seconds": total_elapsed,
            "results": results,
        }, f, indent=2)
    
    print(f"\nSummary saved to: {summary_file}")
    print("="*80)
    print("✓ All done!")


if __name__ == "__main__":
    main()
