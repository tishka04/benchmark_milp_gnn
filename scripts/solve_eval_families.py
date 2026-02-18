"""
Solve all scenarios in the 3 evaluation families using the MILP solver.
Stores reports in a 'reports' subfolder within each family directory.

Usage:
    python -m scripts.solve_eval_families [--workers 4] [--time-limit 1200]
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path
from typing import List, Dict, Any

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.milp.solve import solve_scenario
from src.milp.batch_runner import _serialize_report

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
LOGGER = logging.getLogger("solve_eval_families")


def solve_family(
    family_dir: Path,
    solver_name: str = "highs",
    time_limit_seconds: float = 1200.0,
    tee: bool = False,
) -> Dict[str, Any]:
    """Solve all scenarios in a family directory and save reports."""
    reports_dir = family_dir / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)
    
    scenario_files = sorted(family_dir.glob("scenario_*.json"))
    if not scenario_files:
        LOGGER.warning(f"No scenarios found in {family_dir}")
        return {"count": 0, "success": 0, "failed": 0}
    
    LOGGER.info(f"Solving {len(scenario_files)} scenarios in {family_dir.name}")
    
    results = []
    for idx, scenario_path in enumerate(scenario_files, 1):
        sc_id = scenario_path.stem
        report_path = reports_dir / f"{sc_id}.json"
        
        # Skip if already solved
        if report_path.exists():
            LOGGER.info(f"  [{idx}/{len(scenario_files)}] {sc_id} - SKIPPED (already solved)")
            results.append({"scenario_id": sc_id, "status": "skipped"})
            continue
        
        try:
            t0 = time.perf_counter()
            report = solve_scenario(
                scenario_path,
                solver_name=solver_name,
                tee=tee,
                capture_detail=True,
                time_limit_seconds=time_limit_seconds,
            )
            elapsed = time.perf_counter() - t0
            
            # Serialize and save
            serialized = _serialize_report(report)
            with open(report_path, "w", encoding="utf-8") as f:
                json.dump(serialized, f, indent=2)
            
            obj = report["mip"].objective
            status = report["mip"].termination_condition.name
            solve_time = report["mip"].solve_seconds
            
            LOGGER.info(
                f"  [{idx}/{len(scenario_files)}] {sc_id} - "
                f"obj={obj:,.0f} status={status} time={solve_time:.1f}s"
            )
            results.append({
                "scenario_id": sc_id,
                "status": "ok",
                "objective": obj,
                "solve_time": solve_time,
                "termination": status,
            })
            
        except Exception as e:
            LOGGER.error(f"  [{idx}/{len(scenario_files)}] {sc_id} - FAILED: {e}")
            results.append({"scenario_id": sc_id, "status": "failed", "error": str(e)})
    
    # Save summary
    success_count = sum(1 for r in results if r["status"] in ("ok", "skipped"))
    failed_count = sum(1 for r in results if r["status"] == "failed")
    
    summary = {
        "family": family_dir.name,
        "total": len(scenario_files),
        "success": success_count,
        "failed": failed_count,
        "results": results,
    }
    
    with open(family_dir / "solve_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    
    return summary


def main():
    parser = argparse.ArgumentParser(description="Solve evaluation scenario families.")
    parser.add_argument("--solver", default="highs", help="Solver name")
    parser.add_argument("--time-limit", type=float, default=1200.0, help="Time limit per scenario (seconds)")
    parser.add_argument("--tee", action="store_true", help="Stream solver output")
    parser.add_argument("--families", nargs="*", default=None, 
                        help="Specific families to solve (default: all 3)")
    args = parser.parse_args()
    
    outputs_dir = PROJECT_ROOT / "outputs"
    
    family_names = args.families or [
        "low_criticality_scenarios",
        "medium_criticality_scenarios", 
        "high_criticality_scenarios",
    ]
    
    all_summaries = {}
    for family_name in family_names:
        family_dir = outputs_dir / family_name
        if not family_dir.exists():
            LOGGER.warning(f"Family directory not found: {family_dir}")
            continue
        
        print(f"\n{'='*60}")
        print(f"SOLVING: {family_name}")
        print(f"{'='*60}")
        
        summary = solve_family(
            family_dir,
            solver_name=args.solver,
            time_limit_seconds=args.time_limit,
            tee=args.tee,
        )
        all_summaries[family_name] = summary
        
        print(f"  Completed: {summary['success']}/{summary['total']} success, "
              f"{summary['failed']} failed")
    
    # Global summary
    print(f"\n{'='*60}")
    print("ALL FAMILIES COMPLETE")
    print(f"{'='*60}")
    for name, s in all_summaries.items():
        print(f"  {name}: {s['success']}/{s['total']} success")


if __name__ == "__main__":
    main()
