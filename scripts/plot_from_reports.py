"""Plot scenarios from existing reports without re-solving."""
from __future__ import annotations

import argparse
import json
import logging
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.milp.visualize import plot_scenario_and_dispatch

LOGGER = logging.getLogger("plot_from_reports")


def plot_single(scenario_path: Path, report_path: Path, plots_dir: Path) -> tuple[Path, bool, str]:
    """Plot a single scenario from its report."""
    try:
        with report_path.open('r', encoding='utf-8') as f:
            report = json.load(f)
        
        plots_dir.mkdir(parents=True, exist_ok=True)
        plot_scenario_and_dispatch(scenario_path, report, plots_dir)
        
        return (scenario_path, True, "ok")
    except Exception as exc:
        return (scenario_path, False, str(exc))


def main():
    parser = argparse.ArgumentParser(description="Generate plots from existing MILP reports")
    parser.add_argument("scenarios_dir", type=Path, help="Directory containing scenario JSON files")
    parser.add_argument("--reports-dir", type=Path, required=True, help="Directory containing report JSON files")
    parser.add_argument("--plots-dir", type=Path, required=True, help="Output directory for plots")
    parser.add_argument("--workers", type=int, default=4, help="Number of parallel workers")
    parser.add_argument("--verbose", action="store_true", help="Enable debug logging")
    args = parser.parse_args()
    
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )
    
    # Find all scenario files
    scenario_files = sorted(args.scenarios_dir.glob("scenario_*.json"))
    LOGGER.info("Found %d scenario files", len(scenario_files))
    
    # Match with reports
    tasks = []
    for scenario_path in scenario_files:
        report_path = args.reports_dir / f"{scenario_path.stem}.json"
        if report_path.exists():
            tasks.append((scenario_path, report_path))
        else:
            LOGGER.warning("No report found for %s", scenario_path.name)
    
    LOGGER.info("Found %d scenarios with reports", len(tasks))
    
    if not tasks:
        LOGGER.error("No scenarios with reports found!")
        return
    
    # Plot in parallel
    success_count = 0
    failure_count = 0
    
    if args.workers > 1:
        LOGGER.info("Plotting with %d parallel workers", args.workers)
        with ProcessPoolExecutor(max_workers=args.workers) as executor:
            futures = {
                executor.submit(plot_single, scenario_path, report_path, args.plots_dir): scenario_path
                for scenario_path, report_path in tasks
            }
            
            for idx, future in enumerate(as_completed(futures), start=1):
                scenario_path, success, message = future.result()
                if success:
                    success_count += 1
                    LOGGER.info("[%d/%d] %s OK", idx, len(tasks), scenario_path.name)
                else:
                    failure_count += 1
                    LOGGER.error("[%d/%d] %s FAILED: %s", idx, len(tasks), scenario_path.name, message)
    else:
        LOGGER.info("Plotting sequentially")
        for idx, (scenario_path, report_path) in enumerate(tasks, start=1):
            scenario_path, success, message = plot_single(scenario_path, report_path, args.plots_dir)
            if success:
                success_count += 1
                LOGGER.info("[%d/%d] %s OK", idx, len(tasks), scenario_path.name)
            else:
                failure_count += 1
                LOGGER.error("[%d/%d] %s FAILED: %s", idx, len(tasks), scenario_path.name, message)
    
    LOGGER.info("Complete: %d success, %d failure", success_count, failure_count)
    LOGGER.info("Plots saved to: %s", args.plots_dir)


if __name__ == "__main__":
    main()
