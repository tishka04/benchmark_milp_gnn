"""
Script to find scenarios without corresponding reports in scenarios_v2.
"""
from pathlib import Path

def main():
    base_dir = Path(__file__).parent.parent / "outputs" / "scenarios_v2"
    scenarios_dir = base_dir
    reports_dir = base_dir / "reports"
    
    # Get all scenario files
    scenario_files = {f.stem for f in scenarios_dir.glob("scenario_*.json")}
    
    # Get all report files
    report_files = {f.stem for f in reports_dir.glob("scenario_*.json")}
    
    # Find scenarios without reports
    missing_reports = sorted(scenario_files - report_files)
    
    print(f"Total scenarios: {len(scenario_files)}")
    print(f"Total reports: {len(report_files)}")
    print(f"Missing reports: {len(missing_reports)}")
    
    if missing_reports:
        print("\nScenarios without reports:")
        for s in missing_reports:
            print(f"  - {s}")

if __name__ == "__main__":
    main()
