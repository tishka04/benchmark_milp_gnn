#!/usr/bin/env python3
"""
Classify MILP reports into Gold/Silver/Bronze categories.

Categories:
- Gold: Optimal solution found (mip.termination == "optimal")
- Silver: TimeLimit reached (mip.termination == "maxTimeLimit")
- Bronze: No optimal solution found (any other termination status)
"""

import json
import os
import shutil
from pathlib import Path
from collections import defaultdict


def classify_reports(reports_dir: Path, dry_run: bool = False) -> dict:
    """
    Classify MILP reports into Gold/Silver/Bronze subdirectories.
    
    Args:
        reports_dir: Path to the reports directory
        dry_run: If True, only print what would be done without moving files
        
    Returns:
        Dictionary with classification counts and file lists
    """
    # Create output directories
    gold_dir = reports_dir / "gold"
    silver_dir = reports_dir / "silver"
    bronze_dir = reports_dir / "bronze"
    
    if not dry_run:
        gold_dir.mkdir(exist_ok=True)
        silver_dir.mkdir(exist_ok=True)
        bronze_dir.mkdir(exist_ok=True)
    
    # Track classification results
    results = {
        "gold": [],
        "silver": [],
        "bronze": [],
        "errors": []
    }
    
    # Get all JSON report files (excluding subdirectories)
    report_files = [f for f in reports_dir.glob("*.json") if f.is_file()]
    
    print(f"Found {len(report_files)} report files to classify")
    
    for report_path in sorted(report_files):
        try:
            with open(report_path, "r") as f:
                data = json.load(f)
            
            # Get MIP termination status
            mip_termination = data.get("mip", {}).get("termination", "unknown")
            
            # Classify based on termination status
            if mip_termination == "optimal":
                category = "gold"
                target_dir = gold_dir
            elif mip_termination == "maxTimeLimit":
                category = "silver"
                target_dir = silver_dir
            else:
                category = "bronze"
                target_dir = bronze_dir
            
            results[category].append(report_path.name)
            
            if not dry_run:
                # Copy file to appropriate category directory
                shutil.copy2(report_path, target_dir / report_path.name)
                
        except Exception as e:
            results["errors"].append({
                "file": report_path.name,
                "error": str(e)
            })
    
    return results


def print_summary(results: dict):
    """Print classification summary."""
    print("\n" + "=" * 60)
    print("CLASSIFICATION SUMMARY")
    print("=" * 60)
    print(f"[GOLD]   Optimal:      {len(results['gold']):5d} scenarios")
    print(f"[SILVER] TimeLimit:    {len(results['silver']):5d} scenarios")
    print(f"[BRONZE] Other:        {len(results['bronze']):5d} scenarios")
    print(f"[ERROR]  Failed:       {len(results['errors']):5d} files")
    print("=" * 60)
    
    total = len(results['gold']) + len(results['silver']) + len(results['bronze'])
    if total > 0:
        print(f"\nPercentages:")
        print(f"  Gold:   {100 * len(results['gold']) / total:.1f}%")
        print(f"  Silver: {100 * len(results['silver']) / total:.1f}%")
        print(f"  Bronze: {100 * len(results['bronze']) / total:.1f}%")
    
    if results['errors']:
        print(f"\nErrors encountered:")
        for err in results['errors'][:10]:
            print(f"  - {err['file']}: {err['error']}")
        if len(results['errors']) > 10:
            print(f"  ... and {len(results['errors']) - 10} more errors")


def save_classification_index(results: dict, output_path: Path):
    """Save classification index to JSON file."""
    index = {
        "summary": {
            "gold_count": len(results['gold']),
            "silver_count": len(results['silver']),
            "bronze_count": len(results['bronze']),
            "error_count": len(results['errors'])
        },
        "gold": results['gold'],
        "silver": results['silver'],
        "bronze": results['bronze'],
        "errors": results['errors']
    }
    
    with open(output_path, "w") as f:
        json.dump(index, f, indent=2)
    
    print(f"\nClassification index saved to: {output_path}")


def main():
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Classify MILP reports into Gold/Silver/Bronze categories"
    )
    parser.add_argument(
        "--reports-dir",
        type=Path,
        default=Path(__file__).parent.parent / "outputs" / "scenarios_v3" / "reports",
        help="Path to reports directory"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Only print what would be done without moving files"
    )
    parser.add_argument(
        "--save-index",
        action="store_true",
        help="Save classification index to JSON file"
    )
    
    args = parser.parse_args()
    
    if not args.reports_dir.exists():
        print(f"Error: Reports directory not found: {args.reports_dir}")
        return 1
    
    print(f"Reports directory: {args.reports_dir}")
    if args.dry_run:
        print("DRY RUN - no files will be copied")
    
    results = classify_reports(args.reports_dir, dry_run=args.dry_run)
    print_summary(results)
    
    if args.save_index:
        index_path = args.reports_dir / "classification_index.json"
        save_classification_index(results, index_path)
    
    return 0


if __name__ == "__main__":
    exit(main())
