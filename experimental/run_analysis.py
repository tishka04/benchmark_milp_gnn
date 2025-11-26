"""
Run analysis on existing experiment results.

Use this script to re-run analysis without re-running experiments.
"""
import sys
from pathlib import Path

# Add experimental to path
sys.path.insert(0, str(Path(__file__).parent))

try:
    from analysis import ResultsAnalyzer
except ImportError:
    print("Error: Could not import ResultsAnalyzer")
    sys.exit(1)


def main():
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Analyze existing experimental results"
    )
    parser.add_argument(
        "--results-dir",
        type=str,
        default="../results",
        help="Directory containing experimental results (default: ../results)"
    )
    
    args = parser.parse_args()
    results_dir = Path(args.results_dir)
    
    if not results_dir.exists():
        print(f"Error: Results directory not found: {results_dir}")
        print(f"  Expected: {results_dir.absolute()}")
        sys.exit(1)
    
    print(f"\n{'='*80}")
    print(f"ANALYZING RESULTS FROM: {results_dir}")
    print(f"{'='*80}\n")
    
    # Check if required folders exist
    if not (results_dir / "instances").exists():
        print("Error: instances/ folder not found")
        sys.exit(1)
    
    if not (results_dir / "results_standard").exists():
        print("Error: results_standard/ folder not found")
        sys.exit(1)
    
    # Run analysis
    analyzer = ResultsAnalyzer(results_dir)
    
    print("Computing per-instance metrics...")
    per_instance, aggregated = analyzer.analyze_all()
    
    print("\nGenerating report...")
    analyzer.generate_report()
    
    print(f"\n{'='*80}")
    print(f"ANALYSIS COMPLETE")
    print(f"{'='*80}")
    print(f"\nResults saved to:")
    print(f"  - {results_dir / 'analysis' / 'per_instance_metrics.json'}")
    print(f"  - {results_dir / 'analysis' / 'aggregated_metrics.json'}")
    print(f"  - {results_dir / 'analysis' / 'REPORT.md'}")
    print(f"\nView the report:")
    print(f"  cat {results_dir / 'analysis' / 'REPORT.md'}")


if __name__ == "__main__":
    main()
