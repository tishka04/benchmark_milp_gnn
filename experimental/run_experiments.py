"""
Main entry point for running experiments.

This script runs the full experimental protocol as described in the README.
"""
import argparse
from pathlib import Path
from data_models import ExperimentConfig
from experiment_runner import ExperimentRunner


def main():
    parser = argparse.ArgumentParser(
        description="Run Hybrid vs MILP benchmark experiments"
    )
    
    parser.add_argument(
        "--output-dir",
        type=str,
        default="./results",
        help="Output directory for results (default: ./results)"
    )
    
    parser.add_argument(
        "--scales",
        type=str,
        default="small",
        choices=["small", "medium", "full"],
        help="Experiment scale: small (quick test), medium, full (default: small)"
    )
    
    parser.add_argument(
        "--regenerate",
        action="store_true",
        help="Regenerate instances even if they exist"
    )
    
    parser.add_argument(
        "--instances-per-scale",
        type=int,
        default=None,
        help="Number of instances per scale (overrides preset)"
    )
    
    args = parser.parse_args()
    
    # Configure based on scale
    if args.scales == "small":
        config = ExperimentConfig(
            n_units_list=[20, 50],
            n_periods_list=[24],
            instances_per_scale=args.instances_per_scale or 5,
            time_budget_small=180,  # 3 min
            time_budget_medium=300,  # 5 min
            reference_time_budget=600  # 10 min
        )
    elif args.scales == "medium":
        config = ExperimentConfig(
            n_units_list=[20, 50, 100],
            n_periods_list=[24, 96],
            instances_per_scale=args.instances_per_scale or 10,
            time_budget_small=300,  # 5 min
            time_budget_medium=600,  # 10 min
            time_budget_large=1200,  # 20 min
            reference_time_budget=1800  # 30 min
        )
    else:  # full
        config = ExperimentConfig(
            n_units_list=[20, 50, 100, 200, 400],
            n_periods_list=[24, 96],
            instances_per_scale=args.instances_per_scale or 10,
            time_budget_small=300,  # 5 min
            time_budget_medium=600,  # 10 min
            time_budget_large=1800,  # 30 min
            time_budget_xlarge=3600,  # 60 min
            reference_time_budget=7200  # 2 hours
        )
    
    print("\n" + "="*100)
    print("EXPERIMENTAL PROTOCOL: HYBRID VS MILP")
    print("="*100)
    print(f"\nConfiguration:")
    print(f"  Scale: {args.scales}")
    print(f"  Unit counts: {config.n_units_list}")
    print(f"  Period counts: {config.n_periods_list}")
    print(f"  Instances per scale: {config.instances_per_scale}")
    print(f"  Output directory: {args.output_dir}")
    print(f"  Regenerate instances: {args.regenerate}")
    print("\n" + "="*100)
    
    # Create runner and execute
    output_dir = Path(args.output_dir)
    runner = ExperimentRunner(config, output_dir)
    
    try:
        runner.run_full_protocol(regenerate_instances=args.regenerate)
    except KeyboardInterrupt:
        print("\n\nExperiment interrupted by user.")
        print(f"Partial results saved to: {output_dir}")
    except Exception as e:
        print(f"\n\nError during experiment: {e}")
        import traceback
        traceback.print_exc()
        print(f"\nPartial results may be in: {output_dir}")
    
    print("\n" + "="*100)
    print("EXPERIMENT COMPLETE")
    print("="*100)
    print(f"\nResults available in: {output_dir}")
    print(f"  - Instances: {output_dir}/instances/")
    print(f"  - Reference results: {output_dir}/results_reference/")
    print(f"  - Standard results: {output_dir}/results_standard/")
    print(f"  - Analysis: {output_dir}/analysis/")
    print(f"\nSee {output_dir}/analysis/REPORT.md for summary.\n")


if __name__ == "__main__":
    main()
