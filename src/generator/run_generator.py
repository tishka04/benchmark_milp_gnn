import argparse
from .generator_v1 import generate_scenarios
from .generator_v2 import generate_scenarios_v2


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate MILP scenarios")
    parser.add_argument(
        "--config", "-c",
        default="config/scenario_space.yaml",
        help="Path to scenario space config (default: config/scenario_space.yaml)"
    )
    parser.add_argument(
        "--output", "-o",
        default="outputs/scenarios_v2",
        help="Output directory (default: outputs/scenarios_v2)"
    )
    parser.add_argument(
        "--version", "-v",
        choices=["v1", "v2"],
        default="v2",
        help="Generator version: v1 (greedy cover) or v2 (pool + k-center with LHS)"
    )
    parser.add_argument(
        "--pool-multiplier",
        type=int,
        default=20,
        help="Pool size multiplier for v2 (default: 20x target)"
    )
    parser.add_argument(
        "--no-lhs",
        action="store_true",
        help="Disable Latin Hypercube Sampling in v2"
    )
    parser.add_argument(
        "--no-stratification",
        action="store_true",
        help="Disable stratification/quotas in v2"
    )
    args = parser.parse_args()

    if args.version == "v1":
        manifest = generate_scenarios(args.config, args.output)
    else:
        manifest = generate_scenarios_v2(
            args.config, 
            args.output,
            pool_multiplier=args.pool_multiplier,
            use_lhs=not args.no_lhs,
            use_stratification=not args.no_stratification,
        )
    print(manifest)
