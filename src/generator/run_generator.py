import argparse
from .generator_v1 import generate_scenarios


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate MILP scenarios")
    parser.add_argument(
        "--config", "-c",
        default="config/scenario_space.yaml",
        help="Path to scenario space config (default: config/scenario_space.yaml)"
    )
    parser.add_argument(
        "--output", "-o",
        default="outputs/scenarios_v1",
        help="Output directory (default: outputs/scenarios_v1)"
    )
    args = parser.parse_args()

    manifest = generate_scenarios(args.config, args.output)
    print(manifest)
