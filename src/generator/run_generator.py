from .generator_v1 import generate_scenarios


if __name__ == "__main__":
    manifest = generate_scenarios(
        "config/scenario_space.yaml",
        "outputs/scenarios_v1"
    )
    print(manifest)
