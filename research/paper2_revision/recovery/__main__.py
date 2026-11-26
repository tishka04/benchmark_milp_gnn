import argparse
import json
from pathlib import Path

from . import prepare, verify_prepared


def main():
    parser = argparse.ArgumentParser(description="Resume Paper 2 preparation and training safely")
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--stage", choices=("Prepare", "TrainMultiHorizon"), default="Prepare")
    parser.add_argument("--verify-only", action="store_true")
    args = parser.parse_args()
    run = args.run.resolve()
    if args.stage == "TrainMultiHorizon":
        from .training import inspect_training, run_training
        if args.verify_only:
            print(json.dumps(inspect_training(run), indent=2))
        else:
            run_training(run)
    elif args.verify_only:
        print(json.dumps(verify_prepared(run), indent=2))
    else:
        prepare(run)


if __name__ == "__main__":
    main()
