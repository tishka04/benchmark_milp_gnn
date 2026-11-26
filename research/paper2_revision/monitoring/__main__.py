import argparse
import json
from pathlib import Path

from .estimate import build_report, render


def main():
    parser = argparse.ArgumentParser(description="Read-only Paper 2 progress and time forecast")
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--hours-per-day", type=float, default=24)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()
    if not 0 < args.hours_per_day <= 24:
        parser.error("--hours-per-day must be greater than zero and at most 24")
    report = build_report(args.run.resolve(), hours_per_day=args.hours_per_day)
    print(json.dumps(report, indent=2, ensure_ascii=True) if args.json else render(report))


if __name__ == "__main__":
    main()
