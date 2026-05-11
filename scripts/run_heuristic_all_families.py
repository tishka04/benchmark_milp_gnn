"""Run the RH-MO+LP heuristic on low, medium and high criticality families
sequentially in a single command.

Usage (from the `benchmark_milp_gnn` directory, with the venv activated):

    python -m scripts.run_heuristic_all_families
    python -m scripts.run_heuristic_all_families -W 8 --max-scenarios 20 -v

By default the script mirrors the notebook conventions:
  - scenarios_dir = outputs/<family>_criticality_scenarios
  - output JSON   = outputs/rhmo_<family>.json

Each family is launched as a subprocess calling `python -m src.heuristics.runner`
so that a crash on one family (e.g. OOM on `high`) does not abort the others.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path
from typing import List

FAMILIES = ("low", "medium", "high")
REPO_ROOT = Path(__file__).resolve().parent.parent


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Run RH-MO+LP heuristic on low -> medium -> high sequentially."
    )
    p.add_argument("--window", "-W", type=int, default=6,
                   help="Look-ahead window W (default: 6).")
    p.add_argument("--max-scenarios", type=int, default=None,
                   help="Cap the number of scenarios per family (debug/smoke test).")
    p.add_argument("--solver", type=str, default="appsi_highs")
    p.add_argument("--slack-tol", type=float, default=1.0)
    p.add_argument("--deviation-penalty", type=float, default=10000.0)
    p.add_argument("--max-stages", type=int, default=5)
    p.add_argument("--no-warm-start", action="store_true")
    p.add_argument("--verbose", "-v", action="store_true")
    p.add_argument("--scenarios-root", type=Path,
                   default=REPO_ROOT / "outputs",
                   help="Root containing <family>_criticality_scenarios folders.")
    p.add_argument("--output-root", type=Path,
                   default=REPO_ROOT / "outputs",
                   help="Where to save rhmo_<family>.json files.")
    p.add_argument("--families", nargs="+", choices=FAMILIES, default=list(FAMILIES),
                   help="Subset of families to run (default: all three).")
    p.add_argument("--keep-going", action="store_true",
                   help="If a family fails, continue with the remaining ones "
                        "instead of aborting.")
    return p.parse_args()


def _build_cmd(args: argparse.Namespace, family: str) -> List[str]:
    scenarios_dir = args.scenarios_root / f"{family}_criticality_scenarios"
    out_path = args.output_root / f"rhmo_{family}.json"

    cmd: List[str] = [
        sys.executable, "-m", "src.heuristics.runner",
        str(scenarios_dir),
        "-W", str(args.window),
        "-o", str(out_path),
        "--family", family,
        "--solver", args.solver,
        "--slack-tol", str(args.slack_tol),
        "--deviation-penalty", str(args.deviation_penalty),
        "--max-stages", str(args.max_stages),
    ]
    if args.max_scenarios is not None:
        cmd += ["--max-scenarios", str(args.max_scenarios)]
    if args.no_warm_start:
        cmd += ["--no-warm-start"]
    if args.verbose:
        cmd += ["-v"]
    return cmd


def main() -> int:
    args = _parse_args()

    print("=" * 72)
    print(f"  RH-MO+LP heuristic — sequential run over families: "
          f"{', '.join(args.families)}")
    print(f"  Scenarios root: {args.scenarios_root}")
    print(f"  Output root:    {args.output_root}")
    print(f"  Window W={args.window}, max_stages={args.max_stages}, "
          f"solver={args.solver}")
    print("=" * 72)

    args.output_root.mkdir(parents=True, exist_ok=True)

    failures: List[str] = []
    t_all_start = time.perf_counter()

    for family in args.families:
        scenarios_dir = args.scenarios_root / f"{family}_criticality_scenarios"
        if not scenarios_dir.exists():
            msg = f"[skip] {family}: directory not found -> {scenarios_dir}"
            print(msg)
            failures.append(family)
            if not args.keep_going:
                return 2
            continue

        cmd = _build_cmd(args, family)
        print(f"\n>>> [{family}] {' '.join(cmd)}")
        t_start = time.perf_counter()
        try:
            # Inherit stdout/stderr so tqdm progress bars stream live.
            proc = subprocess.run(cmd, cwd=REPO_ROOT, check=False)
        except KeyboardInterrupt:
            print(f"\n[abort] interrupted during family={family}")
            return 130

        dt = time.perf_counter() - t_start
        if proc.returncode != 0:
            failures.append(family)
            print(f"<<< [{family}] FAILED in {dt:.1f}s (rc={proc.returncode})")
            if not args.keep_going:
                return proc.returncode
        else:
            print(f"<<< [{family}] OK in {dt:.1f}s")

    total = time.perf_counter() - t_all_start
    print("\n" + "=" * 72)
    print(f"  Total wall time: {total:.1f}s ({total/60:.1f} min)")
    if failures:
        print(f"  Failed families: {failures}")
        return 1
    print("  All families completed successfully.")
    print("=" * 72)
    return 0


if __name__ == "__main__":
    sys.exit(main())
