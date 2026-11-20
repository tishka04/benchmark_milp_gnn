"""Command-line entry point for the fixed-system operating-day pilot."""
from __future__ import annotations

import argparse
from pathlib import Path

from .protocol import DEFAULT_RUN, read_manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    actions = parser.add_subparsers(dest="action", required=True)
    prep = actions.add_parser("prepare", help="Generate and freeze the days and predictions without solving.")
    prep.add_argument("--anchor", type=Path, help="JSON of the fixed source system (24 hourly steps).")
    prep.add_argument("--days", type=int, default=30)
    prep.add_argument("--quota", type=int, default=6, help="Days selected for inspection per profile and deadline.")
    prep.add_argument("--deadlines", type=int, nargs="+", default=[30, 120])
    prep.add_argument("--reference-seconds", type=int, default=600)
    prep.add_argument("--threads", type=int, default=16, help="HiGHS threads in one worker; jobs run sequentially.")
    prep.add_argument("--seed", type=int, default=20261119, help="Master seed for operating days and job order.")
    prep.add_argument("--solver-seed", type=int, default=42)
    run = actions.add_parser("run", help="Run all outstanding jobs, or resume a previous invocation.")
    run.add_argument("--kind", choices=["all", "trial", "reference"], default="all")
    run.add_argument("--max-jobs", type=int, help="Maximum number of new jobs in this invocation.")
    status = actions.add_parser("status", help="Count completed observations and valid dispatches.")
    analysis = actions.add_parser("analyze", help="Evaluate the completed pilot and write its report.")
    for sub in (prep, run, status, analysis):
        sub.add_argument("--run-dir", type=Path, default=DEFAULT_RUN, help="Pilot manifest and results directory.")
    args = parser.parse_args()
    run_dir = args.run_dir.resolve()

    try:
        if args.action == "prepare":
            from .design import DEFAULT_ANCHOR
            from .protocol import prepare
            manifest = prepare(run_dir, anchor=args.anchor or DEFAULT_ANCHOR, n_days=args.days,
                               deadlines=args.deadlines, reference_seconds=args.reference_seconds,
                               threads=args.threads, solver_seed=args.solver_seed, seed=args.seed, quota=args.quota)
            print(f"Frozen {len(manifest['days'])} days, deadlines {manifest['deadlines']}, "
                  f"inspection quota {manifest['quota']}. Manifest: {run_dir / 'manifest.json'}")
            return
        manifest = read_manifest(run_dir)
        if args.action == "run":
            from .protocol import verify_inputs
            from .runner import run_jobs
            verify_inputs(manifest)
            counts = run_jobs(run_dir, manifest, max_jobs=args.max_jobs, kind=args.kind)
            print(f"Jobs ({args.kind}): {counts['already_completed']} previously recorded, "
                  f"{counts['completed_now']} recorded now, {counts['remaining']} pending.")
        elif args.action == "status":
            from .runner import iter_jobs, read_job
            for role in ("trial", "reference"):
                jobs = iter_jobs(manifest, kind=role)
                records = [read_job(run_dir, manifest, day["day_id"], kind, budget)
                           for day, kind, budget in jobs]
                completed = [row for row in records if row is not None]
                print(f"{role}: {len(completed)}/{len(jobs)} recorded, "
                      f"{sum(r['feasible'] for r in completed)} valid on time, "
                      f"{len(jobs) - len(completed)} pending.")
        else:
            from .analysis import analyze
            result = analyze(run_dir, manifest)
            print(f"Report: {result['report']}")
    except (OSError, ValueError, RuntimeError, KeyError) as exc:
        parser.exit(1, f"Pilot: {exc}\n")
    except KeyboardInterrupt:
        parser.exit(130, "Pilot interrupted. Use status and run with the same directory to resume.\n")


if __name__ == "__main__":
    main()
