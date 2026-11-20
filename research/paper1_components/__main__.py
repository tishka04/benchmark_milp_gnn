"""Explicit user-launched CLI. No historical solver, router training or source mutation."""
import argparse
from .dataio import DEFAULT_RUN, DEFAULT_SOURCE, lock, prepare
from .parallel import default_workers, worker_count


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", default=DEFAULT_RUN)
    commands = parser.add_subparsers(dest="command", required=True)
    p = commands.add_parser("prepare")
    p.add_argument("--source-run", default=DEFAULT_SOURCE)
    commands.add_parser("audit")
    for name in ("fit", "report"):
        p = commands.add_parser(name)
        p.add_argument("--workers", type=worker_count, default=default_workers(),
                       help="Worker processes (1 = sequential; default: at most 8)")
    return parser


def main():
    args = build_parser().parse_args()
    with lock(args.run):
        if args.command == "prepare":
            prepare(args.run, args.source_run)
        elif args.command == "audit":
            from .audit import audit
            audit(args.run)
        elif args.command == "fit":
            from .fitting import fit
            fit(args.run, workers=args.workers)
        elif args.command == "report":
            from .reporting import report
            report(args.run, workers=args.workers)


if __name__ == "__main__":
    main()
