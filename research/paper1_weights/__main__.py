"""Explicit stages only; importing the package never launches an analysis."""
import argparse
from .parallel import native_threads


def main():
    native_threads()
    from .data import DEFAULT_RUN, DEFAULT_SOURCE, lock
    parser = argparse.ArgumentParser(description='Paper-1 weights from existing historical outcomes only')
    parser.add_argument('--run', default=str(DEFAULT_RUN))
    sub = parser.add_subparsers(dest='stage', required=True)
    prepare = sub.add_parser('prepare')
    prepare.add_argument('--source-run', default=str(DEFAULT_SOURCE))
    prepare.add_argument('--workers', type=int, default=8)
    for stage in ('fit', 'bootstrap'):
        p = sub.add_parser(stage)
        p.add_argument('--workers', type=int, default=8)
    sub.add_parser('report')
    args = parser.parse_args()
    with lock(args.run):
        if args.stage == 'prepare':
            from .data import prepare as prepare_data
            prepare_data(args.run, args.source_run, args.workers)
        else:
            from .analysis import fit_all, bootstrap_all, report
            if args.stage == 'fit':
                fit_all(args.run, args.workers)
            elif args.stage == 'bootstrap':
                bootstrap_all(args.run, args.workers)
            else:
                report(args.run)


if __name__ == '__main__':
    main()
