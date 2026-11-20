"""Prepare the additional comparison, or explicitly run the existing physical jobs."""
import argparse
from threadpoolctl import threadpool_limits


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=['prepare', 'run', 'analyze'])
    parser.add_argument('--max-hours', type=float, default=24.)
    args = parser.parse_args()
    if args.max_hours <= 0:
        parser.error('--max-hours must be positive')
    with threadpool_limits(limits=1):
        if args.action == 'prepare':
            from .realized.__main__ import prepare_all
            prepare_all()
            return
        from .realized.pipeline import verify_freeze, evaluate
        verify_freeze()
        if args.action == 'run':
            from .campaign import execute
            execute('all', args.max_hours)
        from .future_reports import run
        run()
        evaluate()


if __name__ == '__main__':
    main()
