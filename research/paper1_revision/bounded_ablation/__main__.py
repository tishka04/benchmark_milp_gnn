import argparse
import os

os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'


def main():
    parser = argparse.ArgumentParser(description='Historical bounded regression and B9 ablation; no MILP solves.')
    parser.add_argument('action', choices=['run', 'report'], nargs='?', default='run')
    parser.add_argument('--max-jobs', type=int, help='Limit newly fitted checkpoints for a smoke run.')
    args = parser.parse_args()
    if args.max_jobs is not None and args.max_jobs <= 0:
        parser.error('--max-jobs must be positive')
    # Change only this analysis process, never the physical campaign workers.
    import psutil
    if os.name == 'nt':
        psutil.Process().nice(psutil.BELOW_NORMAL_PRIORITY_CLASS)
    from threadpoolctl import threadpool_limits
    from .pipeline import run, report
    with threadpool_limits(limits=1):
        if args.action == 'report':
            report()
        else:
            run(max_jobs=args.max_jobs)


if __name__ == '__main__':
    main()
