import argparse
from threadpoolctl import threadpool_limits


def prepare_all():
    from .storage import prepare
    from .pipeline import fit_all, freeze
    from .report import run
    prepare()
    fit_all()
    freeze()
    run()


def main():
    parser = argparse.ArgumentParser(description='Recomputed MILP descriptors: no physical optimization in prepare/fit/freeze/report.')
    parser.add_argument('stage', choices=['prepare', 'fit', 'freeze', 'report', 'evaluate', 'all'])
    args = parser.parse_args()
    with threadpool_limits(limits=1):
        if args.stage == 'all':
            prepare_all()
            return
        from .storage import prepare
        from .pipeline import fit_all, freeze, evaluate
        from .report import run
        dict(prepare=prepare, fit=fit_all, freeze=freeze, report=run, evaluate=evaluate)[args.stage]()


if __name__ == '__main__':
    main()
