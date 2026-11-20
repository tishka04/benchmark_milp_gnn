import argparse

def main():
    p=argparse.ArgumentParser(description='Paper 1 revision: isolated analyses and physical campaign preparation')
    p.add_argument('stage',choices=['prepare','fit','lofo','probability','robustness','allocation','campaign-prepare','report','all'])
    args=p.parse_args()
    from threadpoolctl import threadpool_limits
    with threadpool_limits(limits=1):
        if args.stage in ('prepare','all'):
            from .data import prepare
            prepare()
        if args.stage in ('fit','lofo','all'):
            from .analysis import fit
            if args.stage in ('fit','all'): fit('nested')
            if args.stage in ('lofo','all'): fit('lofo')
        if args.stage in ('probability','all'):
            from .probability import run
            run()
        if args.stage in ('robustness','all'):
            from .robustness import run
            run()
        if args.stage in ('allocation','all'):
            from .allocation import run
            run()
        if args.stage in ('campaign-prepare','all'):
            from .campaign import prepare
            prepare()
        if args.stage in ('report','all'):
            from .report import run
            run()

if __name__=='__main__': main()
