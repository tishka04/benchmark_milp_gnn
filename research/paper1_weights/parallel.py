"""Bounded Windows-compatible workers; only the parent publishes files."""
from concurrent.futures import ProcessPoolExecutor, wait, FIRST_COMPLETED
from multiprocessing import get_context
import os


def native_threads():
    for name in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS',
                 'NUMEXPR_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS', 'BLIS_NUM_THREADS'):
        os.environ[name] = '1'
    os.environ['PYTHONDONTWRITEBYTECODE'] = '1'


def bounded_map(function, tasks, workers):
    if not 1 <= workers <= 61:
        raise ValueError('workers must be between 1 and 61')
    native_threads()
    if workers == 1:
        for task in tasks:
            yield function(task)
        return
    iterator = iter(tasks)
    pool = ProcessPoolExecutor(max_workers=workers, mp_context=get_context('spawn'))
    pending = set()
    def fill():
        while len(pending) < 2 * workers:
            try:
                item = next(iterator)
            except StopIteration:
                break
            pending.add(pool.submit(function, item))
    try:
        fill()
        while pending:
            done, pending = wait(pending, return_when=FIRST_COMPLETED)
            for future in done:
                yield future.result()
            fill()
    finally:
        for future in pending:
            future.cancel()
        pool.shutdown(wait=True, cancel_futures=True)
