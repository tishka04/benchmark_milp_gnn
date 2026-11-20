"""Bounded Windows-safe processes; native numeric libraries use one thread each."""
from contextlib import contextmanager
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
import multiprocessing
import os
import signal
from threadpoolctl import threadpool_limits

THREAD_VARIABLES = ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
                    "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "BLIS_NUM_THREADS")
_WORKER_LIMITER = None


def worker_count(value):
    if isinstance(value, str):
        value = int(value)
    if isinstance(value, bool) or not isinstance(value, int) or not 1 <= value <= 61:
        raise ValueError("workers must be an integer between 1 and 61")
    return value


def default_workers():
    return min(8, max(1, (os.cpu_count() or 1)-2))


@contextmanager
def single_thread_environment():
    previous = {name: os.environ.get(name) for name in THREAD_VARIABLES}
    try:
        for name in THREAD_VARIABLES:
            os.environ[name] = "1"
        yield
    finally:
        for name, value in previous.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value


def _start_worker(initializer, initargs):
    global _WORKER_LIMITER
    # Environment limits apply before spawn imports; this also limits loaded BLAS.
    _WORKER_LIMITER = threadpool_limits(limits=1)
    signal.signal(signal.SIGINT, signal.SIG_IGN)
    if initializer is not None:
        initializer(*initargs)


def results(function, jobs, workers=1, initializer=None, initargs=()):
    """Yield (input_index, result) as work finishes; at most 2*workers in flight.

    Workers return values only. The caller owns all publication/checkpoint writes.
    Close the generator on consumer errors, so the executor is joined before the
    run lock is released. On failure/interruption, cancel queued work and wait for
    the small number of already-dispatched tasks; completed publications survive.
    """
    workers = worker_count(workers)
    indexed = iter(enumerate(jobs))
    with single_thread_environment():
        if workers == 1:
            with threadpool_limits(limits=1):
                if initializer is not None:
                    initializer(*initargs)
                for index, job in indexed:
                    yield index, function(job)
            return
        executor = ProcessPoolExecutor(
            max_workers=workers, mp_context=multiprocessing.get_context("spawn"),
            initializer=_start_worker, initargs=(initializer, initargs))
        pending = {}

        def submit_one():
            try:
                index, job = next(indexed)
            except StopIteration:
                return False
            pending[executor.submit(function, job)] = index
            return True

        try:
            for _ in range(2*workers):
                if not submit_one():
                    break
            while pending:
                ready, _ = wait(pending, return_when=FIRST_COMPLETED)
                for future in sorted(ready, key=pending.__getitem__):
                    index = pending.pop(future)
                    yield index, future.result()
                    submit_one()
        finally:
            for future in pending:
                future.cancel()
            executor.shutdown(wait=True, cancel_futures=True)
