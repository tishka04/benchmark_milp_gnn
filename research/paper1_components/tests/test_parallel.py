"""Synthetic parallelism checks, including real Windows-spawn compatibility.

These tests are prepared for the user to run. They never read historical results.
"""
from contextlib import closing, ExitStack, redirect_stderr
from concurrent.futures import Future
import io as textio
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import MagicMock, patch

import pandas as pd
from threadpoolctl import threadpool_info

from research.paper1_components import dataio as io, fitting, models, parallel, reporting
from research.paper1_components.__main__ import build_parser
from research.paper1_components.tests.test_paper1_components import config, observation


def worker_probe(value):
    return dict(value=value, pid=os.getpid(),
                environment={n: os.environ.get(n) for n in parallel.THREAD_VARIABLES},
                native_threads=[p["num_threads"] for p in threadpool_info()])


def tiny_fitting_data():
    cfg = config()
    cfg.update(endpoints=["economic_5"], deadlines_s=[30], family_controls=[False],
               outer_folds=2, inner_folds=2, regularization_C=[1.])
    frame = pd.DataFrame([observation(i % 8, key=f"k{i}", group=f"g{i}",
        economic_5=(i//2) % 2, outer_fold=i % 2) for i in range(24)])
    return cfg, frame


def fitting_context(cfg, frame):
    stack = ExitStack()
    stack.enter_context(patch.object(fitting, "frozen", return_value={"sha256":"synthetic"}))
    stack.enter_context(patch.object(fitting, "protocol", return_value=cfg))
    stack.enter_context(patch.object(fitting, "data", return_value=frame))
    stack.enter_context(patch.object(fitting, "specifications",
                                    return_value=[models.Spec("size")]))
    return stack


class ParallelExecution(unittest.TestCase):
    def test_worker_counts_and_cli_are_explicit(self):
        for value in (0, -1, 62, True, 1.5, "2.5"):
            with self.assertRaises((ValueError, TypeError)):
                parallel.worker_count(value)
        self.assertEqual(parallel.worker_count("8"), 8)
        for command in ("fit", "report"):
            self.assertEqual(build_parser().parse_args([command, "--workers", "4"]).workers, 4)
        with patch.object(parallel.os, "cpu_count", return_value=16):
            self.assertEqual(parallel.default_workers(), 8)
        with patch.object(parallel.os, "cpu_count", return_value=1):
            self.assertEqual(parallel.default_workers(), 1)
        with redirect_stderr(textio.StringIO()), self.assertRaises(SystemExit):
            build_parser().parse_args(["fit", "--workers", "0"])

    def test_sequential_mode_does_not_start_processes_and_restores_limits(self):
        before = {n: os.environ.get(n) for n in parallel.THREAD_VARIABLES}
        with patch.object(parallel, "ProcessPoolExecutor",
                          side_effect=AssertionError("Unexpected process")):
            rows = list(parallel.results(worker_probe, [2, 3], workers=1))
        self.assertEqual([i for i, _ in rows], [0, 1])
        self.assertEqual([r["value"] for _, r in rows], [2, 3])
        for _, row in rows:
            self.assertEqual(row["pid"], os.getpid())
            self.assertTrue(all(v == "1" for v in row["environment"].values()))
            self.assertTrue(all(v == 1 for v in row["native_threads"]))
        self.assertEqual(before, {n: os.environ.get(n) for n in parallel.THREAD_VARIABLES})

    def test_spawn_workers_use_child_processes_and_one_native_thread(self):
        before = {n: os.environ.get(n) for n in parallel.THREAD_VARIABLES}
        rows = list(parallel.results(worker_probe, range(4), workers=2))
        self.assertEqual(sorted(i for i, _ in rows), list(range(4)))
        for index, row in rows:
            self.assertEqual(row["value"], index)
            self.assertNotEqual(row["pid"], os.getpid())
            self.assertTrue(all(v == "1" for v in row["environment"].values()))
            self.assertTrue(all(v == 1 for v in row["native_threads"]))
        self.assertEqual(before, {n: os.environ.get(n) for n in parallel.THREAD_VARIABLES})

    def test_queue_is_bounded_and_consumer_close_joins_executor(self):
        submitted = []
        executor = MagicMock()

        def submit(function, job):
            submitted.append(job)
            future = Future()
            future.set_result(function(job))
            return future

        executor.submit.side_effect = submit
        before = {n: os.environ.get(n) for n in parallel.THREAD_VARIABLES}
        with patch.object(parallel, "ProcessPoolExecutor", return_value=executor) as create:
            with closing(parallel.results(abs, range(100), workers=2)) as stream:
                self.assertEqual(next(stream), (0, 0))
                self.assertEqual(len(submitted), 4)
            self.assertEqual(create.call_args.kwargs["max_workers"], 2)
            self.assertEqual(create.call_args.kwargs["mp_context"].get_start_method(), "spawn")
        executor.shutdown.assert_called_once_with(wait=True, cancel_futures=True)
        self.assertEqual(before, {n: os.environ.get(n) for n in parallel.THREAD_VARIABLES})

    def test_worker_failure_stops_submission_and_propagates(self):
        submitted = []
        executor = MagicMock()

        def submit(function, job):
            submitted.append(job)
            future = Future()
            if job == 0:
                future.set_exception(RuntimeError("synthetic worker failure"))
            else:
                future.set_result(job)
            return future

        executor.submit.side_effect = submit
        with patch.object(parallel, "ProcessPoolExecutor", return_value=executor):
            with self.assertRaisesRegex(RuntimeError, "synthetic worker failure"):
                list(parallel.results(abs, range(100), workers=2))
        self.assertEqual(len(submitted), 4)
        executor.shutdown.assert_called_once_with(wait=True, cancel_futures=True)

    def test_sequential_and_parallel_fits_produce_identical_checkpoints(self):
        cfg, frame = tiny_fitting_data()
        (io.HOME/"runs").mkdir(exist_ok=True)
        with tempfile.TemporaryDirectory(dir=io.HOME/"runs") as tmp, fitting_context(cfg, frame):
            first, second = Path(tmp)/"serial", Path(tmp)/"parallel"
            fitting.fit(first, workers=1)
            fitting.fit(second, workers=2)
            self.assertEqual(io.read(first/"fit_receipt.json"),
                             io.read(second/"fit_receipt.json"))
            names = sorted(p.name for p in (first/"checkpoints").glob("*.json"))
            self.assertEqual(len(names), 2)
            for name in names:
                self.assertEqual(io.read(first/"checkpoints"/name),
                                 io.read(second/"checkpoints"/name))

    def test_resume_with_another_worker_count_starts_no_completed_fit(self):
        cfg, frame = tiny_fitting_data()
        (io.HOME/"runs").mkdir(exist_ok=True)
        with tempfile.TemporaryDirectory(dir=io.HOME/"runs") as tmp, fitting_context(cfg, frame):
            fitting.fit(tmp, workers=1)
            before = io.sha(Path(tmp)/"fit_receipt.json")
            with patch.object(fitting, "results", side_effect=AssertionError("Unexpected refit")):
                fitting.fit(tmp, workers=2)
            self.assertEqual(before, io.sha(Path(tmp)/"fit_receipt.json"))

    def test_failure_preserves_checkpoint_without_publishing_final_receipt(self):
        cfg, frame = tiny_fitting_data()

        def interrupted(function, jobs, workers, initializer, initargs):
            initializer(*initargs)
            yield 0, function(jobs[0])
            raise RuntimeError("synthetic interruption")

        (io.HOME/"runs").mkdir(exist_ok=True)
        with tempfile.TemporaryDirectory(dir=io.HOME/"runs") as tmp, fitting_context(cfg, frame):
            with patch.object(fitting, "results", side_effect=interrupted):
                with self.assertRaisesRegex(RuntimeError, "synthetic interruption"):
                    fitting.fit(tmp, workers=2)
            saved = list((Path(tmp)/"checkpoints").glob("*.json"))
            self.assertEqual(len(saved), 1)
            before = io.sha(saved[0])
            self.assertFalse((Path(tmp)/"fit_receipt.json").exists())
            fitting.fit(tmp, workers=2)
            self.assertEqual(before, io.sha(saved[0]))
            self.assertTrue((Path(tmp)/"fit_receipt.json").exists())

    def test_corrupt_checkpoint_blocks_before_starting_any_process(self):
        cfg, frame = tiny_fitting_data()
        (io.HOME/"runs").mkdir(exist_ok=True)
        with tempfile.TemporaryDirectory(dir=io.HOME/"runs") as tmp, fitting_context(cfg, frame):
            identity, _, _ = next(fitting.tasks(frame, cfg))
            io.put(fitting.checkpoint_path(tmp, identity),
                   io.sealed(dict(manifest="other-run", identity=identity)))
            with patch.object(fitting, "results", side_effect=AssertionError("Unexpected process")):
                with self.assertRaisesRegex(ValueError, "provenance mismatch"):
                    fitting.fit(tmp, workers=2)

    def test_parallel_reporting_matches_original_tables_and_row_order(self):
        cfg = config()
        rows = []
        for budget in (5, 30):
            for i in range(12):
                for name, probability in (("size", .5), ("additive", .4), ("interactions", .6)):
                    rows.append(dict(endpoint="economic_5", budget=budget, family_control=False,
                        description=name, key=f"k{i}", group=f"g{i}", family=f"f{i%2}",
                        fold=i%2, y=i%2, p=probability))
        pred = pd.DataFrame(rows)
        base = pd.DataFrame([observation(i) for i in range(12)])[
            ["group"]+cfg["targeted_components"]]
        scores, cal = reporting.score_tables(pred, cfg)
        contrasts, folds, families = reporting.contrast_tables(pred, cfg)
        expected = (scores, cal, contrasts, folds, families,
                    reporting.surface_table(pred, base, cfg))
        for workers in (1, 2):
            actual = reporting.numerical_tables(pred, base, cfg, workers=workers)
            for left, right in zip(expected, actual):
                pd.testing.assert_frame_equal(left, right, check_exact=True)


if __name__ == "__main__":
    unittest.main()
