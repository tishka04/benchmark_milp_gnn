"""Batch scheduling and timing checks with fake solvers/processes only."""
import json
import copy
import tempfile
import unittest
from concurrent.futures import Future
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

from src.milp import batch_runner as batch


class BatchRunnerTests(unittest.TestCase):
    def setUp(self):
        # Any accidental real solve or process creation fails this test suite.
        self.solver_guard = patch.object(batch, "solve_scenario", side_effect=AssertionError("Real solver prohibited"))
        self.pool_guard = patch.object(batch, "ProcessPoolExecutor", side_effect=AssertionError("Real workers prohibited"))
        self.solver_guard.start()
        self.pool_guard.start()
        self.addCleanup(self.solver_guard.stop)
        self.addCleanup(self.pool_guard.stop)
        self.paths = [Path(f"scenario_{i:05}.json") for i in range(12)]

    def fake_executor(self, fail=False, worker_error=False):
        processes = [SimpleNamespace(is_alive=Mock(return_value=True), terminate=Mock()) for _ in range(4)]
        futures = []

        def submit(function, path):
            self.assertIs(function, batch._run_single)
            future = Future()
            futures.append(future)
            if worker_error and len(futures) == 1:
                future.set_exception(RuntimeError("worker disappeared"))
            elif fail and len(futures) == 1:
                future.set_result(batch.BatchResult(path, False, "synthetic failure"))
            elif not (fail or worker_error):
                future.set_result(batch.BatchResult(path, True, "ok"))
            return future

        return SimpleNamespace(_processes=dict(enumerate(processes)),
                               submit=Mock(side_effect=submit), shutdown=Mock()), futures, processes

    def test_fail_fast_cancels_pending_and_terminates_active_workers(self):
        executor, futures, processes = self.fake_executor(fail=True)
        results = []
        with patch.object(batch, "ProcessPoolExecutor", return_value=executor):
            batch._run_parallel(self.paths, 4, lambda p: (p,), results.append, True, 0)
        self.assertEqual(executor.submit.call_count, 4)  # Never enqueue the full cohort.
        self.assertEqual(len(results), 1)
        self.assertFalse(results[0].success)
        self.assertTrue(all(f.cancelled() for f in futures[1:]))
        for process in processes:
            process.terminate.assert_called_once()
        executor.shutdown.assert_called_once_with(wait=True, cancel_futures=True)

    def test_success_refills_at_most_four_slots_and_shuts_down_cleanly(self):
        executor, _, processes = self.fake_executor()
        results = []

        def record(result):
            self.assertLessEqual(executor.submit.call_count - len(results), 4)
            results.append(result)

        with patch.object(batch, "ProcessPoolExecutor", return_value=executor):
            batch._run_parallel(self.paths, 4, lambda p: (p,), record, True, 0)
        self.assertCountEqual([r.scenario for r in results], self.paths)
        executor.shutdown.assert_called_once_with(wait=True)
        for process in processes:
            process.terminate.assert_not_called()

    def test_worker_exception_is_failure_and_stops_other_workers(self):
        executor, _, processes = self.fake_executor(worker_error=True)
        results = []
        with patch.object(batch, "ProcessPoolExecutor", return_value=executor):
            batch._run_parallel(self.paths, 4, lambda p: (p,), results.append, True, 0)
        self.assertIn("worker disappeared", results[0].message)
        self.assertEqual(executor.submit.call_count, 4)
        for process in processes:
            process.terminate.assert_called_once()

    def test_keyboard_interrupt_also_terminates_workers(self):
        executor, _, processes = self.fake_executor()
        with patch.object(batch, "ProcessPoolExecutor", return_value=executor), \
                patch.object(batch, "wait", side_effect=KeyboardInterrupt):
            with self.assertRaises(KeyboardInterrupt):
                batch._run_parallel(self.paths, 4, lambda p: (p,), Mock(), True, 0)
        for process in processes:
            process.terminate.assert_called_once()

    def test_new_python_uses_public_termination_api(self):
        executor = SimpleNamespace(terminate_workers=Mock())
        batch._stop_executor(executor)
        executor.terminate_workers.assert_called_once_with()

    def test_per_scenario_wall_time_includes_more_than_mip(self):
        report = {"mip": SimpleNamespace(objective=10, solve_seconds=8),
                  "lp": SimpleNamespace(solve_seconds=3)}
        with patch.object(batch, "solve_scenario", return_value=report), \
                patch.object(batch.time, "perf_counter", side_effect=[10, 30]):
            result = batch._run_single(self.paths[0], "highs", False, False, None, None, None, False, None)
        self.assertEqual((result.solve_seconds, result.lp_solve_seconds, result.wall_seconds), (8, 3, 20))

    def test_failed_scenario_is_timed(self):
        with patch.object(batch, "solve_scenario", side_effect=RuntimeError("bad input")), \
                patch.object(batch.time, "perf_counter", side_effect=[10, 15]):
            result = batch._run_single(self.paths[0], "highs", False, False, None, None, None, False, None)
        self.assertFalse(result.success)
        self.assertEqual(result.wall_seconds, 5)

    def temporary_workspace(self):
        base = Path(__file__).resolve().parents[1] / "tmp"
        base.mkdir(exist_ok=True)
        temporary = tempfile.TemporaryDirectory(prefix="batch-test-", dir=base)
        self.assertTrue(Path(temporary.name).resolve().is_relative_to(base.resolve()))
        return temporary

    def test_summary_separates_wall_time_from_cumulative_solve_times(self):
        def fake_run(scenarios, workers, make_job, record, fail_fast, started):
            for path in scenarios:
                record(batch.BatchResult(path, True, "ok", objective=10,
                                         solve_seconds=100, lp_solve_seconds=20, wall_seconds=130))

        with self.temporary_workspace() as temp:
            summary_path = Path(temp) / "summary.json"
            with patch("sys.argv", ["batch", "unused", "--workers", "4", "--summary-json", str(summary_path)]), \
                    patch.object(batch, "_collect_scenarios", return_value=self.paths[:4]), \
                    patch.object(batch, "_run_parallel", side_effect=fake_run), \
                    patch.object(batch.time, "perf_counter", side_effect=[0, 100, 110, 120, 130, 140]), \
                    patch.dict("os.environ", {"BENCHMARK_HIGHS_THREADS": "4"}):
                batch.main()
            summary = json.loads(summary_path.read_text())
        self.assertEqual(summary["elapsed_wall_seconds"], 140)
        self.assertEqual(summary["solve_seconds_total"], 400)
        self.assertEqual(summary["lp_solve_seconds_total"], 80)
        self.assertEqual(summary["scenario_wall_seconds_total"], 520)
        self.assertEqual((summary["workers"], summary["highs_threads"]), (4, 4))
        self.assertEqual(summary["scenarios_unfinished"], 0)

    def test_fail_fast_writes_partial_summary_and_exits_nonzero(self):
        executor, _, _ = self.fake_executor(fail=True)
        with self.temporary_workspace() as temp:
            summary_path = Path(temp) / "summary.json"
            # Fake submit only needs the scenario, not the solver/export options.
            original_submit = executor.submit.side_effect
            executor.submit.side_effect = lambda fn, path, *args: original_submit(fn, path)
            with patch("sys.argv", ["batch", "unused", "--workers", "4", "--fail-fast", "--summary-json", str(summary_path)]), \
                    patch.object(batch, "_collect_scenarios", return_value=self.paths), \
                    patch.object(batch, "ProcessPoolExecutor", return_value=executor):
                with self.assertRaises(SystemExit) as raised:
                    batch.main()
            summary = json.loads(summary_path.read_text())
        self.assertEqual(raised.exception.code, 1)
        self.assertEqual(summary["status"], "failed")
        self.assertEqual(summary["scenarios_unfinished"], 11)
        self.assertEqual(summary["failure_count"], 1)

    def test_interruption_writes_timing_before_propagating(self):
        with self.temporary_workspace() as temp:
            summary_path = Path(temp) / "summary.json"
            with patch("sys.argv", ["batch", "unused", "--workers", "4", "--summary-json", str(summary_path)]), \
                    patch.object(batch, "_collect_scenarios", return_value=self.paths), \
                    patch.object(batch, "_run_parallel", side_effect=KeyboardInterrupt):
                with self.assertRaises(KeyboardInterrupt):
                    batch.main()
            summary = json.loads(summary_path.read_text())
        self.assertEqual(summary["status"], "interrupted")
        self.assertEqual(summary["scenarios_unfinished"], 12)
        self.assertGreaterEqual(summary["elapsed_wall_seconds"], 0)

    def synthetic_report(self, termination="optimal"):
        """Feasible two-period shift; no optimization model or solver involved."""
        payload = {
            "scenario_id": "synthetic", "dr_formulation": batch.DR_FORMULATION,
            "mip": {"objective": 10.0, "termination": termination,
                    "status": "ok" if termination == "optimal" else "aborted", "solve_seconds": 8.0},
            "lp": {"objective": 9.0, "termination": "optimal", "status": "ok", "solve_seconds": 3.0},
            "periods": [0, 1], "zones": ["zone"], "cost_components": {"thermal": 10.0}, "lp_duals": {},
            "detail": {"time_steps": [0, 1], "zones": ["zone"], "dt_hours": 1.0,
                       "demand_response": {"zone": [1.0, 0.0]}, "dr_recover": {"zone": [0.0, 1.0]},
                       "dr_rebound": {"zone": [1.0, 0.0]}, "dr_active": {"zone": [1.0, 0.0]}},
        }
        report = copy.deepcopy(payload)
        for kind in ("mip", "lp"):
            report[kind] = SimpleNamespace(**payload[kind], as_dict=Mock(return_value=payload[kind]))
        return report

    def create_checkpoint(self, scenario, destination, termination="optimal"):
        scenario.write_text('{"id": "synthetic"}', encoding="utf-8")
        with patch.object(batch, "solve_scenario", return_value=self.synthetic_report(termination)):
            result = batch._run_single(scenario, "highs", False, True, destination, None, None, False, None)
        self.assertTrue(result.success, result.message)
        return json.loads(destination.read_text(encoding="utf-8"))

    def test_resume_retains_gold_and_feasible_silver_and_schedules_only_missing_or_bad(self):
        for workers in (1, 4):
            with self.subTest(workers=workers), self.temporary_workspace() as temp:
                root = Path(temp)
                reports = root / "reports"
                reports.mkdir()
                paths = [root / f"scenario_{i:05}.json" for i in range(1, 6)]
                for path in paths:
                    path.write_text('{"id": "synthetic"}', encoding="utf-8")
                for path, status in zip(paths[:2], ("optimal", "maxTimeLimit")):
                    self.create_checkpoint(path, reports / path.name, status)
                preserved = [(reports / p.name).read_bytes() for p in paths[:2]]
                (reports / paths[3].name).write_text('{"mip":', encoding="utf-8")  # Interrupted old write.
                (reports / paths[4].name).write_text('{"mip": {"objective": null}}', encoding="utf-8")
                attempted = []

                def fake_single(path, *args):
                    attempted.append(path)
                    return batch.BatchResult(path, True, "ok", objective=10,
                                             solve_seconds=5, lp_solve_seconds=2, wall_seconds=8)

                def fake_parallel(scenarios, workers, make_job, record, *args):
                    for path in scenarios:
                        record(fake_single(*make_job(path)))

                summary_path = root / "summary.json"
                with patch("sys.argv", ["batch", str(root), "--resume", "--workers", str(workers),
                                        "--reports-dir", str(reports), "--summary-json", str(summary_path)]), \
                        patch.object(batch, "_run_single", side_effect=fake_single), \
                        patch.object(batch, "_run_parallel", side_effect=fake_parallel):
                    batch.main()
                self.assertEqual(attempted, paths[2:])
                self.assertEqual(preserved, [(reports / p.name).read_bytes() for p in paths[:2]])
                summary = json.loads(summary_path.read_text())
                self.assertEqual((summary["reused_count"], summary["executed_count"], summary["success_count"]), (2, 3, 5))
                self.assertEqual(summary["scenarios_unfinished"], 0)
                self.assertEqual(summary["solve_seconds_total"], 31)
                self.assertEqual(summary["executed_solve_seconds_total"], 15)
                self.assertEqual(summary["lp_solve_seconds_total"], 12)
                self.assertEqual(summary["executed_lp_solve_seconds_total"], 6)
                self.assertEqual(summary["scenario_wall_seconds_total"], 24)

    def test_all_complete_resume_needs_neither_summary_nor_workers(self):
        with self.temporary_workspace() as temp:
            root = Path(temp)
            scenario, reports = root / "scenario_00001.json", root / "reports"
            self.create_checkpoint(scenario, reports / scenario.name)
            with patch("sys.argv", ["batch", str(root), "--resume", "--workers", "4", "--reports-dir", str(reports)]), \
                    patch.object(batch, "_run_single") as single, patch.object(batch, "_run_parallel") as parallel:
                batch.main()
            single.assert_not_called()
            parallel.assert_not_called()

    def test_resume_rejects_incompatible_incomplete_or_failed_reports(self):
        with self.temporary_workspace() as temp:
            scenario, destination = Path(temp) / "scenario_00001.json", Path(temp) / "report.json"
            original = self.create_checkpoint(scenario, destination)
            mutations = {
                "old report without provenance": lambda p: p.pop("batch_checkpoint"),
                "incomplete checkpoint": lambda p: p["batch_checkpoint"].update(complete=False),
                "different threads": lambda p: p["batch_checkpoint"]["provenance"].update(highs_threads=999),
                "different solver": lambda p: p["batch_checkpoint"]["provenance"].update(solver="other"),
                "changed formulation code": lambda p: p["batch_checkpoint"]["provenance"].update(solve_code_sha256="old"),
                "nonfinite objective": lambda p: p["mip"].update(objective=float("inf")),
                "failed MIP": lambda p: p["mip"].update(termination="infeasible"),
                "failed LP": lambda p: p["lp"].update(status="error"),
                "legacy shedding": lambda p: p.update(dr_formulation="shedding"),
                "missing recovery": lambda p: p["detail"].pop("dr_recover"),
                "incomplete horizon": lambda p: p["detail"].update(time_steps=[0]),
                "edited cost": lambda p: p["cost_components"].update(thermal=99),
            }
            for name, mutate in mutations.items():
                with self.subTest(name=name):
                    payload = copy.deepcopy(original)
                    mutate(payload)
                    destination.write_text(json.dumps(payload), encoding="utf-8")
                    cached, reason = batch._read_completed_result(scenario, destination, "highs")
                    self.assertIsNone(cached)
                    self.assertIn("invalid/incompatible", reason)
            destination.write_text(json.dumps(original), encoding="utf-8")
            scenario.write_text('{"id": "modified"}', encoding="utf-8")
            self.assertIsNone(batch._read_completed_result(scenario, destination, "highs")[0])

    def test_failed_or_incomplete_solve_does_not_publish_a_checkpoint(self):
        for case in ("no incumbent", "LP error", "missing recovery"):
            with self.subTest(case=case), self.temporary_workspace() as temp:
                scenario, destination = Path(temp) / "scenario_00001.json", Path(temp) / "report.json"
                scenario.write_text("{}", encoding="utf-8")
                report = self.synthetic_report()
                if case == "no incumbent":
                    report["mip"].objective = float("inf")
                elif case == "LP error":
                    report["lp"].as_dict.return_value["status"] = "error"
                else:
                    del report["detail"]["dr_recover"]
                with patch.object(batch, "solve_scenario", return_value=report):
                    result = batch._run_single(scenario, "highs", False, True, destination, None, None, False, None)
                self.assertFalse(result.success)
                self.assertFalse(destination.exists())

    def test_interrupted_write_preserves_previous_report(self):
        with self.temporary_workspace() as temp:
            destination = Path(temp) / "report.json"
            destination.write_text("previous report", encoding="utf-8")

            def interrupted(payload, stream, **kwargs):
                stream.write('{"partial":')
                raise KeyboardInterrupt()

            with patch.object(batch.json, "dump", side_effect=interrupted):
                with self.assertRaises(KeyboardInterrupt):
                    batch._write_json_atomic(destination, {"complete": True})
            self.assertEqual(destination.read_text(encoding="utf-8"), "previous report")
            self.assertEqual(list(Path(temp).glob("*.tmp")), [])


if __name__ == "__main__":
    unittest.main()
