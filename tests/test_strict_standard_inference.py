"""Deadline checks use fake clocks/processes: never run a model or solver."""
import json
import math
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from scripts.run_strict_standard_inference import (
    FORMULATION, result_row, run_method, run_pair,
)


class FakeClock:
    def __init__(self):
        self.t = 0.0

    def __call__(self):
        return self.t

    def sleep(self, duration):
        self.t += duration


def publish(job, sequence=1, objective=7):
    path = Path(job["snapshot_file"]).with_name(f"snapshot-{sequence:06d}.json")
    path.write_text(json.dumps({
        "sequence": sequence, "method": job["method"],
        "row": {"scenario_id": job["scenario_id"], "dr_formulation": FORMULATION,
                "success": True, "lp_objective": objective, "time_total": 0.02},
    }), encoding="utf-8")


class StrictInferenceTests(unittest.TestCase):
    def setUp(self):
        self.guard = patch("subprocess.Popen", side_effect=AssertionError("Real processes prohibited"))
        self.guard.start()
        base = Path(__file__).resolve().parents[1] / "tmp"
        base.mkdir(exist_ok=True)
        self.temp = tempfile.TemporaryDirectory(prefix="strict-inference-test-", dir=base)
        self.assertTrue(Path(self.temp.name).resolve().is_relative_to(base.resolve()))
        self.clock = FakeClock()
        self.common = dict(threads=4, seed=42, k=5, scenario_id="scenario_001", method="pipeline")

    def tearDown(self):
        self.guard.stop()
        self.temp.cleanup()

    def run_worker(self, factory, budget=0.2):
        return run_method(self.common, self.temp.name, budget, popen=factory,
                          clock=self.clock, sleep=self.clock.sleep)

    def test_completed_result_includes_process_startup_in_time(self):
        test = self

        class Process:
            def __init__(self, cmd, **kwargs):
                job = json.loads(Path(cmd[-1]).read_text())
                test.assertEqual(job["deadline"], 0.2)
                test.assertEqual(kwargs["env"]["BENCHMARK_HIGHS_THREADS"], "4")
                test.assertEqual(kwargs["env"]["OMP_NUM_THREADS"], "1")
                test.clock.t += 0.05
                publish(job)
                Path(job["done_file"]).write_text('{"status":"complete"}')

            def poll(self):
                return 0

        outcome = self.run_worker(Process)
        self.assertFalse(outcome["hard_timeout"])
        self.assertEqual(outcome["row"]["lp_objective"], 7)
        self.assertEqual(outcome["decision_wall_s"], 0.05)

    def test_timeout_keeps_completed_candidate_and_rejects_late_improvement(self):
        processes = []

        class Process:
            code = None

            def __init__(self, cmd, **kwargs):
                self.job = json.loads(Path(cmd[-1]).read_text())
                publish(self.job, objective=9)
                processes.append(self)

            def poll(self):
                return self.code

            def kill(self):
                publish(self.job, sequence=2, objective=1)  # output during cleanup is too late
                self.code = -9

            def wait(self, timeout):
                return self.code

        outcome = self.run_worker(Process)
        self.assertEqual(processes[0].code, -9)
        self.assertTrue(outcome["hard_timeout"])
        self.assertEqual(outcome["row"]["lp_objective"], 9)
        self.assertEqual(outcome["decision_wall_s"], 0.2)
        self.assertLess(outcome["snapshot_available_after_s"], 0.2)

    def test_latest_on_time_snapshot_is_selected(self):
        clock = self.clock

        class Process:
            code = None
            updated = False

            def __init__(self, cmd, **kwargs):
                self.job = json.loads(Path(cmd[-1]).read_text())
                publish(self.job, objective=9)

            def poll(self):
                if clock.t > 0.05 and not self.updated:
                    self.updated = True
                    publish(self.job, sequence=2, objective=8)
                return self.code

            def kill(self):
                self.code = -9

            def wait(self, timeout):
                return self.code

        outcome = self.run_worker(Process)
        self.assertEqual(outcome["row"]["lp_objective"], 8)
        self.assertGreater(outcome["snapshot_available_after_s"], 0.05)

    def test_result_after_slow_startup_is_rejected_even_when_worker_finished(self):
        clock = self.clock

        class Process:
            def __init__(self, cmd, **kwargs):
                job = json.loads(Path(cmd[-1]).read_text())
                clock.t += 0.3
                publish(job)
                Path(job["done_file"]).write_text('{"status":"complete"}')

            def poll(self):
                return 0

        outcome = self.run_worker(Process)
        self.assertTrue(outcome["hard_timeout"])
        self.assertIsNone(outcome["row"])
        self.assertEqual(outcome["decision_wall_s"], 0.2)
        self.assertAlmostEqual(outcome["deadline_detection_lag_s"], 0.1)

    def test_worker_error_stops_instead_of_becoming_infeasibility(self):
        class Process:
            def __init__(self, *args, **kwargs):
                pass

            def poll(self):
                return 1

        with self.assertRaisesRegex(RuntimeError, "Worker failed"):
            self.run_worker(Process)

    def test_jepa_gets_its_full_budget_after_pipeline_timeout(self):
        calls = []

        def fake_runner(job, directory, budget):
            calls.append((job["method"], self.clock.t, budget))
            self.clock.t += budget
            return {"row": None, "hard_timeout": True}

        pair = run_pair(self.common, self.temp.name, 120, 120, runner=fake_runner)
        self.assertEqual(calls, [("pipeline", 0, 120), ("jepa", 120, 120)])
        self.assertEqual(set(pair), {"pipeline", "jepa"})

    def test_exports_preserve_failures_partial_success_and_external_time(self):
        base = dict(scenario_id="s", dr_formulation=FORMULATION, family="low",
                    success=False, lp_objective=math.inf, time_total=0.0, gap_pct=math.nan)
        outcome = dict(row=None, hard_timeout=True, budget_seconds=120,
                       decision_wall_s=120, cleanup_wall_s=0.1,
                       deadline_detection_lag_s=0.001, snapshot_available_after_s=None)
        failed = result_row(outcome, base, method="pipeline", k=5)
        direct = result_row(outcome, base, method="jepa", k=1)
        self.assertFalse(failed["success"])
        self.assertTrue(math.isinf(failed["lp_objective"]))
        self.assertEqual(failed["lp_status"], "hard_timeout")
        self.assertEqual(direct["total_time_s"], 120)
        self.assertFalse(direct["success"])
        outcome["row"] = dict(base, success=True, lp_objective=10, time_total=12)
        partial = result_row(outcome, base, method="pipeline", k=5)
        self.assertTrue(partial["success"])
        self.assertTrue(partial["hard_timeout"])
        self.assertEqual(partial["time_total"], 120)
        self.assertEqual(partial["compute_time_before_snapshot_s"], 12)
        self.assertEqual(partial["requested_candidates"], 5)

    def test_zero_or_unbounded_budgets_are_rejected(self):
        for budget in (0, -1, math.inf, math.nan):
            with self.assertRaises(ValueError):
                self.run_worker(None, budget)


if __name__ == "__main__":
    unittest.main()
