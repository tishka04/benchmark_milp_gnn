"""Deadline checks use fake clocks/processes: never run a model or solver."""
import json
import math
import copy
import os
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from scripts.run_strict_standard_inference import (
    COMPATIBLE_RUNNER_HASHES, FORMULATION, RUNNER_SOURCE, completed_cases,
    result_row, run_campaign, run_method, run_pair, validate_resume_manifest,
    validate_worker_path,
)
from scripts.benchmark_parallel_best_of_k import atomic_json


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

    def finished_worker(self):
        """Two immutable results and a completed marker, without a real process."""
        class Process:
            def __init__(self, cmd, **kwargs):
                job = json.loads(Path(cmd[-1]).read_text())
                publish(job, sequence=1, objective=9)
                publish(job, sequence=2, objective=8)
                Path(job["done_file"]).write_text('{"status":"complete"}')

            def poll(self):
                return 0

        return Process

    def test_temporary_snapshot_lock_retries_and_waits_for_final_result_after_exit(self):
        original_read = Path.read_text
        observations = []

        def intermittently_locked(path, *args, **kwargs):
            if path.name.startswith("snapshot-"):
                observations.append((path.name, self.clock.t))
            if path.name == "snapshot-000002.json" and self.clock.t < 0.06:
                raise PermissionError(13, "Simulated Windows sharing violation", str(path))
            return original_read(path, *args, **kwargs)

        with patch.object(Path, "read_text", intermittently_locked):
            outcome = self.run_worker(self.finished_worker())
        self.assertFalse(outcome["hard_timeout"])
        self.assertEqual(outcome["row"]["lp_objective"], 8)
        self.assertGreaterEqual(outcome["snapshot_available_after_s"], 0.06)
        self.assertGreater(outcome["file_access_retries"], 0)
        self.assertIn(("snapshot-000001.json", 0.0), observations)

    def test_lock_at_deadline_preserves_older_result_and_never_reads_late(self):
        original_read = Path.read_text
        read_times = []

        def locked_latest(path, *args, **kwargs):
            if path.name.startswith("snapshot-"):
                read_times.append(self.clock.t)
            if path.name == "snapshot-000002.json":
                raise PermissionError(13, "Simulated lock", str(path))
            return original_read(path, *args, **kwargs)

        with patch.object(Path, "read_text", locked_latest):
            outcome = self.run_worker(self.finished_worker())
        self.assertTrue(outcome["hard_timeout"])
        self.assertEqual(outcome["row"]["lp_objective"], 9)
        self.assertEqual(outcome["snapshot_available_after_s"], 0.0)
        self.assertEqual(outcome["decision_wall_s"], 0.2)
        self.assertTrue(all(t < 0.2 for t in read_times))

    def test_done_marker_also_retries_transient_permissions(self):
        original_read = Path.read_text

        def locked_done(path, *args, **kwargs):
            if path.name == "done.json" and self.clock.t < 0.06:
                raise PermissionError(13, "Simulated lock", str(path))
            return original_read(path, *args, **kwargs)

        with patch.object(Path, "read_text", locked_done):
            outcome = self.run_worker(self.finished_worker())
        self.assertFalse(outcome["hard_timeout"])
        self.assertGreaterEqual(outcome["decision_wall_s"], 0.06)
        self.assertEqual(outcome["row"]["lp_objective"], 8)
        self.assertGreater(outcome["file_access_retries"], 0)

    def test_persistent_lock_is_explicit_io_error_and_cleans_up_worker(self):
        original_read = Path.read_text
        processes = []

        class Process(self.finished_worker()):
            code = None

            def __init__(self, *args, **kwargs):
                super().__init__(*args, **kwargs)
                processes.append(self)

            def poll(self):
                return self.code

            def kill(self):
                self.code = -9

            def wait(self, timeout):
                return self.code

        def locked(path, *args, **kwargs):
            if path.name.startswith("snapshot-"):
                raise PermissionError(13, "Simulated permanent access denial", str(path))
            return original_read(path, *args, **kwargs)

        with patch.object(Path, "read_text", locked):
            with self.assertRaisesRegex(PermissionError, "remains inaccessible.*snapshot-"):
                self.run_worker(Process, budget=120)
        self.assertGreaterEqual(self.clock.t, 2.0)
        self.assertLess(self.clock.t, 2.1)
        self.assertEqual(processes[0].code, -9)

    def test_corrupt_or_mismatched_worker_data_remains_fatal(self):
        original_read = Path.read_text
        bad_data = ["{truncated", "null", json.dumps({"method": "wrong"})]
        for text in bad_data:
            with self.subTest(text=text):
                # Give each synthetic run its own output directory.
                directory = Path(self.temp.name) / str(len(text))

                def corrupt(path, *args, **kwargs):
                    if path.name == "snapshot-000002.json":
                        return text
                    return original_read(path, *args, **kwargs)

                with patch.object(Path, "read_text", corrupt):
                    with self.assertRaises(ValueError):
                        run_method(self.common, directory, 0.2, popen=self.finished_worker(),
                                   clock=self.clock, sleep=self.clock.sleep)

    def test_slow_successful_read_that_crosses_deadline_is_not_accepted(self):
        original_read = Path.read_text

        def slow_read(path, *args, **kwargs):
            if path.name == "snapshot-000002.json":
                self.clock.t = 0.21
            return original_read(path, *args, **kwargs)

        with patch.object(Path, "read_text", slow_read):
            outcome = self.run_worker(self.finished_worker())
        self.assertTrue(outcome["hard_timeout"])
        self.assertIsNone(outcome["row"])
        self.assertEqual(outcome["decision_wall_s"], 0.2)

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


class ResumeTests(unittest.TestCase):
    def setUp(self):
        self.guard = patch("subprocess.Popen", side_effect=AssertionError("Real processes prohibited"))
        self.guard.start()
        base = Path(__file__).resolve().parents[1] / "tmp"
        base.mkdir(exist_ok=True)
        self.temp = tempfile.TemporaryDirectory(prefix="resume-", dir=base)
        self.out = Path(self.temp.name) / "eval"
        self.family = "persona_network_operator_scenarios"
        self.common = dict(root=self.temp.name, scenario_id="scenario_00001", threads=16, seed=42,
                           k=5, scenario_root=str(Path(self.temp.name) / self.family))
        self.args = SimpleNamespace(root=self.temp.name, output_dir=str(self.out),
                                    budget=120, jepa_budget=120, k=5, threads=16, resume=True)
        self.manifest = dict(root=self.temp.name, cohort=[dict(family=self.family,
                             scenario_id="scenario_00001")], inputs_sha256={"model.pt": "modelhash"},
                             source_sha256={RUNNER_SOURCE: "current", "src/model.py": "physics"},
                             budget_seconds=120, jepa_budget_seconds=120)

    def tearDown(self):
        self.guard.stop()
        self.temp.cleanup()

    def outcome(self, success=True, timeout=False):
        return dict(row=dict(scenario_id="scenario_00001", family=self.family,
                             dr_formulation=FORMULATION, success=success,
                             lp_objective=7 if success else math.inf),
                    hard_timeout=timeout, budget_seconds=120,
                    decision_wall_s=120 if timeout else 10, cleanup_wall_s=0,
                    deadline_detection_lag_s=0, snapshot_available_after_s=9)

    def save_pair(self, pair=None):
        path = self.out / "cases" / self.family / "scenario_00001.json"
        atomic_json(path, pair or dict(pipeline=self.outcome(), jepa=self.outcome()))
        return path

    def test_known_orchestration_update_is_compatible(self):
        previous = copy.deepcopy(self.manifest)
        previous["source_sha256"][RUNNER_SOURCE] = next(iter(COMPATIBLE_RUNNER_HASHES))
        validate_resume_manifest(previous, self.manifest)
        self.assertNotEqual(previous, self.manifest)  # No mutation to hide the revision.

    def test_changed_budget_model_physics_or_unknown_runner_are_rejected(self):
        for field, key in (("inputs_sha256", "model.pt"), ("source_sha256", "src/model.py"),
                           ("source_sha256", RUNNER_SOURCE), (None, "budget_seconds")):
            with self.subTest(field=field, key=key):
                previous = copy.deepcopy(self.manifest)
                (previous[field] if field else previous)[key] = "changed"
                with self.assertRaisesRegex(ValueError, "Incompatible"):
                    validate_resume_manifest(previous, self.manifest)

    def test_completed_infeasible_and_timeout_results_are_reused(self):
        timeout = self.outcome(False, True)
        timeout.update(row=None, snapshot_available_after_s=None)
        self.save_pair(dict(pipeline=self.outcome(False), jepa=timeout))
        self.assertEqual(completed_cases(self.out, self.manifest, self.args),
                         {(self.family, "scenario_00001")})

    def test_corrupt_wrong_identity_or_late_checkpoint_is_not_silently_skipped(self):
        for change in ("corrupt", "wrong_family", "late", "budget", "missing_method"):
            with self.subTest(change=change):
                path = self.save_pair()
                pair = json.loads(path.read_text())
                if change == "corrupt":
                    path.write_text("{broken")
                else:
                    if change == "wrong_family":
                        pair["pipeline"]["row"]["family"] = "wrong"
                    elif change == "late":
                        pair["pipeline"]["snapshot_available_after_s"] = 120
                    elif change == "budget":
                        pair["pipeline"]["budget_seconds"] = 240
                    else:
                        del pair["jepa"]
                    atomic_json(path, pair)
                with self.assertRaisesRegex(ValueError, "Invalid completed case"):
                    completed_cases(self.out, self.manifest, self.args)

    def test_resume_skips_legacy_completed_pairs_and_preserves_original_files(self):
        original = copy.deepcopy(self.manifest)
        original["source_sha256"][RUNNER_SOURCE] = next(iter(COMPATIBLE_RUNNER_HASHES))
        atomic_json(self.out / "manifest.json", original)
        path = self.save_pair()
        before = path.read_bytes(), (self.out / "manifest.json").read_bytes()
        with patch("scripts.run_strict_standard_inference.build_manifest", return_value=self.manifest), \
             patch("scripts.run_strict_standard_inference.run_pair", side_effect=AssertionError("No rerun")), \
             patch("scripts.run_strict_standard_inference.export_results") as export:
            run_campaign(self.args)
        export.assert_called_once()
        self.assertEqual(before, (path.read_bytes(), (self.out / "manifest.json").read_bytes()))
        self.assertEqual(len(list((self.out / "resumes").glob("*.json"))), 1)

    def test_resume_runs_only_missing_pair_and_rebuilds_aggregate(self):
        self.manifest["cohort"].append(dict(family=self.family, scenario_id="scenario_00002"))
        atomic_json(self.out / "manifest.json", self.manifest)
        self.save_pair()
        pair = dict(pipeline=self.outcome(), jepa=self.outcome())
        for outcome in pair.values():
            outcome["row"]["scenario_id"] = "scenario_00002"
        with patch("scripts.run_strict_standard_inference.build_manifest", return_value=self.manifest), \
             patch("scripts.run_strict_standard_inference.run_pair", return_value=pair) as runner, \
             patch("scripts.run_strict_standard_inference.export_results") as export:
            run_campaign(self.args)
        runner.assert_called_once()
        self.assertEqual(runner.call_args.args[0]["scenario_id"], "scenario_00002")
        self.assertEqual(runner.call_args.args[1], self.out / "jobs/0002")
        self.assertTrue(runner.call_args.kwargs["checkpoint"])
        export.assert_called_once()
        self.assertEqual(len(completed_cases(self.out, self.manifest, self.args)), 2)

    def test_jepa_crash_preserves_pipeline_and_uses_fresh_attempt_on_resume(self):
        calls = []
        work = self.out / "jobs/0401"

        def crash(job, directory, budget):
            calls.append(job["method"])
            (directory / "snapshot-000001.json").write_text("stale interrupted output")
            if job["method"] == "jepa":
                raise RuntimeError("Simulated JEPA crash")
            return self.outcome()

        with self.assertRaisesRegex(RuntimeError, "Simulated JEPA"):
            run_pair(self.common, work, 120, 120, runner=crash, checkpoint=True)
        old_snapshot = work / "jepa/a001/snapshot-000001.json"

        def finish(job, directory, budget):
            calls.append(job["method"])
            self.assertEqual(directory, work / "jepa/a002")
            self.assertFalse(list(directory.glob("snapshot-*.json")))
            return self.outcome()

        pair = run_pair(self.common, work, 120, 120, runner=finish, checkpoint=True)
        self.assertEqual(calls, ["pipeline", "jepa", "jepa"])
        self.assertEqual(old_snapshot.read_text(), "stale interrupted output")
        self.assertEqual(pair["pipeline"]["decision_wall_s"], 10)
        # Both method checkpoints can reconstruct a pair after a parent interruption.
        self.assertEqual(pair, run_pair(self.common, work, 120, 120,
                                       runner=lambda *_: self.fail("No rerun"), checkpoint=True))

    def test_method_checkpoint_with_changed_identity_is_rejected(self):
        work = self.out / "jobs/0401"
        run_pair(self.common, work, 120, 120, runner=lambda *_: self.outcome(), checkpoint=True)
        with self.assertRaisesRegex(ValueError, "identity mismatch"):
            run_pair(dict(self.common, seed=99), work, 120, 120,
                     runner=lambda *_: self.fail("No rerun"), checkpoint=True)

    def test_nonempty_outputs_without_manifest_are_not_overwritten(self):
        path = self.save_pair()
        before = path.read_bytes()
        with patch("scripts.run_strict_standard_inference.build_manifest", return_value=self.manifest):
            with self.assertRaisesRegex(ValueError, "without a manifest"):
                run_campaign(self.args)
        self.assertEqual(path.read_bytes(), before)

    def test_existing_manifest_requires_explicit_resume(self):
        atomic_json(self.out / "manifest.json", self.manifest)
        self.args.resume = False
        with patch("scripts.run_strict_standard_inference.build_manifest", return_value=self.manifest):
            with self.assertRaisesRegex(ValueError, "use --resume"):
                run_campaign(self.args)

    def test_long_family_summary_can_be_written_using_compact_worker_layout(self):
        work = self.out / "jobs/0401"
        validate_worker_path(work, self.family)

        def write_export(job, directory, budget):
            path = (directory / "artifacts/family_results"
                    / f"direct_baseline_family_{self.family}_summary.json")
            self.assertLess(len(str(path.resolve())), 260)
            path.parent.mkdir(parents=True)
            path.write_text("{}")
            self.assertEqual(path.read_text(), "{}")
            return self.outcome()

        run_pair(self.common, work, 120, 120, runner=write_export, checkpoint=True)

    @unittest.skipUnless(os.name == "nt", "Windows path limit guard")
    def test_overlong_output_root_fails_before_any_worker(self):
        with self.assertRaisesRegex(ValueError, "MAX_PATH"):
            validate_worker_path(self.out / ("x" * 160), self.family)


if __name__ == "__main__":
    unittest.main()
