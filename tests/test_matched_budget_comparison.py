"""No solves, training or inference: fake processes and tiny synthetic matrices."""
import copy
import json
import math
import os
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace as NS
import unittest
from unittest.mock import patch

from scripts import benchmark_matched_budgets as bench
from scripts.feasible_candidate_selection import corrected_selection, select_sample_index
from scripts.matched_budget_milp import linear_validator, solve_streaming


class Clock:
    def __init__(self): self.t = 0.0
    def __call__(self): return self.t
    def sleep(self, duration): self.t += duration


def row(cost=5, kind="repair", success=True, energy=2):
    return dict(scenario_id="scenario_00001", family="low", dr_formulation=bench.FORMULATION,
                kind=kind, lp_objective=cost, success=success, energy=energy,
                dr_metrics={"dr_feasible": success}, slack_mwh=0, status="optimal",
                max_constraint_violation=0)


def publish(job, result, sequence=1):
    path = Path(job["snapshot_file"]).with_name(f"snapshot-{sequence:06d}.json")
    bench.atomic_json(path, dict(method=job["method"], kind=job["kind"],
                                sequence=sequence, row=result))


class SelectionTests(unittest.TestCase):
    def test_invalid_minimum_energy_cannot_hide_feasible_dispatch(self):
        self.assertEqual(select_sample_index("energy", [1, 3, 2], [math.inf, 20, 50]), 2)
        # Energy ranking is preserved even when the other feasible cost is cheaper.
        self.assertNotEqual(select_sample_index("energy", [1, 3, 2], [math.inf, 20, 50]), 1)

    def test_all_invalid_remains_failure_and_nan_energy_does_not_discard_success(self):
        self.assertEqual(select_sample_index("energy", [1, 2], [math.inf, math.nan]), 0)
        self.assertEqual(select_sample_index("energy", [0, math.nan], [math.inf, 5]), 1)
        self.assertEqual(select_sample_index("energy", [2, 2], [5, 5]), 0)
        with self.assertRaises(ValueError): select_sample_index("energy", [1], [])

    def test_parent_also_filters_failure_flags_and_dr_violations(self):
        invalid = row(1, energy=0)
        invalid["dr_metrics"]["dr_feasible"] = False
        chosen = bench.select_repairs([invalid, row(5, energy=3), row(10, energy=2)])
        self.assertEqual(chosen["best_sample_idx"], 2)
        self.assertEqual(chosen["lp_objective"], 10)
        self.assertTrue(chosen["success"])
        self.assertFalse(bench.select_repairs([None, invalid])["success"])
        with self.assertRaises(ValueError): bench.select_repairs([])

    def test_process_local_adapter_restores_original_even_on_error(self):
        original = lambda *_: 0
        module = NS(_select_sample_index=original)
        with self.assertRaisesRegex(RuntimeError, "fake"):
            with corrected_selection(module):
                self.assertEqual(module._select_sample_index("energy", [0, 3], [math.inf, 5]), 1)
                raise RuntimeError("fake")
        self.assertIs(module._select_sample_index, original)


class WorkerFixture(unittest.TestCase):
    def setUp(self):
        self.guard = patch("subprocess.Popen", side_effect=AssertionError("Real workers prohibited"))
        self.guard.start()
        base = Path(__file__).resolve().parents[1] / "tmp"
        base.mkdir(exist_ok=True)
        self.temp = tempfile.TemporaryDirectory(prefix="mb-", dir=base)
        self.directory = Path(self.temp.name)
        self.clock = Clock()
        self.common = dict(scenario_id="scenario_00001", family_alias="low", seed=42, threads=16)
        self.job = dict(self.common, method="milp", kind="milp")

    def tearDown(self):
        self.guard.stop()
        self.temp.cleanup()

    def phase(self, factory, *, jobs=None, reducer=None, budget=.2):
        return bench.run_phase(jobs or [self.job], self.directory, 0, budget,
                               popen=factory, clock=self.clock, sleep=self.clock.sleep, reducer=reducer)

    def finished(self):
        test = self
        class Process:
            def __init__(self, cmd, **kwargs):
                test.assertIn("scripts.benchmark_matched_budgets", cmd)
                self.job = json.loads(Path(cmd[-1]).read_text())
                self.returncode = 0
                publish(self.job, row(kind=self.job["kind"]))
                bench.atomic_json(self.job["done_file"], {"status": "complete"})
            def poll(self): return self.returncode
            def wait(self, timeout): return self.returncode
            def kill(self): self.returncode = -9
        return Process

class ProcessTests(WorkerFixture):
    def test_all_six_budgets_are_positive_and_exact(self):
        self.assertEqual(bench.parse_budgets("1,2,5,10,60,120"), list(bench.DEFAULT_BUDGETS))
        for bad in ("", "0,1", "nan", "inf", "-2"):
            with self.assertRaises(ValueError): bench.parse_budgets(bad)

    def test_parent_preserves_streamed_incumbent_when_solver_is_killed(self):
        processes = []
        class Process(self.finished()):
            def __init__(self, *args, **kwargs):
                super().__init__(*args, **kwargs)
                self.returncode = None
                processes.append(self)
            def kill(self):
                publish(self.job, row(cost=1, kind="milp"), sequence=2)
                self.returncode = -9
        outcome = self.phase(Process)
        self.assertTrue(outcome["hard_timeout"])
        self.assertEqual(outcome["members"][0]["row"]["lp_objective"], 5)
        self.assertEqual(outcome["decision_wall_s"], .2)
        self.assertEqual(processes[0].returncode, -9)

    def test_startup_crossing_deadline_does_not_accept_a_finished_result(self):
        clock = self.clock
        class Process(self.finished()):
            def __init__(self, *args, **kwargs):
                super().__init__(*args, **kwargs)
                clock.t = .3
        outcome = self.phase(Process)
        self.assertIsNone(outcome["members"][0]["row"])
        self.assertEqual(outcome["decision_wall_s"], .2)

    def test_selection_itself_must_finish_before_deadline(self):
        def slow_selection(rows):
            self.clock.t = .21
            return rows[0]
        outcome = self.phase(self.finished(), reducer=slow_selection)
        self.assertIsNone(outcome["aggregate"])
        self.assertIsNone(outcome["members"][0]["row"])

    def test_all_16_repairs_launch_before_polling_at_one_thread_each(self):
        launched = []
        test = self
        class Process(self.finished()):
            def __init__(self, cmd, **kwargs):
                super().__init__(cmd, **kwargs)
                test.assertEqual(kwargs["env"]["BENCHMARK_HIGHS_THREADS"], "1")
                test.assertEqual(kwargs["env"]["CUDA_VISIBLE_DEVICES"], "-1")
                launched.append(self.job)
            def poll(self):
                test.assertEqual(len(launched), 16)
                return 0
        jobs = [dict(self.common, method="pipeline", kind="repair", threads=1, candidate_idx=i) for i in range(16)]
        outcome = self.phase(Process, jobs=jobs, reducer=bench.select_repairs)
        self.assertEqual({j["deadline"] for j in launched}, {.2})
        self.assertFalse(outcome["hard_timeout"])
        self.assertEqual(outcome["aggregate"]["n_repairs_completed"], 16)

    def test_generation_and_repairs_share_full_absolute_deadline(self):
        calls = []
        def phase(jobs, directory, started, deadline, **kwargs):
            calls.append((jobs, started, deadline))
            if jobs[0]["kind"] == "generate":
                self.clock.t = 7
                return dict(members=[dict(row={"ready": True, "n_candidates": 16, "candidate_file": "fake"})],
                            phase_wall_s=7)
            self.assertEqual(len(jobs), 16)
            self.assertEqual({j["threads"] for j in jobs}, {1})
            self.clock.t = 119
            return dict(aggregate=row(kind="pipeline"), aggregate_after_s=118, members=[],
                        phase_wall_s=112, decision_wall_s=119, hard_timeout=False)
        outcome = bench.run_experiment(self.common, "pipeline", 16, 120, self.directory,
                                       phase=phase, clock=self.clock)
        self.assertEqual([(a, b) for _, a, b in calls], [(0, 120), (0, 120)])
        self.assertEqual(outcome["row"]["kind"], "pipeline")
        self.assertEqual(outcome["budget_seconds"], 120)

    def test_expired_generation_starts_no_repairs(self):
        calls = []
        def phase(jobs, directory, started, deadline, **kwargs):
            calls.append(jobs)
            self.clock.t = 2.01
            return dict(members=[dict(row=None)], phase_wall_s=2, decision_wall_s=2, hard_timeout=True)
        outcome = bench.run_experiment(self.common, "pipeline", 16, 2, self.directory,
                                       phase=phase, clock=self.clock)
        self.assertEqual(len(calls), 1)
        self.assertIsNone(outcome["row"])

    def test_generation_cleanup_cannot_extend_the_shared_budget(self):
        def phase(jobs, directory, started, deadline, **kwargs):
            self.assertEqual(jobs[0]["kind"], "generate")
            self.clock.t = 2.1
            return dict(members=[dict(row={"ready": True, "n_candidates": 16, "candidate_file": "fake"})],
                        phase_wall_s=1.9, decision_wall_s=1.9, hard_timeout=False)
        outcome = bench.run_experiment(self.common, "pipeline", 16, 2, self.directory,
                                       phase=phase, clock=self.clock)
        self.assertTrue(outcome["hard_timeout"])
        self.assertEqual(outcome["decision_wall_s"], 2)
        self.assertIsNone(outcome["row"])

    def test_slow_read_crossing_deadline_is_rejected(self):
        original = Path.read_text
        def slow(path, *args, **kwargs):
            if path.name.startswith("snapshot-"):
                self.clock.t = .21
            return original(path, *args, **kwargs)
        with patch.object(Path, "read_text", slow):
            outcome = self.phase(self.finished())
        self.assertTrue(outcome["hard_timeout"])
        self.assertIsNone(outcome["members"][0]["row"])

    def test_milp_and_jepa_each_receive_fresh_deadline_and_16_threads(self):
        deadlines = []
        def phase(jobs, directory, started, deadline, **kwargs):
            deadlines.append((started, deadline, jobs[0]["threads"]))
            self.clock.t += 120
            return dict(members=[dict(row=None, received_after_s=None)], decision_wall_s=120, hard_timeout=True)
        for method in ("milp", "jepa"):
            bench.run_experiment(self.common, method, 16, 120, self.directory / method,
                                 phase=phase, clock=self.clock)
        self.assertEqual(deadlines, [(0, 120, 16), (120, 240, 16)])

    def test_transient_snapshot_lock_retries_within_deadline(self):
        original = Path.read_text
        def locked(path, *args, **kwargs):
            if path.name.startswith("snapshot-") and self.clock.t < .06:
                raise PermissionError("simulated lock")
            return original(path, *args, **kwargs)
        with patch.object(Path, "read_text", locked):
            outcome = self.phase(self.finished())
        self.assertFalse(outcome["hard_timeout"])
        self.assertGreater(outcome["file_access_retries"], 0)
        self.assertGreaterEqual(outcome["members"][0]["received_after_s"], .06)

    def test_corrupt_output_and_worker_crash_are_fatal(self):
        class Crashed(self.finished()):
            def __init__(self, *args, **kwargs):
                super().__init__(*args, **kwargs)
                self.returncode = 1
        with self.assertRaisesRegex(RuntimeError, "Worker failed"):
            self.phase(Crashed)

    def test_active_campaign_lock_refuses_before_workers(self):
        (self.directory / "_pipeline").mkdir()
        with patch.object(Path, "open", side_effect=PermissionError("legacy holds lock")):
            with self.assertRaisesRegex(RuntimeError, "Another campaign"):
                with bench.campaign_lock(self.directory): self.fail("Must not enter")


class MatrixTests(unittest.TestCase):
    def lp(self):
        return NS(num_col_=2, num_row_=1, a_matrix_=NS(format_="kRowwise", value_=[1., 1.],
                   index_=[0, 1], start_=[0, 2]), row_lower_=[1.], row_upper_=[10.],
                   col_lower_=[0., 0.], col_upper_=[1., 10.], integrality_=[1, 0],
                   col_cost_=[2., 3.], offset_=7., sense_=1)

    def test_matrix_validation_catches_constraint_bound_integer_and_nan_errors(self):
        validate = linear_validator(self.lp())
        self.assertEqual(validate([1, 0]), (9, 0))
        for values in ([0, 0], [2, 0], [.5, 1], [math.nan, 1]):
            self.assertGreater(validate(values)[1], 1e-3)

    def test_columnwise_matrix_and_objective_offset(self):
        lp = self.lp()
        lp.a_matrix_ = NS(format_="kColwise", value_=[1., 1.], index_=[0, 0], start_=[0, 1, 2])
        self.assertEqual(linear_validator(lp)([0, 1]), (10, 0))

    def test_streaming_callback_keeps_early_incumbent_without_post_deadline_recovery(self):
        clock, captured = Clock(), []
        variable_a, variable_b = object(), object()
        model = NS(unserved={0: variable_a}, overgen_spill={0: variable_b}, dt_hours=1)
        lp = self.lp()
        class Event:
            def subscribe(self, callback): self.callback = callback
        class Native:
            def __init__(self):
                self.cbMipImprovingSolution, self.cbMipInterrupt = Event(), Event()
                self.cbSimplexInterrupt, self.cbIpmInterrupt = Event(), Event()
            def getLp(self): return lp
            def setOptionValue(self, *args): return 0
            def run(self):
                event = NS(data_out=NS(mip_solution=[0, 1], mip_dual_bound=7, mip_gap=.3), interrupt=lambda: None)
                clock.t = .05
                self.cbMipImprovingSolution.callback(event)
                clock.t = .21
                event.data_out.mip_solution = [1, 0]
                self.cbMipImprovingSolution.callback(event)
                return 0
            def getSolution(self): return NS(value_valid=True, col_value=[1, 0])
            def getModelStatus(self): return "time_limit"
            def getInfo(self): return NS(mip_dual_bound=7, mip_gap=.2)
        native = Native()
        adapter = NS(set_instance=lambda _: None, _solver_model=native,
                     _pyomo_var_to_solver_var_map={id(variable_a): 0, id(variable_b): 1})
        modules = {"highspy": NS(ObjSense=NS(kMinimize=1), HighsStatus=NS(kError=-1)),
                   "pyomo.environ": NS(value=lambda v: v),
                   "pyomo.contrib.appsi.solvers.highs": NS(Highs=lambda: adapter),
                   "src.milp.model": NS(build_uc_model=lambda *a, **kw: model),
                   "src.milp.scenario_loader": NS(load_scenario_data=lambda _: None)}
        job = dict(scenario_file="unused.json", deadline=.2, threads=16, seed=42,
                   scenario_id="scenario_00001", family_alias="low")
        def publish_result(method, result):
            captured.append(result)
            return True
        with patch.dict(sys.modules, modules):
            solve_streaming(job, publish_result, clock=clock)
        self.assertEqual([r["lp_objective"] for r in captured], [10])
        self.assertEqual(captured[0]["slack_mwh"], 1)


class MetricsAndResumeTests(WorkerFixture):
    def manifest(self):
        root = self.directory / "input"
        out = root / "evaluation/new"
        args = NS(root=str(root), output_dir=str(out), families=["low_criticality_scenarios"],
                  budgets=[1., 2.], threads=16, k=16, resume=False)
        manifest = dict(root=str(root), protocol=bench.PROTOCOL, k=16, budgets=args.budgets,
                        families=args.families, cohort=[dict(family=args.families[0], scenario_id="scenario_00001")])
        return args, manifest, out

    def outcome(self, budget):
        return dict(row=None, protocol=bench.PROTOCOL, budget_seconds=budget, decision_wall_s=budget,
                    hard_timeout=True, cleanup_wall_s=.01, snapshot_available_after_s=None)

    def test_resume_keeps_completed_failures_and_runs_only_missing_method(self):
        args, manifest, out = self.manifest()
        bench.atomic_json(out / "manifest.json", manifest)
        paths = []
        for b, budget in enumerate(args.budgets):
            for method in bench.METHODS:
                if (b, method) == (1, "pipeline"): continue
                payload = dict(identity=bench.identity(manifest["cohort"][0], budget, method, manifest),
                               manifest_sha256=bench.digest(manifest), outcome=self.outcome(budget))
                payload["sha256"] = bench.digest(payload)
                path = bench.checkpoint_path(out, 1, b, method)
                bench.atomic_json(path, payload)
                paths.append((path, path.read_bytes()))
        args.resume = True
        with patch.object(bench, "campaign_lock"), patch.object(bench, "build_manifest", return_value=manifest), \
             patch.object(bench, "run_experiment", return_value=self.outcome(2)) as run, \
             patch.object(bench, "analyze") as analyze:
            bench.run_campaign(args)
        run.assert_called_once()
        self.assertEqual(run.call_args.args[1:4], ("pipeline", 16, 2))
        analyze.assert_called_once()
        for path, before in paths: self.assertEqual(path.read_bytes(), before)

    def test_changed_manifest_blocks_resume_without_any_workers(self):
        args, manifest, out = self.manifest()
        bench.atomic_json(out / "manifest.json", dict(manifest, k=5))
        args.resume = True
        with patch.object(bench, "campaign_lock"), patch.object(bench, "build_manifest", return_value=manifest), \
             patch.object(bench, "run_experiment", side_effect=AssertionError("No worker")):
            with self.assertRaisesRegex(ValueError, "Manifest mismatch"):
                bench.run_campaign(args)

    def compatible_manifests(self, manifest):
        shared = "scripts/benchmark_parallel_best_of_k.py"
        before, after = next(iter(bench.COMPATIBLE_SOURCE_TRANSITIONS[shared]))
        old = dict(manifest, source_sha256={
            bench.RUNNER_SOURCE: next(iter(bench.COMPATIBLE_RUNNER_HASHES)),
            shared: before, "src/milp/model.py": "physics-unchanged",
        })
        current = dict(manifest, source_sha256={
            bench.RUNNER_SOURCE: bench.sha256(bench.__file__),
            shared: after, "src/milp/model.py": "physics-unchanged",
        })
        return old, current

    def test_audited_source_migration_is_allowed_with_all_other_fields_identical(self):
        _, manifest, _ = self.manifest()
        old, current = self.compatible_manifests(manifest)
        changed = bench.validate_resume_manifest(old, current)
        self.assertEqual(set(changed), {bench.RUNNER_SOURCE, "scripts/benchmark_parallel_best_of_k.py"})
        self.assertEqual(bench.validate_resume_manifest(current, current), [])

    def test_unknown_source_edits_inputs_versions_and_budget_changes_still_fail(self):
        _, manifest, _ = self.manifest()
        old, current = self.compatible_manifests(manifest)
        changes = [dict(current, k=5), dict(current, budgets=[1., 120.]),
                   dict(current, inputs_sha256={"checkpoint": "changed"}),
                   dict(current, versions={"torch": "changed"}),
                   dict(current, platform="different"),
                   dict(current, cohort=[])]
        for name in (bench.RUNNER_SOURCE, "scripts/benchmark_parallel_best_of_k.py", "src/milp/model.py"):
            changes.append(dict(current, source_sha256=dict(current["source_sha256"], **{name: "unknown"})))
        for changed in changes:
            with self.subTest(changed=changed), self.assertRaisesRegex(ValueError, "Manifest mismatch"):
                bench.validate_resume_manifest(old, changed)
        for name in (bench.RUNNER_SOURCE, "scripts/benchmark_parallel_best_of_k.py"):
            unknown_old = dict(old, source_sha256=dict(old["source_sha256"], **{name: "unknown"}))
            with self.assertRaisesRegex(ValueError, name):
                bench.validate_resume_manifest(unknown_old, current)

    def test_compatible_resume_keeps_original_hash_and_new_checkpoints_survive_next_resume(self):
        args, manifest, out = self.manifest()
        old, current = self.compatible_manifests(manifest)
        manifest_path = out / "manifest.json"
        bench.atomic_json(manifest_path, old)
        preserved = [(manifest_path, manifest_path.read_bytes())]
        for b, budget in enumerate(args.budgets):
            for method in bench.METHODS:
                if (b, method) == (1, "pipeline"):
                    continue
                payload = dict(identity=bench.identity(old["cohort"][0], budget, method, old),
                               manifest_sha256=bench.digest(old), outcome=self.outcome(budget))
                payload["sha256"] = bench.digest(payload)
                path = bench.checkpoint_path(out, 1, b, method)
                bench.atomic_json(path, payload)
                preserved.append((path, path.read_bytes()))
        args.resume = True
        with patch.object(bench, "campaign_lock"), patch.object(bench, "build_manifest", return_value=current), \
             patch.object(bench, "run_experiment", return_value=self.outcome(2)) as run, \
             patch.object(bench, "analyze"), patch("builtins.print"):
            bench.run_campaign(args)
            run.assert_called_once()
            self.assertEqual(run.call_args.args[1:4], ("pipeline", 16, 2))
            run.reset_mock()
            bench.run_campaign(args)
            run.assert_not_called()
        for path, raw in preserved:
            self.assertEqual(path.read_bytes(), raw)
        new_path = bench.checkpoint_path(out, 1, 1, "pipeline")
        new = json.loads(new_path.read_text())
        self.assertEqual(new["manifest_sha256"], bench.digest(old))
        self.assertEqual(new["execution_manifest_sha256"], bench.digest(current))
        expected = bench.identity(old["cohort"][0], 2, "pipeline", old)
        self.assertEqual(bench.load_checkpoint(new_path, expected, bench.digest(old)), self.outcome(2))
        resumes = list((out / "resumes").glob("*.json"))
        self.assertEqual(len(resumes), 2)
        audit = json.loads(resumes[0].read_text())
        self.assertEqual(audit["manifest"], current)
        self.assertEqual(audit["original_manifest_sha256"], bench.digest(old))
        self.assertEqual(len(audit["compatible_source_changes"]), 2)

    def test_existing_manifest_requires_explicit_resume_even_if_compatible(self):
        args, manifest, out = self.manifest()
        old, current = self.compatible_manifests(manifest)
        bench.atomic_json(out / "manifest.json", old)
        with patch.object(bench, "campaign_lock"), patch.object(bench, "build_manifest", return_value=current), \
             patch.object(bench, "run_experiment") as run:
            with self.assertRaisesRegex(ValueError, "requires --resume"):
                bench.run_campaign(args)
            run.assert_not_called()
        self.assertFalse((out / "resumes").exists())

    def test_parent_checkpoint_is_flushed_before_publication(self):
        path = self.directory / "parent.json"
        with patch.object(os, "fsync", wraps=os.fsync) as flush:
            bench.save_parent_json(path, {"example": 1})
        flush.assert_called_once()
        self.assertEqual(json.loads(path.read_text()), {"example": 1})

    def test_corrupt_checkpoint_is_never_silently_reused(self):
        args, manifest, out = self.manifest()
        expected = bench.identity(manifest["cohort"][0], 1, "milp", manifest)
        payload = dict(identity=expected, manifest_sha256=bench.digest(manifest), outcome=self.outcome(1))
        payload["sha256"] = bench.digest(payload)
        payload["outcome"]["decision_wall_s"] = 99
        path = out / "test.json"
        bench.atomic_json(path, payload)
        with self.assertRaisesRegex(ValueError, "corrupt"):
            bench.load_checkpoint(path, expected, bench.digest(manifest))

    def test_metrics_pair_identical_scenarios_and_keep_pending_separate_from_failure(self):
        rows = []
        for sid, costs in (("a", [10, 12, 11]), ("b", [10, None, 5]), ("c", [None, None, None])):
            for method, cost in zip(bench.METHODS, costs):
                rows.append(dict(family="low", scenario_id=sid, budget_seconds=1, method=method,
                                 completed=sid != "c", feasible=cost is not None, objective=cost,
                                 gap_reference_pct=bench.gap(cost, 10), zero_slack=cost is not None,
                                 hard_timeout=cost is None and sid != "c", decision_wall_s=1))
        summaries, comparisons = bench.summarize(rows, ["low"], [1])
        p = next(s for s in summaries if s["family"] == "low" and s["method"] == "pipeline")
        self.assertEqual((p["n_completed"], p["n_pending"], p["n_feasible"]), (2, 1, 1))
        self.assertEqual(p["feasibility_rate_completed"], .5)
        self.assertEqual(p["n_common_all_three"], 1)
        pair = next(s for s in comparisons if s["family"] == "low" and s["left"] == "pipeline" and s["right"] == "jepa")
        self.assertEqual(pair["n_paired"], 1)
        self.assertEqual(pair["mean_right_gap_reference"], 10)
        self.assertEqual(pair["right_wins"], 1)
        self.assertIsNone(bench.gap(1, 0))
        self.assertEqual(bench.gap(-8, -10), 20)


if __name__ == "__main__":
    unittest.main()
