"""Solver-free checks: fake processes only; no inference or training."""
import json
import math
import tempfile
import unittest
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

from scripts.benchmark_parallel_best_of_k import (
    PROTOCOL, analyze, comparison_rows, gap_percent, parse_k_values, run_benchmark,
    run_gpu_portfolio, run_group, sample_decode_batch,
)
from scripts import benchmark_parallel_best_of_k as benchmark
from src.milp.solver_resources import configure_highs_threads


class FakeClock:
    def __init__(self):
        self.t = 100.0

    def __call__(self):
        self.t += 0.001
        return self.t


class ParallelBenchmarkTests(unittest.TestCase):
    def setUp(self):
        # A regression must never accidentally create a real worker process.
        self.process_guard = patch("subprocess.Popen", side_effect=AssertionError("Real process prohibited"))
        self.process_guard.start()

    def tearDown(self):
        self.process_guard.stop()

    def temporary_workspace(self):
        base = Path(__file__).resolve().parents[1] / "tmp"
        base.mkdir(exist_ok=True)
        temporary = tempfile.TemporaryDirectory(prefix="parallel-benchmark-test-", dir=base)
        # Check the absolute cleanup target before TemporaryDirectory removes it.
        self.assertTrue(Path(temporary.name).resolve().is_relative_to(base.resolve()))
        return temporary

    def test_k_grid_and_caps(self):
        self.assertEqual(parse_k_values("", 4), [1, 2])
        self.assertEqual(parse_k_values("", 16), [1, 2, 5, 8, 12, 16])
        self.assertEqual(parse_k_values("4,1,4", 4), [1, 4])
        for values in ("0,1", "5", "-1"):
            with self.assertRaises(ValueError):
                parse_k_values(values, 4)

    def test_highs_configuration_does_not_construct_a_solver(self):
        solver = SimpleNamespace(highs_options={})
        with patch.dict("os.environ", {"BENCHMARK_HIGHS_THREADS": "4"}):
            result = configure_highs_threads(solver, "appsi_highs")
        self.assertEqual(result["highs_threads"], 4)
        self.assertEqual(solver.highs_options, {"threads": 4, "parallel": "on"})
        with patch.dict("os.environ", {"BENCHMARK_HIGHS_THREADS": "1"}):
            configure_highs_threads(solver, "appsi_highs")
        self.assertEqual(solver.highs_options["parallel"], "off")
        with patch.dict("os.environ", {"BENCHMARK_HIGHS_THREADS": "0"}):
            with self.assertRaises(ValueError):
                configure_highs_threads(solver, "appsi_highs")

    def test_launches_all_members_before_waiting_and_selects_feasible_min(self):
        events, deadlines = [], []
        test = self

        class Process:
            returncode = 0

            def __init__(self, cmd, **kwargs):
                job = json.loads(Path(cmd[-1]).read_text())
                index = len(events)
                events.append("launch")
                deadlines.append(job["deadline"])
                test.assertEqual(kwargs["env"]["BENCHMARK_HIGHS_THREADS"], "1")
                result = {"status": "ok", "feasible": index != 0,
                          "objective": [1, 8, 5][index], "cpu_seconds": 0.2}
                Path(job["result_file"]).write_text(json.dumps(result))

            def poll(self):
                test.assertEqual(events.count("launch"), 3)
                return 0

            def wait(self, timeout):
                return 0

        with self.temporary_workspace() as temp:
            result = run_group([{"threads": 1, "seed": i} for i in range(3)],
                               temp, 120, popen=Process, clock=FakeClock())
        self.assertEqual(len(set(deadlines)), 1)
        self.assertEqual(result["objective"], 5)  # infeasible cheaper result excluded
        self.assertEqual(len(result["members"]), 3)
        self.assertTrue(result["cpu_accounting_complete"])

    def test_deadline_kills_active_processes_and_keeps_failure_in_cohort(self):
        processes = []

        class HangingProcess:
            returncode = None
            killed = False

            def __init__(self, *args, **kwargs):
                processes.append(self)

            def poll(self):
                return self.returncode

            def kill(self):
                self.killed = True
                self.returncode = -9

            def wait(self, timeout):
                return self.returncode

        with self.temporary_workspace() as temp, patch("time.sleep"):
            result = run_group([{"threads": 1, "seed": 42}], temp, 0.01,
                               popen=HangingProcess, clock=FakeClock())
        self.assertTrue(processes[0].killed)
        self.assertFalse(result["feasible"])
        self.assertTrue(math.isinf(result["objective"]))
        self.assertFalse(result["cpu_accounting_complete"])
        self.assertLessEqual(result["decision_wall_s"], 0.01)

    def test_unexpected_worker_failure_is_not_treated_as_infeasibility(self):
        class FailedProcess:
            returncode = 1

            def __init__(self, *args, **kwargs):
                pass

            def poll(self):
                return 1

            def wait(self, timeout):
                return 1

        with self.temporary_workspace() as temp:
            with self.assertRaisesRegex(RuntimeError, "Worker failed"):
                run_group([{"threads": 1, "seed": 42}], temp, 1,
                          popen=FailedProcess, clock=FakeClock())

    def test_gpu_batch_and_repairs_share_deadline_without_dividing_budget(self):
        jobs, repair_remaining = [], []
        clock = FakeClock()
        test = self
        gpu_finished = False
        k = 3

        class Process:
            returncode = 0

            def __init__(self, cmd, **kwargs):
                self.job = job = json.loads(Path(cmd[-1]).read_text())
                jobs.append(job)
                test.assertEqual(kwargs["env"]["BENCHMARK_HIGHS_THREADS"], "1")
                if job["kind"] == "gpu_generate":
                    test.assertEqual(job["k"], k)
                    test.assertEqual(kwargs["env"]["CUDA_VISIBLE_DEVICES"], "0")
                    result = {"status": "candidates_ready", "ready": True, "feasible": False,
                              "objective": math.inf, "candidate_file": "fake.npz", "n_candidates": k,
                              "cpu_seconds": 1.0, "gpu_batch_wall_s": 7.0}
                else:
                    test.assertTrue(gpu_finished)
                    test.assertEqual(kwargs["env"]["CUDA_VISIBLE_DEVICES"], "-1")
                    test.assertEqual(job["candidate_file"], "fake.npz")
                    repair_remaining.append(job["deadline"] - clock.t)
                    result = {"status": "ok", "feasible": job["candidate_idx"] > 0,
                              "objective": [1, 9, 5][job["candidate_idx"]],
                              "candidate_idx": job["candidate_idx"], "cpu_seconds": 2.0}
                Path(job["result_file"]).write_text(json.dumps(result))

            def poll(self):
                nonlocal gpu_finished
                if self.job["kind"] == "gpu_generate":
                    if not gpu_finished:
                        clock.t += 8.0
                    gpu_finished = True
                else:
                    test.assertEqual(len(jobs), k + 1)  # all repairs launched before waiting
                return 0

        with self.temporary_workspace() as temp, patch.dict("os.environ", {"CUDA_VISIBLE_DEVICES": "0"}):
            result = run_gpu_portfolio({}, k, 42, temp, 120, popen=Process, clock=clock)
        self.assertEqual([j["kind"] for j in jobs], ["gpu_generate", "repair", "repair", "repair"])
        self.assertEqual(len({j["deadline"] for j in jobs}), 1)
        self.assertTrue(all(111 < remaining < 112 for remaining in repair_remaining))
        self.assertEqual(result["best_candidate_idx"], 2)
        self.assertEqual(result["objective"], 5)
        self.assertGreater(result["first_feasible_s"], 8)
        self.assertGreater(result["decision_wall_s"], 8)
        self.assertGreater(result["generation_wall_s"], 8)
        self.assertEqual(result["cpu_seconds_completed_members"], 7.0)
        self.assertEqual(result["n_candidates_generated"], k)
        self.assertEqual(result["n_repairs_completed"], k)

    def test_gpu_timeout_does_not_launch_repairs(self):
        jobs = []

        class Process:
            returncode = None
            killed = False

            def __init__(self, cmd, **kwargs):
                jobs.append(json.loads(Path(cmd[-1]).read_text()))

            def poll(self):
                return self.returncode

            def kill(self):
                self.killed = True
                self.returncode = -9

            def wait(self, timeout):
                return self.returncode

        with self.temporary_workspace() as temp, patch("time.sleep"):
            result = run_gpu_portfolio({}, 4, 42, temp, 0.02, popen=Process, clock=FakeClock())
        self.assertEqual(len(jobs), 1)
        self.assertEqual(jobs[0]["kind"], "gpu_generate")
        self.assertEqual(result["n_candidates_generated"], 0)
        self.assertEqual(result["n_repairs_completed"], 0)
        self.assertEqual(len(result["members"]), 4)
        self.assertFalse(result["feasible"])
        self.assertFalse(result["cpu_accounting_complete"])

    def test_batch_uses_one_sampler_call_and_one_decode_with_gradients_for_sampling(self):
        # Tensor shapes and model calls only: never import torch or execute a model.
        grad_enabled = True
        k = 4
        test = self

        @contextmanager
        def no_grad():
            nonlocal grad_enabled
            old = grad_enabled
            grad_enabled = False
            try:
                yield
            finally:
                grad_enabled = old

        class Tensor:
            def __init__(self, *shape):
                self.shape = shape

            def __getitem__(self, index):
                test.assertEqual(index, 0)
                return Tensor(*self.shape[1:])

            def expand(self, *shape):
                return Tensor(*(old if new == -1 else new for old, new in zip(self.shape, shape)))

            def detach(self):
                return self

            def cpu(self):
                return self

            def numpy(self):
                return SimpleNamespace(shape=self.shape)

        def predict(h, **kwargs):
            test.assertFalse(grad_enabled)
            return {"z_pred": Tensor(1, 16, 128)}

        def sample(**kwargs):
            test.assertTrue(grad_enabled)
            test.assertEqual(kwargs["n_samples"], k)
            return Tensor(1, k, 16, 128), Tensor(1, k), None

        def decode(z, **kwargs):
            test.assertFalse(grad_enabled)
            test.assertEqual(z.shape, (k, 16, 128))
            test.assertEqual(kwargs["h_scenario"].shape, (k, 3, 24, 128))
            test.assertEqual(kwargs["n_zones"], 3)
            test.assertEqual(kwargs["n_timesteps"], 24)
            return Tensor(k, 3, 24, 8)

        torch = SimpleNamespace(no_grad=no_grad, sigmoid=lambda x: x,
                                isfinite=lambda x: SimpleNamespace(all=lambda: True),
                                cuda=SimpleNamespace(synchronize=Mock()))
        utils = SimpleNamespace(apply_thresholds=lambda x, thresholds: x)
        predictor, sampler, ae = Mock(side_effect=predict), SimpleNamespace(sample=Mock(side_effect=sample)), SimpleNamespace(decode=Mock(side_effect=decode))
        with patch.dict("sys.modules", {"torch": torch, "src.uc_jepa.latent_utils": utils}):
            candidates, energies = sample_decode_batch(
                sampler, object(), ae, predictor, Tensor(1, 3, 24, 128), Tensor(1, 3), k, object(),
            )
        predictor.assert_called_once()
        sampler.sample.assert_called_once()
        ae.decode.assert_called_once()
        torch.cuda.synchronize.assert_called_once()
        self.assertEqual(candidates.shape, (k, 3, 24, 8))
        self.assertEqual(energies.shape, (k,))

    def test_resume_rejects_old_cpu_protocol(self):
        with self.temporary_workspace() as temp:
            path = Path(temp)
            (path / "manifest.json").write_text(json.dumps({"schema_version": 1, "pipeline_device": "cpu"}))
            with self.assertRaisesRegex(ValueError, "Incompatible comparison protocol"):
                analyze(path)
            args = SimpleNamespace(root=temp, output_dir=temp, k_values="", threads=2,
                                   families=["low"], budget=120, resume=True)
            with patch("scripts.benchmark_parallel_best_of_k.make_manifest",
                       return_value={"schema_version": 2, "protocol": PROTOCOL}):
                with self.assertRaisesRegex(ValueError, "Existing benchmark differs"):
                    run_benchmark(args)

    @staticmethod
    def completed_result(feasible=True):
        return dict(complete=True, feasible=feasible, objective=10 if feasible else math.inf,
                    decision_wall_s=1, cleanup_wall_s=0)

    def checkpoint_record(self, sid="scenario_00001"):
        return dict(family="low", scenario_id=sid, dr_formulation=benchmark.FORMULATION,
                    milp=self.completed_result(), runs={"1": self.completed_result(False)})

    def test_durable_record_flushes_and_keeps_previous_version_with_checksum(self):
        with self.temporary_workspace() as temp:
            path = Path(temp) / "case.json"
            record = self.checkpoint_record()
            item = dict(family="low", scenario_id=record["scenario_id"])
            with patch.object(benchmark.os, "fsync", wraps=benchmark.os.fsync) as fsync:
                benchmark.save_record(path, record, item, 120, [1, 2])
                first = path.read_bytes()
                record["runs"]["2"] = self.completed_result()
                benchmark.save_record(path, record, item, 120, [1, 2])
            self.assertEqual(fsync.call_count, 3)  # first save, backup, second save
            self.assertEqual(path.with_suffix(".json.bak").read_bytes(), first)
            saved, _ = benchmark.read_record(path, item, 120, [1, 2])
            self.assertEqual(saved["checkpoint_sha256"], benchmark.record_checksum(saved))
            self.assertEqual(set(saved["runs"]), {"1", "2"})

    def test_corrupt_json_is_preserved_without_reusing_worker_outputs(self):
        for raw in (b"", b"\x00" * 1053, b'{"family":', b'\xff'):
            with self.subTest(raw=raw[:12]), self.temporary_workspace() as temp:
                path = Path(temp) / "case.json"
                path.write_bytes(raw)
                item = dict(family="low", scenario_id="scenario_00047")
                with patch("builtins.print"):
                    self.assertIsNone(benchmark.load_record(path, item, 120, [1, 2]))
                self.assertFalse(path.exists())
                archives = list(path.parent.glob("case.json.corrupt-*"))
                self.assertEqual(len(archives), 1)
                self.assertEqual(archives[0].read_bytes(), raw)

    def test_corrupt_current_checkpoint_restores_parent_backup(self):
        with self.temporary_workspace() as temp:
            path = Path(temp) / "case.json"
            record = self.checkpoint_record()
            item = dict(family="low", scenario_id=record["scenario_id"])
            benchmark.save_record(path, record, item, 120, [1, 2])
            record["runs"]["2"] = self.completed_result()
            benchmark.save_record(path, record, item, 120, [1, 2])
            # Valid JSON with silently changed content must also be rejected.
            damaged = json.loads(path.read_bytes())
            damaged["runs"]["2"]["objective"] = -999
            path.write_text(json.dumps(damaged))
            with patch("builtins.print"):
                restored = benchmark.load_record(path, item, 120, [1, 2])
            self.assertEqual(set(restored["runs"]), {"1"})
            self.assertFalse(restored["runs"]["1"]["feasible"])
            self.assertEqual(path.read_bytes(), path.with_suffix(".json.bak").read_bytes())

    def test_missing_current_checkpoint_restores_backup_and_bad_backup_is_preserved(self):
        with self.temporary_workspace() as temp:
            path = Path(temp) / "case.json"
            backup = path.with_suffix(".json.bak")
            record = self.checkpoint_record()
            item = dict(family="low", scenario_id=record["scenario_id"])
            backup.write_text(json.dumps(record))
            with patch("builtins.print"):
                self.assertEqual(benchmark.load_record(path, item, 120, [1]), record)
                path.write_bytes(b"\x00")
                backup.write_bytes(b"broken")
                self.assertIsNone(benchmark.load_record(path, item, 120, [1]))
            self.assertEqual(len(list(path.parent.glob("*.corrupt-*"))), 2)

    def test_semantic_mismatch_does_not_silently_recompute_or_overwrite(self):
        with self.temporary_workspace() as temp:
            path = Path(temp) / "case.json"
            record = self.checkpoint_record()
            path.write_text(json.dumps(record))
            original = path.read_bytes()
            with self.assertRaisesRegex(ValueError, "mismatch"):
                benchmark.load_record(path, dict(family="other", scenario_id=record["scenario_id"]), 120, [1])
            self.assertEqual(path.read_bytes(), original)
            with self.assertRaisesRegex(ValueError, "exceeds"):
                benchmark.read_record(path, dict(family="low", scenario_id=record["scenario_id"]), 0.5, [1])

    def test_checkpoint_read_retries_permission_error(self):
        record = self.checkpoint_record()
        item = dict(family="low", scenario_id=record["scenario_id"])
        raw = json.dumps(record).encode()
        with patch.object(Path, "read_bytes", side_effect=[PermissionError("locked"), raw]) as read, \
                patch.object(benchmark.time, "sleep"):
            self.assertEqual(benchmark.read_record(Path("unused.json"), item, 120, [1])[0], record)
        self.assertEqual(read.call_count, 2)

    def test_checkpoint_publication_retries_lock_and_flush_failure_keeps_original(self):
        with self.temporary_workspace() as temp:
            path = Path(temp) / "case.json"
            path.write_bytes(b"original")
            replace = Path.replace
            calls = []

            def locked_once(source, target):
                calls.append(source)
                if len(calls) == 1:
                    raise PermissionError("locked")
                return replace(source, target)

            with patch.object(Path, "replace", locked_once), patch.object(benchmark.time, "sleep"):
                benchmark.durable_checkpoint(path, b"new")
            self.assertEqual(len(calls), 2)
            with patch.object(benchmark.os, "fsync", side_effect=OSError("disk failure")):
                with self.assertRaisesRegex(OSError, "disk failure"):
                    benchmark.durable_checkpoint(path, b"not published")
            self.assertEqual(path.read_bytes(), b"new")

    def test_manifest_only_allows_known_runner_migration(self):
        previous = dict(protocol=PROTOCOL, source_sha256={benchmark.RUNNER_SOURCE:
                        next(iter(benchmark.COMPATIBLE_RUNNER_HASHES)), "physics.py": "same"},
                        inputs_sha256={"model": "same"}, budget_seconds=120)
        current = dict(previous, source_sha256={benchmark.RUNNER_SOURCE: "new", "physics.py": "same"})
        benchmark.validate_resume_manifest(previous, current)
        for changed in (dict(current, budget_seconds=60),
                        dict(current, inputs_sha256={"model": "different"}),
                        dict(current, source_sha256={benchmark.RUNNER_SOURCE: "new", "physics.py": "changed"})):
            with self.assertRaisesRegex(ValueError, "Existing benchmark differs"):
                benchmark.validate_resume_manifest(previous, changed)
        with self.assertRaisesRegex(ValueError, "Existing benchmark differs"):
            benchmark.validate_resume_manifest(dict(previous, source_sha256={benchmark.RUNNER_SOURCE: "unknown"}), current)

    def test_resume_keeps_completed_failures_and_only_recomputes_missing_groups(self):
        with self.temporary_workspace() as temp:
            root = Path(temp) / "root"
            out = Path(temp) / "out"
            cases = out / "cases/low"
            cases.mkdir(parents=True)
            records = [self.checkpoint_record(f"scenario_{i:05d}") for i in (1, 2, 3)]
            records[0]["runs"]["2"] = self.completed_result()
            for record in records:
                (cases / f"{record['scenario_id']}.json").write_text(json.dumps(record))
            first_bytes = (cases / "scenario_00001.json").read_bytes()
            (cases / "scenario_00002.json").write_bytes(b"\x00" * 1053)
            reports = root / "low/reports"
            reports.mkdir(parents=True)
            for record in records:
                (reports / f"{record['scenario_id']}.json").write_text(json.dumps(dict(
                    dr_formulation=benchmark.FORMULATION, mip=dict(objective=10, termination="optimal"))))
            # A worker-only artifact is deliberately not a reusable parent checkpoint.
            worker_file = out / "jobs/low/scenario_00002/milp/attempt-old/0/result.json"
            worker_file.parent.mkdir(parents=True)
            worker_file.write_text(json.dumps(self.completed_result()))
            cohort = [dict(family=r["family"], scenario_id=r["scenario_id"]) for r in records]
            previous = dict(protocol=PROTOCOL, cohort=cohort,
                            source_sha256={benchmark.RUNNER_SOURCE: next(iter(benchmark.COMPATIBLE_RUNNER_HASHES))})
            current = dict(previous, source_sha256={benchmark.RUNNER_SOURCE: "new"})
            manifest_path = out / "manifest.json"
            manifest_path.write_text(json.dumps(previous))
            manifest_bytes = manifest_path.read_bytes()
            args = SimpleNamespace(root=str(root), output_dir=str(out), families=["low"],
                                   threads=2, k_values="1,2", budget=120, resume=True)
            with patch.object(benchmark, "make_manifest", return_value=current), \
                    patch.object(benchmark, "run_group", return_value=self.completed_result()) as milp, \
                    patch.object(benchmark, "run_gpu_portfolio", return_value=self.completed_result()) as gpu, \
                    patch("builtins.print"):
                run_benchmark(args)
            milp.assert_called_once()
            self.assertEqual(Path(milp.call_args.args[0][0]["scenario_file"]).stem, "scenario_00002")
            self.assertEqual([(Path(c.args[0]["scenario_file"]).stem, c.args[1]) for c in gpu.call_args_list],
                             [("scenario_00002", 1), ("scenario_00002", 2), ("scenario_00003", 2)])
            self.assertEqual((cases / "scenario_00001.json").read_bytes(), first_bytes)
            self.assertEqual(manifest_path.read_bytes(), manifest_bytes)
            self.assertEqual(len(list((out / "resumes").glob("*.json"))), 1)
            self.assertFalse(json.loads((cases / "scenario_00003.json").read_bytes())["runs"]["1"]["feasible"])

    def test_resume_validates_all_cases_before_any_worker(self):
        with self.temporary_workspace() as temp:
            root = Path(temp)
            cases = root / "cases/low"
            cases.mkdir(parents=True)
            bad = self.checkpoint_record()
            (cases / "scenario_00002.json").write_text(json.dumps(bad))
            manifest = dict(cohort=[dict(family="low", scenario_id=f"scenario_{i:05d}") for i in (1, 2)])
            (root / "manifest.json").write_text(json.dumps(manifest))
            args = SimpleNamespace(root=temp, output_dir=temp, families=["low"],
                                   threads=2, k_values="1,2", budget=120, resume=True)
            with patch.object(benchmark, "make_manifest", return_value=manifest), \
                    patch.object(benchmark, "run_group") as worker:
                with self.assertRaisesRegex(ValueError, "mismatch"):
                    run_benchmark(args)
                worker.assert_not_called()

    def test_comparisons_use_common_cohorts_and_retain_failed_scenarios(self):
        def outcome(feasible, objective):
            return {"feasible": feasible, "objective": objective, "decision_wall_s": 1,
                    "first_feasible_s": 0.8 if feasible else None, "cleanup_wall_s": 0,
                    "cpu_seconds_completed_members": 1, "cpu_accounting_complete": True}

        records = [
            {"family": "low", "scenario_id": "a", "reference_objective": 10, "reference_status": "optimal",
             "milp": outcome(True, 10), "runs": {"1": outcome(False, math.inf), "2": outcome(True, 12)}},
            {"family": "low", "scenario_id": "b", "reference_objective": 10, "reference_status": "maxTimeLimit",
             "milp": outcome(True, 10), "runs": {"1": outcome(True, 13), "2": outcome(True, 11)}},
        ]
        rows = comparison_rows(records, [1, 2])
        self.assertEqual(len(rows), 4)
        self.assertFalse(rows[0]["paired_feasible"])
        self.assertIsNone(rows[0]["gap_vs_milp_budget_pct"])
        self.assertFalse(rows[1]["common_all_k_feasible"])
        self.assertTrue(rows[2]["common_all_k_feasible"])
        self.assertEqual(rows[3]["gap_vs_milp_budget_pct"], 10)
        self.assertIsNone(gap_percent(1, 0))
        with self.temporary_workspace() as temp:
            root = Path(temp)
            cohort = [{"family": r["family"], "scenario_id": r["scenario_id"]} for r in records]
            (root / "manifest.json").write_text(json.dumps({
                "schema_version": 2, "protocol": PROTOCOL,
                "cohort": cohort, "families": ["low"], "k_values": [1, 2],
                "highs_threads": 2, "budget_seconds": 120,
            }))
            (root / "cases/low").mkdir(parents=True)
            for record in records:
                (root / "cases/low" / f"{record['scenario_id']}.json").write_text(json.dumps(record))
            analyze(root)
            summary = json.loads((root / "best_of_k_parallel_summary.json").read_text())
            self.assertEqual(summary[0]["n_scenarios"], 2)
            self.assertEqual(summary[0]["pipeline_feasibility_rate"], 0.5)
            self.assertEqual(summary[0]["n_common_all_k_feasible"], 1)


if __name__ == "__main__":
    unittest.main()
