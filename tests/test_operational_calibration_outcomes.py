"""Deadline failures are counted once and never abort or retry calibration."""
from pathlib import Path
import unittest
from unittest.mock import patch

from test_operational_pilot_io import workspace_temp
from src.operational_exit.common import METHODS, atomic_json, digest, load_record, save_record, sha, seed_for
from src.operational_exit.campaign import calibrate
from src.operational_exit.calibration_outcomes import (check_sources, register_policy, verify_policy,
    RECEIPT_NAME, PREVIOUS_RECEIPT)


class CalibrationOutcomeTests(unittest.TestCase):
    def fixture(self, out, statuses):
        models = {}
        for name in ("hte", "hte_features", "ae", "jepa", "ebm"):
            path = out / (name + ".fixture")
            path.write_text(name, encoding="utf-8")
            models[name] = str(path)
        keys = ["validation/one", "validation/two"] if len(statuses) == 8 else ["validation/one"]
        manifest = dict(train=[dict(key=k) for k in keys], cohort=[], calibration_cases=keys,
            splits={"assignments": {k: "validation" for k in keys}}, hte_sha256=sha(models["hte"]),
            sources={}, versions={})
        paths = []
        for i, status in enumerate(statuses):
            key, method = keys[i // 4], METHODS[i % 4]
            ident = dict(key=key, method=method, budget=120, repetition=0, seed=seed_for(key),
                purpose="calibration", runtime_revision="unchanged-runtime", manifest_hash=digest(manifest))
            timed = status in ("forced_timeout", "late_exit")
            path = out / "calibration_runs" / (digest(ident) + ".json")
            record = dict(identity=ident, status=status, complete=True, success=status == "success",
                forced=status == "forced_timeout", all_processes_exited=True, exit_s=120.001 if timed else 3.,
                finalization_s=None if timed else 1., preload_metadata={"repair_processes": 16 if method == "pipeline" else 1},
                result=None if timed else dict(generated_candidates=16 if method == "pipeline" else 1, gpu_peak_bytes=1),
                attempt=str(out / ("attempt" + str(i))))
            if status == "cooperative_deadline":
                record["result"] = {"status": "cooperative_deadline"}
            save_record(path, record)
            paths.append(path)
        return manifest, models, paths

    def run_calibration(self, manifest, out, models):
        progress = []
        with patch("src.operational_exit.campaign.model_files", return_value=models), \
             patch("src.operational_exit.solution_transfer_repair.calibration_revision", return_value="unchanged-runtime"), \
             patch("src.operational_exit.supervisor.run_experiment") as run:
            calibrate(manifest, out, lambda *args: progress.append(args))
            run.assert_not_called()  # Reuse complete failures as well as successes.
        return load_record(out / "calibration.json"), progress

    def test_resume_counts_failures_and_reaches_later_cases_without_retry(self):
        with workspace_temp() as out:
            statuses = ["forced_timeout", "forced_timeout", "success", "success",
                        "late_exit", "success", "cooperative_deadline", "success"]
            manifest, models, paths = self.fixture(out, statuses)
            hashes = {p: sha(p) for p in paths}
            result, progress = self.run_calibration(manifest, out, models)
            self.assertEqual(result["runtime_revision"], "unchanged-runtime")
            self.assertEqual(result["status_counts"], {"forced_timeout": 2, "success": 4, "late_exit": 1, "cooperative_deadline": 1})
            self.assertEqual(result["observations"], 8)
            self.assertEqual(len(progress), 8)
            self.assertEqual(result["finalization_observations"], 5)
            self.assertEqual(result["unobserved_finalizations"], 3)
            self.assertEqual(result["reserve_s"], 2.)
            self.assertEqual({p: sha(p) for p in paths}, hashes)
            self.assertIsNone(load_record(paths[0])["result"])

    def test_all_forced_timeouts_keep_provisional_reserve_and_unknown_measurements(self):
        with workspace_temp() as out:
            manifest, models, _ = self.fixture(out, ["forced_timeout"] * 4)
            result, progress = self.run_calibration(manifest, out, models)
            self.assertEqual(result["status_counts"], {"forced_timeout": 4})
            self.assertEqual(result["finalization_observations"], 0)
            self.assertEqual(result["unobserved_finalizations"], 4)
            self.assertIsNone(result["maximum_finalization_s"])
            self.assertEqual(result["reserve_s"], 30.)
            self.assertEqual(len(progress), 4)

    def test_infrastructure_error_or_unclosed_process_is_not_a_completed_deadline_failure(self):
        for changes in (dict(status="worker_error", forced=False), dict(all_processes_exited=False)):
            with self.subTest(changes=changes), workspace_temp() as out:
                manifest, models, paths = self.fixture(out, ["forced_timeout"] * 4)
                record = load_record(paths[0])
                record.update(changes)
                save_record(paths[0], record)
                with self.assertRaises(ValueError):
                    self.run_calibration(manifest, out, models)
                self.assertFalse((out / "calibration.json").exists())

    def test_execution_changes_are_not_allowed_by_outcome_policy(self):
        for name in ("solving.py", "worker.py", "supervisor.py", "common.py", "generation.py", "physics.py"):
            path = "src/operational_exit/" + name
            with self.subTest(path=path), self.assertRaises(ValueError):
                check_sources({path: "before"}, {path: "after"})

    def test_policy_preserves_execution_revision_models_and_completed_failures(self):
        with workspace_temp() as out:
            manifest, models, paths = self.fixture(out, ["forced_timeout"] * 4)
            # This test isolates the new policy layer; predecessor validation is
            # separately covered by the runtime amendment integration tests.
            manifest["train"] = []
            before = {"src/operational_exit/campaign.py": "stop-on-timeout"}
            current = dict(before, **{"src/operational_exit/campaign.py": "count-timeout",
                "src/operational_exit/calibration_outcomes.py": "new"})
            fingerprints = {k: sha(v) for k, v in models.items()}
            save_record(out / PREVIOUS_RECEIPT, dict(complete=True, sources=before, models=fingerprints))
            previous = load_record(out / PREVIOUS_RECEIPT)
            atomic_json(out / "state.json", dict(steps={str(s): "completed" for s in range(1, 9)}))
            hashes = {p: sha(p) for p in paths + [out / PREVIOUS_RECEIPT, *map(Path, models.values())]}
            with patch("src.operational_exit.protocol.verify_source_history"), \
                 patch("src.operational_exit.protocol.source_signature", return_value=current), \
                 patch("src.operational_exit.protocol.versions", return_value={}), \
                 patch("src.operational_exit.campaign.model_files", return_value=models):
                atomic_json(out / "calibration.json.bak", dict(complete=True))
                with self.assertRaisesRegex(ValueError, "calibrated/evaluated"):
                    register_policy(manifest, out)
                (out / "calibration.json.bak").unlink()
                register_policy(manifest, out)
                receipt = verify_policy(manifest, out, before, current, check_artifacts=True)
                self.assertEqual(receipt["execution_runtime_revision"], previous["sha256"])
                self.assertTrue(receipt["pilot_identities_unchanged"])
                self.assertEqual({p: sha(p) for p in hashes}, hashes)
                self.assertEqual({load_record(p)["status"] for p in paths}, {"forced_timeout"})
                Path(models["ae"]).write_text("changed", encoding="utf-8")
                with self.assertRaises(ValueError):
                    verify_policy(manifest, out, before, current, check_artifacts=True)


if __name__ == "__main__":
    unittest.main()
