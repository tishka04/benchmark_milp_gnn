"""Windows publication and provenance regressions; synthetic files/processes."""
from contextlib import contextmanager
from pathlib import Path
import os
import shutil
import tempfile
import unittest
from unittest.mock import patch

from src.operational_exit.common import (ROOT, atomic_json, digest, io_path, load_record,
                                         read_json, save_record, sha)
from src.operational_exit.pilot_io_repair import (RECEIPT_NAME, calibration_revision,
    check_sources, register_repair)
from src.operational_exit.protocol import effective_sources
from src.operational_exit.supervisor import run_experiment


@contextmanager
def workspace_temp():
    parent = (ROOT / "tmp").resolve()
    parent.mkdir(exist_ok=True)
    root = Path(tempfile.mkdtemp(dir=parent, prefix="pilot_io_repair_fixture_")).resolve()
    try:
        yield root
    finally:
        # Delete only this verified, newly created workspace test directory.
        root.relative_to(parent)
        assert root != parent
        shutil.rmtree(io_path(root))


@unittest.skipUnless(os.name == "nt", "Windows path semantics")
class LongPathTests(unittest.TestCase):
    def attempt_path(self, root):
        return root / ("x" * (236 - len(str(root)) - 1))

    def test_atomic_result_with_284_character_temporary_path(self):
        with workspace_temp() as root:
            path = self.attempt_path(root) / "final.json"
            self.assertEqual(len(str(path)), 247)
            atomic_json(path, {"synthetic": True})
            self.assertEqual(read_json(path), {"synthetic": True})

    def test_long_record_backup_and_corruption_recovery(self):
        with workspace_temp() as root:
            path = self.attempt_path(root) / ("r" * 40 + ".json")
            self.assertGreater(len(str(path)), 260)
            save_record(path, dict(identity={"test": 1}, complete=True, success=False))
            expected = load_record(path, {"test": 1})
            io_path(path).write_text("interrupted write", encoding="utf-8")
            self.assertEqual(load_record(path, {"test": 1}), expected)
            self.assertEqual(read_json(path), expected)
            self.assertEqual(sha(path), sha(str(path) + ".bak"))
            self.assertTrue(list(io_path(path.parent).glob(path.name + ".corrupt-*")))

    def test_relative_extended_and_unc_paths(self):
        self.assertTrue(str(io_path("tmp/a.json")).startswith("\\\\?\\"))
        self.assertEqual(io_path(io_path("tmp/a.json")), io_path("tmp/a.json"))
        self.assertEqual(str(io_path(r"\\server\share\a.json")), r"\\?\UNC\server\share\a.json")

    def test_supervised_process_can_publish_at_the_failing_path_length(self):
        with workspace_temp() as root:
            attempt = self.attempt_path(root)
            result = run_experiment({"mode": "fast"}, {"budget": 3, "identity": {"test": "long_path"}},
                                    attempt, worker_module="tests.operational_fake_worker")
            self.assertTrue(result["success"], result)
            self.assertTrue(result["all_processes_exited"])
            self.assertIsNotNone(result["finalization_s"])
            self.assertLessEqual(result["exit_s"], 3)


@unittest.skipUnless(os.name == "nt", "Diagnosed Windows pilot failure")
class PilotRepairTests(unittest.TestCase):
    def fixture(self, out):
        before = {"src/operational_exit/common.py": "before"}
        current = dict(before, **{"src/operational_exit/common.py": "after",
                                 "src/operational_exit/pilot_io_repair.py": "new"})
        models = {}
        for name in ("hte", "hte_features", "ae", "jepa", "ebm"):
            path = out / "training" / (name + ".fixture")
            path.parent.mkdir(exist_ok=True)
            path.write_text("synthetic " + name, encoding="utf-8")
            models[name] = str(path)
        scenario = out / "scenario.json"
        atomic_json(scenario, {"synthetic": True})
        key = "scenarios_v3/synthetic"
        manifest = dict(sources=before, versions={}, train=[dict(key=key, file=str(scenario), scenario_sha256=sha(scenario))],
                        cohort=[], calibration_cases=[key], hte_sha256=sha(models["hte"]),
                        splits={"assignments": {key: "validation"}})
        save_record(out / "manifest.json", manifest)
        manifest = load_record(out / "manifest.json")
        for name in ("audit.json", "dataset.json"):
            save_record(out / name, dict(complete=True, manifest_hash=digest(manifest)))
        atomic_json(out / "state.json", dict(steps={str(s): "completed" for s in range(1, 9)}))
        identity = dict(key=key, method="highs", budget=120, repetition=0, seed=42,
                        purpose="calibration", manifest_hash=digest(manifest))
        checkpoint = out / "calibration_runs" / (digest(identity) + ".json")
        attempt = out / "calibration_runs_attempts" / digest(identity) / ("a" * 32)
        io_path(attempt).mkdir(parents=True)
        io_path(attempt / "worker.log").write_text("atomic_json: FileNotFoundError final.json.synthetic.tmp", encoding="utf-8")
        failure = dict(identity=identity, complete=True, success=False, forced=False, returncode=1,
                       status="worker_error", all_processes_exited=True, result=None, attempt=str(attempt))
        save_record(checkpoint, failure)
        return manifest, before, current, models, checkpoint

    def test_repair_preserves_failure_and_training_then_uses_new_pilot_identities(self):
        from src.operational_exit.campaign import calibrate
        with workspace_temp() as out:
            manifest, before, current, models, checkpoint = self.fixture(out)
            protected = {p: sha(p) for p in [checkpoint, out / "manifest.json", out / "dataset.json", *map(Path, models.values())]}
            with patch("src.operational_exit.protocol.source_signature", return_value=current), \
                 patch("src.operational_exit.protocol.versions", return_value={}), \
                 patch("src.operational_exit.campaign.model_files", return_value=models):
                with self.assertRaises(ValueError):
                    effective_sources(manifest, out)
                register_repair(manifest, out)
                self.assertEqual(effective_sources(manifest, out, check_artifacts=True), current)
                receipt = load_record(out / RECEIPT_NAME)
                self.assertEqual(len(receipt["superseded_pilots"]), 1)
                fingerprints = {k: sha(v) for k, v in models.items()}
                revision = calibration_revision(out, fingerprints)
                self.assertEqual(revision, receipt["sha256"])
                original_identity = load_record(checkpoint)["identity"]
                self.assertNotEqual(digest(original_identity), digest(dict(original_identity, runtime_revision=revision)))
                seen = []
                def fake_run(manifest, out, case, item, *args):
                    seen.append(item)
                    return dict(status="success", forced=False, finalization_s=.1, all_processes_exited=True,
                                preload_metadata={"repair_processes": 16 if item["method"] == "pipeline" else 1},
                                result={"generated_candidates": 16 if item["method"] == "pipeline" else 1, "gpu_peak_bytes": 1})
                with patch("src.operational_exit.campaign.one_run", side_effect=fake_run):
                    calibrate(manifest, out, lambda *args: None)
                self.assertEqual(len(seen), 4)
                self.assertTrue(all(item["runtime_revision"] == revision for item in seen))
                self.assertEqual(load_record(out / "calibration.json")["runtime_revision"], revision)
                self.assertEqual({p: sha(p) for p in protected}, protected)
                Path(models["ae"]).write_text("changed model", encoding="utf-8")
                with self.assertRaises(ValueError):
                    effective_sources(manifest, out, check_artifacts=True)

    def test_repair_refuses_to_retry_timeout_or_success(self):
        for status, forced, success in (("forced_timeout", True, False), ("success", False, True)):
            with self.subTest(status=status), workspace_temp() as out:
                manifest, before, current, models, checkpoint = self.fixture(out)
                record = load_record(checkpoint)
                record.update(status=status, forced=forced, success=success)
                save_record(checkpoint, record)
                with patch("src.operational_exit.protocol.source_signature", return_value=current), \
                     patch("src.operational_exit.protocol.versions", return_value={}), \
                     patch("src.operational_exit.campaign.model_files", return_value=models):
                    with self.assertRaisesRegex(ValueError, "success, timeout"):
                        register_repair(manifest, out)

    def test_repair_refuses_completed_calibration(self):
        with workspace_temp() as out:
            manifest, before, current, models, checkpoint = self.fixture(out)
            atomic_json(out / "calibration.json", {"complete": True})
            with patch("src.operational_exit.protocol.source_signature", return_value=current), \
                 patch("src.operational_exit.protocol.versions", return_value={}):
                with self.assertRaisesRegex(ValueError, "calibrated or evaluated"):
                    register_repair(manifest, out)

    def test_repair_refuses_an_interrupted_evaluation_backup(self):
        with workspace_temp() as out:
            manifest, before, current, models, checkpoint = self.fixture(out)
            atomic_json(out / "experiments/interrupted.json.bak", {"complete": False})
            with patch("src.operational_exit.protocol.source_signature", return_value=current), \
                 patch("src.operational_exit.protocol.versions", return_value={}):
                with self.assertRaisesRegex(ValueError, "calibrated or evaluated"):
                    register_repair(manifest, out)

    def test_repair_cannot_change_physics_training_or_inference(self):
        for path in ("src/milp/model.py", "src/operational_exit/solving.py",
                     "src/operational_exit/generation.py", "src/uc_jepa/training/train_jepa.py"):
            with self.subTest(path=path), self.assertRaises(ValueError):
                check_sources({path: "before"}, {path: "after"})

    def test_pilot_error_reports_worker_log_before_generic_finalization_error(self):
        from src.operational_exit.campaign import calibrate
        with workspace_temp() as out:
            manifest, before, current, models, checkpoint = self.fixture(out)
            with patch("src.operational_exit.campaign.model_files", return_value=models), \
                 patch("src.operational_exit.campaign.one_run", return_value=load_record(checkpoint)):
                with self.assertRaisesRegex(ValueError, "Pilot worker_error.*worker.log"):
                    calibrate(manifest, out, lambda *args: None)


if __name__ == "__main__":
    unittest.main()
