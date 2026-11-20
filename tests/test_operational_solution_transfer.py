"""Native HiGHS solution transfer must copy the column vector only once."""
import os
os.environ["BENCHMARK_HIGHS_THREADS"] = "1"
import unittest
from pathlib import Path
from unittest.mock import patch

import pandas  # noqa: F401 -- Windows DLL order.
import highspy

from test_milp_load_shifting import make_data, scenario_input
from test_operational_pilot_io import workspace_temp
from src.milp.scenario_loader import load_scenario_data
from src.operational_exit.common import atomic_json, digest, load_record, save_record, sealed, sha
from src.operational_exit.physics import validate_dispatch
from src.operational_exit.solving import solve_data
from src.operational_exit.protocol import effective_sources
from src.operational_exit.solution_transfer_repair import (check_sources, check_pilots, register_repair,
    calibration_revision, RECEIPT_NAME, STRUCTURAL_RECEIPT)


class SolutionTransferTests(unittest.TestCase):
    def test_copy_count_is_constant_and_dispatch_preserves_native_solution(self):
        original = highspy.Highs.getSolution
        for zones in (1, 8):
            with self.subTest(zones=zones):
                reads = []
                class ObservedSolution:
                    def __init__(self, solution):
                        self.solution = solution

                    def __getattr__(self, name):
                        return getattr(self.solution, name)

                    @property
                    def col_value(self):
                        values = self.solution.col_value
                        reads.append(len(values))
                        return values

                def get_solution(native):
                    return ObservedSolution(original(native))

                data = make_data(zones=zones)
                with patch.object(highspy.Highs, "getSolution", get_solution):
                    result = solve_data(data, threads=1, seconds=5)
                self.assertTrue(result["proven_optimal"])
                self.assertTrue(validate_dispatch(data, result["dispatch"])["valid"])
                self.assertEqual(len(reads), 1, "Copying the vector per variable makes transfer quadratic")
                self.assertGreater(reads[0], zones * len(data.periods))
                self.assertAlmostEqual(result["objective"], 4. * zones)
                for row in result["dispatch"]["values"]["dr_shed"]:
                    self.assertEqual(row, [4., 0.])
                for row in result["dispatch"]["values"]["dr_recover"]:
                    self.assertEqual(row, [0., 4.])


class TransferRevisionTests(unittest.TestCase):
    def fixture(self, out):
        case_path = out / "scenario.json"
        atomic_json(case_path, scenario_input(n=2, dr_units=0))
        case = dict(key="validation/synthetic", file=str(case_path), scenario_sha256=sha(case_path))
        before = {"src/operational_exit/solving.py": "old"}
        manifest = dict(sources=before, versions={}, train=[case], cohort=[], calibration_cases=[case["key"]],
            splits={"assignments": {case["key"]: "validation"}})
        save_record(out / "manifest.json", manifest)
        manifest = load_record(out / "manifest.json")
        atomic_json(out / "state.json", dict(steps={str(s): "completed" for s in range(1, 9)}))
        models = {}
        for name in ("hte", "hte_features", "ae", "jepa", "ebm"):
            path = out / (name + ".fixture")
            path.write_text(name, encoding="utf-8")
            models[name] = str(path)
        hashes = {k: sha(v) for k, v in models.items()}
        for receipt_name in ("pilot_io_amendment.json", "repair_integrality_amendment.json", STRUCTURAL_RECEIPT):
            save_record(out / receipt_name, dict(complete=True, models=hashes, sources=before))
        previous = load_record(out / STRUCTURAL_RECEIPT)
        ident = dict(method="highs", key=case["key"], budget=120, purpose="calibration",
                     runtime_revision=previous["sha256"], manifest_hash=digest(manifest))
        attempt = out / "calibration_runs_attempts/diagnostic"
        result = solve_data(load_scenario_data(case_path), threads=1, seconds=5)
        raw = dict(result, identity=ident, status="solution")
        atomic_json(attempt / "final.json", sealed(raw))
        checkpoint = out / "calibration_runs/failed.json"
        record = dict(identity=ident, complete=True, status="forced_timeout", forced=True, success=False,
            all_processes_exited=True, finalization_s=None, result=None, attempt=str(attempt))
        save_record(checkpoint, record)
        diagnostic = dict(case=case, observations=[dict(cached=cached, columns=100, values_digest="same", seconds=1.)
                                                  for cached in (False, True)])
        path = out / "diagnostic.json"
        atomic_json(path, diagnostic)
        return manifest, before, models, previous, checkpoint, record, path

    def test_budget_validator_supervision_and_training_sources_cannot_change(self):
        for name in ("common.py", "supervisor.py", "physics.py", "worker.py", "generation.py", "workflows.py"):
            path = "src/operational_exit/" + name
            with self.subTest(path=path), self.assertRaises(ValueError):
                check_sources({path: "old"}, {path: "new"})

    def test_raw_diagnostic_never_replaces_timeout_and_unrelated_failures_are_rejected(self):
        with workspace_temp() as out:
            manifest, _, _, previous, checkpoint, record, _ = self.fixture(out)
            original = sha(checkpoint)
            identities, failures = check_pilots(manifest, out, previous["sha256"])
            self.assertEqual(failures, manifest["calibration_cases"])
            self.assertEqual(len(identities), 1)
            self.assertEqual(sha(checkpoint), original)
            self.assertFalse(load_record(checkpoint)["success"])
            self.assertIsNone(load_record(checkpoint)["result"])
            for changes in (dict(status="worker_error"), dict(all_processes_exited=False), dict(result={"status": "solution"})):
                save_record(checkpoint, dict(record, **changes))
                with self.assertRaises(ValueError):
                    check_pilots(manifest, out, previous["sha256"])
            save_record(checkpoint, record)
            (Path(record["attempt"]) / "final.json").unlink()
            with self.assertRaisesRegex(ValueError, "Missing post-solve"):
                check_pilots(manifest, out, previous["sha256"])

    def test_new_revision_keeps_models_and_failures_and_disallows_completed_calibration(self):
        with workspace_temp() as out:
            manifest, before, models, previous, checkpoint, _, diagnostic = self.fixture(out)
            current = dict(before, **{"src/operational_exit/solving.py": "single-copy",
                "src/operational_exit/solution_transfer_repair.py": "new"})
            protected = [checkpoint, out / STRUCTURAL_RECEIPT, out / "manifest.json", out / "state.json", *map(Path, models.values())]
            hashes = {p: sha(p) for p in protected}
            # Earlier amendment layers have their own integration regressions.
            with patch("src.operational_exit.protocol.source_signature", return_value=current), \
                 patch("src.operational_exit.protocol.hte_runtime_sources", return_value={}), \
                 patch("src.operational_exit.protocol.versions", return_value={}), \
                 patch("src.operational_exit.pilot_io_repair.verify_repair"), \
                 patch("src.operational_exit.repair_integrality.verify_repair"), \
                 patch("src.operational_exit.repair_structural_fixes.verify_repair"), \
                 patch("src.operational_exit.campaign.model_files", return_value=models), \
                 patch("src.operational_exit.solution_transfer_repair.structural_revision", return_value=previous["sha256"]):
                # Distinct source states force verification through every layer.
                for name, sources in (("pilot_io_amendment.json", {"src/operational_exit/solving.py": "io"}),
                                      ("repair_integrality_amendment.json", {"src/operational_exit/solving.py": "integer"})):
                    save_record(out / name, dict(sources=sources))
                with self.assertRaises(ValueError):
                    effective_sources(manifest, out)
                atomic_json(out / "calibration.json.bak", dict(complete=True))
                with self.assertRaisesRegex(ValueError, "calibrated/evaluated"):
                    register_repair(manifest, out, diagnostic)
                (out / "calibration.json.bak").unlink()
                register_repair(manifest, out, diagnostic)
                self.assertEqual(effective_sources(manifest, out, check_artifacts=True), current)
                fingerprints = {k: sha(v) for k, v in models.items()}
                revision = calibration_revision(out, fingerprints)
                self.assertEqual(revision, load_record(out / RECEIPT_NAME)["sha256"])
                self.assertNotEqual(revision, previous["sha256"])
                register_repair(manifest, out, diagnostic)
                self.assertEqual(calibration_revision(out, fingerprints), revision)
                self.assertEqual({p: sha(p) for p in protected}, hashes)
                Path(models["ae"]).write_text("changed", encoding="utf-8")
                with self.assertRaises(ValueError):
                    effective_sources(manifest, out, check_artifacts=True)


if __name__ == "__main__":
    unittest.main()
