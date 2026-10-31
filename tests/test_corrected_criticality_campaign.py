"""Control-flow tests only: solver/process creation and statistical fitting guarded."""
import copy
import json
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from unittest.mock import Mock, patch

import numpy as np
import pandas as pd

from scripts import recompute_thermal_references as campaign
from scripts import analyze_corrected_criticality as audit
from scripts import analyze_criticality_existing as diagnostic


class CorrectedCampaignTests(unittest.TestCase):
    def setUp(self):
        base = campaign.ROOT/"tmp"
        base.mkdir(exist_ok=True)
        self.temp = tempfile.TemporaryDirectory(prefix="corrected-control-test-", dir=base)
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.source, self.out = self.root/"old", self.root/"new"
        self.source.mkdir(); self.out.mkdir()
        self.process_guard = patch.object(campaign.subprocess, "Popen", side_effect=AssertionError("Real process forbidden"))
        self.process_guard.start(); self.addCleanup(self.process_guard.stop)

    def test_refuses_overlapping_output(self):
        for out in (self.source, self.source/"reports", self.source.parent):
            with self.assertRaises(ValueError):
                campaign.validate_paths(self.source, out)
        campaign.validate_paths(self.source, self.out)

    def test_manifest_resume_requires_identical_inputs_and_solver(self):
        inv = dict(source_root=str(self.source), cases=[], original_manifest_sha256="old")
        with patch.object(campaign, "inventory", return_value=inv), patch.object(campaign, "solver_signature", return_value={"threads": 16}):
            first = campaign.prepare_manifest(self.source, self.out, False)
            self.assertEqual(first, campaign.prepare_manifest(self.source, self.out, True))
            with self.assertRaisesRegex(ValueError, "-Resume"):
                campaign.prepare_manifest(self.source, self.out, False)
        with patch.object(campaign, "inventory", return_value=inv), patch.object(campaign, "solver_signature", return_value={"threads": 4}):
            with self.assertRaisesRegex(ValueError, "solver"):
                campaign.prepare_manifest(self.source, self.out, True)

    def test_manifest_does_not_claim_nonempty_unowned_directory(self):
        (self.out/"some-result.json").write_text("{}")
        with patch.object(campaign, "inventory", return_value={}), patch.object(campaign, "solver_signature", return_value={}):
            with self.assertRaisesRegex(ValueError, "non vide"):
                campaign.prepare_manifest(self.source, self.out, False)

    def test_preview_never_creates_files_or_calls_solver(self):
        args = SimpleNamespace(source_root=self.source, output_dir=self.out/"absent", preview=True)
        with patch.object(campaign, "inventory", return_value={"cases": [1]*600}), \
             patch.object(campaign, "solver_signature", return_value={}), patch.object(campaign, "exclusive_lock", side_effect=AssertionError("No lock in preview")):
            campaign.run(args)
        self.assertFalse(args.output_dir.exists())

    def test_damaged_auxiliary_state_is_rebuilt_not_trusted(self):
        p = self.out/"state.json"
        p.write_text("")
        self.assertEqual(campaign.optional_checkpoint(p), {})
        with patch.object(campaign, "read_json", side_effect=PermissionError("locked")):
            with self.assertRaises(PermissionError):
                campaign.optional_checkpoint(p)

    def test_locked_output_is_retried_and_not_reported_as_corrupt(self):
        with patch.object(Path, "read_text", side_effect=[PermissionError("locked"), '{"ok": true}']), \
             patch.object(campaign.time, "sleep") as wait:
            self.assertEqual(campaign.read_json(self.out/"fake.json"), {"ok": True})
        wait.assert_called_once()

    def test_reference_checksum_and_campaign_identity_are_required(self):
        from src.milp import batch_runner as batch
        case = dict(family="low_criticality_scenarios", scenario_id="scenario_00001", scenario_path="unused.json")
        manifest = dict(source_root=str(self.source), cases=[case])
        payload = dict(reference_campaign_sha256=campaign.digest(manifest), thermal_formulation=campaign.THERMAL)
        payload["batch_checkpoint"] = dict(complete=True, provenance={"test": True}, report_sha256=batch._report_hash(payload))
        target = campaign.report_path(self.out, case)
        campaign.atomic_json(target, payload)
        with patch.object(batch, "_provenance", return_value={"test": True}), \
             patch.object(batch, "_validate_completed_report"), patch.object(campaign, "thermal_check"):
            self.assertIsNotNone(campaign.read_completed(self.out, case, manifest))
            changed = copy.deepcopy(payload); changed["thermal_formulation"] = "old"
            campaign.atomic_json(target, changed)
            self.assertIsNone(campaign.read_completed(self.out, case, manifest))
            target.write_text("")
            self.assertIsNone(campaign.read_completed(self.out, case, manifest))
            with patch.object(campaign, "read_json", side_effect=PermissionError("locked")):
                with self.assertRaises(PermissionError):
                    campaign.read_completed(self.out, case, manifest)

    def test_resume_skips_valid_reference_and_keeps_cached_analysis(self):
        cases = [dict(family="low_criticality_scenarios", scenario_id=f"scenario_{i:05}") for i in (1, 2)]
        inv = dict(source_root=str(self.source), original_manifest_sha256="old", cases=cases)
        manifest = dict(**inv, solver={"test": True})
        present = {"scenario_00001"}
        report = self.out/"analysis/rapport.html"
        report.parent.mkdir(); report.write_text("fake report")
        signature = {"test": True}
        campaign.atomic_json(self.out/"analysis_complete.json", dict(signature=signature, outputs={"analysis/rapport.html": campaign.sha(report)}))
        commands = []
        def child(command, *args, **kwargs):
            commands.append(command)
            self.assertIn("--worker", command)
            present.add("scenario_00002")
        args = SimpleNamespace(source_root=self.source, output_dir=self.out, preview=False, resume=True, start_step=1, repeats=10)
        with patch.object(campaign, "prepare_manifest", return_value=manifest), \
             patch.object(campaign, "inventory", return_value=inv), \
             patch.object(campaign, "solver_signature", return_value=manifest["solver"]), \
             patch.object(campaign, "read_completed", side_effect=lambda out, case, manifest: {} if case["scenario_id"] in present else None), \
             patch.object(campaign, "analyze_signature", return_value=signature), \
             patch.object(campaign, "call_child", side_effect=child):
            campaign.run(args)
        self.assertEqual(len(commands), 1)
        self.assertEqual(commands[0][-1], "1")
        self.assertEqual(campaign.read_json(self.out/"state.json")["status"], "completed")

    def test_worker_failure_stops_before_analysis_and_marks_failure(self):
        case = dict(family="low_criticality_scenarios", scenario_id="scenario_00001")
        manifest = dict(source_root=str(self.source), original_manifest_sha256="old", cases=[case], solver={})
        args = SimpleNamespace(source_root=self.source, output_dir=self.out, preview=False, resume=True, start_step=1, repeats=10)
        with patch.object(campaign, "prepare_manifest", return_value=manifest), \
             patch.object(campaign, "read_completed", return_value=None), \
             patch.object(campaign, "call_child", side_effect=RuntimeError("fake worker failed")) as child, \
             patch.object(campaign, "analyze_signature", side_effect=AssertionError("Analysis must not start")):
            with self.assertRaisesRegex(RuntimeError, "fake worker failed"):
                campaign.run(args)
        self.assertEqual(child.call_count, 1)
        state = campaign.read_json(self.out/"state.json")
        self.assertEqual((state["status"], state["stage"]), ("failed", 1))

    def test_analysis_only_refuses_missing_references(self):
        manifest = dict(cases=[{}])
        args = SimpleNamespace(source_root=self.source, output_dir=self.out, preview=False, resume=True, start_step=2, repeats=10)
        with patch.object(campaign, "prepare_manifest", return_value=manifest), patch.object(campaign, "read_completed", return_value=None), \
             patch.object(campaign, "call_child", side_effect=AssertionError("No worker")):
            with self.assertRaisesRegex(ValueError, "references manquantes"):
                campaign.run(args)

    def test_nonzero_child_exit_and_interruption_cleanup(self):
        child = Mock()
        child.poll.return_value = 7
        child.returncode = 7
        with patch.object(campaign.subprocess, "Popen", return_value=child):
            with self.assertRaisesRegex(RuntimeError, "code 7"):
                campaign.call_child(["fake"], self.out/"child.log", {}, Mock())
        child = Mock(); child.poll.return_value = None
        with patch.object(campaign.subprocess, "Popen", return_value=child):
            with self.assertRaises(KeyboardInterrupt):
                campaign.call_child(["fake"], self.out/"interrupt.log", {}, Mock(side_effect=KeyboardInterrupt))
        child.terminate.assert_called_once()


class DiagnosticGuardTests(unittest.TestCase):
    def test_constant_and_rare_responses_are_explicitly_unestimable(self):
        d = pd.DataFrame({"unserved_log": [0.]*100, "ref_timeout": [0]*99+[1], "runtime_log": np.linspace(1, 3, 100)})
        self.assertEqual(diagnostic.endpoint_support(d, "unserved_log", 5), "constant response")
        self.assertIn("fewer than 5", diagnostic.endpoint_support(d, "ref_timeout", 5))
        self.assertIsNone(diagnostic.endpoint_support(d, "runtime_log", 5))
        d["unserved_log"] = np.linspace(.1, 1, 100)
        self.assertIsNone(diagnostic.endpoint_support(d, "unserved_log", 5))

    def test_no_regression_runs_when_corrected_response_is_constant(self):
        with tempfile.TemporaryDirectory(dir=campaign.ROOT/"tmp") as tmp:
            frame = pd.DataFrame({"family": [diagnostic.FAMILIES[0]]*20, "ref_optimal": [True]*20,
                "thermal_startup_blocked": [False]*20, "unserved_log": [0.]*20})
            skipped = []
            with patch.object(diagnostic, "fit_predict", side_effect=AssertionError("No fit")), \
                 patch.object(diagnostic, "size_basis", side_effect=AssertionError("No design")):
                associations = diagnostic.associations(frame, [], Path(tmp), endpoints=["unserved_log"], minimum_events=5, skipped=skipped)
                metrics = diagnostic.validation(frame, [], Path(tmp), endpoints=["unserved_log"], minimum_events=5, skipped=skipped)
            self.assertTrue(associations.empty and metrics.empty)
            self.assertTrue(skipped)
            self.assertIn("model", pd.read_csv(Path(tmp)/"validation.csv").columns)

    def test_timeout_is_not_converted_to_an_optimal_result(self):
        report = {"mip": {"objective": -100, "solve_seconds": 605, "termination": "maxTimeLimit"},
            "detail": {"dt_hours": 1, "demand": {"z": [10, 10]}, "unserved": {"z": [0, 1]},
                       "overgen_spill": {"z": [0, 0]}, "thermal": {"z": [2, 3]}}}
        result = audit.outcome_row(report)
        self.assertTrue(result["ref_timeout"])
        self.assertFalse(result["ref_optimal"])
        self.assertEqual(result["runtime_log"], np.log1p(600))
        self.assertEqual(result["unserved_pct"], 5)
        self.assertEqual(result["reference_objective"], -100)

    def test_no_historical_neural_responses_requested(self):
        self.assertEqual(set(audit.ENDPOINTS), {"unserved_log", "overgen_log", "runtime_log", "ref_timeout"})

    def test_before_after_pairs_join_by_identity_and_use_absolute_cost(self):
        base = dict(family="f", unserved_mwh=0, overgen_mwh=0, ref_seconds=2, ref_optimal=True, ref_timeout=False, thermal_mwh=0)
        before = pd.DataFrame([dict(base, scenario_id="a", reference_objective=-100), dict(base, scenario_id="b", reference_objective=100)])
        after = pd.DataFrame([dict(base, scenario_id="b", reference_objective=50), dict(base, scenario_id="a", reference_objective=-150)])
        compared = audit.compare_rows(before, after)
        self.assertEqual(compared.delta_cost_normalized_pct.tolist(), [-50, -50])
        with self.assertRaises(pd.errors.MergeError):
            audit.compare_rows(before, pd.concat([after, after]))

    def test_report_handles_unestimable_responses_without_fitting(self):
        # The data are artificial: this validates rendering, not any experiment.
        rows = []
        for family in diagnostic.FAMILIES:
            rows.append(dict(family=family, scenario_id="fake", ref_optimal=True, ref_timeout=False,
                unserved_mwh=0., overgen_mwh=0., ref_seconds=1., reference_objective=-10., thermal_mwh=0.))
        frame = pd.DataFrame(rows)
        results = {v: {"validation": pd.DataFrame(columns=["scheme", "endpoint", "model", "r2_cv", "delta_r2", "delta_low", "delta_high"]),
            "associations": pd.DataFrame(columns=["scope", "adjustment", "feature", "rho", "endpoint", "n", "q_bh"])} for v in ("before", "after")}
        with tempfile.TemporaryDirectory(dir=campaign.ROOT/"tmp") as tmp, \
             patch.object(diagnostic, "fit_predict", side_effect=AssertionError("No fitting")):
            out = Path(tmp)
            audit.render(out, frame, frame, results, pd.DataFrame([dict(reason="constant response")]), pd.DataFrame())
            self.assertTrue((out/"figures/02_apport_au_dela_taille.png").is_file())
            self.assertIn("NaN signifie non", (out/"rapport.html").read_text(encoding="utf-8"))


if __name__ == "__main__":
    unittest.main()
