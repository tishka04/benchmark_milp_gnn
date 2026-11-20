"""Regression for fractional import/export in the legacy hard-fix relaxation."""
import os
os.environ["BENCHMARK_HIGHS_THREADS"] = "1"
from pathlib import Path
import time
import unittest
from unittest.mock import patch

import pandas  # noqa: F401 -- Windows DLL order.
import torch
from pyomo.environ import SolverFactory, Var, value
from pyomo.core import TransformationFactory

from test_milp_load_shifting import make_data
from test_thermal_transitions import fixture as thermal_fixture
import test_operational_pilot_io as pilot_io_tests
from test_operational_pilot_io import workspace_temp
from src.milp.model import build_uc_model
from src.milp.lp_worker_two_stage import LPWorkerTwoStage
from src.operational_exit.common import atomic_json, digest, load_record, save_record, sha
from src.operational_exit.physics import extract_dispatch, validate_dispatch
from src.operational_exit.solving import make_repair_worker
from src.operational_exit.pilot_io_repair import register_repair as register_io, RECEIPT_NAME as IO_RECEIPT
from src.operational_exit.repair_integrality import (register_repair, check_sources, check_pilots,
    calibration_revision, RECEIPT_NAME)
from src.operational_exit.protocol import effective_sources

SETTINGS = dict(max_stages=3, slack_tol_mwh=1., deviation_penalty=10000., flip_budget_20=100, flip_budget_100=1000)


def arbitrage_data():
    data = make_data(demand=(0, 0), solar=(0, 0), limits=(0, 0))
    data.import_capacity, data.import_cost, data.export_cost = 100., 10., -20.
    return data


def dispatch_pair():
    data = arbitrage_data()
    model = build_uc_model(data)
    TransformationFactory("core.relax_integer_vars").apply_to(model)
    solver = SolverFactory("highs")
    solver.options.update({"threads": 1, "time_limit": 5})
    solver.solve(model)
    before = extract_dispatch(model)
    worker = make_repair_worker(SETTINGS)
    worker.deadline = time.perf_counter() + 5
    _, _, has = worker._solve_with_timeout(model, 5)
    assert has
    return data, before, extract_dispatch(model)


class RepairIntegralityTests(unittest.TestCase):
    def test_fractional_legacy_dispatch_is_rejected_and_corrected_dispatch_is_valid(self):
        data, before, after = dispatch_pair()
        old_check, new_check = [validate_dispatch(data, d) for d in (before, after)]
        self.assertEqual(old_check["violations"], {})
        self.assertAlmostEqual(old_check["max_integrality_residual"], .5)
        self.assertFalse(old_check["valid"])
        self.assertGreater(sum(before["values"]["net_import"]), 0)
        self.assertGreater(sum(before["values"]["net_export"]), 0)
        self.assertTrue(new_check["valid"], new_check)
        self.assertEqual(new_check["max_integrality_residual"], 0)
        self.assertEqual(sum(after["values"]["net_import"]), 0)
        self.assertEqual(sum(after["values"]["net_export"]), 0)

    def test_original_domains_and_fixed_decoder_values_survive_restoration(self):
        model = build_uc_model(thermal_fixture())
        zone = model.Z.first()
        model.u_thermal[zone, 0].fix(1)
        original = {v.name: (v.is_binary(), v.fixed, v.value if v.fixed else None)
                    for v in model.component_data_objects(Var)}
        TransformationFactory("core.relax_integer_vars").apply_to(model)
        worker = make_repair_worker(SETTINGS)
        def inspect(_worker, restored, seconds):
            self.assertGreater(seconds, 0)
            for v in restored.component_data_objects(Var):
                binary, fixed, val = original[v.name]
                self.assertEqual(v.is_binary(), binary, v.name)
                self.assertEqual(v.fixed, fixed, v.name)
                if fixed:
                    self.assertEqual(value(v), val, v.name)
            return None, 0, False
        with patch.object(LPWorkerTwoStage, "_solve_with_timeout", inspect):
            worker._solve_with_timeout(model, 5)

    def test_full_three_stage_cascade_returns_integral_dispatch(self):
        data = arbitrage_data()
        worker = make_repair_worker(SETTINGS)
        worker.deadline = time.perf_counter() + 5
        worker.dt_hours = data.dt_hours
        with patch.object(worker, "_find_scenario_path", return_value="synthetic.json"), \
             patch("src.milp.lp_worker_two_stage.load_scenario_data", return_value=data):
            result = worker.solve("synthetic", torch.zeros(1, 2, 7), max_stages=3, candidate_time_budget_s=5)
        self.assertEqual(str(result.stage_used), "SolveStage.HARD_FIX")
        check = validate_dispatch(data, result.continuous_vars["operational_dispatch"])
        self.assertTrue(check["valid"], check)

    def test_expired_deadline_still_prevents_solver_call(self):
        worker = make_repair_worker(SETTINGS)
        worker.deadline = 0
        model = build_uc_model(arbitrage_data())
        TransformationFactory("core.relax_integer_vars").apply_to(model)
        with patch.object(LPWorkerTwoStage, "_solve_with_timeout") as solve:
            with self.assertRaises(TimeoutError):
                worker._solve_with_timeout(model, 5)
            solve.assert_not_called()


class IntegralityRevisionTests(unittest.TestCase):
    def test_validator_models_and_training_sources_cannot_change(self):
        for path in ("src/operational_exit/physics.py", "src/operational_exit/generation.py", "src/milp/model.py",
                     "src/uc_jepa/training/train_autoencoder.py", "configs/uc_jepa/autoencoder.yaml"):
            with self.subTest(path=path), self.assertRaises(ValueError):
                check_sources({path: "before"}, {path: "after"})

    def test_completed_timeouts_are_not_retried_under_this_correction(self):
        with workspace_temp() as out:
            manifest = dict(calibration_cases=["v"])
            save_record(out / "calibration_runs/failed.json", dict(identity=dict(runtime_revision="old", key="v",
                purpose="calibration", manifest_hash=digest(manifest)), complete=True, forced=True,
                all_processes_exited=True, status="forced_timeout"))
            with self.assertRaisesRegex(ValueError, "timeout/unrelated"):
                check_pilots(manifest, out, "old")

    def test_revision_preserves_previous_models_success_and_failures(self):
        with workspace_temp() as out:
            manifest, _, io_sources, models, original_failure = pilot_io_tests.PilotRepairTests().fixture(out)
            with patch("src.operational_exit.protocol.versions", return_value={}), \
                 patch("src.operational_exit.campaign.model_files", return_value=models):
                with patch("src.operational_exit.protocol.source_signature", return_value=io_sources):
                    register_io(manifest, out)
                old = load_record(out / IO_RECEIPT)
                for method, status in (("highs", "success"), ("pipeline", "invalid_dispatch")):
                    ident = dict(method=method, key=manifest["calibration_cases"][0], purpose="calibration",
                                 runtime_revision=old["sha256"], manifest_hash=digest(manifest))
                    result = dict(generated_candidates=16, candidate_diagnostics=[dict(status="invalid_dispatch",
                                  repair_status="optimal", repair_stage="SolveStage.HARD_FIX") for _ in range(16)])
                    save_record(out / "calibration_runs" / (method + ".json"), dict(identity=ident, complete=True,
                                forced=False, all_processes_exited=True, status=status, result=result))
                data, before, after = dispatch_pair()
                paths = [out / "before.json", out / "after.json"]
                for path, dispatch in zip(paths, (before, after)):
                    atomic_json(path, dict(case=manifest["train"][0], dispatch=dispatch))
                protected = [original_failure, out / IO_RECEIPT, out / "manifest.json",
                             out / "calibration_runs/highs.json", out / "calibration_runs/pipeline.json", *map(Path, models.values())]
                hashes = {p: sha(p) for p in protected}
                current = dict(io_sources, **{"src/operational_exit/solving.py": "corrected",
                                              "src/operational_exit/repair_integrality.py": "new"})
                with patch("src.operational_exit.protocol.source_signature", return_value=current), \
                     patch("src.milp.scenario_loader.load_scenario_data", return_value=data):
                    with self.assertRaises(ValueError):
                        effective_sources(manifest, out)
                    register_repair(manifest, out, *paths)
                    self.assertEqual(effective_sources(manifest, out, check_artifacts=True), current)
                    receipt = load_record(out / RECEIPT_NAME)
                    revision = calibration_revision(out, {k: sha(v) for k, v in models.items()})
                    self.assertEqual(revision, receipt["sha256"])
                    self.assertNotEqual(revision, old["sha256"])
                    self.assertEqual({p: sha(p) for p in protected}, hashes)
                    self.assertEqual(len(receipt["superseded_pilots"]), 2)
                    changed = dict(current, **{"src/operational_exit/physics.py": "altered validator"})
                    with patch("src.operational_exit.protocol.source_signature", return_value=changed), self.assertRaises(ValueError):
                        effective_sources(manifest, out)


if __name__ == "__main__":
    unittest.main()
