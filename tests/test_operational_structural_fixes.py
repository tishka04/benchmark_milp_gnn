"""Repair must retain builder-fixed decisions for unavailable equipment."""
import os
os.environ["BENCHMARK_HIGHS_THREADS"] = "1"
import time
import unittest
from pathlib import Path
from unittest.mock import patch

import pandas  # noqa: F401 -- Windows DLL order.
import torch
from pyomo.environ import SolverFactory, Var, value

from test_milp_load_shifting import make_data
from test_operational_repair_integrality import SETTINGS
import test_operational_repair_integrality as integer_tests
import test_operational_pilot_io as io_tests
from src.milp.model import build_uc_model
from src.milp.lp_worker_two_stage import LPWorkerTwoStage
from src.operational_exit.physics import extract_dispatch, validate_dispatch
from src.operational_exit.solving import make_repair_worker
from src.operational_exit.common import atomic_json, digest, load_record, save_record, sha
from src.operational_exit.protocol import effective_sources
from src.operational_exit.repair_structural_fixes import (register_repair, calibration_revision,
    check_sources, check_pilots, RECEIPT_NAME)


def failed_pilot(manifest, revision):
    return dict(identity=dict(runtime_revision=revision, key=manifest["calibration_cases"][0], method="jepa",
        purpose="calibration", manifest_hash=digest(manifest)), complete=True, forced=False, all_processes_exited=True,
        status="invalid_dispatch", result=dict(candidate_diagnostics=[dict(status="invalid_dispatch",
        repair_stage="SolveStage.REPAIR_20", validation=dict(valid=False, violations={"zero_capacity:b_charge_mode": 1.},
        max_integrality_residual=0., integer_tolerance=1e-5))]))


def targets_for(model):
    return {name: {idx: 1. for idx in getattr(model, name)}
            for name in ("u_thermal", "b_charge_mode", "pumped_charge_mode", "dr_active")}


class StructuralFixTests(unittest.TestCase):
    def test_unavailable_devices_stay_fixed_but_decoder_choices_can_be_repaired(self):
        data = make_data(limits=(0, 4), zones=2)
        present = data.zones[1]
        data.thermal_capacity[present] = 100.
        data.battery_power[present] = data.pumped_power[present] = 10.
        for release in ("_unfix_subset", "_unfix_subset_all"):
            with self.subTest(release=release):
                model = build_uc_model(data)
                original = {v.name: value(v) for v in model.component_data_objects(Var) if v.fixed}
                targets = targets_for(model)
                worker = make_repair_worker(SETTINGS)
                worker._fix_all_binaries(model, targets)
                # Repeated calls must not mistake decoder choices for original fixes.
                worker._fix_all_binaries(model, targets)
                getattr(worker, release)(model, {(z, t) for z in model.Z for t in model.T})
                for var in model.component_data_objects(Var):
                    if var.name in original:
                        self.assertTrue(var.fixed, var.name)
                        self.assertEqual(value(var), original[var.name], var.name)
                        if var.parent_component().local_name in targets:
                            self.assertEqual(targets[var.parent_component().local_name][var.index()], original[var.name])
                self.assertFalse(model.u_thermal[present, 0].fixed)
                self.assertEqual(targets["u_thermal"][present, 0], 1.)
                if release == "_unfix_subset_all":
                    for name in ("b_charge_mode", "pumped_charge_mode", "dr_active"):
                        self.assertFalse(getattr(model, name)[present, 1].fixed, name)
                        self.assertEqual(targets[name][present, 1], 1.)

    def test_legacy_penalty_reactivates_absent_storage_but_corrected_repair_is_valid(self):
        data = make_data(demand=(0, 0), solar=(0, 0), limits=(0, 0))
        checks = []
        for corrected in (False, True):
            model = build_uc_model(data)
            worker = make_repair_worker(SETTINGS)
            targets = targets_for(model)
            for name in ("u_thermal", "dr_active"):
                targets[name] = {idx: 0. for idx in targets[name]}
            subset = set(targets["u_thermal"])
            if corrected:
                worker._fix_all_binaries(model, targets)
                worker._unfix_subset_all(model, subset)
            else:
                LPWorkerTwoStage._fix_all_binaries(worker, model, targets)
                LPWorkerTwoStage._unfix_subset_all(worker, model, subset)
            worker._add_deviation_penalty_on_subset(model, targets, subset, 10000., penalize_all_binaries=True)
            worker._add_flip_budget_constraint(model, targets, subset, 0, include_all_binaries=True)
            solver = SolverFactory("highs")
            solver.options.update({"threads": 1, "time_limit": 5})
            solver.solve(model)
            checks.append(validate_dispatch(data, extract_dispatch(model)))
            self.assertAlmostEqual(value(model.flip_budget.body), 0.)
            self.assertAlmostEqual(value(model.penalty_term_expr), 0.)
        self.assertEqual(checks[0]["violations"], {"zero_capacity:b_charge_mode": 1., "zero_capacity:pumped_charge_mode": 1.})
        self.assertFalse(checks[0]["valid"])
        self.assertTrue(checks[1]["valid"], checks[1])

    def test_full_cascade_with_impossible_targets_and_zero_flip_budgets(self):
        data = make_data(demand=(10, 10), solar=(0, 0), limits=(0, 0))
        settings = dict(SETTINGS, flip_budget_20=0, flip_budget_100=0)
        worker = make_repair_worker(settings)
        worker.deadline = time.perf_counter() + 10
        worker.dt_hours = data.dt_hours
        candidate = torch.ones(1, 2, 7)
        candidate[:, :, (1, 3)] = 0  # Request charge on absent storage.
        with patch.object(worker, "_find_scenario_path", return_value="synthetic.json"), \
             patch("src.milp.lp_worker_two_stage.load_scenario_data", return_value=data), \
             patch.object(worker, "_unfix_subset_all", wraps=worker._unfix_subset_all) as unfix:
            result = worker.solve("synthetic", candidate, max_stages=3, candidate_time_budget_s=10)
        self.assertGreaterEqual(unfix.call_count, 2)  # Both repair stages ran.
        check = validate_dispatch(data, result.continuous_vars["operational_dispatch"])
        self.assertTrue(check["valid"], check)
        self.assertEqual(check["ens_mwh"], 20.)


class StructuralRevisionTests(unittest.TestCase):
    def test_unrelated_source_changes_are_rejected(self):
        for path in ("src/operational_exit/physics.py", "src/milp/model.py", "src/operational_exit/worker.py",
                     "src/operational_exit/generation.py", "src/uc_jepa/training/train_jepa.py"):
            with self.subTest(path=path), self.assertRaises(ValueError):
                check_sources({path: "before"}, {path: "after"})

    def test_unrelated_or_unclean_pilot_failures_are_rejected(self):
        with io_tests.workspace_temp() as out:
            manifest = dict(calibration_cases=["validation"])
            record = failed_pilot(manifest, "old")
            checkpoint = out / "calibration_runs/failed.json"
            save_record(checkpoint, record)
            self.assertEqual(len(check_pilots(manifest, out, "old")), 1)
            for changes in (dict(forced=True), dict(all_processes_exited=False), dict(status="forced_timeout")):
                save_record(checkpoint, dict(record, **changes))
                with self.assertRaises(ValueError):
                    check_pilots(manifest, out, "old")
            for violations, integral in (({"nodal_balance": 10.}, 0.), ({"zero_capacity:b_charge_mode": 1.}, .2), ({}, 0.)):
                record["result"]["candidate_diagnostics"][0]["validation"].update(
                    violations=violations, max_integrality_residual=integral)
                save_record(checkpoint, record)
                with self.assertRaises(ValueError):
                    check_pilots(manifest, out, "old")

    def test_chained_revision_preserves_history_and_requires_fresh_calibration(self):
        with io_tests.workspace_temp() as out:
            manifest, _, io_sources, models, old_failure = io_tests.PilotRepairTests().fixture(out)
            with patch("src.operational_exit.protocol.versions", return_value={}), \
                 patch("src.operational_exit.campaign.model_files", return_value=models):
                with patch("src.operational_exit.protocol.source_signature", return_value=io_sources):
                    integer_tests.register_io(manifest, out)
                io_receipt = load_record(out / integer_tests.IO_RECEIPT)
                identity = dict(method="pipeline", key=manifest["calibration_cases"][0], purpose="calibration",
                                runtime_revision=io_receipt["sha256"], manifest_hash=digest(manifest))
                save_record(out / "calibration_runs/fractional.json", dict(identity=identity, complete=True,
                    forced=False, all_processes_exited=True, status="invalid_dispatch", result=dict(generated_candidates=16,
                    candidate_diagnostics=[dict(status="invalid_dispatch", repair_status="optimal",
                    repair_stage="SolveStage.HARD_FIX") for _ in range(16)])))
                data, before, after = integer_tests.dispatch_pair()
                paths = [out / "before.json", out / "after.json"]
                for path, dispatch in zip(paths, (before, after)):
                    atomic_json(path, dict(case=manifest["train"][0], dispatch=dispatch))
                integer_sources = dict(io_sources, **{"src/operational_exit/solving.py": "integral",
                    "src/operational_exit/repair_integrality.py": "new"})
                with patch("src.operational_exit.protocol.source_signature", return_value=integer_sources), \
                     patch("src.milp.scenario_loader.load_scenario_data", return_value=data):
                    integer_tests.register_repair(manifest, out, *paths)
                previous = load_record(out / integer_tests.RECEIPT_NAME)
                save_record(out / "calibration_runs/structural.json", failed_pilot(manifest, previous["sha256"]))
                success = dict(failed_pilot(manifest, previous["sha256"]), status="success")
                success["identity"] = dict(success["identity"], method="highs")
                save_record(out / "calibration_runs/success.json", success)
                protected = [old_failure, out / "manifest.json", out / "state.json", *map(Path, models.values()),
                    out / integer_tests.IO_RECEIPT, out / integer_tests.RECEIPT_NAME,
                    *sorted((out / "calibration_runs").glob("*.json"))]
                hashes = {p: sha(p) for p in protected}
                current = dict(integer_sources, **{"src/operational_exit/solving.py": "structural",
                    "src/operational_exit/repair_structural_fixes.py": "new"})
                with patch("src.operational_exit.protocol.source_signature", return_value=current):
                    with self.assertRaises(ValueError):
                        effective_sources(manifest, out)
                    # A partially published calibration is sufficient to disallow revision.
                    atomic_json(out / "calibration.json.bak", dict(complete=True))
                    with self.assertRaisesRegex(ValueError, "calibrated/evaluated"):
                        register_repair(manifest, out)
                    (out / "calibration.json.bak").unlink()
                    register_repair(manifest, out)
                    self.assertEqual(effective_sources(manifest, out, check_artifacts=True), current)
                    fingerprints = {k: sha(v) for k, v in models.items()}
                    revision = calibration_revision(out, fingerprints)
                    self.assertNotEqual(revision, previous["sha256"])
                    receipt = load_record(out / RECEIPT_NAME)
                    self.assertEqual(len(receipt["superseded_pilots"]), 2)
                    register_repair(manifest, out)  # Idempotent; retain the same pilot identity.
                    self.assertEqual(calibration_revision(out, fingerprints), revision)
                    self.assertEqual({p: sha(p) for p in protected}, hashes)
                    seen = []
                    def fake_run(manifest, out, case, item, *args):
                        seen.append(item)
                        return dict(status="success", forced=False, finalization_s=.1, all_processes_exited=True,
                            preload_metadata={"repair_processes": 16 if item["method"] == "pipeline" else 1},
                            result={"generated_candidates": 16 if item["method"] == "pipeline" else 1, "gpu_peak_bytes": 1})
                    from src.operational_exit.campaign import calibrate
                    with patch("src.operational_exit.campaign.one_run", side_effect=fake_run):
                        calibrate(manifest, out, lambda *args: None)
                    self.assertEqual(len(seen), 4)
                    self.assertTrue(all(item["runtime_revision"] == revision for item in seen))
                    self.assertEqual({p: sha(p) for p in protected}, hashes)
                    Path(models["jepa"]).write_text("changed model", encoding="utf-8")
                    with self.assertRaises(ValueError):
                        effective_sources(manifest, out, check_artifacts=True)


if __name__ == "__main__":
    unittest.main()
