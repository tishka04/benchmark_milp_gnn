"""Only synthetic data/processes. Never launches real-case pilots or training."""
from __future__ import annotations

import copy
import json
import os
os.environ["BENCHMARK_HIGHS_THREADS"] = "1"
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import pandas  # Windows DLL import order.
import numpy as np
import torch
from pyomo.environ import SolverFactory, Var, value
from pyomo.core import TransformationFactory

from test_milp_load_shifting import make_data, scenario_input
from test_thermal_transitions import fixture
from src.milp.model import build_uc_model
from src.milp.scenario_loader import NetworkLine
from src.operational_exit.common import (BUDGETS, METHODS, FAMILIES, PHYSICS, digest,
    atomic_json, save_record, load_record, read_json, seed_for, sha)
from src.operational_exit.physics import (extract_dispatch, validate_dispatch, apply_cost_units,
                                          from_report, to_report)
from src.operational_exit.solving import solve_data
from src.operational_exit.protocol import schedule, freeze_splits, size_selection
from src.operational_exit.supervisor import classify, run_experiment
from src.operational_exit.analysis import gap, summarize
from src.operational_exit.training_runtime import (TrainingRuntime, backward_step, rng_state,
                                                   restore_rng, save_training_checkpoint)


def tempdir():
    # The managed Windows temporary directory permits file creation but may
    # deny renames; exercise actual atomic replacement in the writable workspace.
    root = Path(__file__).resolve().parents[1] / "tmp"
    root.mkdir(exist_ok=True)
    return tempfile.TemporaryDirectory(dir=root, prefix="operational_test_")


class PhysicsTests(unittest.TestCase):
    def solve(self, data, fixed=None):
        m = build_uc_model(data)
        apply_cost_units(m)
        if fixed:
            for v in m.component_data_objects(Var):
                if v.is_binary():
                    v.fix(round(fixed[v.name]))
            TransformationFactory("core.relax_integer_vars").apply_to(m)
        solver = SolverFactory("highs")
        solver.options.update({"threads": 1, "time_limit": 5})
        solver.solve(m)
        return m, extract_dispatch(m)

    def test_thermal_ramp_below_pmin_and_fixed_lp(self):
        data = fixture()
        m, d = self.solve(data)
        check = validate_dispatch(data, d)
        self.assertTrue(check["valid"], check)
        self.assertEqual(d["values"]["p_thermal"], [[45, 65, 45, 0]])
        fixed = {v.name: value(v) for v in m.component_data_objects(Var) if v.is_binary()}
        lp, dl = self.solve(data, fixed)
        self.assertTrue(validate_dispatch(data, dl)["valid"])
        self.assertAlmostEqual(value(m.obj), value(lp.obj))
        self.assertAlmostEqual(check["cost"], 162)

    def test_initial_on_and_shutdown_from_high_power(self):
        data = fixture(demand=(70, 50, 45, 0), initial=90)
        _, d = self.solve(data)
        self.assertTrue(validate_dispatch(data, d)["valid"])
        self.assertEqual(sum(d["values"]["v_thermal_startup"][0]), 0)
        d["values"]["p_thermal"][0][0] = 0
        d["values"]["u_thermal"][0][0] = 0
        check = validate_dispatch(data, d)
        self.assertFalse(check["valid"])
        self.assertIn("thermal_ramp_down", check["violations"])

    def test_cost_units_and_startup_not_scaled(self):
        for dt in (.25, .5, 1, 2):
            with self.subTest(dt=dt):
                data = fixture(demand=(45,))
                data.dt_hours = dt
                m, d = self.solve(data)
                check = validate_dispatch(data, d)
                self.assertTrue(check["valid"], check)
                self.assertAlmostEqual(check["cost"], 45 * dt + 7)
                self.assertAlmostEqual(value(m.obj), check["cost"])

    def test_cost_units_correct_active_soft_repair_objective(self):
        from pyomo.environ import Objective
        data = fixture(demand=(45,))
        data.dt_hours = .5
        m = build_uc_model(data)
        m.base_obj_expr = m.obj.expr
        m.obj.deactivate()
        m.obj_with_penalty = Objective(expr=m.base_obj_expr + 100 * m.v_thermal_startup[data.zones[0], 0])
        apply_cost_units(m)
        solver = SolverFactory("highs")
        solver.options.update({"threads": 1, "time_limit": 5})
        solver.solve(m)
        check = validate_dispatch(data, extract_dispatch(m))
        self.assertTrue(check["valid"])
        self.assertAlmostEqual(check["cost"], 29.5)
        self.assertAlmostEqual(value(m.obj_with_penalty), check["cost"] + 100)
        self.assertAlmostEqual(value(m.base_obj_expr), check["cost"])

    def test_load_shifting_and_native_optional_lp(self):
        for dt in (.5, 1, 2):
            data = make_data(dt=dt)
            result = solve_data(data, threads=1, seconds=5, relaxation=False)
            self.assertTrue(result["valid"], result)
            self.assertEqual(result["lp"]["status"], "skipped")
            self.assertAlmostEqual(result["validation"]["ens_mwh"], 0)
            self.assertIsNotNone(result["bound"])
            self.assertTrue(result["proven_optimal"])
        result = solve_data(make_data(), threads=1, seconds=5, relaxation=True)
        self.assertIsNotNone(result["lp"]["objective"])

    def test_zero_capacities_allow_measured_ens(self):
        data = make_data(demand=(10,), solar=(0,), limits=(0,))
        _, d = self.solve(data)
        check = validate_dispatch(data, d)
        self.assertTrue(check["valid"], check)
        self.assertAlmostEqual(check["ens_mwh"], 10)
        self.assertAlmostEqual(check["cost"], data.voll * 10)

    def test_injected_full_dispatch_violations(self):
        data = make_data()
        _, original = self.solve(data)
        for name in ("p_thermal", "b_soc", "pumped_level", "h_level", "dr_rebound", "dr_recover",
                     "unserved", "dr_shed_block", "net_import", "spill_wind"):
            with self.subTest(name=name):
                d = copy.deepcopy(original)
                a = np.array(d["values"][name])
                a.flat[0] += 2
                d["values"][name] = a.tolist()
                self.assertFalse(validate_dispatch(data, d)["valid"])
        for name in ("dr_active", "u_thermal", "b_charge_mode", "pumped_charge_mode", "import_mode"):
            d = copy.deepcopy(original)
            a = np.array(d["values"][name])
            a.flat[0] = .00005  # Below physical tol, ABOVE integrality tol.
            d["values"][name] = a.tolist()
            self.assertFalse(validate_dispatch(data, d)["valid"])

    def test_line_capacity_and_nodal_balance(self):
        data = make_data(zones=2)
        a, b = data.zones
        data.lines = {"L": NetworkLine("L", a, b, 1., 10.)}
        data.lines_from_index = {a: ("L",)}
        data.lines_to_index = {b: ("L",)}
        _, d = self.solve(data)
        self.assertTrue(validate_dispatch(data, d)["valid"])
        d["values"]["flow"][0][0] = 2
        check = validate_dispatch(data, d)
        self.assertIn("line_capacity:L", check["violations"])
        self.assertIn("nodal_balance", check["violations"])

    def test_storage_efficiencies_and_terminal_states(self):
        data = make_data()
        z = data.zones[0]
        for prefix in ("battery", "pumped"):
            getattr(data, prefix + "_power")[z] = 3
            getattr(data, prefix + "_energy")[z] = 5
            getattr(data, prefix + "_initial")[z] = 2
            getattr(data, prefix + "_final_min")[z] = 2
            getattr(data, prefix + "_final_max")[z] = 3
            setattr(data, prefix + "_eta_charge", .9)
            setattr(data, prefix + "_eta_discharge", .8)
        _, d = self.solve(data)
        check = validate_dispatch(data, d)
        self.assertTrue(check["valid"], check)
        d["values"]["b_soc"][0][-1] = 1
        self.assertIn("battery_final_min", validate_dispatch(data, d)["violations"])

    def test_strict_extraction_and_report_roundtrip(self):
        data = fixture()
        _, d = self.solve(data)
        self.assertEqual(from_report(to_report(d, data), data), d)
        broken = copy.deepcopy(d)
        del broken["values"]["flow"]
        self.assertFalse(validate_dispatch(data, broken)["valid"])
        broken = copy.deepcopy(d)
        broken["values"]["unserved"][0][0] = float("nan")
        self.assertFalse(validate_dispatch(data, broken)["valid"])

    def test_no_incumbent_without_solver_call_when_deadline_expired(self):
        r = solve_data(fixture(), threads=1, seconds=5, deadline=0)
        self.assertEqual(r["status"], "no_incumbent")
        self.assertFalse(r["valid"])


class ProtocolTests(unittest.TestCase):
    def manifest(self):
        cohort = [dict(key=f"{f}/s{i:03}", family=f, graph_size=i) for f in FAMILIES for i in range(100)]
        return dict(cohort=cohort, variability_cases=sum((size_selection([c for c in cohort if c["family"] == f], 5) for f in FAMILIES), []))

    def test_exact_counts_budget_sums_seeds_and_balanced_order(self):
        m = self.manifest()
        main, extra = list(schedule(m)), list(schedule(m, extra=True))
        self.assertEqual(len(main), 16800)
        self.assertEqual(len(extra), 1680)
        self.assertEqual(sum(r["budget"] for r in main) / 3600, 152)
        self.assertEqual(sum(r["budget"] for r in extra) / 3600, 15.2)
        self.assertEqual(len({digest(r) for r in main + extra}), 18480)
        first = main[0]
        self.assertEqual(len({r["seed"] for r in main if r["key"] == first["key"]}), 1)
        self.assertNotEqual(seed_for("key", 0, 0), seed_for("key", 0, 1))
        self.assertEqual({r["method"] for r in main[:4]}, set(METHODS))

    def test_historical_split_frozen_before_new_classification(self):
        with tempdir() as temp:
            source = Path(temp)
            for i in range(10):
                p = source / "uc_jepa/dataset_train" / ("gold" if i < 4 else "silver") / f"scenario_{i:05}_uc.npz"
                p.parent.mkdir(parents=True, exist_ok=True)
                p.touch()
            train = [dict(key=f"scenarios_v3/scenario_{i:05}") for i in range(12)]
            split = freeze_splits(train, source)
            self.assertEqual(split["historical_count"], 10)
            self.assertEqual(len(split["assignments"]), 12)
            expected = torch.randperm(10, generator=torch.Generator().manual_seed(42))[-1].item()
            self.assertEqual(split["assignments"][f"scenarios_v3/scenario_{expected:05}"], "validation")
            self.assertEqual(split, freeze_splits(train, source))

    def test_atomic_failure_retained_backup_corruption_recovered(self):
        with tempdir() as tmp:
            p = Path(tmp) / "record.json"
            record = dict(identity={"x": 1}, complete=True, success=False, status="forced_timeout")
            save_record(p, record)
            p.write_text("partial", encoding="utf-8")
            got = load_record(p, {"x": 1})
            self.assertEqual(got["status"], "forced_timeout")
            self.assertTrue(list(Path(tmp).glob("*.corrupt-*")))
            with self.assertRaises(ValueError):
                load_record(p, {"x": 2})

    def test_transient_onedrive_replace_retry(self):
        with tempdir() as tmp:
            real = os.replace
            calls = []
            def replace(a, b):
                calls.append(1)
                if len(calls) == 1:
                    raise PermissionError("synthetic lock")
                return real(a, b)
            with patch("src.operational_exit.common.os.replace", side_effect=replace):
                atomic_json(Path(tmp) / "a.json", {"ok": True})
            self.assertEqual(len(calls), 2)

    def test_completed_backup_wins_over_old_incomplete_primary(self):
        from src.operational_exit.common import sealed
        with tempdir() as tmp:
            p = Path(tmp) / "record.json"
            identity = {"run": 1}
            atomic_json(p, sealed(dict(identity=identity, complete=False)))
            atomic_json(str(p) + ".bak", sealed(dict(identity=identity, complete=True, success=False, status="forced_timeout")))
            result = load_record(p, identity)
            self.assertTrue(result["complete"])
            self.assertFalse(result["success"])
            self.assertEqual(result, load_record(p, identity))

    def test_negative_and_zero_reference_gaps_and_common_support(self):
        self.assertEqual(gap(-110, -100), (-10, -10))
        self.assertEqual(gap(3, 0), (None, 3))
        self.assertEqual(gap(3, None), (None, None))
        rows = [dict(key="a", family=FAMILIES[0], budget=1, method=m, repetition=0, success=True,
                     relative_gap_percent=-2, ens_mwh=0, reference_valid=True, reference_cost=-100) for m in METHODS]
        summary, common, pairs = summarize(rows)
        self.assertEqual(len(common), 4)
        self.assertEqual(len(pairs), 6)
        self.assertEqual(summary[0]["denominator"], 100)
        self.assertEqual(summary[0]["common_support"], 1)
        self.assertEqual(summary[0]["coverage_gap_1_percent"], 1)
        rows[-1]["success"] = False
        self.assertEqual(len(summarize(rows)[1]), 0)

    def test_exact_exit_acceptance_and_no_forced_recovery(self):
        payload = dict(status="solution", validation={"valid": True}, dispatch={"x": 1})
        for exit_s, forced, expected in ((1, False, True), (1.00001, False, False), (.5, True, False)):
            self.assertEqual(classify(payload, budget=1, exit_s=exit_s, forced=forced, returncode=0)[0], expected)


@unittest.skipUnless(os.name == "nt", "Windows Job Object tests")
class ProcessTreeTests(unittest.TestCase):
    def test_actual_highs_and_rh_workers_on_two_step_synthetic_input(self):
        repair = dict(max_stages=3, slack_tol_mwh=1., deviation_penalty=10000., flip_budget_20=100, flip_budget_100=1000)
        for method in ("highs", "rh"):
            with self.subTest(method=method), tempdir() as tmp:
                path = Path(tmp) / "scenario_synthetic.json"
                atomic_json(path, scenario_input(n=2, dr_units=0))
                request = dict(budget=10, key="synthetic/two_step", repetition=0, seed=42,
                               identity={"synthetic": method}, file=str(path), scenario_sha256=sha(path))
                result = run_experiment(dict(method=method, reserve_s=2, repair=repair), request, Path(tmp) / "attempt")
                self.assertTrue(result["success"], (result, (Path(tmp) / "attempt/worker.log").read_text()))
                self.assertTrue(result["result"]["validation"]["valid"])
                self.assertTrue(result["all_processes_exited"])

    def test_real_process_tree_faults(self):
        for mode, success, status in (("fast", True, "success"), ("invalid", False, "invalid_dispatch"),
            ("no_incumbent", False, "no_incumbent"), ("written_but_active", False, "forced_timeout"),
            ("blocked_child", False, "forced_timeout"), ("validation_overrun", False, "forced_timeout"),
            ("write_overrun", False, "forced_timeout"), ("crash", False, "worker_error")):
            with self.subTest(mode=mode), tempdir() as tmp:
                result = run_experiment(dict(mode=mode), dict(budget=.6, identity={"test": mode}), Path(tmp) / "attempt",
                                        worker_module="tests.operational_fake_worker")
                self.assertEqual(result["success"], success, result)
                self.assertEqual(result["status"], status, result)
                self.assertTrue(result["all_processes_exited"])
                if result["forced"]:
                    self.assertIsNone(result["result"])


class TrainingContractTests(unittest.TestCase):
    def test_batched_generation_with_independent_candidate_streams(self):
        from types import SimpleNamespace
        from src.operational_exit.generation import Generator
        from src.uc_jepa.sampling.latent_langevin import LatentLangevinConfig, LatentLangevinSampler
        gen = Generator.__new__(Generator)
        gen.device = torch.device("cpu")
        gen.hte = SimpleNamespace(encode=lambda data: torch.zeros(1, 3, 2, 4))
        gen.predictor = lambda h, zone_mask: {"z_pred": torch.zeros(1, 2, 8)}
        gen.ebm = lambda z, anchor, **kw: z.square().mean(dim=(1, 2))
        gen.ae = SimpleNamespace(decode=lambda z, **kw: z.mean(dim=1)[:, None, None, :].expand(-1, 3, 2, -1))
        gen.sampler = LatentLangevinSampler(LatentLangevinConfig(num_steps=2))
        gen.prior_mean = gen.prior_std = None
        gen.thresholds = torch.full((8,), .5)
        data = SimpleNamespace(zones=["a", "b", "c"], periods=[0, 1])
        with patch("torch.cuda.synchronize"):
            a, seeds = gen.candidates(data, dict(key="synthetic", repetition=0), float("inf"))
            b, seeds2 = gen.candidates(data, dict(key="synthetic", repetition=0), float("inf"))
        np.testing.assert_array_equal(a, b)
        self.assertEqual(a.shape, (16, 3, 2, 8))
        self.assertEqual(len(set(seeds)), 16)
        self.assertEqual(seeds, seeds2)

    def test_physical_cost_selection_ignores_ebm_energy(self):
        from src.operational_exit.worker import select_candidates
        rows = [dict(candidate=0, energy=-100, validation=dict(valid=True, cost=20)),
                dict(candidate=1, energy=5, validation=dict(valid=True, cost=-5)),
                dict(candidate=2, energy=-200, validation=dict(valid=False, cost=-200))]
        self.assertEqual(select_candidates(rows)["candidate"], 1)

    def test_graph_adapter_and_zone_ordering(self):
        from src.operational_exit.embedding import graph_data, align_zone_embeddings
        from src.gnn.hetero_graph_dataset import build_hetero_temporal_record, save_graph_record
        from src.gnn.embeddings.dataset import TemporalGraphDatasetV3
        data = make_data(zones=12)
        record = build_hetero_temporal_record(data)
        memory = graph_data(record)
        with tempdir() as tmp:
            p = Path(tmp) / "synthetic.npz"
            save_graph_record(record, p)
            historical = TemporalGraphDatasetV3([p])[0]
            for name in ("x", "edge_index", "node_type", "asset_to_zone", "zone_to_region"):
                torch.testing.assert_close(getattr(memory, name), getattr(historical, name))
        h = torch.tensor([sorted(data.zones).index(z) for z in sorted(data.zones)])[:, None, None]
        result = align_zone_embeddings(h, data.zones)
        self.assertEqual(result[:, 0, 0].tolist(), [sorted(data.zones).index(z) for z in data.zones])

    def test_gradient_accumulation_matches_effective_batch(self):
        from torch.utils.data import DataLoader, TensorDataset
        x = torch.arange(10.).reshape(5, 2) / 10
        for micro in (1, 2):
            model = torch.nn.Linear(2, 1, bias=False)
            other = copy.deepcopy(model)
            opt = torch.optim.SGD(model.parameters(), lr=.1)
            opt._op_effective = 8
            loader = DataLoader(TensorDataset(x), batch_size=micro)
            for step, (batch,) in enumerate(loader):
                backward_step(model(batch).square().mean(), opt, model.parameters(), 0, len(batch), step, loader)
            ref = torch.optim.SGD(other.parameters(), lr=.1)
            ref.zero_grad()
            other(x).square().mean().backward()
            ref.step()
            torch.testing.assert_close(model.weight, other.weight)

    def test_rng_checkpoint_roundtrip(self):
        before = rng_state()
        expected = torch.rand(3)
        restore_rng(before)
        torch.testing.assert_close(expected, torch.rand(3))

    def test_resume_restores_optimizer_scheduler_epoch_early_stop(self):
        from src.uc_jepa.training.train_autoencoder import TrainConfig
        with tempdir() as tmp:
            cfg = TrainConfig(dict(batch_size=8), {}, {}, dict(operational=True, manifest_hash="test", patience=3, epochs=5), {})
            m = torch.nn.Linear(2, 1)
            opt = torch.optim.AdamW(m.parameters())
            scheduler = torch.optim.lr_scheduler.StepLR(opt, 1)
            runtime = TrainingRuntime(cfg, m, opt, scheduler, tmp)
            m(torch.ones(2)).sum().backward()
            opt.step()
            scheduler.step()
            runtime.save(1, 2., 1, [{"epoch": 1}], torch.tensor([.4]))
            expected = copy.deepcopy(m.state_dict())
            fresh = torch.nn.Linear(2, 1)
            opt2 = torch.optim.AdamW(fresh.parameters())
            sch2 = torch.optim.lr_scheduler.StepLR(opt2, 1)
            rt2 = TrainingRuntime(cfg, fresh, opt2, sch2, tmp)
            self.assertEqual(rt2.start_epoch, 2)
            self.assertEqual(rt2.values(0, 0, []), (2., 1, [{"epoch": 1}]))
            self.assertEqual(scheduler.state_dict(), sch2.state_dict())
            for k in expected:
                torch.testing.assert_close(expected[k], fresh.state_dict()[k])
            self.assertEqual(len(opt2.state), len(opt.state))

    def test_legacy_training_not_forced_onto_gpu(self):
        from src.operational_exit.training_runtime import training_device
        from types import SimpleNamespace
        self.assertEqual(str(training_device(SimpleNamespace(optim={}), "cpu")), "cpu")
        with self.assertRaises(ValueError):
            training_device(SimpleNamespace(optim={"operational": True}), "cpu")

    def test_interrupted_checkpoint_recovers_last_committed_epoch(self):
        from src.uc_jepa.training.train_autoencoder import TrainConfig
        from src.operational_exit.training_runtime import atomic_torch_save
        with tempdir() as tmp:
            cfg = TrainConfig(dict(batch_size=8), {}, {}, dict(operational=True, manifest_hash="test", patience=3, epochs=5), {})
            def fresh():
                model = torch.nn.Linear(2, 1)
                opt = torch.optim.AdamW(model.parameters())
                return TrainingRuntime(cfg, model, opt, torch.optim.lr_scheduler.StepLR(opt, 1), tmp)
            first = fresh()
            first.save(0, 1., 0, [])
            old_commit = (Path(tmp) / "resume.json").read_bytes()
            first.save(1, .8, 0, [])
            (Path(tmp) / "resume.json").write_bytes(old_commit)  # interrupted before second commit
            recovered = fresh()
            self.assertEqual(recovered.start_epoch, 1)
            self.assertEqual(fresh().start_epoch, 1)  # recovery itself is durable
        with tempdir() as tmp:
            atomic_torch_save({"uncommitted": True}, Path(tmp) / "resume.pt")
            self.assertEqual(fresh().start_epoch, 0)
            self.assertTrue(list(Path(tmp).glob("resume.pt.uncommitted-*")))


if __name__ == "__main__":
    unittest.main()
