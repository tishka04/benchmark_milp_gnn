"""Solver-free migration checks. No optimisation, training or benchmark runs.

Run only this file:
  python -B -m unittest discover -s tests -p test_pipeline_load_shifting.py -v
"""
import copy
import json
import pickle
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
# Initialise pandas/PyArrow before Torch's DLLs in the Windows test environment.
# Otherwise the optional Arrow extension can terminate the interpreter natively.
import pandas as pd
import torch

from src.demand_response import (
    DR_FORMULATION, FormulationMismatchError, physics_dr_schedule,
    require_current_formulation, schedule_shifting, shifting_metrics, validate_dr_report,
)
from src.ebm.dataset_v3 import _extract_binaries_from_report
from src.ebm.feasibility import HierarchicalFeasibilityDecoder, ScenarioPhysics
from src.ebm.embedding_loader import EmbeddingLoader
from src.eval.pipeline_runner import PipelineRunner
from src.gnn.dispatch_dataset import extract_dispatch_from_report
from src.gnn.dispatch_model import DISPATCH_CHANNELS, DispatchGNN, N_DISPATCH
from src.gnn.graph_dataset import build_graph_record
from src.gnn.training.metrics import compute_constraint_violation_rate
from src.gnn.training.decoder import FeasibilityDecoder
from src.gnn.training.config import DecoderConfig
from src.gnn.hetero_graph_dataset import build_hetero_temporal_record, EDGE_TYPE_TEMPORAL_DR
from src.gnn.dispatch_predictor import GNNDispatchPredictor, GNNDispatchResult
from src.milp.model import build_uc_model
from src.milp.lp_worker_two_stage import LPWorkerTwoStage, TwoStageResult, SolveStage
from src.eval.time_constrained_heuristic import _result_to_payload
from src.eval.time_constrained_milp import _hard_timeout_payload
from src.uc_jepa.data.extract_uc_labels import extract_u_star, process_one
from src.uc_jepa.data.uc_label_dataset import UCLabelDataset
from src.uc_jepa.latent_utils import load_frozen_ae, load_frozen_jepa, load_latent_ebm
from src.eval.metrics import build_comparison_dataframe, load_milp_reports

# This helper constructs input data only. Never run its HiGHS TestCase here.
from test_milp_load_shifting import make_data


def report_fixture():
    series = {
        "demand": [10, 10], "solar": [6, 14], "demand_response": [4, 0],
        "dr_recover": [0, 4], "dr_rebound": [4, 0], "dr_active": [1, 0],
    }
    for key in ("thermal", "nuclear", "wind", "hydro_release", "hydro_ror",
                "battery_soc", "battery_charge", "battery_discharge", "pumped_level",
                "pumped_charge", "pumped_discharge", "unserved", "solar_spill",
                "wind_spill", "hydro_spill", "overgen_spill", "thermal_commitment",
                "thermal_startup", "battery_charge_mode", "pumped_charge_mode"):
        series[key] = [0, 0]
    detail = {k: {"R1Z1": v} for k, v in series.items()}
    detail.update({
        "dr_formulation": DR_FORMULATION, "zones": ["R1Z1"], "time_steps": [0, 1],
        "time_hours": [0, 1], "dt_hours": 1.0, "flows": {},
        "net_import": {"values": [0, 0]}, "net_export": {"values": [0, 0]},
    })
    return {
        "dr_formulation": DR_FORMULATION, "scenario_id": "scenario_00001",
        "zones": ["R1Z1"], "mip": {"objective": 4, "solve_seconds": 1},
        "detail": detail,
    }


def physics_fixture():
    return ScenarioPhysics(
        n_zones=1, n_timesteps=2, demand=torch.tensor([[10., 10.]]),
        solar_available=torch.tensor([[6., 14.]]), dt_hours=0.5,
        dr_limit_mw=torch.tensor([[4., 4.]]),
        dr_recovery_limit_mw=torch.tensor([[4., 4.]]),
        dr_max_active_periods=torch.tensor([1]), dr_ramp_mw=torch.tensor([4.]),
    )


class PipelineLoadShiftingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Any accidental solver invocation in these tests is an error.
        cls.solve_guard = patch("pyomo.opt.base.solvers.SolverFactoryClass.__call__",
                                side_effect=AssertionError("Solver invocation forbidden"))
        cls.solve_guard.start()

    @classmethod
    def tearDownClass(cls):
        cls.solve_guard.stop()

    def test_report_contract_rejects_legacy_and_missing_recovery(self):
        valid = report_fixture()
        validate_dr_report(valid, 2)
        for payload in ({}, {"dr_formulation": "load_shedding"}):
            with self.assertRaises(FormulationMismatchError):
                require_current_formulation(payload)
        del valid["detail"]["dr_recover"]
        with self.assertRaisesRegex(ValueError, "dr_recover"):
            validate_dr_report(valid)

    def test_truncated_horizon_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "complete horizon"):
            extract_u_star(report_fixture(), n_timesteps=1)

    def test_metrics_detect_missing_repayment_and_prepayment(self):
        ok = shifting_metrics([[4, 0]], [[0, 4]], 0.5, backlog=[[2, 0]])
        self.assertTrue(ok["dr_feasible"])
        self.assertEqual(ok["dr_deferred_mwh"], 2)
        for down, up in (([[4, 0]], [[0, 0]]), ([[0, 4]], [[4, 0]]), ([[4, 0]], [[4, 0]])):
            self.assertFalse(shifting_metrics(down, up, 1)["dr_feasible"])

    def test_metrics_do_not_cancel_energy_between_zones(self):
        metrics = shifting_metrics([[4, 0], [0, 0]], [[0, 0], [0, 4]], 1)
        self.assertFalse(metrics["dr_feasible"])
        self.assertEqual(metrics["dr_energy_error_mwh"], 4)

    def test_schedule_reserves_real_future_capacity(self):
        p = physics_fixture()
        down, up, backlog = physics_dr_schedule(p)
        np.testing.assert_allclose(down, [[4, 0]])
        np.testing.assert_allclose(up, [[0, 4]])
        np.testing.assert_allclose(backlog, [[2, 0]])
        p.dr_recovery_limit_mw.zero_()
        self.assertEqual(physics_dr_schedule(p)[0].sum(), 0)

    def test_schedule_randomized_physical_invariants(self):
        rng = np.random.default_rng(31)
        for _ in range(30):
            requested = rng.uniform(0, 10, (3, 12))
            available = rng.uniform(0, 10, (3, 12))
            cap = rng.uniform(0, 5, (3, 12))
            recovery = rng.uniform(0, 5, (3, 12))
            down, up, backlog = schedule_shifting(requested, available, cap, recovery, .25, [3, 4, 2], [1, 2, 3])
            metric = shifting_metrics(
                down, up, .25, backlog=backlog, active=(down > 0),
                down_limit=cap, up_limit=recovery, max_periods=[3, 4, 2], ramp=[1, 2, 3],
            )
            self.assertTrue(metric["dr_feasible"], metric)
            self.assertTrue(np.all(up <= available + 1e-8))

    def test_both_heuristic_decoders_include_recovery_consumption(self):
        for method in ("decode", "decode_passthrough"):
            with self.subTest(method=method):
                p = physics_fixture()
                u = torch.zeros(1, 2, 7)
                u[0, 0, 4] = 1
                plan = getattr(HierarchicalFeasibilityDecoder(p), method)(u)
                self.assertEqual(float(plan.dr_recover[0, 1]), 4)
                self.assertEqual(float(plan.unserved_energy.sum()), 0)
                self.assertTrue(shifting_metrics(plan.demand_response, plan.dr_recover, p.dt_hours,
                                                backlog=plan.dr_rebound)["dr_feasible"])
                torch.testing.assert_close(plan.solar_dispatch + plan.demand_response,
                                           p.demand + plan.dr_recover)
                self.assertEqual(plan.to_tensor().shape[-1], 7)
                self.assertEqual(float(plan.to_tensor()[0, 1, 4]), 0)

    def test_binary_targets_keep_deferral_separate_from_recovery(self):
        binary = _extract_binaries_from_report(report_fixture(), 2)
        self.assertEqual(tuple(binary.shape), (1, 2, 7))
        torch.testing.assert_close(binary[0, :, 4], torch.tensor([1., 0.]))
        uc, zones = extract_u_star(report_fixture(), 2)
        self.assertEqual(tuple(uc.shape), (1, 2, 8))
        self.assertEqual(zones, ["R1Z1"])

    def test_dispatch_and_graph_targets_include_recovery(self):
        dispatch, _ = extract_dispatch_from_report(report_fixture(), 2)
        self.assertEqual(dispatch.shape[-1], 12)
        self.assertEqual(float(dispatch[0, 1, DISPATCH_CHANNELS.index("dr_recover")]), 4)
        graph = build_graph_record(make_data(), report_fixture())
        self.assertEqual(graph["node_labels"].shape, (2, 1, 14))
        self.assertEqual(graph["node_labels"][1, 0, -1], 4)
        self.assertEqual(graph["node_time"].shape, (2, 1, 20))
        self.assertEqual(graph["node_time"][1, 0, -1], 4)
        require_current_formulation(graph)

    def test_graph_inputs_use_recovery_capacity_not_observed_recovery(self):
        data = make_data()
        data.dr_recovery_limit_factor = 2
        graph = build_graph_record(data, report_fixture())
        np.testing.assert_allclose(graph["node_time"][:, :, -1], [[8], [8]])
        np.testing.assert_allclose(graph["node_labels"][:, :, -1], [[0], [4]])

    def test_heterogeneous_graph_keeps_spatial_and_temporal_edges_aligned(self):
        graph = build_hetero_temporal_record(make_data())
        require_current_formulation(graph)
        self.assertEqual(graph["edge_index"].shape[0], 2)
        self.assertEqual(graph["edge_index"].shape[1], len(graph["edge_types"]))
        self.assertEqual(graph["edge_features"].shape[0], len(graph["edge_types"]))
        dr_edges = graph["edge_index"][:, graph["edge_types"] == EDGE_TYPE_TEMPORAL_DR]
        self.assertGreater(dr_edges.shape[1], 0)
        np.testing.assert_array_equal(dr_edges[1] - dr_edges[0], graph["meta"]["N_base"])
        self.assertLess(graph["edge_index"].max(), len(graph["node_types"]))

    def test_period_decoder_keeps_recovered_consumption_in_balance(self):
        decoder = FeasibilityDecoder(DecoderConfig())
        pred = torch.zeros(1, 14)
        pred[0, 2], pred[0, 13] = 14, 4
        batch = SimpleNamespace(node_time=torch.tensor([[10.]]))
        out = decoder._balance_supply(batch, pred, {})
        self.assertEqual(float(out[0, 2]), 14)
        self.assertEqual(float(decoder._final_trim(batch, out)[0, 2]), 14)

    def test_lp_warm_start_and_diagnostics_without_a_solver(self):
        from pyomo.environ import Var, value
        model = build_uc_model(make_data(dt=.5))
        for var in model.component_data_objects(Var):
            var.set_value(0, skip_validation=True)
        plan = HierarchicalFeasibilityDecoder(physics_fixture()).decode(torch.zeros(1, 2, 7))
        worker = LPWorkerTwoStage.__new__(LPWorkerTwoStage)
        worker.verbose = False
        worker._warm_start_from_plan(model, plan, ["R1Z1"])
        model.dr_active["R1Z1", 0].set_value(1)
        self.assertEqual(value(model.dr_recover["R1Z1", 1]), 4)
        self.assertEqual(value(model.dr_rebound["R1Z1", 0]), 2)
        result = TwoStageResult("synthetic", "fixture", SolveStage.HARD_FIX, 0, 0,
                                worker._extract_solution(model))
        self.assertTrue(result.dr_metrics["dr_feasible"])
        self.assertEqual(result.dr_metrics["dr_recovered_mwh"], 2)
        result.continuous_vars["dr_recover"][:] = 0
        self.assertFalse(result.dr_metrics["dr_feasible"])

    def test_raw_gnn_prediction_is_unverified_and_uses_subhourly_dt(self):
        predictor = GNNDispatchPredictor.__new__(GNNDispatchPredictor)
        predictor.device = "cpu"
        predictor.dispatch_channels = DISPATCH_CHANNELS
        def forward(u, h, mask):
            dispatch = torch.zeros(1, 1, 2, N_DISPATCH)
            dispatch[0, 0, 0, DISPATCH_CHANNELS.index("demand_response")] = 4
            dispatch[0, 0, 1, DISPATCH_CHANNELS.index("dr_recover")] = 4
            return dispatch
        predictor.model = forward
        u = torch.zeros(1, 2, 7)
        u[0, 0, 4] = 1
        result = predictor.predict(u, torch.zeros(1, 2, 8), dt_hours=.5, physics=physics_fixture())
        self.assertEqual(result.status, "predicted_unverified")
        self.assertTrue(np.isnan(result.objective_value))
        self.assertEqual(result.dr_metrics["dr_deferred_mwh"], 2)
        self.assertTrue(result.dr_metrics["dr_feasible"])
        self.assertEqual(GNNDispatchResult("empty").status, "predicted_unverified")

    def test_budget_payloads_preserve_provenance_and_dr_metrics(self):
        metrics = {"dr_feasible": True, "dr_recovered_mwh": 2}
        payload = _result_to_payload(
            {"dr_formulation": DR_FORMULATION, "success": True, "lp_objective": 4, "dr_metrics": metrics},
            scenario_path=Path("synthetic.json"), family="fixture", budget_seconds=1, elapsed_seconds=.5,
        )
        require_current_formulation(payload)
        self.assertEqual(payload["dr_metrics"], metrics)
        require_current_formulation(_hard_timeout_payload(Path("synthetic.json"), 1, 1))
        invalid = _result_to_payload(
            {"dr_formulation": DR_FORMULATION, "success": True, "lp_objective": 4,
             "dr_metrics": {"dr_feasible": False}},
            scenario_path=Path("synthetic.json"), family="fixture", budget_seconds=1, elapsed_seconds=.5,
        )
        self.assertFalse(invalid["heur_found_feasible"])

    def test_embedding_loader_checks_version_in_full_and_lazy_modes(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "embeddings.pt"
            for mode in ("full", "lazy"):
                torch.save({"scenario_00001": torch.zeros(2, 8)}, path)
                with self.assertRaises(FormulationMismatchError):
                    EmbeddingLoader(str(path), loading_mode=mode)
                torch.save({"dr_formulation": DR_FORMULATION, "scenario_00001": torch.zeros(2, 8)}, path)
                loader = EmbeddingLoader(str(path), loading_mode=mode)
                self.assertEqual(loader.scenario_ids, ["scenario_00001"])
                self.assertEqual(loader.get_embedding("scenario_00001").shape, (2, 8))

    def test_pipeline_cache_checks_version_on_load(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "results.pkl"
            path.write_bytes(pickle.dumps([{"scenario_id": "legacy"}]))
            with self.assertRaises(FormulationMismatchError):
                PipelineRunner.load_results(path)
            path.write_bytes(pickle.dumps([{"scenario_id": "current", "dr_formulation": DR_FORMULATION}]))
            self.assertEqual(len(PipelineRunner.load_results(path)), 1)

    def test_instantaneous_balance_counts_recovered_load(self):
        pred = torch.zeros(1, 14)
        pred[0, 2] = 10
        pred[0, 13] = 4
        batch = SimpleNamespace(node_time=torch.tensor([[10.]]), batch_size=1, node_batch=torch.tensor([0]))
        self.assertEqual(compute_constraint_violation_rate(pred, batch).value, 1)
        pred[0, 2] = 14
        self.assertEqual(compute_constraint_violation_rate(pred, batch).value, 0)

    def test_dispatch_network_outputs_twelve_channels_without_training(self):
        model = DispatchGNN(embed_dim=8, hidden_dim=16, n_heads=2, gru_layers=1, n_gat_layers=1, dropout=0)
        model.eval()
        with torch.no_grad():
            out = model(torch.zeros(1, 1, 2, 7), torch.zeros(1, 1, 2, 8), torch.ones(1, 1))
        self.assertEqual(tuple(out.shape), (1, 1, 2, N_DISPATCH))

    def test_uc_artifacts_are_versioned_and_legacy_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            path = root / "scenario_00001.json"
            path.write_text(json.dumps(report_fixture()), encoding="utf-8")
            payload = process_one(str(path), 2, {}, 8)
            self.assertEqual(str(payload["dr_formulation"]), DR_FORMULATION)
            self.assertEqual(payload["dr_recover"][0, 1], 4)
            npz = root / "scenario_00001_uc.npz"
            np.savez(npz, **payload)
            dataset = UCLabelDataset([directory])
            self.assertEqual(dataset[0].n_timesteps, 2)
            del payload["dr_formulation"]
            np.savez(npz, **payload)
            with self.assertRaises(FormulationMismatchError):
                dataset[0]

    def test_legacy_latent_checkpoints_are_rejected_before_model_construction(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "legacy.pt"
            torch.save({}, path)
            for loader in (load_frozen_ae, load_frozen_jepa, load_latent_ebm):
                with self.assertRaises(FormulationMismatchError):
                    loader(str(path), torch.device("cpu"))

    def test_reference_comparison_rejects_mixed_formulations(self):
        row = {"scenario_id": "scenario_00001", "dr_formulation": DR_FORMULATION}
        with self.assertRaises(FormulationMismatchError):
            build_comparison_dataframe([row], {"scenario_00001": {"milp_objective": 1}})
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "scenario_00001.json"
            path.write_text(json.dumps({"mip": {"objective": 4}}), encoding="utf-8")
            with self.assertRaises(FormulationMismatchError):
                load_milp_reports(Path(directory))
            path.write_text(json.dumps(report_fixture()), encoding="utf-8")
            references = load_milp_reports(Path(directory))
            result = build_comparison_dataframe([row], references)
            self.assertEqual(float(result.iloc[0]["milp_dr_recovered_mwh"]), 4)


if __name__ == "__main__":
    unittest.main()
