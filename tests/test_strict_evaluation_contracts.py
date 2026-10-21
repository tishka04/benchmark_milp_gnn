"""Evaluation plumbing with fake models/repairs and tiny synthetic CPU tensors."""
from contextlib import ExitStack
from pathlib import Path
import tempfile
import time
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

# Windows import ordering; no model or solver is instantiated by these imports.
import pandas  # noqa: F401
import torch
import src.uc_jepa.evaluation.evaluate_latent_pipeline as evaluation
from scripts.run_strict_standard_inference import FORMULATION


class EvaluationContracts(unittest.TestCase):
    def setUp(self):
        base = Path(__file__).resolve().parents[1] / "tmp"
        base.mkdir(exist_ok=True)
        self.temp = tempfile.TemporaryDirectory(prefix="strict-contract-test-", dir=base)
        self.assertTrue(Path(self.temp.name).resolve().is_relative_to(base.resolve()))
        self.stack = ExitStack()
        self.stack.enter_context(patch("subprocess.Popen", side_effect=AssertionError("Real processes prohibited")))
        self.sample = SimpleNamespace(
            u_star=torch.zeros(1, 24, 8), h_scenario=torch.zeros(1, 24, 128),
            n_zones=1, n_timesteps=24, family="low_criticality_scenarios",
            scenario_id="scenario_001", criticality=0.1, objective=10.0,
        )

        class Dataset(list):
            files = ["low_criticality_scenarios/scenario_001_uc.npz"]

        self.stack.enter_context(patch.object(evaluation, "UCLabelDataset", return_value=Dataset([self.sample])))
        self.stack.enter_context(patch.object(evaluation, "tqdm"))
        self.ae = SimpleNamespace(n_features=8, decode=Mock(return_value=torch.zeros(1, 1, 24, 8)))
        self.predictor = Mock(return_value={"z_pred": torch.zeros(1, 4, 16)})
        self.stack.enter_context(patch.object(evaluation, "load_frozen_ae", return_value=self.ae))
        self.stack.enter_context(patch.object(evaluation, "load_frozen_jepa", return_value=(self.predictor, {})))
        self.metadata = {"dr_formulation": FORMULATION}
        self.ebm_loader = self.stack.enter_context(patch.object(
            evaluation, "load_latent_ebm", return_value=(Mock(), self.metadata),
        ))
        self.checkpoint_loader = self.stack.enter_context(patch.object(torch, "load", return_value=self.metadata))
        self.sampler = Mock()
        self.sampler.sample.side_effect = [
            (torch.zeros(1, 1, 4, 16), torch.tensor([[energy]]), None) for energy in (2.0, 1.0)
        ]
        self.stack.enter_context(patch.object(evaluation, "LatentLangevinSampler", return_value=self.sampler))
        self.lp = Mock()
        self.stack.enter_context(patch.object(evaluation, "MILPWarmWorker", return_value=self.lp))
        self.record = dict(
            scenario_id="scenario_001", family=self.sample.family, dr_formulation=FORMULATION,
            dr_metrics={"dr_feasible": True}, lp_objective=12.0, gap_pct=20.0,
            status="optimal", stage_used="hard_fix", solve_time=0.01,
        )
        self.lp.evaluate_candidates.return_value = [self.record]
        self.snapshots = []

    def tearDown(self):
        self.stack.close()
        self.temp.cleanup()

    def evaluate(self, **kwargs):
        return evaluation.evaluate(
            latent_ebm_path="fake-ebm", ae_path="fake-ae", jepa_path="fake-jepa",
            data_roots=["fake-label"], scenario_roots=[self.temp.name], output_dir=self.temp.name,
            device="cpu", k=2, analyze=False, deadline_monotonic=time.monotonic() + 120,
            result_callback=lambda method, row: self.snapshots.append(
                (method, row, self.lp.evaluate_candidates.call_count)
            ), **kwargs,
        )

    def test_standard_publishes_before_next_repair_and_skips_direct_decode(self):
        self.evaluate(skip_direct_baseline=True)
        self.assertEqual(self.sampler.sample.call_count, 2)
        self.assertEqual(self.ae.decode.call_count, 2)
        self.assertEqual(self.lp.evaluate_candidates.call_count, 2)
        self.assertEqual([x[0] for x in self.snapshots], ["pipeline"] * 3)
        self.assertEqual([x[2] for x in self.snapshots], [1, 2, 2])
        self.assertEqual(self.snapshots[0][1]["n_samples"], 1)
        self.assertEqual(self.snapshots[-1][1]["best_sample_idx"], 1)
        self.checkpoint_loader.assert_not_called()
        for call in self.lp.evaluate_candidates.call_args_list:
            self.assertGreater(call.kwargs["candidate_time_budget_s"], 0)
            self.assertLessEqual(call.kwargs["candidate_time_budget_s"], 120)

    def test_jepa_only_never_constructs_or_samples_ebm(self):
        self.evaluate(direct_only=True)
        self.ebm_loader.assert_not_called()
        self.sampler.sample.assert_not_called()
        self.ae.decode.assert_called_once()
        self.lp.evaluate_candidates.assert_called_once()
        self.assertEqual([x[0] for x in self.snapshots], ["jepa"])
        self.assertTrue(self.snapshots[0][1]["success"])
        self.assertEqual(self.checkpoint_loader.call_args.kwargs["map_location"], "cpu")


if __name__ == "__main__":
    unittest.main()
