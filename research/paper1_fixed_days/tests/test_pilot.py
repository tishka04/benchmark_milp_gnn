"""Regression checks for comparisons that should hold before running a single MILP."""
import unittest
from pathlib import Path
from unittest.mock import patch

import pandas as pd

from research.paper1_fixed_days import analysis
from research.paper1_fixed_days.design import RULE_DIRECTIONS, inspect_day, load_anchor, load_day, static_sha256
from research.paper1_fixed_days.predict import load_frozen, predict_day
from research.paper1_revision.realized.storage import DEST
from research.paper1_weights.data import config as loss_config


ROOT = Path(__file__).resolve().parents[3]
SMALL_ANCHOR = ROOT / "outputs/low_criticality_scenarios/scenario_00001.json"


class FixedDayTests(unittest.TestCase):
    def test_same_seed_replays_chronology_and_different_days_keep_infrastructure(self):
        first = load_day(SMALL_ANCHOR, 42)
        repeat = load_day(SMALL_ANCHOR, 42)
        second = load_day(SMALL_ANCHOR, 43)
        assert first.demand == repeat.demand
        assert first.solar_available == repeat.solar_available
        assert first.wind_available == repeat.wind_available
        assert first.demand != second.demand
        assert first.wind_available != second.wind_available
        for name in ("lines", "thermal_capacity", "solar_capacity", "wind_capacity",
                     "battery_initial", "pumped_initial", "hydro_ror_generation"):
            assert getattr(first, name) == getattr(second, name), name

        left, right = inspect_day(first), inspect_day(second)
        for key in ("static_sha256", "compact_trio", "model_dimensions"):
            assert left[key] == right[key], key
        self.assertEqual(left["static_sha256"], static_sha256(load_anchor(SMALL_ANCHOR)))
        assert left["profiles_sha256"] != right["profiles_sha256"]
        assert left["residual_peak_mw"] != right["residual_peak_mw"]

        models, _ = load_frozen()

        def structural_predictions(features):
            return {
                (row["budget"], row["target"], row["representation"]): row["prediction_raw"]
                for row in predict_day(models, features, [30, 120])
                if row["representation"] in {"realized_compact", "model_size"}
            }
        fixed = structural_predictions(left)
        assert len(fixed) == 2 * 7 * 2
        assert fixed == structural_predictions(right)


    def test_all_pilot_forecasts_reproduce_an_archived_external_case(self):
        group = "external/unfiltered/0000"
        row = pd.read_parquet(DEST / "features.parquet", filters=[("group", "==", group)]).iloc[0]
        features = {
            "compact_trio": [row.real_B1, row.real_B3, row.real_B4],
            "model_dimensions": [
                row.real_B1, row.real_B2, row.audit_lines, row.audit_variables,
                row.audit_free_binary_variables, row.audit_active_constraints,
            ],
            "realized_features": {name: row[name] for name in row.index if name.startswith("real_")},
        }
        forecasts = predict_day(load_frozen()[0], features, [30, 120])
        expected = pd.read_parquet(DEST / "external_predictions.parquet", filters=[
            ("group", "==", group), ("budget", "in", [30, 120]),
            ("kind", "in", ["positive", "logistic"]),
            ("representation", "in", ["realized_compact", "model_size", "realized"]),
        ]).set_index(["budget", "target", "representation"])
        assert len(forecasts) == len(expected) == 42
        for forecast in forecasts:
            key = forecast["budget"], forecast["target"], forecast["representation"]
            raw = expected.loc[key, "p"]
            self.assertAlmostEqual(forecast["prediction_raw"], raw, delta=1e-12)
            self.assertAlmostEqual(forecast["prediction"], max(0.0, min(1.0, raw)), delta=1e-12)

    def test_expected_recall_and_regret_under_ties(self):
        ids = ["d0", "d1", "d2", "d3"]
        y = [0.1, 0.7, 0.7, 0.9]
        weights = analysis.top_k_weights(y, 2)
        self.assertEqual(weights.tolist(), [0.0, 0.5, 0.5, 1.0])
        obs = pd.DataFrame(dict(day_id=ids, budget=30, feasible=True, ens_mwh=0.0,
                                cost_excess_eur=0.0, co2_direct_t=0.0, exit_s=2.0))
        scores = pd.DataFrame(dict(day_id=ids, target="equal4", score=y, budget=30))
        predictions = pd.DataFrame(dict(day_id=ids, budget=30, target="equal4",
                                        representation="realized_compact", prediction=0.2))
        manifest = dict(quota=2, days=[dict(day_id=i, features=dict.fromkeys(RULE_DIRECTIONS, 2))
                                        for i in ids])
        _, result, _ = analysis._shortlists(manifest, obs, scores, predictions)
        for strategy in ("uniform_random", "realized_compact"):
            row = result.set_index("strategy").loc[strategy]
            self.assertAlmostEqual(row.expected_hits, 1.0)
            self.assertAlmostEqual(row.recall, 0.5)
            self.assertAlmostEqual(row.mean_regret, 0.2)
            self.assertAlmostEqual(row.gain_vs_random, 0.0)

    def test_invalid_reference_keeps_cost_unknown_and_observed_failures_known(self):
        features = dict.fromkeys(RULE_DIRECTIONS, 2.0)
        features["demand_mwh"] = 200.0
        manifest = dict(days=[dict(day_id="d0", features=features, predictions=[])],
                        deadlines=[30, 60, 120], reference_seconds=600)
        invalid = dict(feasible=False, status="no_incumbent", cost=None, proven_optimal=False,
                       bound=None, gap=None)
        trial = dict(day_id="d0", feasible=True, status="success", exit_s=2.0, cost=100.0,
                     ens_mwh=0.0, demand_mwh=200.0, served_mwh=200.0, thermal_mwh=1.0,
                     co2_direct_t=0.72, co2_t_per_served_mwh=0.0036, imports_mwh=0.0, exports_mwh=0.0)
        failure = dict(trial, budget=30, feasible=False, status="forced_timeout", cost=None,
                       ens_mwh=None, served_mwh=None, thermal_mwh=None, co2_direct_t=None,
                       co2_t_per_served_mwh=None, imports_mwh=None, exports_mwh=None)
        cfg = loss_config()
        with patch("research.paper1_fixed_days.runner.read_job",
                   side_effect=[invalid, failure, dict(trial, budget=60, ens_mwh=2.0),
                                dict(trial, budget=120)]):
            obs, scores, _ = analysis._collect(Path("."), manifest, cfg)
        self.assertTrue(obs.loc[obs.budget == 120, ["E", "ES"]].isna().all().all())
        self.assertTrue((obs.loc[obs.budget == 30, ["E", "ES"]] == 0).all().all())
        self.assertTrue(obs.loc[obs.budget == 60, "E"].isna().all())
        self.assertEqual(obs.loc[obs.budget == 60, "ES"].item(), 0)
        self.assertTrue(scores.loc[scores.budget == 120, "score"].isna().all())
        self.assertTrue((scores.loc[scores.budget == 30, "score"] == 1).all())

