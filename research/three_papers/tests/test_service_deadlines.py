"""Synthetic checks for separate 5/10/30/60/120-second decisions; no campaign."""
import copy
import unittest
from unittest.mock import patch
import pandas as pd
from research.three_papers import core, requirements as req, routing, experiments
from research.three_papers.joint_analysis import expanded
from research.three_papers.tests.test_service_routing import row, predictions


def comparison(**updates):
    r = dict(selector="oracle", groups=20, coverage_gain=0., coverage_lower=0.,
             coverage_upper=0., secondary_gain=0., secondary_lower=0.,
             paired_success_groups=20, success=.99)
    r.update(updates)
    return r


class DeadlineSemantics(unittest.TestCase):
    def test_grid_and_signature(self):
        self.assertEqual(req.deadlines(), [5, 10, 30, 60, 120])
        cfg = copy.deepcopy(core.config())
        cfg["service_routing"]["deadlines_s"] = [30, 120]
        self.assertNotEqual(req.profile_signature(), req.profile_signature(cfg))
        cfg["service_routing"]["deadlines_s"] = [30, 30]
        with self.assertRaises(ValueError):
            req.deadlines(cfg)
        with self.assertRaises(ValueError):
            req.request_key("urgent_continuity", 2.)

    def test_only_declared_allocations_enter_service_analysis_and_training(self):
        data = pd.DataFrame([row(budget=b) for b in [1, 2, 5, 10, 30, 60, 120]])
        for x in (expanded(data), routing.training_table(data)):
            self.assertEqual(sorted(x.budget.unique().tolist()), req.deadlines())

    def test_mixed_deadlines_cannot_be_scored_together(self):
        q = pd.DataFrame([dict(profile_id="urgent_continuity", budget=b) for b in [5, 30]])
        with self.assertRaisesRegex(ValueError, "one deadline"):
            routing._paired_gain(q)

    def test_gain_at_30_seconds_is_not_diluted_by_other_deadlines(self):
        cells = pd.DataFrame([dict(profile_id="urgent_continuity", budget=b, group=str(g))
                              for b in req.deadlines() for g in range(20)])
        def report(q):
            if q.empty:
                return []
            b = float(q.budget.iloc[0])
            return [dict(comparison(coverage_gain=.03 if b == 30 else 0.,
                                    coverage_lower=.01 if b == 30 else 0.),
                         profile_id="urgent_continuity", budget=b)]
        with patch.object(routing, "_paired_gain", side_effect=report):
            decisions, _ = routing._request_diagnostics(cells, cells, cells)
        enabled = [(r["profile_id"], r["deadline_s"]) for r in decisions if r["proceed"]]
        self.assertEqual(enabled, [("urgent_continuity", 30)])
        self.assertEqual(len(decisions), len(req.profiles())*len(req.deadlines()))
        self.assertTrue(all(not r["proceed"] for r in decisions if r["expected"] == 0))

    def test_secondary_gate_keeps_risk_threshold_within_one_deadline(self):
        p = req.profile("balanced_service")
        good = comparison(secondary_gain=1., secondary_lower=.6)
        bad = dict(good, success=.8)
        self.assertTrue(routing._continuation(p, 20, 20, good)["secondary_gate"])
        self.assertFalse(routing._continuation(p, 20, 20, bad)["secondary_gate"])
        self.assertFalse(routing._continuation(p, 30, 20, good)["proceed"])

    def test_development_scores_and_enabled_cells_are_request_specific(self):
        cells = pd.DataFrame([dict(profile_id="urgent_continuity", budget=b) for b in req.deadlines()])
        enabled = [dict(profile_id="urgent_continuity", deadline_s=30)]
        self.assertEqual(routing._enabled_cells(cells, enabled).budget.tolist(), [30])
        scores = pd.DataFrame([dict(profile_id="urgent_continuity", budget=b, selector=k,
            success=.99, coverage_lower=0., secondary_gain=gain)
            for b, k, gain in [(30, "tree", 3.), (30, "logistic", 1.),
                               (60, "tree", 1.), (60, "logistic", 3.)]])
        selected = routing._select_models(scores)
        self.assertEqual(selected, {req.request_key("urgent_continuity", 30): "tree",
                                    req.request_key("urgent_continuity", 60): "logistic"})

    def test_online_permission_uses_nominal_deadline_not_remaining_allowance(self):
        p = req.request_for("urgent_continuity", 30.)
        pack = dict(schema=req.SCHEMA, profile_signature=req.profile_signature(), kind="selected",
                    enabled_requests=[dict(profile_id=p["id"], deadline_s=30)],
                    selected={req.request_key(p["id"], 30): "tree"}, candidates={"tree": {}})
        with patch.object(routing, "predict_many", return_value=[predictions()]) as predict:
            d = routing.choose(pack, {}, 29.7, p)
            self.assertEqual(d["selector_kind"], "tree")
            self.assertEqual(predict.call_args.args[1].budget.iloc[0], 29.7)
            with self.assertRaisesRegex(ValueError, "profile/deadline"):
                routing.choose(pack, {}, 59.7, req.request_for(p["id"], 60.))
            self.assertEqual(predict.call_count, 1)

    def test_calibration_export_keeps_distinct_deadlines(self):
        ev = pd.DataFrame([dict(profile_id="urgent_continuity", budget=b, selector="tree",
                               predicted_success=.95, joint_success=y, group=str(g))
                           for b, y in [(5, 0), (30, 1)] for g in range(2)])
        rows = routing._calibration_rows(ev)
        self.assertEqual({r["budget"]: r["observed"] for r in rows}, {5.: 0., 30.: 1.})

    def test_experiments_schedule_only_enabled_pairs(self):
        enabled = [dict(profile_id="urgent_continuity", deadline_s=30),
                   dict(profile_id="balanced_service", deadline_s=10)]
        receipt = core.sealed(dict(schema=req.SCHEMA, profile_signature=req.profile_signature(),
                                   enabled_requests=enabled))
        case = dict(key="k", group="g", expected_horizon=24, scenario_sha256="scenario")
        with patch.object(experiments, "frozen", return_value={"sha256": "freeze"}), \
             patch.object(experiments, "read", return_value=receipt), \
             patch.object(experiments, "select", return_value=[case]), \
             patch.object(experiments, "bind_models", return_value=({}, {})):
            jobs = list(experiments.identities(core.HOME/"runs"/"synthetic", "final", "router"))
        self.assertEqual({(item["profile_id"], item["budget"]) for _, item, _ in jobs},
                         {("urgent_continuity", 30), ("balanced_service", 10)})
        self.assertEqual(len(jobs), 2)


if __name__ == "__main__":
    unittest.main()
