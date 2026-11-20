"""Synthetic contract tests for service routing. No campaign, solver or GPU invocation."""
import copy
import unittest
from unittest.mock import patch
import numpy as np
import pandas as pd
from research.three_papers import core, requirements as req
from research.three_papers import routing
from research.three_papers.joint_analysis import expanded, paired

def row(**updates):
    r = dict(partition="synthetic", condition="base", family="synthetic", group="g", key="s",
             method="highs", repetition=0, budget=30, observed=True, feasible=True,
             cost=103., reference_cost=100., ens_mwh=0., exit_s=20., criticality=.3)
    r.update(updates)
    return r

def predictions():
    return {m:dict(p=.98, cost=100.+i, time=10.+i, supported=True,
                   in_budget_range=True, interpolated_budget=False, calibration_status="synthetic")
            for i,m in enumerate(core.METHODS)}

class RequirementSemantics(unittest.TestCase):
    def test_valid_is_not_zero_unserved(self):
        p=req.request_for("balanced_service",30.)
        self.assertEqual(req.outcome(row(ens_mwh=4.),p)["joint_success"],0)
        self.assertEqual(req.outcome(row(),p)["joint_success"],1)

    def test_observed_failure_is_not_missing(self):
        p=req.request_for("balanced_service",30.)
        self.assertEqual(req.outcome(row(feasible=False,cost=None,reference_cost=None,ens_mwh=None,exit_s=None),p)["joint_success"],0)
        self.assertIsNone(req.outcome(row(observed=False,feasible=None),p)["joint_success"])
        self.assertIsNone(req.outcome(row(reference_cost=None),p)["joint_success"])

    def test_known_violation_dominates_unknown_reference(self):
        p=req.request_for("balanced_service",30.)
        self.assertEqual(req.outcome(row(reference_cost=None,ens_mwh=10.),p)["joint_success"],0)

    def test_inactive_constraints_do_not_need_reference_or_ens(self):
        p=req.request_for("urgent_continuity",30.)
        self.assertEqual(req.outcome(row(reference_cost=None,cost=None),p)["joint_success"],1)
        p=req.request_for("economic_precision",30.)
        self.assertEqual(req.outcome(row(cost=100.,ens_mwh=None),p)["joint_success"],1)

    def test_negative_gap_and_near_zero_reference(self):
        p=req.request_for("balanced_service",30.)
        self.assertEqual(req.outcome(row(cost=90.),p)["joint_success"],1)
        self.assertEqual(req.outcome(row(cost=1.5,reference_cost=.5),p)["joint_success"],1)
        self.assertEqual(req.outcome(row(cost=1.51,reference_cost=.5),p)["joint_success"],0)

    def test_deadline_and_energy_boundary(self):
        p=req.request_for("balanced_service",30.)
        self.assertEqual(req.outcome(row(exit_s=30.,ens_mwh=p["ens_cap_mwh"]),p)["joint_success"],1)
        self.assertEqual(req.outcome(row(exit_s=30.001),p)["joint_success"],0)
        self.assertEqual(req.outcome(row(ens_mwh=p["ens_cap_mwh"]*2),p)["joint_success"],0)

    def test_never_relabel_a_120_second_allocation(self):
        with self.assertRaises(ValueError):
            req.outcome(row(budget=120,exit_s=10),req.request_for("balanced_service",30.))

    def test_validate_profiles_and_requests(self):
        cfg=copy.deepcopy(core.config())
        cfg["service_routing"]["profiles"][0]["alpha"]=1.1
        with self.assertRaises(ValueError): req.profiles(cfg)
        p=req.request_for("balanced_service",30.); p["alpha"]=.8
        with self.assertRaises(ValueError): req.check_request(p)

    def test_absolute_ceiling(self):
        p=req.request_for("urgent_continuity",30.)
        p["cost_ceiling_eur"]=102.
        self.assertEqual(req.outcome(row(reference_cost=None),p)["joint_success"],0)

class JointReporting(unittest.TestCase):
    def test_routed_execution_keeps_its_own_request(self):
        data=pd.DataFrame([row(),row(method="router",profile_id="balanced_service")])
        x=expanded(data)
        self.assertEqual(len(x[x.method=="highs"]),len(req.profiles()))
        self.assertEqual(len(x[x.method=="router"]),1)
        self.assertEqual(x[x.method=="router"].profile_id.iloc[0],"balanced_service")

    def test_failures_not_zero_cost_in_paired_contrasts(self):
        records=[row(),row(method="rh",cost=110.),
                 row(group="h",key="t",feasible=False,cost=None),
                 row(group="h",key="t",method="rh",cost=101.)]
        result=paired(expanded(pd.DataFrame(records)))
        z=result[(result.profile_id=="urgent_continuity")&(result.metric=="cost")].iloc[0]
        self.assertEqual(z.common_known,2)
        self.assertEqual(z.paired_support,1)
        self.assertAlmostEqual(z.difference,-7.)

    def test_thresholds_and_profiles_share_scenario_groups(self):
        from research.three_papers.statistics import folds
        x=expanded(pd.DataFrame([row(budget=5),row(budget=30)]))
        self.assertEqual(len(set(folds(x))),1)

class Selection(unittest.TestCase):
    def test_cost_priority_is_not_always_maximum_probability(self):
        p=req.request_for("urgent_continuity",30.)
        pred=predictions()
        pred["highs"].update(p=.99,cost=100.)
        pred["rh"].update(p=.96,cost=80.)
        self.assertEqual(routing.select_prediction(pred,p)["method"],"rh")

    def test_time_priority_and_joint_risk_constraint(self):
        p=req.request_for("balanced_service",30.)
        pred=predictions()
        pred["rh"].update(p=.90,time=1.)
        pred["jepa"].update(p=.97,time=4.)
        self.assertEqual(routing.select_prediction(pred,p)["method"],"jepa")

    def test_fallback_does_not_relax_risk_target(self):
        p=req.request_for("balanced_service",30.)
        pred=predictions()
        for v in pred.values(): v["p"]=.2
        pred["rh"]["p"]=.8
        d=routing.select_prediction(pred,p)
        self.assertEqual(d["method"],"rh")
        self.assertTrue(d["fallback"])
        self.assertFalse(d["risk_supported"])

    def test_missing_secondary_prediction_is_unsupported(self):
        p=req.request_for("urgent_continuity",30.)
        pred=predictions()
        for v in pred.values(): v["cost"]=None
        self.assertFalse(routing.select_prediction(pred,p)["risk_supported"])

    def test_remaining_allowance_and_feature_whitelist(self):
        p=req.request_for("balanced_service",30.)
        pack=dict(schema=req.SCHEMA,profile_signature=req.profile_signature(),kind="fixed",
                  enabled_requests=[dict(profile_id=p["id"],deadline_s=30)],candidates={"fixed":{}})
        seen=[]
        def mock_predict(model,cells):
            seen.append(cells)
            return [predictions()]
        with patch.object(routing,"predict_many",side_effect=mock_predict):
            routing.choose(pack,{"reference_cost":1.,"cost":2.,"n_zones":3},29.7,p)
        self.assertEqual(seen[0].budget.iloc[0],29.7)
        self.assertNotIn("reference_cost",seen[0].columns)
        self.assertNotIn("cost",seen[0].columns)

    def test_old_model_rejected(self):
        with self.assertRaises(ValueError):
            routing.choose({"kind":"fixed"},{},30.,req.request_for("balanced_service",30.))

    def test_calibration_never_sees_a_training_group_in_its_test_fold(self):
        q=pd.DataFrame([dict(group=str(g),joint_success=0,**{f:float(g) for f in routing.FEATURES})
                        for g in range(9) for _ in range(2)])
        checked=[]
        def fit(train,y,features,kind):
            return set(train.n_zones)
        def predict(model,test):
            self.assertTrue(model.isdisjoint(set(test.n_zones)))
            checked.append(len(test))
            return np.zeros(len(test))
        with patch.object(routing,"seed",side_effect=lambda prefix,g:int(g)), \
             patch.object(routing,"fit_probability",side_effect=fit), \
             patch.object(routing,"probability",side_effect=predict):
            routing._calibrated(q,"logistic")
        self.assertGreaterEqual(len(checked),2)


    def test_nested_profiles_receive_coherent_probabilities(self):
        def fake(model,cells):
            out=[]
            rates={"urgent_continuity":.6,"balanced_service":.9,
                   "economic_precision":.8,"precision_continuity":.95}
            for r in cells.to_dict("records"):
                pred=predictions()
                for v in pred.values(): v["p"]=rates[r["profile_id"]]
                out.append(pred)
            return out
        cells=pd.DataFrame([dict(profile_id=p["id"],budget=30.) for p in req.profiles()])
        with patch.object(routing,"_predict_many_raw",side_effect=fake):
            pred=routing.predict_many({},cells)
        p={r["profile_id"]:z["highs"]["p"] for r,z in zip(cells.to_dict("records"),pred)}
        self.assertLessEqual(p["balanced_service"],p["urgent_continuity"])
        self.assertLessEqual(p["precision_continuity"],p["balanced_service"])
        self.assertLessEqual(p["precision_continuity"],p["economic_precision"])

    def test_restricted_portfolio_never_selects_learned_method(self):
        p=req.request_for("balanced_service",30.)
        pred=predictions(); pred["jepa"]["time"]=.1
        d=routing.select_prediction(pred,p,allowed=("highs","rh"))
        self.assertIn(d["method"],("highs","rh"))

class IntegrationContractsWithoutExecution(unittest.TestCase):
    def test_request_changes_experiment_identity(self):
        from research.three_papers import experiments
        gate=core.sealed(dict(schema=req.SCHEMA,profile_signature=req.profile_signature(),
                              enabled_requests=[dict(profile_id=pid,deadline_s=b)
                                  for pid in ("urgent_continuity","balanced_service") for b in req.deadlines()]))
        case=dict(key="k",group="g",expected_horizon=24,scenario_sha256="scenario")
        with patch.object(experiments,"frozen",return_value={"sha256":"freeze"}), \
             patch.object(experiments,"read",return_value=gate), \
             patch.object(experiments,"select",return_value=[case]), \
             patch.object(experiments,"bind_models",return_value=({},{})):
            jobs=list(experiments.identities(core.HOME/"runs"/"synthetic","final","router"))
        self.assertEqual(len(jobs),2*len(req.deadlines()))
        self.assertEqual(len({core.digest(j[2]) for j in jobs}),len(jobs))
        for _,item,identity in jobs:
            self.assertEqual(item["request"]["deadline_s"],item["budget"])
            self.assertEqual(identity["request"],item["request"])

    def test_extraction_retains_profile_and_routing_flags(self):
        from research.three_papers import extract
        item=dict(key="k",method="router",budget=30,repetition=0,profile_id="balanced_service")
        result=dict(complete=True,status="valid",exit_s=25.,result={
            "selected_method":"rh","routing_decision":{"risk_supported":False,
                "fallback":True,"predicted_success":.7,"method_allowance_s":29.8}})
        with patch.object(extract,"accepted",return_value=(True,{"cost":100.,"ens_mwh":0.})):
            r=extract.observation({"key":"k"},item,result,
                {"reference_cost":100.,"reference_bound":None,"reference_certified":False},{},None)
        self.assertEqual(r["profile_id"],"balanced_service")
        self.assertTrue(r["routing_fallback"])
        self.assertEqual(r["routing_method_allowance_s"],29.8)

if __name__ == "__main__":
    unittest.main()
