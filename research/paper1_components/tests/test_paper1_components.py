"""Synthetic validation only. These tests do not read campaign outcomes or run solvers."""
import ast
import copy
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import numpy as np
import pandas as pd
from research.paper1_components import dataio as io, models, reporting, audit, fitting

def config():
    cfg = copy.deepcopy(io.protocol())
    cfg["bootstrap_samples"] = 30
    return cfg

def observation(i=0, budget=30, **updates):
    row = dict(key=f"k{i}", group=f"g{i}", family=f"f{i%2}",
        method="highs", repetition=0, partition="historical_development", condition="historical",
        budget=budget, observed=True, feasible=True, exit_s=2., cost=100.,
        reference_cost=100., ens_mwh=0., n_zones=10+i, n_steps=24, n_lines=9+i,
        zone_steps=24*(10+i), criticality=.4, stress=.5, hardness=.3)
    row.update({c:.1+.05*i for c in io.COMPONENTS})
    row.update({"raw_"+c:float(i) for c in io.COMPONENTS})
    row.update({"sat_"+c:0 for c in io.COMPONENTS})
    row.update(updates)
    return row

class IsolationAndData(unittest.TestCase):
    def test_seed_accepts_pandas_group_key_scalars(self):
        self.assertEqual(io.seed("e", np.int64(30), np.bool_(False)), io.seed("e", 30, False))

    def test_syntax_and_no_frozen_project_imports(self):
        for file in io.HOME.rglob("*.py"):
            if "runs" in file.relative_to(io.HOME).parts:
                continue
            tree = ast.parse(file.read_text(encoding="utf8"), filename=str(file))
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    names = [a.name for a in node.names]
                elif isinstance(node, ast.ImportFrom):
                    names = [node.module or ""]
                else:
                    continue
                self.assertFalse(any(n.startswith(("src", "research.three_papers")) for n in names), file)

    def test_write_boundary_and_immutable_publication(self):
        with self.assertRaises(ValueError):
            io.writable(io.ROOT/"research/three_papers/runs/anything.json")
        (io.HOME/"runs").mkdir(exist_ok=True)
        with tempfile.TemporaryDirectory(dir=io.HOME/"runs") as tmp:
            p = Path(tmp)/"test.json"
            io.put(p, {"x":1}); io.put(p, {"x":1})
            with self.assertRaises(ValueError):
                io.put(p, {"x":2})
            self.assertEqual(io.read(p), {"x":1})

    def test_labels_respect_time_cost_service_and_unknowns(self):
        cfg = config()
        self.assertEqual(io.labels(observation(),cfg), {"economic_5":1,"economic_5_ens":1})
        self.assertEqual(io.labels(observation(ens_mwh=1.),cfg)["economic_5_ens"],0)
        self.assertEqual(io.labels(observation(exit_s=30.01),cfg)["economic_5"],0)
        self.assertEqual(io.labels(observation(feasible=False,cost=None,reference_cost=None),cfg)["economic_5"],0)
        self.assertIsNone(io.labels(observation(observed=False),cfg)["economic_5"])
        self.assertIsNone(io.labels(observation(ens_mwh=None),cfg)["economic_5_ens"])
        self.assertEqual(io.labels(observation(cost=1.5,reference_cost=.5),cfg)["economic_5"],1)
        self.assertEqual(io.labels(observation(cost=-105.,reference_cost=-100.),cfg)["economic_5"],1)

    def test_only_five_independent_primary_allocations(self):
        cfg=config(); cfg.update(expected_scenarios=4,expected_families=2,expected_per_family=2)
        rows=[observation(i,b) for i in range(4) for b in [1,2,5,10,30,60,120]]
        rows.append(observation(method="rh"))
        rows.append(observation(repetition=1))
        df=io.select_rows(rows,cfg)
        self.assertEqual(len(df),20)
        self.assertEqual(sorted(df.budget.unique()),cfg["deadlines_s"])
        with self.assertRaises(ValueError):
            io.select_rows(rows+[observation()],cfg)

    def test_pre_solve_descriptor_cannot_change_with_deadline(self):
        cfg=config(); cfg.update(expected_scenarios=4,expected_families=2,expected_per_family=2)
        rows=[observation(i,b) for i in range(4) for b in cfg["deadlines_s"]]
        rows[0]["criticality"]=.99
        with self.assertRaisesRegex(ValueError,"varies across deadlines"):
            io.select_rows(rows,cfg)

    def test_all_deadlines_of_a_group_share_the_fold(self):
        frame=pd.DataFrame([observation(i,b) for i in range(12) for b in [5,30,120]])
        frame["fold"]=io.group_folds(frame,3,"test")
        self.assertTrue((frame.groupby("group").fold.nunique()==1).all())
        self.assertEqual(set(frame.fold),{0,1,2})

class PredictiveDesign(unittest.TestCase):
    def test_component_removal_also_removes_its_products(self):
        cfg=config()
        spec=next(s for s in models.specifications(cfg) if s.name=="drop_component_A2")
        self.assertNotIn("A2_residual_volatility",spec.components)
        self.assertFalse(any("A2_residual_volatility" in p for p in spec.pairs))
        reduced=next(s for s in models.specifications(cfg) if s.name=="drop_interaction_1")
        self.assertIn("A2_residual_volatility",reduced.components)
        self.assertIn("A8_inv_storage_energy",reduced.components)
        self.assertNotIn(tuple(cfg["interactions"][0]),reduced.pairs)

    def test_design_learns_only_from_training_features(self):
        cfg=config()
        train=pd.DataFrame([observation(i) for i in range(6)])
        train.loc[0,"A2_residual_volatility"]=np.nan
        spec=next(s for s in models.specifications(cfg) if s.name=="interactions")
        design=models.Design(spec,True,cfg).fit(train)
        old=design.medians.copy()
        test=pd.DataFrame([observation(99,cost=1e20,reference_cost=-1e20)])
        a=design.transform(test)
        test["economic_5"]=999; test["cost"]=-1e30
        np.testing.assert_array_equal(a,design.transform(test))
        pd.testing.assert_series_equal(old,design.medians)
        self.assertFalse(any("cost" in n or "reference" in n for n in design.names))
        # An unseen family must not create a new learned column.
        test["family"]="unseen"
        self.assertEqual(design.transform(test).shape,a.shape)

    def test_constant_parent_has_no_identifiable_product(self):
        cfg=config()
        train=pd.DataFrame([observation(i) for i in range(8)])
        a,b=cfg["interactions"][0]; train[a]=.5
        spec=next(s for s in models.specifications(cfg) if s.name=="interactions")
        design=models.Design(spec,False,cfg).fit(train)
        self.assertIn((a,b),design.inactive_pairs)
        self.assertNotIn(a+"*"+b,design.names)

    def test_interaction_can_capture_a_synthetic_xor_beyond_main_effects(self):
        cfg=config(); a,b=cfg["interactions"][0]
        records=[]
        for repeat in range(12):
            for x,z in [(0.1,0.1),(0.1,0.9),(0.9,0.1),(0.9,0.9)]:
                r=observation(0,**{a:x,b:z})
                r.update(group=str(len(records)),key=str(len(records)),y=int(x==z))
                records.append(r)
        train=pd.DataFrame(records)
        specs={s.name:s for s in models.specifications(cfg)}
        additive=models.fit_one(train,train.y,specs["additive"],False,10.,cfg)
        joint=models.fit_one(train,train.y,specs["interactions"],False,10.,cfg)
        # Distinct observations of the same four feature combinations.
        test=train.iloc[:4].copy(); test["group"]=["held1","held2","held3","held4"]
        self.assertLess(np.mean((test.y-joint.predict(test))**2),
                        np.mean((test.y-additive.predict(test))**2)-.1)

    def test_nested_tuning_never_predicts_on_a_training_group(self):
        cfg=config()
        frame=pd.DataFrame([observation(i%8,group=str(i),key=str(i),economic_5=i%2) for i in range(18)])
        calls=[]
        class Fake:
            def __init__(self, groups): self.groups=set(groups)
            def predict(self, valid):
                self.assertion(valid)
                return np.full(len(valid),.5)
            def assertion(self,valid):
                if self.groups & set(valid.group):
                    raise AssertionError("Group leakage")
        def fit(train,y,spec,control,C,cfg):
            calls.append(set(train.group))
            return Fake(train.group)
        with patch.object(models,"fit_one",side_effect=fit):
            _, tuning=models.tune_fit(frame,"economic_5",models.Spec("size"),False,cfg)
        self.assertEqual(tuning["C"],min(cfg["regularization_C"]))
        self.assertEqual(len(calls),len(cfg["regularization_C"])*cfg["inner_folds"]+1)

    def test_constant_label_returns_explicit_constant_predictor(self):
        cfg=config(); frame=pd.DataFrame([observation(i) for i in range(8)])
        model=models.fit_one(frame,np.zeros(len(frame)),models.Spec("size"),False,1.,cfg)
        self.assertEqual(model.status,"constant_label")
        np.testing.assert_array_equal(model.predict(frame),np.zeros(len(frame)))

class ReportingAndResume(unittest.TestCase):
    def test_contrasts_are_separate_and_have_the_documented_sign(self):
        cfg=config()
        rows=[]
        for budget in [5,30]:
            for i in range(8):
                for name,p in [("size",.5),("additive",.5),("interactions",.9 if budget==30 else .1)]:
                    rows.append(dict(endpoint="economic_5",budget=budget,family_control=False,
                        description=name,group=str(i),key=str(i),family="f",fold=i%2,y=1,p=p))
        contrasts,folds,families=reporting.contrast_tables(pd.DataFrame(rows),cfg)
        q=contrasts[(contrasts.contrast_kind=="interaction_set")&(contrasts.metric=="brier")]
        self.assertGreater(q[q.budget==30].gain.iloc[0],0)
        self.assertLess(q[q.budget==5].gain.iloc[0],0)
        self.assertEqual(set(folds.budget),{5,30})
        self.assertEqual(set(families.family),{"f"})

    def test_sparse_pair_cells_are_not_filled_with_predictions(self):
        cfg=config();cfg["minimum_surface_groups"]=3
        base=pd.DataFrame([observation(i) for i in range(2)])[["group"]+cfg["targeted_components"]]
        pred=pd.DataFrame([dict(endpoint="economic_5",budget=30,family_control=False,
            description=name,group=f"g{i}",y=1,p=.8) for i in range(2) for name in ["additive","interactions"]])
        table=reporting.surface_table(pred,base,cfg)
        self.assertFalse(table.supported.any())
        self.assertTrue(table.probability_difference.isna().all())

    def test_tied_pairs_are_not_counted_as_rank_reversals(self):
        r=audit.rank_change([0.,0.,1.],[1.,1.,0.])
        self.assertEqual(r["reversed_pairs"],2)
        self.assertEqual(r["comparable_pairs"],2)
        self.assertEqual(r["baseline_tied_pairs"],1)

    def test_score_variants_do_not_redefine_stored_components(self):
        cfg=config();base=pd.DataFrame([observation(i) for i in range(4)])
        snapshot=base.copy(deep=True)
        original,variants=audit.score_variants(base,cfg)
        pd.testing.assert_frame_equal(base,snapshot)
        np.testing.assert_allclose(original,variants["mix_0.5"])
        self.assertEqual(len(variants),5+len(io.COMPONENTS))

    def test_checkpoint_resume_does_not_refit_completed_work(self):
        cfg=config();cfg.update(endpoints=["economic_5"],deadlines_s=[30],family_controls=[False],outer_folds=2)
        frame=pd.DataFrame([observation(i,economic_5=i%2,outer_fold=i%2) for i in range(4)])
        class Fake:
            def predict(self,test): return np.full(len(test),.5)
            def metadata(self): return dict(status="synthetic",active_terms=[],dropped_constant_terms=[],
                                           inactive_interactions=[],coefficients=[])
        tuning=dict(C=1.,inner_brier=.25,inner_scores=[],tuning_status="synthetic")
        (io.HOME/"runs").mkdir(exist_ok=True)
        with tempfile.TemporaryDirectory(dir=io.HOME/"runs") as tmp, \
             patch.object(fitting,"frozen",return_value={"sha256":"synthetic"}), \
             patch.object(fitting,"protocol",return_value=cfg), \
             patch.object(fitting,"data",return_value=frame), \
             patch.object(fitting,"specifications",return_value=[models.Spec("size")]), \
             patch.object(fitting,"tune_fit",return_value=(Fake(),tuning)) as fit:
            fitting.fit(tmp); fitting.fit(tmp)
            self.assertEqual(fit.call_count,2)
            self.assertTrue((Path(tmp)/"fit_receipt.json").exists())

if __name__=="__main__":
    unittest.main()
