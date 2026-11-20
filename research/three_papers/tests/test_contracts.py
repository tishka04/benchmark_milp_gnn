import ast
import copy
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
from research.three_papers import core

class Contracts(unittest.TestCase):
    def test_negative_and_near_zero_reference(self):
        self.assertEqual(core.gap(90.,100.),(-.1,-10.))
        self.assertEqual(core.economic(True,90.,100.,.01),1)
        self.assertEqual(core.economic(True,1.5,.5,.05),1)
        self.assertEqual(core.economic(True,1.51,.5,.05),0)
        self.assertIsNone(core.economic(None,None,100.,.05))
        self.assertIsNone(core.economic(False,None,None,.05))
        self.assertEqual(core.economic(False,None,100.,.05),0)

    def test_record_is_read_only_and_disagreement_is_not_repaired(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/"record.json"; b=Path(str(p)+".bak")
            original=core.sealed(dict(complete=False,identity={"k":1}))
            p.write_text(json.dumps(original),encoding="utf8")
            b.write_text(json.dumps(core.sealed(dict(complete=True,identity={"k":1}))),encoding="utf8")
            before={q.name:q.read_bytes() for q in Path(d).iterdir()}
            with self.assertRaises(ValueError): core.record(p)
            self.assertEqual(before,{q.name:q.read_bytes() for q in Path(d).iterdir()})

    def test_short_budget_identity_changes_but_120_is_preserved(self):
        from src.operational_exit.budget_reserve import evaluation_schedule, POLICY, KIND
        manifest={"id":"synthetic"}
        items=[dict(key="a",method="highs",budget=b,repetition=0,seed=1) for b in (5,120)]
        cal=core.sealed(dict(reserve_s=38.))
        amend=core.sealed(dict(kind=KIND,policy=POLICY,manifest_hash=core.digest(manifest)))
        with patch("src.operational_exit.protocol.schedule",return_value=iter(copy.deepcopy(items))):
            actual=list(evaluation_schedule(manifest,cal,amend))
        self.assertNotEqual(actual[0],items[0]); self.assertEqual(actual[1],items[1])

    def test_exit_contract_forced_late_and_valid(self):
        from src.operational_exit.supervisor import classify
        payload=dict(status="solution",validation={"valid":True},dispatch={"exists":True})
        self.assertTrue(classify(payload,budget=5,exit_s=4.9,forced=False,returncode=0)[0])
        self.assertFalse(classify(payload,budget=5,exit_s=5.1,forced=False,returncode=0)[0])
        self.assertFalse(classify(payload,budget=5,exit_s=4.9,forced=True,returncode=0)[0])

    def test_all_prepared_python_parses(self):
        # User-executed static syntax check; no source module is run here.
        for p in core.HOME.rglob("*.py"):
            if "runs" not in p.parts: ast.parse(p.read_text(encoding="utf8"),filename=str(p))

    def test_namespace_cannot_write_campaign(self):
        with self.assertRaises(ValueError): core.writable(core.campaign()/"forbidden.json")

    def test_feature_groups_do_not_split_budgets(self):
        import pandas as pd
        from research.three_papers.statistics import folds
        df=pd.DataFrame(dict(group=["base1","base1","base2","base2"],budget=[5,30,5,30]))
        f=folds(df); self.assertEqual(f[0],f[1]); self.assertEqual(f[2],f[3])

    def test_storage_targets_match_activity_convention(self):
        from research.three_papers.worker import uc_from_dispatch
        values={k:[[0.,2.]] for k in ("b_charge","pumped_discharge")}
        values.update({k:[[0.,0.]] for k in ("b_discharge","pumped_charge","dr_active","v_thermal_startup","u_thermal")})
        values["import_mode"]=[0.,1.]
        u=uc_from_dispatch({"values":values})
        self.assertEqual(u.shape,(1,2,8)); self.assertEqual(u[0,0,:4].sum(),0.)
        self.assertEqual(u[0,1,0],1.); self.assertEqual(u[0,1,3],1.)

if __name__=="__main__": unittest.main()
