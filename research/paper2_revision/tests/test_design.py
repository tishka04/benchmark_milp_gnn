import ast
import tempfile
from pathlib import Path
import unittest
from unittest.mock import patch
from research.paper2_revision.core import config, FAMILIES, economic, sealed, verify, put, record, sha
from research.paper2_revision.design import schedule, planned_blocks, CONDITIONS
from research.paper2_revision.experiments import guard_test

class DesignTests(unittest.TestCase):
    def cases(self,n,partition,controlled=False):
        return [dict(key=f"{partition}/{f}/{i}/{c}",group=f"{partition}/{f}/{i}",family=f,
                     partition=partition,condition=c,expected_horizon=int(c[7:]) if c.startswith("horizon") else 24)
                for f in FAMILIES for i in range(n) for c in (CONDITIONS if controlled else ["base"])]
    def test_exact_plan_counts(self):
        p=planned_blocks(); self.assertEqual(p["trials"],23700); self.assertEqual(p["references"],1200)
        self.assertEqual(p["allocated_trial_hours"],311.25)
        self.assertEqual(len(list(schedule(self.cases(20,"final"),"four"))),2400)
        self.assertEqual(len(list(schedule(self.cases(20,"final"),"ablation"))),1200)
        self.assertEqual(len(list(schedule(self.cases(5,"controlled_test",True),"four"))),3600)
        self.assertEqual(len(list(schedule(self.cases(5,"controlled_test",True),"multi"))),3150)
    def test_stability_counts_and_no_historical_overflow(self):
        cases=self.cases(20,"final")
        self.assertEqual(sum(len(list(schedule(cases,s,extra_repeats=True))) for s in ("four","ablation")),720)
        controlled=self.cases(5,"controlled_test",True)
        self.assertEqual(len(list(schedule(controlled,"multi",training_seed=20261113,sensitivity=True))),540)
        self.assertTrue(all(c["expected_horizon"]<=24 or r["method"] in ("highs","rh") for c,r in schedule(controlled,"four")))
    def test_failure_without_reference(self):
        self.assertEqual(economic(False,None,None,.05),0)
        self.assertIsNone(economic(True,3,None,.05))
        self.assertEqual(economic(True,-12,-10,.05),1)
        self.assertEqual(economic(True,.5,0,.05),1)
    def test_final_guard_requires_lock(self):
        with tempfile.TemporaryDirectory() as folder, patch("research.paper2_revision.experiments.frozen",return_value={"sha256":"x"}):
            with self.assertRaises(FileNotFoundError): guard_test(folder,"final")
            guard_test(folder,"development")
    def test_restart_preserves_failures_and_rejects_changed_identity(self):
        with tempfile.TemporaryDirectory(dir=Path(__file__).parents[1]) as folder, patch("research.paper2_revision.core.HOME",Path(folder)):
            path=Path(folder)/"trial.json"
            identity={"protocol":"fixed","model":"abc"}
            terminal=sealed(dict(identity=identity,complete=True,success=False))
            self.assertIsNone(record(path,identity,optional=True))
            put(path,terminal); original=sha(path)
            put(path,terminal)
            self.assertEqual(sha(path),original)
            self.assertFalse(record(path,identity)["success"])
            with self.assertRaises(ValueError): record(path,dict(identity,model="changed"))
            with self.assertRaises(ValueError): put(path,sealed(dict(terminal,success=True)))
            self.assertEqual(sha(path),original)
    def test_final_guard_rejects_model_modified_after_lock(self):
        with tempfile.TemporaryDirectory(dir=Path(__file__).parents[1]) as folder, patch("research.paper2_revision.core.HOME",Path(folder)), patch("research.paper2_revision.experiments.frozen",return_value={"sha256":"x"}):
            run=Path(folder); model=run/"model.pt"; model.write_bytes(b"original")
            put(run/"cases.json",{"cases":[]})
            put(run/"design_lock.json",sealed(dict(freeze="x",cases=sha(run/"cases.json"),artifacts={str(model):sha(model)})))
            guard_test(run,"final")
            model.write_bytes(b"changed")
            with self.assertRaises(ValueError): guard_test(run,"final")
    def test_seal_and_syntax(self):
        x=sealed({"x":1}); x["x"]=2
        with self.assertRaises(ValueError): verify(x)
        root=Path(__file__).parents[1]
        for p in root.glob("*.py"): ast.parse(p.read_text(encoding="utf-8-sig"),filename=str(p))

if __name__=="__main__": unittest.main()
