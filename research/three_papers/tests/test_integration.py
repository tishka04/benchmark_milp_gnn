"""Opt-in synthetic worker/Gurobi integration, prepared but not run."""
import os
import sys
import tempfile
import time
import unittest
from pathlib import Path
from research.three_papers.core import ROOT, HOME, source_hashes, sha
sys.path.insert(0,str(ROOT/"tests"))
from test_milp_load_shifting import scenario_input, make_data

@unittest.skipUnless(os.environ.get("THREE_PAPERS_INTEGRATION")=="1","explicit integration opt-in")
class WorkerIntegration(unittest.TestCase):
    def test_new_worker_exit_and_dispatch(self):
        import json
        from src.operational_exit.supervisor import run_experiment
        folder=HOME/"test_artifacts"; folder.mkdir(exist_ok=True)
        with tempfile.TemporaryDirectory(dir=folder) as tmp:
            tmp=Path(tmp); case=tmp/"tiny.json"
            case.write_text(json.dumps(scenario_input(n=2)),encoding="utf8")
            settings=dict(method="highs",manifest_hash="synthetic",repair={},reserve_s=2.,
                          extension_sources=source_hashes(),model_hashes={})
            request=dict(key="tiny",family="synthetic",file=str(case),scenario_sha256=sha(case),
                         budget=10,repetition=0,seed=42,identity={"test":"new-worker"})
            r=run_experiment(settings,request,tmp/"attempt",worker_module="research.three_papers.worker")
            self.assertTrue(r["success"],r["status"])
            self.assertTrue(r["all_processes_exited"]); self.assertFalse(r["forced"])

@unittest.skipUnless(os.environ.get("THREE_PAPERS_GUROBI_TEST")=="1","explicit licensed Gurobi opt-in")
class GurobiAdapter(unittest.TestCase):
    def test_same_physical_cost_as_highs(self):
        import gurobipy as gp
        from research.three_papers.gurobi import solve
        from src.operational_exit.solving import solve_data
        d=make_data(demand=[10,10],solar=[6,14],limits=[4,4],zones=1)
        h=solve_data(d,threads=1,seconds=5)
        with gp.Env(params={"OutputFlag":0}) as env:
            g=solve(d,seconds=5,deadline=time.perf_counter()+30,seed=42,env=env)
        self.assertTrue(h["valid"]); self.assertTrue(g["valid"])
        self.assertAlmostEqual(h["validation"]["cost"],g["validation"]["cost"],places=4)

if __name__=="__main__": unittest.main()
