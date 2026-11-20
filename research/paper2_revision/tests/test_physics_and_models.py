import sys
from pathlib import Path
import time
import unittest
from unittest.mock import patch
import pandas
import numpy as np
import torch
from research.paper2_revision.core import ROOT
sys.path.insert(0,str(ROOT/"tests"))
from test_milp_load_shifting import make_data
from research.paper2_revision.cases import transform
from research.paper2_revision.physics_adapter import physics_from_data
from research.paper2_revision.multihorizon import VariableAE, Pool
from research.paper2_revision.worker import repair_change
from research.paper2_revision.extract import accepted
from src.operational_exit.solving import solve_data

class PhysicsTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls): torch.set_num_threads(1)
    def test_horizon_and_heuristic_see_same_realized_data(self):
        d=make_data(demand=[10]*24,solar=[10]*24,limits=[2]*24,zones=2)
        from src.milp.scenario_loader import NetworkLine
        from research.paper2_revision.cases import rebuild_lines
        d.lines={"L0":NetworkLine("L0",d.zones[0],d.zones[1],20.,10.)}
        rebuild_lines(d)
        for condition in ("base","size","lines","fleet","permutation","horizon48","horizon72"):
            data=transform(d,condition,42); p=physics_from_data(data)
            self.assertEqual(p.n_timesteps,len(data.periods)); self.assertEqual(p.zone_names,data.zones)
            expected=[[data.demand[z,t] for t in data.periods] for z in data.zones]
            np.testing.assert_allclose(p.demand.numpy(),expected,rtol=1e-6)
            if condition.startswith("horizon"):
                for z in d.zones:
                    for t in range(24): self.assertEqual(data.demand[z,t],d.demand[z,t])
            for z in data.zones:
                self.assertLessEqual(data.battery_initial[z],data.battery_energy[z])
    def test_permutation_preserves_physical_optimum(self):
        d=make_data(demand=[10,10],solar=[6,14],limits=[4,4],zones=3)
        a=solve_data(d,threads=1,seconds=2); b=solve_data(transform(d,"permutation",23),threads=1,seconds=2)
        self.assertTrue(a["valid"]); self.assertTrue(b["valid"])
        self.assertAlmostEqual(a["validation"]["cost"],b["validation"]["cost"],places=3)
    def test_model_equivariance_and_horizons(self):
        cfg=dict(latent_dim=16,tokens=4,heads=2); ae=VariableAE(6,cfg).eval(); predictor=Pool(6,16,4,2).eval()
        permutation=torch.tensor([2,0,1])
        for t in (12,24,48,72):
            h=torch.randn(1,3,t,6); u=torch.rand(1,3,t,8)
            result=ae(u,h); changed=ae(u[:,permutation],h[:,permutation])
            self.assertEqual(tuple(result.shape),(1,3,t,8))
            self.assertTrue(torch.allclose(result[:,permutation],changed,atol=2e-5))
            self.assertTrue(torch.allclose(predictor(h),predictor(h[:,permutation]),atol=2e-5))
    def test_applicability_mask_excludes_absent_storage(self):
        d=make_data(demand=[10,10],solar=[10,10],limits=[0,0],zones=1)
        d.battery_power={z:0. for z in d.zones}; d.battery_energy={z:0. for z in d.zones}
        a=solve_data(d,threads=1,seconds=2); self.assertTrue(a["valid"])
        from research.paper2_revision.worker import uc_from_dispatch
        candidate=uc_from_dispatch(a["dispatch"]); candidate[:,:,0]=1-candidate[:,:,0]
        out=repair_change(d,candidate,a["dispatch"])
        self.assertEqual(out["repair_changed_count"],0); self.assertGreater(out["repair_changed_fraction_all"],0)
    def test_late_or_forced_success_rejected(self):
        for attrs in (dict(exit_s=6),dict(forced=True),dict(all_processes_exited=False)):
            r=dict(complete=True,success=True,forced=False,all_processes_exited=True,returncode=0,exit_s=4)
            r.update(attrs)
            with self.assertRaises(ValueError): accepted(r,5,None)
        self.assertEqual(accepted(dict(complete=True,success=False),5,None)[0],False)
    def test_gurobi_matches_highs_when_available(self):
        try: import gurobipy as gp
        except ImportError: self.skipTest("Gurobi is not installed")
        from research.paper2_revision.gurobi import solve
        d=make_data(demand=[10,10],solar=[6,14],limits=[4,4],zones=2)
        try:
            with gp.Env(params={"OutputFlag":0}) as env:
                b=solve(d,seconds=2,deadline=time.perf_counter()+8,seed=7,env=env)
        except gp.GurobiError as exc: self.skipTest(f"Gurobi license unavailable: {exc}")
        a=solve_data(d,threads=1,seconds=2)
        self.assertTrue(b["valid"]); self.assertAlmostEqual(a["validation"]["cost"],b["validation"]["cost"],places=3)

class AblationTests(unittest.TestCase):
    def test_zero_gradient_pipeline_matches_control_candidate_for_candidate(self):
        from types import SimpleNamespace
        from research.paper2_revision.fixed_generator import Generator
        from src.uc_jepa.sampling.latent_langevin import LatentLangevinConfig
        class Sampler:
            config=LatentLangevinConfig(num_steps=3)
            def _as_prior_tensor(self,v,z,default): return torch.full_like(z,float(default))
            def _get_annealed_scale(self,v,step): return v
        class AE:
            def decode(self,z,**kw): return z[:,:2,:8].unsqueeze(2).expand(-1,2,2,8)
        g=Generator.__new__(Generator); g.device=torch.device("cpu"); g.sampler=Sampler()
        g.prior_mean=None; g.prior_std=None; g.thresholds=torch.full((8,),.5); g.ae=AE()
        g.hte=SimpleNamespace(encode=lambda data:torch.ones(1,2,2,8))
        g.predictor=lambda *a,**k:{"z_pred":torch.zeros(1,2,8)}
        g.ebm=lambda z,*a,**k:z.sum(dim=(1,2))*0
        data=SimpleNamespace(zones=["a","b"],periods=[0,1]); req=dict(key="paired",repetition=0)
        with patch("torch.cuda.synchronize"):
            g.control=False; a,sa=g.candidates(data,req,time.perf_counter()+5)
            g.control=True; b,sb=g.candidates(data,req,time.perf_counter()+5)
        np.testing.assert_array_equal(a,b); self.assertEqual(sa,sb); self.assertEqual(len(sa),16)

if __name__=="__main__": unittest.main()
