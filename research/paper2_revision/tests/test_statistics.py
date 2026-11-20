import unittest
import numpy as np
import pandas as pd
from research.paper2_revision.analysis import endpoint, interval, holm, frame, primary_inference, paired

class StatisticsTests(unittest.TestCase):
    def test_unknown_and_failure_are_distinct(self):
        self.assertEqual(endpoint(dict(feasible=False),.05,.001),0)
        self.assertTrue(np.isnan(endpoint(dict(feasible=True,ens_mwh=0),.05,.001)))
        self.assertEqual(endpoint(dict(feasible=True,ens_mwh=2),.05,.001),0)
        self.assertEqual(endpoint(dict(feasible=True),None,None),1)
        self.assertTrue(np.isnan(endpoint(dict(feasible=None),None,None)))
    def test_near_zero_and_negative_cost(self):
        self.assertEqual(endpoint(dict(feasible=True,cost=-12,reference_cost=-10),.05),1)
        self.assertEqual(endpoint(dict(feasible=True,cost=1.1,reference_cost=0),.05),0)
    def test_clustered_bootstrap_keeps_replicates_together(self):
        a=interval([0,1],["a","b"],["f","f"],samples=200,key="same")
        b=interval([0,0,1,1],["a","a","b","b"],["f"]*4,samples=200,key="same")
        for k in ("estimate","lower","upper","groups"): self.assertEqual(a[k],b[k])
    def test_holm_and_unavailable_primary_comparisons(self):
        np.testing.assert_allclose(holm([.01,.04,.2]),[.03,.08,.2])
        df=frame([dict(key="a",group="a",family="f",method="highs",budget=5,feasible=False,repetition=0)])
        table=primary_inference(df)
        self.assertEqual(len(table),25); self.assertFalse(table.complete.any()); self.assertFalse(table.significant.any())
    def test_duplicate_pairs_rejected(self):
        df=pd.DataFrame(dict(key=["x","x"],repetition=[0,0],group=["x","x"],family=["f","f"],y=[0,1]))
        with self.assertRaises(ValueError): paired(df,df,"y")

if __name__=="__main__": unittest.main()
