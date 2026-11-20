"""Small CPU architecture tests; no training, solver or GPU needed."""
import unittest
import pandas
import torch
from research.three_papers.multihorizon import VariableAE, Pool, Energy

class VariableLength(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(7)
        self.cfg=dict(latent_dim=16,tokens=4,heads=2)
        self.ae=VariableAE(6,self.cfg).eval()
        self.predictor=Pool(6,16,4,2).eval()

    def test_horizons_and_zone_equivariance(self):
        permutation=torch.tensor([2,0,1])
        for t in (12,24,48,72):
            h=torch.randn(1,3,t,6); u=torch.rand(1,3,t,8)
            z=self.ae.encode(u); out=self.ae.decode(z,h)
            permuted=self.ae.decode(self.ae.encode(u[:,permutation]),h[:,permutation])
            self.assertEqual(tuple(out.shape),(1,3,t,8))
            self.assertTrue(torch.allclose(out[:,permutation],permuted,atol=2e-5))
            self.assertTrue(torch.allclose(self.predictor(h),self.predictor(h[:,permutation]),atol=2e-5))

if __name__=="__main__": unittest.main()
