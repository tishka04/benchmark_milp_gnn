"""Synthetic tests prepared for the user; no historical data or campaign writes."""
import sys
import unittest
from research.three_papers.core import ROOT
sys.path.insert(0,str(ROOT/"tests"))
from test_milp_load_shifting import make_data
from research.three_papers.cases import transform
from src.operational_exit.solving import solve_data

class PhysicalAdapters(unittest.TestCase):
    def test_horizon_preserves_first_day_and_axes(self):
        d=make_data(demand=[10]*24,solar=[10]*24,limits=[0]*24,zones=2)
        out=transform(d,"horizon72",42)
        self.assertEqual(len(out.periods),72)
        for z in d.zones:
            for t in d.periods: self.assertEqual(d.demand[z,t],out.demand[z,t])
        self.assertEqual(len(out.demand),len(d.zones)*72)
        self.assertEqual(len(d.periods),24)

    def test_permutation_is_an_optimization_isomorphism(self):
        d=make_data(demand=[10,10],solar=[6,14],limits=[4,4],zones=3)
        d.demand[d.zones[0],0]=12
        a=solve_data(d,threads=1,seconds=5)
        p=transform(d,"permutation",23)
        b=solve_data(p,threads=1,seconds=5)
        self.assertTrue(a["valid"]); self.assertTrue(b["valid"])
        self.assertAlmostEqual(a["validation"]["cost"],b["validation"]["cost"],places=4)

    def test_size_replication_and_fleet_bounds(self):
        d=make_data(demand=[10,10],solar=[10,10],limits=[0,0],zones=2)
        out=transform(d,"size",42)
        self.assertEqual(len(out.zones),4); self.assertIn("bridge",out.lines)
        self.assertEqual(sum(out.demand.values()),2*sum(d.demand.values()))
        out=transform(d,"fleet",42)
        for z in out.zones:
            self.assertLessEqual(out.thermal_min_power[z],out.thermal_capacity[z])
            self.assertLessEqual(out.battery_initial[z],out.battery_energy[z])

if __name__=="__main__": unittest.main()
