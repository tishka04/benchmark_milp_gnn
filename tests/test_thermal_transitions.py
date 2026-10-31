"""Only synthetic, one-zone tests (1--4 periods); never touch campaign data.

python -B -m unittest discover -s tests -p test_thermal_transitions.py -v
"""
import os
os.environ["BENCHMARK_HIGHS_THREADS"] = "1"

from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import pandas  # Windows: initialise Arrow before Torch's DLLs.
import torch
from pyomo.environ import Constraint, SolverFactory, TerminationCondition, value
from pyomo.core import TransformationFactory

from src.thermal import THERMAL_FORMULATION, scenario_thermal_limits, thermal_limits, thermal_metrics
from src.ebm.feasibility import HierarchicalFeasibilityDecoder, ScenarioPhysics
from src.milp.model import build_uc_model
from src.milp.lp_worker_two_stage import LPWorkerTwoStage, SolveStage
from src.milp.solve import _apply_mip_warm_start, solve_scenario
from src.milp.run_milp import _serialize_report
from src.uc_jepa.workers.milp_warm_worker import _serialize_lp_result
from test_milp_load_shifting import make_data, load_input, scenario_input


def fixture(demand=(45, 65, 45, 0), *, ramp=20, initial=0, startup=None, shutdown=None):
    data = make_data(demand=demand, solar=[0] * len(demand), limits=[0] * len(demand))
    z = data.zones[0]
    data.thermal_capacity[z] = 100
    data.thermal_min_power[z] = 45
    data.thermal_ramp[z] = ramp
    data.thermal_initial_output[z] = initial
    data.thermal_cost[z] = 1
    data.thermal_startup_cost[z] = 7
    if startup is not None:
        data.thermal_startup_limit[z] = startup
    if shutdown is not None:
        data.thermal_shutdown_limit[z] = shutdown
    return data


def candidates(states):
    u = torch.zeros(1, len(states), 7)
    u[0, :, 6] = torch.tensor(states)
    return u


class ThermalTransitionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.solver = SolverFactory("highs")
        if not cls.solver.available(exception_flag=False):
            raise RuntimeError("Small physical tests require HiGHS")
        cls.solver.options["threads"] = 1
        cls.solver.options["time_limit"] = 5

    def solve(self, data, *, states=None, powers=None, feasible=True, relax=False):
        m = build_uc_model(data)
        z = data.zones[0]
        if states is not None:
            for t, u in enumerate(states):
                m.u_thermal[z, t].fix(u)
        if powers is not None:
            for t, p in enumerate(powers):
                m.p_thermal[z, t].fix(p)
        if relax:
            TransformationFactory("core.relax_integer_vars").apply_to(m)
        result = self.solver.solve(m, load_solutions=False)
        expected = TerminationCondition.optimal if feasible else TerminationCondition.infeasible
        self.assertEqual(result.solver.termination_condition, expected)
        if feasible:
            m.solutions.load_from(result)
            # Check every active original row, not only the thermal helper.
            for c in m.component_data_objects(Constraint, active=True):
                body = value(c.body)
                if c.has_lb():
                    self.assertGreaterEqual(body + 1e-6, value(c.lower), c.name)
                if c.has_ub():
                    self.assertLessEqual(body - 1e-6, value(c.upper), c.name)
            self.assertTrue(self.metrics(m)["thermal_feasible"])
        return m

    @staticmethod
    def series(m, name):
        return [value(getattr(m, name)[m.Z.first(), t]) for t in m.T]

    def metrics(self, m):
        z = m.Z.first()
        p = lambda name: [value(getattr(m, name)[z])]
        return thermal_metrics(
            [self.series(m, "p_thermal")], [self.series(m, "u_thermal")],
            [self.series(m, "v_thermal_startup")], capacity=p("thermal_capacity"),
            minimum=p("thermal_min"), ramp=p("thermal_ramp"), startup_limit=p("thermal_startup_limit"),
            shutdown_limit=p("thermal_shutdown_limit"), initial_power=p("thermal_initial"),
            initial_on=p("thermal_initial_on"),
        )

    def test_start_ramp_and_stop_with_ramp_below_minimum(self):
        m = self.solve(fixture())
        np.testing.assert_allclose(self.series(m, "p_thermal"), [45, 65, 45, 0])
        np.testing.assert_allclose(self.series(m, "v_thermal_startup"), [1, 0, 0, 0])
        np.testing.assert_allclose(self.series(m, "w_thermal_shutdown"), [0, 0, 0, 1])
        self.assertEqual(sum(self.series(m, "unserved")), 0)
        self.assertAlmostEqual(value(m.obj), 45 + 65 + 45 + 7)

    def test_startup_cannot_jump_to_full_capacity(self):
        m = self.solve(fixture((100,)))
        self.assertAlmostEqual(self.series(m, "p_thermal")[0], 45)
        self.assertAlmostEqual(self.series(m, "unserved")[0], 55)
        self.solve(fixture((100,)), states=[1], powers=[100], feasible=False)

    def test_online_up_and_down_ramps_are_preserved(self):
        self.solve(fixture((45, 90)), states=[1, 1], powers=[45, 90], feasible=False)
        self.solve(fixture((90, 45), initial=90), states=[1, 1], powers=[90, 45], feasible=False)

    def test_shutdown_from_high_output_requires_ramping_down(self):
        self.solve(fixture((0,), initial=90), states=[0], feasible=False)
        m = self.solve(fixture((0, 0, 0, 0), initial=90))
        np.testing.assert_allclose(self.series(m, "p_thermal"), [70, 50, 45, 0])
        self.assertEqual(sum(self.series(m, "v_thermal_startup")), 0)

    def test_initially_online_unit_has_no_fake_startup(self):
        m = self.solve(fixture((45,), initial=45))
        self.assertEqual(self.series(m, "v_thermal_startup"), [0])
        self.assertAlmostEqual(value(m.obj), 45)
        stopped = self.solve(fixture((0,), initial=45), states=[0])
        self.assertEqual(self.series(stopped, "w_thermal_shutdown"), [1])

    def test_zero_startup_cost_cannot_create_spurious_transitions(self):
        data = fixture((45, 45, 0, 45))
        data.thermal_startup_cost[data.zones[0]] = 0
        m = self.solve(data, states=[1, 1, 0, 1], relax=True)
        self.assertEqual(self.series(m, "v_thermal_startup"), [1, 0, 0, 1])
        self.assertEqual(self.series(m, "w_thermal_shutdown"), [0, 0, 1, 0])

    def test_zero_online_ramp_still_allows_startup_and_shutdown(self):
        m = self.solve(fixture((45, 45, 0), ramp=0))
        np.testing.assert_allclose(self.series(m, "p_thermal"), [45, 45, 0])
        self.solve(fixture((45, 50), ramp=0), states=[1, 1], powers=[45, 50], feasible=False)

    def test_explicit_startup_and_shutdown_limits(self):
        m = self.solve(fixture((60, 55, 0), startup=60, shutdown=55))
        np.testing.assert_allclose(self.series(m, "p_thermal"), [60, 55, 0])

    def test_limits_and_initial_state_validation(self):
        for kwargs in ({"startup": 20}, {"shutdown": 110}, {"initial_power": 10},
                       {"initial_power": 45, "initial_on": 0}, {"ramp": -1}, {"startup": float("nan")}):
            args = dict(capacity=100, minimum=45, ramp=20)
            args.update(kwargs)
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                thermal_limits(**args)

    def test_scenario_json_overrides_reach_model(self):
        payload = scenario_input(2)
        payload["assets"]["thermal_per_zone"] = [1]
        payload["tech"].update(thermal_ramp_pct=.2, thermal_startup_limit_pct=.6,
                               thermal_shutdown_limit_pct=.55)
        data = load_input(payload)
        unit = scenario_thermal_limits(data)[data.zones[0]]
        self.assertAlmostEqual(unit.startup, .6 * unit.capacity)
        self.assertAlmostEqual(unit.shutdown, .55 * unit.capacity)

    def test_zero_capacity_stays_off(self):
        data = make_data()
        m = self.solve(data)
        self.assertEqual(self.series(m, "u_thermal"), [0, 0])

    def test_mip_and_fixed_commitment_lp_agree(self):
        mip = self.solve(fixture())
        lp = self.solve(fixture(), states=[1, 1, 1, 0], relax=True)
        self.assertAlmostEqual(value(mip.obj), value(lp.obj))

    def test_both_repair_entry_paths_use_corrected_physics(self):
        with tempfile.TemporaryDirectory() as directory:
            worker = LPWorkerTwoStage(directory, verbose=False, time_limit_hard_fix=5,
                                      time_limit_repair_20=5, time_limit_repair_100=5,
                                      deviation_penalty=1, flip_budget_20=10)
            for force in (False, True):
                with self.subTest(force_repair=force), \
                     patch.object(worker, "_find_scenario_path", return_value=Path(directory)/"toy.json"), \
                     patch("src.milp.lp_worker_two_stage.load_scenario_data", return_value=fixture()):
                    result = worker.solve("toy", candidates([1, 1, 1, 0]), max_stages=3,
                                          force_repair_20=force, candidate_time_budget_s=10)
                    self.assertNotEqual(result.status, "error", result.message)
                    self.assertEqual(result.stage_used, SolveStage.REPAIR_20 if force else SolveStage.HARD_FIX)
                    payload = _serialize_lp_result(result)
                    self.assertTrue(payload["thermal_metrics"]["thermal_feasible"])
                    self.assertEqual(payload["thermal_formulation"], THERMAL_FORMULATION)
                    self.assertAlmostEqual(payload["lp_objective"], 162)
                    np.testing.assert_allclose(result.continuous_vars["p_thermal"], [[45, 65, 45, 0]])
                    # The boundary refuses a corrupted ramp, even with valid DR.
                    result.continuous_vars["p_thermal"][0, 0] = 100
                    self.assertEqual(_serialize_lp_result(result)["lp_objective"], float("inf"))

    def test_round_and_refix_derives_startups_from_rounded_states(self):
        m = build_uc_model(fixture((45, 45), initial=45))
        TransformationFactory("core.relax_integer_vars").apply_to(m)
        z = m.Z.first()
        for t in m.T:
            m.u_thermal[z, t].value = .6
            m.v_thermal_startup[z, t].value = .8  # Deliberately inconsistent relaxed proposal.
        targets = {"u_thermal": {(z, 0): 1, (z, 1): 1}, "b_charge_mode": {},
                   "pumped_charge_mode": {}, "dr_active": {}}
        rounded = LPWorkerTwoStage._round_solution_to_targets(object(), m, targets)
        self.assertEqual(list(rounded["v_thermal_startup"].values()), [0, 0])

    def test_warm_start_respects_initial_online_state(self):
        for supplied in ({"binary_sample": candidates([1, 0]).numpy()},
                         {"continuous_vars": {"u_thermal": np.array([[1, 0]])}}):
            m = build_uc_model(fixture((45, 0), initial=45))
            _apply_mip_warm_start(m, supplied)
            self.assertEqual(self.series(m, "v_thermal_startup"), [0, 0])

    def test_report_marks_new_formulation_and_exports_shutdown(self):
        with patch("src.milp.solve.load_scenario_data", return_value=fixture((45, 0))):
            report = solve_scenario(Path("synthetic-only.json"), solver_name="highs",
                                    capture_detail=True, time_limit_seconds=5)
        saved = _serialize_report(report)
        self.assertEqual(saved["thermal_formulation"], THERMAL_FORMULATION)
        self.assertIn("thermal_shutdown", saved["detail"])
        self.assertIn("thermal_parameters", saved["detail"])

    def test_repair_corrects_impossible_immediate_shutdown_candidate(self):
        with tempfile.TemporaryDirectory() as directory:
            worker = LPWorkerTwoStage(directory, verbose=False, time_limit_hard_fix=5,
                                      time_limit_repair_20=5, time_limit_repair_100=5,
                                      deviation_penalty=1, flip_budget_20=10)
            with patch.object(worker, "_find_scenario_path", return_value=Path(directory)/"toy.json"), \
                 patch("src.milp.lp_worker_two_stage.load_scenario_data",
                       return_value=fixture((70, 50, 45, 0), initial=90)):
                result = worker.solve("toy", candidates([0, 0, 0, 0]), max_stages=3,
                                      candidate_time_budget_s=10)
            self.assertTrue(result.thermal_metrics["thermal_feasible"], result.message)
            self.assertNotEqual(result.stage_used, SolveStage.HARD_FIX)
            np.testing.assert_allclose(result.continuous_vars["p_thermal"], [[70, 50, 45, 0]])
            self.assertGreaterEqual(result.n_flips, 3)

    def test_heuristic_warm_starts_obey_the_same_thermal_equations(self):
        for demand, initial, ramp in (((45, 65, 45, 0), 0, 20), ((0, 0, 0, 0), 90, 20),
                                      ((45, 45, 0), 0, 0), ((45, 0), 45, 20)):
            for method in ("decode", "decode_passthrough"):
                with self.subTest(method=method, demand=demand, initial=initial, ramp=ramp):
                    p = ScenarioPhysics(n_zones=1, n_timesteps=len(demand), demand=torch.tensor([demand], dtype=torch.float32),
                                        thermal_capacity_mw=torch.tensor([100.]), thermal_min_mw=torch.tensor([45.]),
                                        thermal_ramp_mw=torch.tensor([float(ramp)]),
                                        thermal_initial_output=torch.tensor([float(initial)]))
                    plan = getattr(HierarchicalFeasibilityDecoder(p), method)(candidates([int(x > 0) for x in demand]))
                    binary = plan.to_tensor().numpy()
                    result = thermal_metrics(plan.thermal_dispatch, plan.thermal_on, binary[:, :, 5],
                                             capacity=[100], minimum=[45], ramp=[ramp],
                                             startup_limit=[45], shutdown_limit=[45],
                                             initial_power=[initial], initial_on=[int(initial > 0)])
                    self.assertTrue(result["thermal_feasible"], result)
                    torch.testing.assert_close(plan.thermal_dispatch + plan.unserved_energy,
                                               p.demand + plan.curtailment)


if __name__ == "__main__":
    unittest.main()
