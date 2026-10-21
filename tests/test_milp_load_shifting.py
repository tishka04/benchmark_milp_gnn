"""Physical regression tests for the full UC model, using small HiGHS solves.

Run from the repository root: python -m unittest discover -s tests -v
No historical benchmark data or trained checkpoints are needed.
"""
from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd
from pyomo.environ import SolverFactory, TerminationCondition, value

from src.milp.batch_runner import _serialize_report as serialize_batch
from src.milp.export_binary_variables import extract_all_variables
from src.milp.model import DR_FORMULATION, build_uc_model
from src.milp.run_milp import _serialize_report as serialize_single
from src.milp.scenario_loader import load_scenario_data
from src.milp.solve import _apply_mip_warm_start, _relax_integrality, solve_scenario


def scenario_input(n=2, dt=1.0, dr_units=1, zones=1):
    """Minimal generator-format input, without storage or external imports."""
    return {
        "id": "00000000-0000-0000-0000-000000000001",
        "horizon_hours": n * dt,
        "dt_minutes": 60 * dt,
        "graph": {"zones_per_region": [zones], "intertie_density": 0.0, "neighbor_nations": 0},
        "assets": {
            "thermal_per_zone": [0] * zones, "solar_per_zone": [1] * zones,
            "wind_per_zone": [0] * zones, "battery_per_zone": [0] * zones,
            "dr_per_zone": [dr_units] * zones, "hydro_ror_per_zone": [0] * zones,
        },
        "econ_policy": {"co2_price": 0, "import_export_caps_factor": 0, "price_cap": 100},
        "tech": {
            "battery_roundtrip_eff": 1.0, "battery_e_to_p_hours": 1.0,
            "thermal_marg_cost": 0, "thermal_ramp_pct": 1.0,
            "dr_max_shed_share": 0.4, "dr_num_blocks": 1,
            "dr_max_events": n, "dr_ramp_limit_factor": 1.0,
        },
        "operation_costs": {
            "renewable_spill_cost_eur_per_mwh": 0,
            "hydro_spill_cost_eur_per_mwh": 0,
            "overgen_spill_cost_eur_per_mwh": 100,
            "demand_response_cost_eur_per_mwh": 1,
            "value_of_lost_load_eur_per_mwh": 1000,
            "thermal_fuel_eur_per_mwh": 100,
            "nuclear_fuel_eur_per_mwh": 10,
        },
        "exogenous": {
            "demand_profile": "evening_peak", "weather_profile": "sunny",
            "inflow_factor": 0, "demand_scale_factor": 1.0,
        },
    }


def load_input(payload):
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "scenario.json"
        path.write_text(json.dumps(payload), encoding="utf-8")
        return load_scenario_data(path)


def make_data(demand=(10, 10), solar=(6, 14), dt=1.0, limits=None, zones=1):
    data = load_input(scenario_input(len(demand), dt, zones=zones))
    # Prescribe exact profiles so the expected physical dispatch is known.
    limits = [4.0] * len(demand) if limits is None else limits
    for z in data.zones:
        data.dr_ramp_limit[z] = 100.0
        for t in data.periods:
            data.demand[z, t] = float(demand[t])
            data.solar_available[z, t] = float(solar[t])
            data.dr_limit[z, t] = float(limits[t])
            data.dr_block_limit[z, t, 0] = float(limits[t])
    # Keep multi-zone fixtures independent (no cross-zone energy settlement).
    data.lines = {}
    data.lines_from_index = {}
    data.lines_to_index = {}
    return data


class LoadShiftingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.solver = SolverFactory("highs")
        if not cls.solver.available(exception_flag=False):
            raise RuntimeError("These regression tests require HiGHS (highspy)")

    def solve(self, model, feasible=True):
        result = self.solver.solve(model, load_solutions=False)
        expected = TerminationCondition.optimal if feasible else TerminationCondition.infeasible
        self.assertEqual(result.solver.termination_condition, expected)
        if feasible:
            model.solutions.load_from(result)
        return model

    def assert_energy_conserved(self, model):
        for z in model.Z:
            deferred = sum(value(model.dr_shed[z, t]) for t in model.T) * value(model.dt_hours)
            recovered = sum(value(model.dr_recover[z, t]) for t in model.T) * value(model.dt_hours)
            self.assertAlmostEqual(deferred, recovered, places=7)
            self.assertAlmostEqual(value(model.dr_rebound[z, model.T.last()]), 0.0)

    def test_deficit_is_shifted_and_recovered_in_power_balance(self):
        model = self.solve(build_uc_model(make_data()))
        z = model.Z.first()
        self.assertAlmostEqual(value(model.dr_shed[z, 0]), 4.0)
        self.assertAlmostEqual(value(model.dr_recover[z, 1]), 4.0)
        self.assertAlmostEqual(value(model.p_solar[z, 1]), 14.0)
        self.assertAlmostEqual(sum(value(model.unserved[z, t]) for t in model.T), 0.0)
        self.assert_energy_conserved(model)

    def test_subhourly_backlog_is_in_mwh(self):
        for dt in (0.25, 0.5, 2.0):
            with self.subTest(dt=dt):
                model = self.solve(build_uc_model(make_data(dt=dt)))
                self.assertAlmostEqual(value(model.dr_rebound[model.Z.first(), 0]), 4 * dt)
                self.assert_energy_conserved(model)

    def test_backlog_does_not_decay_during_idle_period(self):
        data = make_data(demand=(10, 10, 10), solar=(8, 10, 12))
        data.dr_rebound_decay = 1.0
        data.dr_rebound_tolerance = 1000.0
        model = build_uc_model(data)
        z = data.zones[0]
        model.dr_shed[z, 0].fix(2)
        model.dr_shed[z, 1].fix(0)
        model.dr_recover[z, 1].fix(0)
        self.solve(model)
        self.assertAlmostEqual(value(model.dr_rebound[z, 1]), 2.0)
        self.assertAlmostEqual(value(model.dr_recover[z, 2]), 2.0)
        self.assert_energy_conserved(model)

    def test_legacy_tolerance_cannot_allow_unpaid_terminal_energy(self):
        data = make_data()
        data.dr_rebound_decay = 1.0
        data.dr_rebound_tolerance = 1000.0
        model = build_uc_model(data)
        z = data.zones[0]
        model.dr_shed[z, 0].fix(1)
        for t in model.T:
            model.dr_recover[z, t].fix(0)
        self.solve(model, feasible=False)

    def test_last_period_cannot_be_shed_instead_of_recovered(self):
        model = self.solve(build_uc_model(make_data(demand=(10,), solar=(6,))))
        z = model.Z.first()
        self.assertAlmostEqual(value(model.dr_shed[z, 0]), 0.0)
        self.assertAlmostEqual(value(model.dr_recover[z, 0]), 0.0)
        self.assertAlmostEqual(value(model.unserved[z, 0]), 4.0)

    def test_cannot_recover_before_deferring_even_in_lp(self):
        for relax in (False, True):
            with self.subTest(relax=relax):
                model = build_uc_model(make_data())
                if relax:
                    _relax_integrality(model)
                model.dr_recover[model.Z.first(), 0].fix(1)
                self.solve(model, feasible=False)

    def test_surplus_before_deficit_cannot_be_used_for_forward_shifting(self):
        model = self.solve(build_uc_model(make_data(solar=(14, 6))))
        z = model.Z.first()
        self.assertAlmostEqual(sum(value(model.dr_shed[z, t]) for t in model.T), 0.0)
        self.assertAlmostEqual(value(model.unserved[z, 1]), 4.0)

    def test_deferral_and_recovery_cannot_overlap(self):
        model = build_uc_model(make_data(demand=(10, 10, 10), solar=(10, 10, 10)))
        z = model.Z.first()
        model.dr_shed[z, 0].fix(1)
        model.dr_shed[z, 1].fix(1)
        model.dr_recover[z, 1].fix(1)
        self.solve(model, feasible=False)

    def test_recovery_capacity_limits_deferral(self):
        data = make_data()
        data.dr_recovery_limit_factor = 0.5
        model = self.solve(build_uc_model(data))
        z = model.Z.first()
        self.assertAlmostEqual(value(model.dr_shed[z, 0]), 2.0)
        self.assertAlmostEqual(value(model.dr_recover[z, 1]), 2.0)
        self.assertAlmostEqual(value(model.unserved[z, 0]), 2.0)
        self.assert_energy_conserved(model)

    def test_no_recovery_capacity_disables_shifting(self):
        data = make_data()
        data.dr_recovery_limit_factor = 0
        model = self.solve(build_uc_model(data))
        self.assertAlmostEqual(sum(value(v) for v in model.dr_shed.values()), 0.0)

    def test_zero_dr_capacity(self):
        model = self.solve(build_uc_model(make_data(limits=(0, 0))))
        for variable in (model.dr_shed, model.dr_recover, model.dr_rebound):
            self.assertAlmostEqual(sum(value(v) for v in variable.values()), 0.0)

    def test_energy_cannot_be_repaid_in_another_zone(self):
        model = build_uc_model(make_data(zones=2))
        first, second = list(model.Z)
        model.dr_shed[first, 0].fix(2)
        for t in model.T:
            model.dr_recover[first, t].fix(0)
            model.dr_shed[second, t].fix(0)
        model.dr_recover[second, 1].fix(2)
        self.solve(model, feasible=False)

    def test_lp_and_fixed_binary_projection_conserve_energy(self):
        mip = self.solve(build_uc_model(make_data()))
        for fixed in (False, True):
            with self.subTest(fixed=fixed):
                model = build_uc_model(make_data())
                if fixed:
                    for key in model.dr_active:
                        model.dr_active[key].fix(round(value(mip.dr_active[key])))
                _relax_integrality(model)
                self.solve(model)
                self.assert_energy_conserved(model)
                self.assertAlmostEqual(value(model.dr_recover[model.Z.first(), 1]), 4.0)

    def test_loader_respects_absent_dr_assets_and_recovery_factor(self):
        payload = scenario_input(dr_units=0)
        payload["tech"]["dr_recovery_limit_factor"] = 0.75
        data = load_input(payload)
        self.assertEqual(data.dr_recovery_limit_factor, 0.75)
        self.assertTrue(all(v == 0 for v in data.dr_limit.values()))
        self.assertTrue(all(v == 0 for v in data.dr_block_limit.values()))

    def test_invalid_recovery_factor_rejected(self):
        for factor in (-1, float("inf"), float("nan")):
            with self.subTest(factor=factor):
                data = make_data()
                data.dr_recovery_limit_factor = factor
                with self.assertRaisesRegex(ValueError, "dr_recovery_limit_factor"):
                    build_uc_model(data)
                payload = scenario_input()
                payload["tech"]["dr_recovery_limit_factor"] = factor
                with self.assertRaisesRegex(ValueError, "dr_recovery_limit_factor"):
                    load_input(payload)

    def test_exports_and_warm_start_keep_recovery(self):
        data = make_data()
        model = self.solve(build_uc_model(data))
        exported = extract_all_variables(model, data)
        z = data.zones[0]
        self.assertEqual(exported["metadata"]["dr_formulation"], DR_FORMULATION)
        self.assertAlmostEqual(exported["continuous_variables"]["dr_recover"][z][1], 4)
        fresh = build_uc_model(data)
        count = _apply_mip_warm_start(fresh, {"continuous_vars": {"dr_recover": np.array([[0, 4]])}})
        self.assertEqual(count, 2)
        self.assertAlmostEqual(value(fresh.dr_recover[z, 1]), 4)

    def test_solve_report_and_csv_include_both_sides_of_shift(self):
        data = make_data()
        with tempfile.TemporaryDirectory() as directory:
            prefix = Path(directory) / "dispatch"
            with patch("src.milp.solve.load_scenario_data", return_value=data):
                report = solve_scenario(Path("synthetic.json"), capture_detail=True, export_csv_prefix=prefix)
            detail = report["detail"]
            z = data.zones[0]
            self.assertAlmostEqual(detail["dr_recover"][z][1], 4)
            self.assertAlmostEqual(sum(detail["demand_response_net"][z]), 0)
            self.assertEqual(detail["pre_dispatch"]["dr_recover"], detail["dr_recover"])
            for serialize in (serialize_single, serialize_batch):
                self.assertEqual(serialize(report)["dr_formulation"], DR_FORMULATION)
            zone = pd.read_csv(prefix.with_name("dispatch_zone.csv"))
            system = pd.read_csv(prefix.with_name("dispatch_system.csv"))
            self.assertAlmostEqual(zone["dr_recover"].sum(), 4)
            self.assertAlmostEqual(system["demand_response_net_total"].sum(), 0)
            self.assertAlmostEqual(system["dr_rebound_total"].iloc[-1], 0)


if __name__ == "__main__":
    unittest.main()
