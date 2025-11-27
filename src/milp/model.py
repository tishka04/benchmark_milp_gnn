from __future__ import annotations

from typing import Dict, Tuple

from pyomo.environ import (
    Binary,
    ConcreteModel,
    Constraint,
    NonNegativeReals,
    Objective,
    Param,
    RangeSet,
    Reals,
    Set,
    Suffix,
    Var,
    inequality,
    minimize,
)
from pyomo.environ import value  # noqa: F401  (used by callers)

from .scenario_loader import ScenarioData


def build_uc_model(data: ScenarioData, enable_duals: bool = False) -> ConcreteModel:
    m = ConcreteModel(name=f"UC_{data.scenario_id}")

    m.Z = Set(initialize=data.zones, ordered=True)
    m.T = RangeSet(0, len(data.periods) - 1)
    m.L = Set(initialize=list(data.lines.keys()))

    m.dt_hours = Param(initialize=data.dt_hours, mutable=False)

    m.demand = Param(m.Z, m.T, initialize=data.demand, mutable=False)
    m.dr_limit = Param(m.Z, m.T, initialize=data.dr_limit, mutable=False)
    m.solar_available = Param(m.Z, m.T, initialize=data.solar_available, mutable=False)
    m.wind_available = Param(m.Z, m.T, initialize=data.wind_available, mutable=False)
    m.hydro_ror = Param(m.Z, m.T, initialize=data.hydro_ror_generation, mutable=False)
    m.hydro_inflow = Param(m.Z, m.T, initialize=data.hydro_inflow_power, mutable=False)

    m.thermal_capacity = Param(m.Z, initialize=data.thermal_capacity, mutable=False)
    m.thermal_min = Param(m.Z, initialize=data.thermal_min_power, mutable=False)
    m.thermal_cost = Param(m.Z, initialize=data.thermal_cost, mutable=False)
    m.thermal_ramp = Param(m.Z, initialize=data.thermal_ramp, mutable=False)
    m.thermal_initial = Param(m.Z, initialize=data.thermal_initial_output, mutable=False)
    m.thermal_startup_cost = Param(m.Z, initialize=data.thermal_startup_cost, mutable=False)

    m.nuclear_capacity = Param(m.Z, initialize=data.nuclear_capacity, mutable=False)
    m.nuclear_min = Param(m.Z, initialize=data.nuclear_min_power, mutable=False)
    m.nuclear_cost = Param(m.Z, initialize=data.nuclear_cost, mutable=False)
    m.nuclear_startup_cost = Param(m.Z, initialize=data.nuclear_startup_cost, mutable=False)

    m.battery_power = Param(m.Z, initialize=lambda _, z: data.battery_power.get(z, 0.0), mutable=False)
    m.battery_energy = Param(m.Z, initialize=lambda _, z: data.battery_energy.get(z, 0.0), mutable=False)
    m.battery_initial = Param(m.Z, initialize=lambda _, z: data.battery_initial.get(z, 0.0), mutable=False)
    m.battery_cycle_cost = Param(m.Z, initialize=lambda _, z: data.battery_cycle_cost.get(z, 0.0), mutable=False)
    m.battery_retention = Param(m.Z, initialize=lambda _, z: data.battery_retention.get(z, 1.0), mutable=False)
    m.battery_final_min = Param(m.Z, initialize=lambda _, z: data.battery_final_min.get(z, 0.0), mutable=False)
    m.battery_final_max = Param(m.Z, initialize=lambda _, z: data.battery_final_max.get(z, 0.0), mutable=False)

    m.hydro_capacity = Param(m.Z, initialize=lambda _, z: data.hydro_res_capacity.get(z, 0.0), mutable=False)
    m.hydro_energy = Param(m.Z, initialize=lambda _, z: data.hydro_res_energy.get(z, 0.0), mutable=False)
    m.hydro_initial = Param(m.Z, initialize=lambda _, z: data.hydro_initial.get(z, 0.0), mutable=False)

    m.pumped_power = Param(m.Z, initialize=lambda _, z: data.pumped_power.get(z, 0.0), mutable=False)
    m.pumped_energy = Param(m.Z, initialize=lambda _, z: data.pumped_energy.get(z, 0.0), mutable=False)
    m.pumped_initial = Param(m.Z, initialize=lambda _, z: data.pumped_initial.get(z, 0.0), mutable=False)
    m.pumped_cycle_cost = Param(m.Z, initialize=lambda _, z: data.pumped_cycle_cost.get(z, 0.0), mutable=False)
    m.pumped_retention = Param(m.Z, initialize=lambda _, z: data.pumped_retention.get(z, 1.0), mutable=False)
    m.pumped_final_min = Param(m.Z, initialize=lambda _, z: data.pumped_final_min.get(z, 0.0), mutable=False)
    m.pumped_final_max = Param(m.Z, initialize=lambda _, z: data.pumped_final_max.get(z, 0.0), mutable=False)

    m.import_capacity = Param(initialize=data.import_capacity, mutable=False)
    m.import_cost = Param(initialize=data.import_cost, mutable=False)
    m.export_cost = Param(initialize=data.export_cost, mutable=False)
    m.overgen_spill_cost = Param(initialize=data.overgen_spill_penalty, mutable=False)
    m.voll = Param(initialize=data.voll, mutable=False)
    m.dr_cost = Param(initialize=data.dr_cost_per_mwh, mutable=False)
    m.res_spill_cost = Param(initialize=data.variable_spill_cost, mutable=False)
    m.hydro_spill_cost = Param(initialize=data.hydro_spill_cost, mutable=False)

    # Indexed lookup helpers
    lines_from: Dict[str, Tuple[str, ...]] = {
        z: data.lines_from_index.get(z, tuple()) for z in data.zones
    }
    lines_to: Dict[str, Tuple[str, ...]] = {z: data.lines_to_index.get(z, tuple()) for z in data.zones}

    m.p_thermal = Var(m.Z, m.T, within=NonNegativeReals)
    m.u_thermal = Var(m.Z, m.T, within=Binary)
    m.v_thermal_startup = Var(m.Z, m.T, within=Binary)  # Startup indicator

    m.p_nuclear = Var(m.Z, m.T, within=NonNegativeReals)
    m.u_nuclear = Var(m.Z, m.T, within=Binary)  # Commitment binary for nuclear
    m.v_nuclear_startup = Var(m.Z, m.T, within=Binary)  # Startup indicator

    m.p_solar = Var(m.Z, m.T, within=NonNegativeReals)
    m.spill_solar = Var(m.Z, m.T, within=NonNegativeReals)
    m.p_wind = Var(m.Z, m.T, within=NonNegativeReals)
    m.spill_wind = Var(m.Z, m.T, within=NonNegativeReals)

    m.dr_shed = Var(m.Z, m.T, within=NonNegativeReals)
    m.unserved = Var(m.Z, m.T, within=NonNegativeReals)

    m.b_charge = Var(m.Z, m.T, within=NonNegativeReals)
    m.b_discharge = Var(m.Z, m.T, within=NonNegativeReals)
    m.b_soc = Var(m.Z, m.T, within=NonNegativeReals)

    m.h_release = Var(m.Z, m.T, within=NonNegativeReals)
    m.h_spill = Var(m.Z, m.T, within=NonNegativeReals)
    m.h_level = Var(m.Z, m.T, within=NonNegativeReals)

    m.pumped_charge = Var(m.Z, m.T, within=NonNegativeReals)
    m.pumped_discharge = Var(m.Z, m.T, within=NonNegativeReals)
    m.pumped_level = Var(m.Z, m.T, within=NonNegativeReals)

    m.flow = Var(m.L, m.T, within=Reals)
    m.net_import = Var(m.T, within=NonNegativeReals, bounds=(0.0, data.import_capacity))
    m.net_export = Var(m.T, within=NonNegativeReals, bounds=(0.0, data.import_capacity))
    m.overgen_spill = Var(m.Z, m.T, within=NonNegativeReals)

    # Thermal constraints
    def _thermal_capacity_rule(model, z, t):
        return model.p_thermal[z, t] <= model.thermal_capacity[z] * model.u_thermal[z, t]

    m.thermal_capacity_limit = Constraint(m.Z, m.T, rule=_thermal_capacity_rule)

    def _thermal_min_rule(model, z, t):
        return model.p_thermal[z, t] >= model.thermal_min[z] * model.u_thermal[z, t]

    m.thermal_min_limit = Constraint(m.Z, m.T, rule=_thermal_min_rule)

    def _thermal_ramp_up_rule(model, z, t):
        if t == model.T.first():
            prev = model.thermal_initial[z]
        else:
            prev = model.p_thermal[z, t - 1]
        return model.p_thermal[z, t] - prev <= model.thermal_ramp[z]

    def _thermal_ramp_down_rule(model, z, t):
        if t == model.T.first():
            prev = model.thermal_initial[z]
        else:
            prev = model.p_thermal[z, t - 1]
        return prev - model.p_thermal[z, t] <= model.thermal_ramp[z]

    m.thermal_ramp_up = Constraint(m.Z, m.T, rule=_thermal_ramp_up_rule)
    m.thermal_ramp_down = Constraint(m.Z, m.T, rule=_thermal_ramp_down_rule)

    # Thermal startup logic
    def _thermal_startup_rule(model, z, t):
        if t == model.T.first():
            # Assume initially off (thermal_initial = 0 for most zones)
            # If unit turns on at t=0, count as startup
            return model.v_thermal_startup[z, t] >= model.u_thermal[z, t]
        else:
            # Startup occurs when u[t] > u[t-1]
            return model.v_thermal_startup[z, t] >= model.u_thermal[z, t] - model.u_thermal[z, t - 1]

    m.thermal_startup_detection = Constraint(m.Z, m.T, rule=_thermal_startup_rule)

    # Nuclear limits with commitment variable
    def _nuclear_cap_rule(model, z, t):
        return model.p_nuclear[z, t] <= model.nuclear_capacity[z] * model.u_nuclear[z, t]

    def _nuclear_min_rule(model, z, t):
        return model.p_nuclear[z, t] >= model.nuclear_min[z] * model.u_nuclear[z, t]

    m.nuclear_capacity_limit = Constraint(m.Z, m.T, rule=_nuclear_cap_rule)
    m.nuclear_min_limit = Constraint(m.Z, m.T, rule=_nuclear_min_rule)

    # Nuclear startup logic
    def _nuclear_startup_rule(model, z, t):
        if t == model.T.first():
            # Assume nuclear initially on (baseload) - only count startup if going from off to on
            return model.v_nuclear_startup[z, t] >= model.u_nuclear[z, t] - 1.0
        else:
            return model.v_nuclear_startup[z, t] >= model.u_nuclear[z, t] - model.u_nuclear[z, t - 1]

    m.nuclear_startup_detection = Constraint(m.Z, m.T, rule=_nuclear_startup_rule)

    # Renewable curtailment
    def _solar_balance_rule(model, z, t):
        return model.p_solar[z, t] + model.spill_solar[z, t] == model.solar_available[z, t]

    def _wind_balance_rule(model, z, t):
        return model.p_wind[z, t] + model.spill_wind[z, t] == model.wind_available[z, t]

    m.solar_balance = Constraint(m.Z, m.T, rule=_solar_balance_rule)
    m.wind_balance = Constraint(m.Z, m.T, rule=_wind_balance_rule)

    # Demand response bounds
    def _dr_limit_rule(model, z, t):
        return model.dr_shed[z, t] <= model.dr_limit[z, t]

    m.dr_limit_con = Constraint(m.Z, m.T, rule=_dr_limit_rule)

    # Battery power limits
    def _battery_power_charge_rule(model, z, t):
        return model.b_charge[z, t] <= model.battery_power[z]

    def _battery_power_discharge_rule(model, z, t):
        return model.b_discharge[z, t] <= model.battery_power[z]

    m.battery_charge_power = Constraint(m.Z, m.T, rule=_battery_power_charge_rule)
    m.battery_discharge_power = Constraint(m.Z, m.T, rule=_battery_power_discharge_rule)

    # Battery state of charge
    def _battery_energy_rule(model, z, t):
        eta_c = data.battery_eta_charge
        eta_d = data.battery_eta_discharge
        if t == model.T.first():
            prev = model.battery_initial[z]
        else:
            prev = model.b_soc[z, t - 1]
        retention = model.battery_retention[z]
        return model.b_soc[z, t] == retention * prev + model.dt_hours * (
            eta_c * model.b_charge[z, t] - model.b_discharge[z, t] / max(eta_d, 1e-4)
        )

    m.battery_soc_evolution = Constraint(m.Z, m.T, rule=_battery_energy_rule)

    def _battery_soc_bounds_rule(model, z, t):
        return model.b_soc[z, t] <= model.battery_energy[z]

    m.battery_soc_limit = Constraint(m.Z, m.T, rule=_battery_soc_bounds_rule)

    def _battery_final_min_rule(model, z):
        if value(model.battery_energy[z]) <= 1e-6:
            return Constraint.Skip
        if value(model.battery_final_min[z]) <= 1e-6:
            return Constraint.Skip
        return model.b_soc[z, model.T.last()] >= model.battery_final_min[z]

    def _battery_final_max_rule(model, z):
        if value(model.battery_energy[z]) <= 1e-6:
            return Constraint.Skip
        if value(model.battery_final_max[z]) <= 1e-6:
            return Constraint.Skip
        return model.b_soc[z, model.T.last()] <= model.battery_final_max[z]

    m.battery_final_min_con = Constraint(m.Z, rule=_battery_final_min_rule)
    m.battery_final_max_con = Constraint(m.Z, rule=_battery_final_max_rule)

    # Hydro reservoir
    def _hydro_release_cap_rule(model, z, t):
        return model.h_release[z, t] <= model.hydro_capacity[z]

    m.hydro_release_cap = Constraint(m.Z, m.T, rule=_hydro_release_cap_rule)

    def _hydro_level_rule(model, z, t):
        if model.hydro_energy[z] <= 1e-6:
            return model.h_level[z, t] == 0.0
        if t == model.T.first():
            prev = model.hydro_initial[z]
        else:
            prev = model.h_level[z, t - 1]
        inflow = model.dt_hours * model.hydro_inflow[z, t]
        return model.h_level[z, t] == prev + inflow - model.dt_hours * (
            model.h_release[z, t] + model.h_spill[z, t]
        )

    m.hydro_level_balance = Constraint(m.Z, m.T, rule=_hydro_level_rule)

    def _hydro_level_cap_rule(model, z, t):
        return model.h_level[z, t] <= model.hydro_energy[z]

    m.hydro_level_cap = Constraint(m.Z, m.T, rule=_hydro_level_cap_rule)

    # Pumped storage
    def _pumped_charge_limit_rule(model, z, t):
        return model.pumped_charge[z, t] <= model.pumped_power[z]

    def _pumped_discharge_limit_rule(model, z, t):
        return model.pumped_discharge[z, t] <= model.pumped_power[z]

    m.pumped_charge_limit = Constraint(m.Z, m.T, rule=_pumped_charge_limit_rule)
    m.pumped_discharge_limit = Constraint(m.Z, m.T, rule=_pumped_discharge_limit_rule)

    def _pumped_level_rule(model, z, t):
        eta_c = data.pumped_eta_charge
        eta_d = data.pumped_eta_discharge
        if model.pumped_energy[z] <= 1e-6:
            return model.pumped_level[z, t] == 0.0
        if t == model.T.first():
            prev = model.pumped_initial[z]
        else:
            prev = model.pumped_level[z, t - 1]
        retention = model.pumped_retention[z]
        return model.pumped_level[z, t] == retention * prev + model.dt_hours * (
            eta_c * model.pumped_charge[z, t] - model.pumped_discharge[z, t] / max(eta_d, 1e-4)
        )

    m.pumped_level_balance = Constraint(m.Z, m.T, rule=_pumped_level_rule)

    def _pumped_level_cap_rule(model, z, t):
        return model.pumped_level[z, t] <= model.pumped_energy[z]

    m.pumped_level_cap = Constraint(m.Z, m.T, rule=_pumped_level_cap_rule)

    def _pumped_final_min_rule(model, z):
        if value(model.pumped_energy[z]) <= 1e-6:
            return Constraint.Skip
        if value(model.pumped_final_min[z]) <= 1e-6:
            return Constraint.Skip
        return model.pumped_level[z, model.T.last()] >= model.pumped_final_min[z]

    def _pumped_final_max_rule(model, z):
        if value(model.pumped_energy[z]) <= 1e-6:
            return Constraint.Skip
        if value(model.pumped_final_max[z]) <= 1e-6:
            return Constraint.Skip
        return model.pumped_level[z, model.T.last()] <= model.pumped_final_max[z]

    m.pumped_final_min_con = Constraint(m.Z, rule=_pumped_final_min_rule)
    m.pumped_final_max_con = Constraint(m.Z, rule=_pumped_final_max_rule)

    # Transmission bounds
    def _flow_upper_rule(model, l, t):
        return model.flow[l, t] <= data.lines[l].capacity_mw

    def _flow_lower_rule(model, l, t):
        return -data.lines[l].capacity_mw <= model.flow[l, t]

    m.flow_upper = Constraint(m.L, m.T, rule=_flow_upper_rule)
    m.flow_lower = Constraint(m.L, m.T, rule=_flow_lower_rule)

    # Power balance
    def _power_balance_rule(model, z, t):
        generation = (
            model.p_thermal[z, t]
            + model.p_nuclear[z, t]
            + model.p_solar[z, t] + model.p_wind[z, t]
            + model.h_release[z, t]
            + model.hydro_ror[z, t]
            + model.b_discharge[z, t]
            + model.pumped_discharge[z, t]
        )
        inflow = sum(model.flow[l, t] for l in lines_to[z])
        outflow = sum(model.flow[l, t] for l in lines_from[z])
        net_exchange = 0.0
        if z == data.import_anchor_zone:
            net_exchange = model.net_import[t] - model.net_export[t]
        lhs = generation + model.dr_shed[z, t] + model.unserved[z, t] + inflow + net_exchange
        rhs = model.demand[z, t] + model.b_charge[z, t] + model.pumped_charge[z, t] + outflow + model.overgen_spill[z, t]
        return lhs == rhs

    m.power_balance = Constraint(m.Z, m.T, rule=_power_balance_rule)

    # Objective
    def _objective_rule(model):
        gen_cost = sum(
            model.thermal_cost[z] * model.p_thermal[z, t]
            + model.nuclear_cost[z] * model.p_nuclear[z, t]
            for z in model.Z for t in model.T
        )
        startup_cost = sum(
            model.thermal_startup_cost[z] * model.v_thermal_startup[z, t]
            + model.nuclear_startup_cost[z] * model.v_nuclear_startup[z, t]
            for z in model.Z for t in model.T
        )
        response_cost = sum(
            model.dr_cost * model.dr_shed[z, t] + model.voll * model.unserved[z, t]
            for z in model.Z for t in model.T
        )
        spill_cost = sum(
            model.res_spill_cost * (model.spill_solar[z, t] + model.spill_wind[z, t]) + model.hydro_spill_cost * model.h_spill[z, t]
            for z in model.Z for t in model.T
        )
        storage_cost = sum(
            model.battery_cycle_cost[z] * (model.b_charge[z, t] + model.b_discharge[z, t])
            + model.pumped_cycle_cost[z] * (model.pumped_charge[z, t] + model.pumped_discharge[z, t])
            for z in model.Z for t in model.T
        )
        import_cost = sum(model.import_cost * model.net_import[t] for t in model.T)
        export_cost = sum(model.export_cost * model.net_export[t] for t in model.T)
        overgen_cost = sum(model.overgen_spill_cost * model.overgen_spill[z, t] for z in model.Z for t in model.T)
        return gen_cost + startup_cost + response_cost + spill_cost + storage_cost + import_cost + export_cost + overgen_cost

    m.obj = Objective(rule=_objective_rule, sense=minimize)

    if enable_duals:
        m.dual = Suffix(direction=Suffix.IMPORT)

    # Fix binaries for zones without capacity
    for z in data.zones:
        if data.thermal_capacity.get(z, 0.0) <= 1e-6:
            for t in data.periods:
                m.u_thermal[z, t].fix(0.0)
                m.v_thermal_startup[z, t].fix(0.0)
        if data.nuclear_capacity.get(z, 0.0) <= 1e-6:
            for t in data.periods:
                m.u_nuclear[z, t].fix(0.0)
                m.v_nuclear_startup[z, t].fix(0.0)
        else:
            # Nuclear typically stays on (baseload) - initialize commitment to 1
            for t in data.periods:
                m.u_nuclear[z, t].setlb(1.0)  # Force nuclear to stay on

    return m

