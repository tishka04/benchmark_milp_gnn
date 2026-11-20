"""Direct ScenarioData conversion, adapted from the historical physical loader.
No alternate loader or simplified fallback is permitted.
"""
import torch
from src.ebm.feasibility import ScenarioPhysics
from src.thermal import scenario_thermal_limits

def physics_from_data(data):
    zones = data.zones
    Z = len(zones)
    T = len(data.periods)

    demand = torch.zeros(Z, T)
    solar_available = torch.zeros(Z, T)
    wind_available = torch.zeros(Z, T)
    hydro_ror_ts = torch.zeros(Z, T)
    hydro_inflow_ts = torch.zeros(Z, T)

    for z_idx, zone in enumerate(zones):
        for t_idx in range(T):
            demand[z_idx, t_idx] = data.demand.get((zone, t_idx), 0.0)
            solar_available[z_idx, t_idx] = data.solar_available.get((zone, t_idx), 0.0)
            wind_available[z_idx, t_idx] = data.wind_available.get((zone, t_idx), 0.0)
            hydro_ror_ts[z_idx, t_idx] = data.hydro_ror_generation.get((zone, t_idx), 0.0)
            hydro_inflow_ts[z_idx, t_idx] = data.hydro_inflow_power.get((zone, t_idx), 0.0)

    thermal_units = scenario_thermal_limits(data)
    thermal_capacity = torch.tensor(
        [data.thermal_capacity.get(z, 0.0) for z in zones], dtype=torch.float32
    )
    thermal_min = torch.tensor(
        [data.thermal_min_power.get(z, 0.0) for z in zones], dtype=torch.float32
    )
    nuclear_capacity = torch.tensor(
        [data.nuclear_capacity.get(z, 0.0) for z in zones], dtype=torch.float32
    )
    battery_power = torch.tensor(
        [data.battery_power.get(z, 0.0) for z in zones], dtype=torch.float32
    )
    battery_capacity = torch.tensor(
        [data.battery_energy.get(z, 0.0) for z in zones], dtype=torch.float32
    )
    battery_initial_soc = torch.tensor(
        [
            data.battery_initial.get(z, 0.0)
            / max(data.battery_energy.get(z, 1.0), 1e-6)
            for z in zones
        ],
        dtype=torch.float32,
    )
    pumped_power = torch.tensor(
        [data.pumped_power.get(z, 0.0) for z in zones], dtype=torch.float32
    )
    pumped_capacity = torch.tensor(
        [data.pumped_energy.get(z, 0.0) for z in zones], dtype=torch.float32
    )
    pumped_initial_soc = torch.tensor(
        [
            data.pumped_initial.get(z, 0.0)
            / max(data.pumped_energy.get(z, 1.0), 1e-6)
            for z in zones
        ],
        dtype=torch.float32,
    )
    hydro_capacity_mw = torch.tensor(
        [data.hydro_res_capacity.get(z, 0.0) for z in zones], dtype=torch.float32
    )
    hydro_capacity_mwh = torch.tensor(
        [data.hydro_res_energy.get(z, 0.0) for z in zones], dtype=torch.float32
    )
    hydro_initial = torch.tensor(
        [data.hydro_initial.get(z, 0.0) for z in zones], dtype=torch.float32
    )
    dr_capacity = torch.zeros(Z)
    for z_idx, zone in enumerate(zones):
        max_dr = max(data.dr_limit.get((zone, t_idx), 0.0) for t_idx in range(T))
        dr_capacity[z_idx] = max_dr

    zone_to_region = {z: data.region_of_zone.get(z, "R1") for z in zones}
    n_regions = len(set(zone_to_region.values()))

    # Anchor zone index (Fix #3)
    anchor_idx = 0
    if data.import_anchor_zone in zones:
        anchor_idx = zones.index(data.import_anchor_zone)

    return ScenarioPhysics(
        n_zones=Z,
        n_timesteps=T,
        n_regions=n_regions,
        dt_hours=data.dt_hours,
        zone_names=zones,
        zone_to_region=zone_to_region,
        demand=demand,
        solar_available=solar_available,
        wind_available=wind_available,
        hydro_ror=hydro_ror_ts,
        battery_power_mw=battery_power,
        battery_capacity_mwh=battery_capacity,
        battery_initial_soc=battery_initial_soc,
        battery_efficiency=data.battery_eta_charge,
        pumped_power_mw=pumped_power,
        pumped_capacity_mwh=pumped_capacity,
        pumped_initial_soc=pumped_initial_soc,
        pumped_efficiency=data.pumped_eta_charge,
        thermal_capacity_mw=thermal_capacity,
        thermal_min_mw=thermal_min,
        nuclear_capacity_mw=nuclear_capacity,
        hydro_capacity_mw=hydro_capacity_mw,
        hydro_capacity_mwh=hydro_capacity_mwh,
        hydro_initial=hydro_initial,
        hydro_inflow=hydro_inflow_ts,
        dr_capacity_mw=dr_capacity,
        dr_limit_mw=torch.tensor([[data.dr_limit[z, t] for t in range(T)] for z in zones]),
        dr_recovery_limit_mw=torch.tensor([
            [data.dr_limit[z, t] * data.dr_recovery_limit_factor for t in range(T)] for z in zones
        ]),
        dr_max_active_periods=torch.tensor([data.dr_max_events[z] for z in zones]),
        dr_ramp_mw=torch.tensor([data.dr_ramp_limit[z] for z in zones]),
        dr_max_duration_hours=4.0,
        import_capacity_mw=data.import_capacity,
        # Fix #3: anchor zone
        import_anchor_zone_idx=anchor_idx,
        export_cost_per_mwh=data.export_cost,
        # Fix #4: separate storage efficiencies + retention
        battery_eta_charge=data.battery_eta_charge,
        battery_eta_discharge=data.battery_eta_discharge,
        battery_retention_rate=torch.tensor(
            [data.battery_retention.get(z, 1.0) for z in zones], dtype=torch.float32
        ),
        pumped_eta_charge=data.pumped_eta_charge,
        pumped_eta_discharge=data.pumped_eta_discharge,
        pumped_retention_rate=torch.tensor(
            [data.pumped_retention.get(z, 1.0) for z in zones], dtype=torch.float32
        ),
        # Fix #5: thermal startup cost
        thermal_startup_cost=torch.tensor(
            [data.thermal_startup_cost.get(z, 0.0) for z in zones], dtype=torch.float32
        ),
        thermal_cost_per_mwh=torch.tensor(
            [data.thermal_cost.get(z, 0.0) for z in zones], dtype=torch.float32
        ),
        # Fix #6: thermal ramp + initial state
        thermal_ramp_mw=torch.tensor(
            [data.thermal_ramp.get(z, 1e6) for z in zones], dtype=torch.float32
        ),
        thermal_initial_output=torch.tensor(
            [data.thermal_initial_output.get(z, 0.0) for z in zones], dtype=torch.float32
        ),
        thermal_initial_on=torch.tensor([thermal_units[z].initial_on for z in zones]),
        thermal_startup_limit_mw=torch.tensor([thermal_units[z].startup for z in zones]),
        thermal_shutdown_limit_mw=torch.tensor([thermal_units[z].shutdown for z in zones]),
    )
