# ==============================================================================
# FEASIBILITY DECODER V4 - Cost-Aware Merit Order
# ==============================================================================
# Hierarchical feasibility decoder that transforms relaxed EBM samples
# into physically feasible dispatch plans.
#
# Merit Order DEFICIT: VRE > RoR > Nuclear > Hydro > Storage > Thermal > Import > DR > Unserved
# Merit Order SURPLUS: Export (revenue) > Reduce Thermal > Curtail VRE > Storage charge > Spill
# ==============================================================================

from __future__ import annotations

import os
import json
import torch
from dataclasses import dataclass, field
from typing import Optional, List, Dict
from pathlib import Path


@dataclass
class ScenarioPhysics:
    """Physics and constraints for a scenario."""

    n_zones: int
    n_timesteps: int
    n_regions: int = 1
    dt_hours: float = 1.0

    zone_names: Optional[List[str]] = None
    zone_to_region: Optional[Dict[str, str]] = None

    # Time series [Z, T]
    demand: Optional[torch.Tensor] = None
    solar_available: Optional[torch.Tensor] = None
    wind_available: Optional[torch.Tensor] = None
    hydro_ror: Optional[torch.Tensor] = None

    # Storage [Z]
    battery_power_mw: Optional[torch.Tensor] = None
    battery_capacity_mwh: Optional[torch.Tensor] = None
    battery_initial_soc: Optional[torch.Tensor] = None
    battery_efficiency: float = 0.90

    pumped_power_mw: Optional[torch.Tensor] = None
    pumped_capacity_mwh: Optional[torch.Tensor] = None
    pumped_initial_soc: Optional[torch.Tensor] = None
    pumped_efficiency: float = 0.80

    # Thermal [Z]
    thermal_capacity_mw: Optional[torch.Tensor] = None
    thermal_min_mw: Optional[torch.Tensor] = None

    # Nuclear [Z]
    nuclear_capacity_mw: Optional[torch.Tensor] = None

    # Hydro reservoir [Z]
    hydro_capacity_mw: Optional[torch.Tensor] = None
    hydro_capacity_mwh: Optional[torch.Tensor] = None
    hydro_initial: Optional[torch.Tensor] = None
    hydro_inflow: Optional[torch.Tensor] = None

    # DR [Z]
    dr_capacity_mw: Optional[torch.Tensor] = None
    dr_max_duration_hours: float = 4.0

    # Import/export (Fix #3: anchor zone)
    import_capacity_mw: float = 0.0
    import_anchor_zone_idx: int = 0
    export_cost_per_mwh: float = 0.0

    # Storage: separate charge/discharge efficiencies (Fix #4)
    battery_eta_charge: Optional[float] = None
    battery_eta_discharge: Optional[float] = None
    battery_retention_rate: Optional[torch.Tensor] = None
    pumped_eta_charge: Optional[float] = None
    pumped_eta_discharge: Optional[float] = None
    pumped_retention_rate: Optional[torch.Tensor] = None

    # Thermal dynamics (Fix #5: startup cost, Fix #6: ramp)
    thermal_startup_cost: Optional[torch.Tensor] = None
    thermal_cost_per_mwh: Optional[torch.Tensor] = None
    thermal_ramp_mw: Optional[torch.Tensor] = None
    thermal_initial_output: Optional[torch.Tensor] = None


@dataclass
class FeasiblePlan:
    """Feasible dispatch plan - key decision variables."""

    # Binary/relaxed decisions [Z, T]
    thermal_on: torch.Tensor
    nuclear_on: torch.Tensor
    battery_charging: torch.Tensor
    battery_discharging: torch.Tensor
    pumped_charging: torch.Tensor
    pumped_discharging: torch.Tensor
    dr_active: torch.Tensor

    # Continuous dispatch [Z, T]
    thermal_dispatch: torch.Tensor
    nuclear_dispatch: torch.Tensor
    battery_charge: torch.Tensor
    battery_discharge: torch.Tensor
    pumped_charge: torch.Tensor
    pumped_discharge: torch.Tensor
    demand_response: torch.Tensor

    # VRE dispatch [Z, T]
    solar_dispatch: torch.Tensor
    wind_dispatch: torch.Tensor
    hydro_dispatch: torch.Tensor

    # Balancing [Z, T]
    unserved_energy: torch.Tensor
    curtailment: torch.Tensor
    net_import: torch.Tensor

    # Storage state [Z, T+1]
    battery_soc: torch.Tensor
    pumped_level: torch.Tensor

    def to_tensor(self) -> torch.Tensor:
        """Convert to binary tensor [Z, T, 7] for EBM."""
        Z, T = self.thermal_dispatch.shape
        u = torch.zeros(Z, T, 7)
        u[:, :, 0] = self.battery_charging
        u[:, :, 1] = self.battery_discharging
        u[:, :, 2] = self.pumped_charging
        u[:, :, 3] = self.pumped_discharging
        u[:, :, 4] = self.dr_active
        # Feature 5: thermal start-up (off→on transition)
        for t in range(T):
            if t == 0:
                u[:, t, 5] = self.thermal_on[:, t]  # on at t=0 counts as start-up
            else:
                u[:, t, 5] = ((self.thermal_on[:, t] == 1.0) &
                              (self.thermal_on[:, t - 1] == 0.0)).float()
        u[:, :, 6] = self.thermal_on
        return u


class HierarchicalFeasibilityDecoder:
    """
    v5 - EBM-Guided Feasibility Decoder.

    Takes relaxed Langevin samples [Z, T, F] and projects them into
    physically feasible dispatch plans. Uses EBM binary suggestions to
    guide resource commitment, with merit-order fallback for feasibility:
      Phase 1: Dispatch must-run and free resources (nuclear, VRE)
      Phase 2: Handle deficit (EBM-guided, then merit-order fallback)
      Phase 3: Handle surplus (export, thermal reduction, curtail, storage)

    Matches MILP model: separate storage efficiencies, thermal ramp limits,
    anchor-zone exports, self-discharge retention.
    """

    FEAT_BATTERY_CHARGE = 0
    FEAT_BATTERY_DISCHARGE = 1
    FEAT_PUMPED_CHARGE = 2
    FEAT_PUMPED_DISCHARGE = 3
    FEAT_DR = 4
    FEAT_THERMAL_SU = 5
    FEAT_THERMAL = 6

    def __init__(
        self,
        physics: ScenarioPhysics,
        nuclear_must_run_fraction: float = 0.20,
        verbose: bool = False,
    ):
        self.physics = physics
        self.nuclear_must_run_fraction = nuclear_must_run_fraction
        self.verbose = verbose

    def decode(self, u_relax: torch.Tensor) -> FeasiblePlan:
        """
        Project relaxed EBM sample to feasible dispatch.

        Uses EBM binary suggestions to guide resource commitment,
        with merit-order fallback when EBM suggestions are insufficient.

        Args:
            u_relax: [Z, T, F] relaxed binary decisions from EBM/Langevin

        Returns:
            FeasiblePlan with feasible dispatch decisions
        """
        Z, T, F = u_relax.shape
        p = self.physics
        dt = p.dt_hours

        # ── Extract EBM binary suggestions (Fix #1) ──
        ebm_therm = u_relax[:, :, self.FEAT_THERMAL] > 0.5
        ebm_b_ch = u_relax[:, :, self.FEAT_BATTERY_CHARGE] > 0.5
        ebm_b_dis = u_relax[:, :, self.FEAT_BATTERY_DISCHARGE] > 0.5
        ebm_p_ch = u_relax[:, :, self.FEAT_PUMPED_CHARGE] > 0.5
        ebm_p_dis = u_relax[:, :, self.FEAT_PUMPED_DISCHARGE] > 0.5
        ebm_dr = u_relax[:, :, self.FEAT_DR] > 0.5

        # ── Capacities ──
        batt_power = p.battery_power_mw if p.battery_power_mw is not None else torch.zeros(Z)
        batt_cap = p.battery_capacity_mwh if p.battery_capacity_mwh is not None else torch.zeros(Z)
        pump_power = p.pumped_power_mw if p.pumped_power_mw is not None else torch.zeros(Z)
        pump_cap = p.pumped_capacity_mwh if p.pumped_capacity_mwh is not None else torch.zeros(Z)
        dr_cap = p.dr_capacity_mw if p.dr_capacity_mw is not None else torch.zeros(Z)
        therm_cap = p.thermal_capacity_mw if p.thermal_capacity_mw is not None else torch.zeros(Z)
        therm_min = p.thermal_min_mw if p.thermal_min_mw is not None else therm_cap * 0.3
        nuc_cap = p.nuclear_capacity_mw if p.nuclear_capacity_mw is not None else torch.zeros(Z)
        hydro_cap_mw = p.hydro_capacity_mw if p.hydro_capacity_mw is not None else torch.zeros(Z)
        hydro_cap_mwh = p.hydro_capacity_mwh if p.hydro_capacity_mwh is not None else torch.zeros(Z)
        import_cap = p.import_capacity_mw

        nuc_must_run = nuc_cap * self.nuclear_must_run_fraction

        # ── Storage efficiencies (Fix #4) ──
        eta_bc = p.battery_eta_charge if p.battery_eta_charge is not None else p.battery_efficiency
        eta_bd = p.battery_eta_discharge if p.battery_eta_discharge is not None else p.battery_efficiency
        eta_pc = p.pumped_eta_charge if p.pumped_eta_charge is not None else p.pumped_efficiency
        eta_pd = p.pumped_eta_discharge if p.pumped_eta_discharge is not None else p.pumped_efficiency
        batt_ret = p.battery_retention_rate
        pump_ret = p.pumped_retention_rate

        # ── Thermal dynamics (Fix #5, #6) ──
        therm_ramp = p.thermal_ramp_mw if p.thermal_ramp_mw is not None else therm_cap.clone()
        # If ramp is 0 for zones WITH thermal capacity, use capacity as fallback (unrestricted)
        therm_ramp = torch.where((therm_ramp < 1e-6) & (therm_cap > 0), therm_cap, therm_ramp)
        prev_therm_disp = p.thermal_initial_output.clone() if p.thermal_initial_output is not None else torch.zeros(Z)

        # ── Anchor zone for export (Fix #3) ──
        anchor_z = p.import_anchor_zone_idx

        # ── Initialize outputs ──
        thermal_on = torch.zeros(Z, T)
        nuclear_on = torch.zeros(Z, T)
        battery_charging = torch.zeros(Z, T)
        battery_discharging = torch.zeros(Z, T)
        pumped_charging = torch.zeros(Z, T)
        pumped_discharging = torch.zeros(Z, T)
        dr_active = torch.zeros(Z, T)

        thermal_dispatch = torch.zeros(Z, T)
        nuclear_dispatch = torch.zeros(Z, T)
        battery_charge = torch.zeros(Z, T)
        battery_discharge = torch.zeros(Z, T)
        pumped_charge = torch.zeros(Z, T)
        pumped_discharge = torch.zeros(Z, T)
        demand_response = torch.zeros(Z, T)

        solar_dispatch = torch.zeros(Z, T)
        wind_dispatch = torch.zeros(Z, T)
        hydro_dispatch = torch.zeros(Z, T)

        unserved_energy = torch.zeros(Z, T)
        curtailment = torch.zeros(Z, T)
        net_import = torch.zeros(Z, T)

        battery_soc = torch.zeros(Z, T + 1)
        pumped_level = torch.zeros(Z, T + 1)

        b_soc = torch.zeros(Z)
        p_level = torch.zeros(Z)
        h_level = torch.zeros(Z)
        dr_used_hours = torch.zeros(Z)

        if p.battery_initial_soc is not None and batt_cap is not None:
            b_soc = (p.battery_initial_soc * batt_cap).clone()
        if p.pumped_initial_soc is not None and pump_cap is not None:
            p_level = (p.pumped_initial_soc * pump_cap).clone()
        if p.hydro_initial is not None:
            h_level = p.hydro_initial.clone()

        battery_soc[:, 0] = b_soc.clone()
        pumped_level[:, 0] = p_level.clone()

        for t in range(T):
            demand_t = p.demand[:, t].clone() if p.demand is not None else torch.zeros(Z)
            solar_avail = p.solar_available[:, t].clone() if p.solar_available is not None else torch.zeros(Z)
            wind_avail = p.wind_available[:, t].clone() if p.wind_available is not None else torch.zeros(Z)
            ror_t = p.hydro_ror[:, t].clone() if p.hydro_ror is not None else torch.zeros(Z)
            hydro_inflow_t = p.hydro_inflow[:, t] if p.hydro_inflow is not None else torch.zeros(Z)

            h_level = h_level + hydro_inflow_t * dt
            h_level = torch.clamp(h_level, max=hydro_cap_mwh)

            # Apply self-discharge retention (Fix #4)
            if batt_ret is not None:
                b_soc = b_soc * batt_ret
            if pump_ret is not None:
                p_level = p_level * pump_ret

            residual = demand_t.clone()

            # ── PHASE 1: MUST-RUN AND FREE RESOURCES ──
            nuc_dispatch_t = nuc_must_run.clone()
            nuclear_dispatch[:, t] = nuc_dispatch_t
            nuclear_on[:, t] = (nuc_cap > 0).float()
            residual = residual - nuc_dispatch_t

            solar_dispatch[:, t] = solar_avail
            wind_dispatch[:, t] = wind_avail
            residual = residual - solar_avail - wind_avail - ror_t

            # ── PHASE 2: HANDLE DEFICIT (EBM-guided, then fallback) ──
            for z in range(Z):
                if residual[z] <= 0:
                    continue

                # Nuclear remaining
                nuc_headroom = nuc_cap[z] - nuc_dispatch_t[z]
                if nuc_headroom > 0:
                    nuc_add = min(nuc_headroom.item(), residual[z].item())
                    nuclear_dispatch[z, t] += nuc_add
                    residual[z] -= nuc_add
                if residual[z] <= 0:
                    continue

                # Hydro reservoir
                if hydro_cap_mw[z] > 0 and h_level[z] > 0:
                    max_release = min(hydro_cap_mw[z].item(), h_level[z].item() / dt)
                    release = min(max_release, residual[z].item())
                    hydro_dispatch[z, t] = release
                    h_level[z] -= release * dt
                    residual[z] -= release
                if residual[z] <= 0:
                    continue

                # --- EBM-guided storage discharge ---
                if ebm_p_dis[z, t] and pump_power[z] > 0 and p_level[z] > 0:
                    max_dis = min(pump_power[z].item(), p_level[z].item() * eta_pd / dt)
                    dis = min(max(max_dis, 0), residual[z].item())
                    if dis > 0:
                        pumped_discharge[z, t] = dis
                        pumped_discharging[z, t] = 1.0
                        p_level[z] -= dis * dt / max(eta_pd, 1e-4)
                        residual[z] -= dis
                if residual[z] <= 0:
                    continue

                if ebm_b_dis[z, t] and batt_power[z] > 0 and b_soc[z] > 0:
                    max_dis = min(batt_power[z].item(), b_soc[z].item() * eta_bd / dt)
                    dis = min(max(max_dis, 0), residual[z].item())
                    if dis > 0:
                        battery_discharge[z, t] = dis
                        battery_discharging[z, t] = 1.0
                        b_soc[z] -= dis * dt / max(eta_bd, 1e-4)
                        residual[z] -= dis
                if residual[z] <= 0:
                    continue

                # --- EBM-guided thermal (with ramp limits - Fix #5, #6) ---
                if ebm_therm[z, t] and therm_cap[z] > 0:
                    ramp_up_max = prev_therm_disp[z].item() + therm_ramp[z].item()
                    if prev_therm_disp[z].item() < 1e-6:
                        max_disp = min(therm_cap[z].item(), max(ramp_up_max, therm_min[z].item()))
                    else:
                        max_disp = min(therm_cap[z].item(), ramp_up_max)
                    dispatch = min(max_disp, residual[z].item())
                    if 0 < dispatch < therm_min[z].item():
                        dispatch = min(therm_min[z].item(), max_disp)
                    if dispatch > 0:
                        thermal_dispatch[z, t] = dispatch
                        thermal_on[z, t] = 1.0
                        residual[z] -= dispatch
                if residual[z] <= 0:
                    continue

                # --- Fallback: resources NOT suggested by EBM ---
                if pumped_discharge[z, t] == 0 and pump_power[z] > 0 and p_level[z] > 0:
                    max_dis = min(pump_power[z].item(), p_level[z].item() * eta_pd / dt)
                    dis = min(max(max_dis, 0), residual[z].item())
                    if dis > 0:
                        pumped_discharge[z, t] = dis
                        pumped_discharging[z, t] = 1.0
                        p_level[z] -= dis * dt / max(eta_pd, 1e-4)
                        residual[z] -= dis
                if residual[z] <= 0:
                    continue

                if battery_discharge[z, t] == 0 and batt_power[z] > 0 and b_soc[z] > 0:
                    max_dis = min(batt_power[z].item(), b_soc[z].item() * eta_bd / dt)
                    dis = min(max(max_dis, 0), residual[z].item())
                    if dis > 0:
                        battery_discharge[z, t] = dis
                        battery_discharging[z, t] = 1.0
                        b_soc[z] -= dis * dt / max(eta_bd, 1e-4)
                        residual[z] -= dis
                if residual[z] <= 0:
                    continue

                if thermal_dispatch[z, t] == 0 and therm_cap[z] > 0:
                    ramp_up_max = prev_therm_disp[z].item() + therm_ramp[z].item()
                    if prev_therm_disp[z].item() < 1e-6:
                        max_disp = min(therm_cap[z].item(), max(ramp_up_max, therm_min[z].item()))
                    else:
                        max_disp = min(therm_cap[z].item(), ramp_up_max)
                    dispatch = min(max_disp, residual[z].item())
                    if 0 < dispatch < therm_min[z].item():
                        dispatch = min(therm_min[z].item(), max_disp)
                    if dispatch > 0:
                        thermal_dispatch[z, t] = dispatch
                        thermal_on[z, t] = 1.0
                        residual[z] -= dispatch
                if residual[z] <= 0:
                    continue

                # Imports (all zones as proxy for network flows)
                if import_cap > 0:
                    imp = min(import_cap, residual[z].item())
                    net_import[z, t] = imp
                    residual[z] -= imp
                if residual[z] <= 0:
                    continue

                # Demand Response
                if dr_cap[z] > 0 and dr_used_hours[z] < p.dr_max_duration_hours:
                    dr_amt = min(dr_cap[z].item(), residual[z].item())
                    demand_response[z, t] = dr_amt
                    dr_active[z, t] = 1.0 if dr_amt > 0 else 0.0
                    dr_used_hours[z] += dt
                    residual[z] -= dr_amt
                if residual[z] <= 0:
                    continue

                # Unserved (last resort)
                if residual[z] > 1e-6:
                    unserved_energy[z, t] = max(0, residual[z].item())
                    residual[z] = 0

            # ── PHASE 3: HANDLE SURPLUS ──
            # Export (anchor only) > Reduce thermal > Curtail VRE > Storage charge > Spill
            surplus = torch.clamp(-residual, min=0)

            for z in range(Z):
                if surplus[z] <= 0:
                    continue

                # Export (anchor zone only - Fix #3)
                if z == anchor_z and import_cap > 0:
                    exp = min(import_cap, surplus[z].item())
                    net_import[z, t] -= exp
                    surplus[z] -= exp
                if surplus[z] <= 0:
                    continue

                # Reduce thermal (ramp-down aware - Fix #6)
                therm_running = thermal_dispatch[z, t].item()
                if therm_running > 0:
                    ramp_dn_floor = max(prev_therm_disp[z].item() - therm_ramp[z].item(), 0)
                    min_disp = max(therm_min[z].item(), ramp_dn_floor) if thermal_on[z, t] > 0 else 0
                    reduce = min(therm_running - min_disp, surplus[z].item())
                    if reduce > 0:
                        thermal_dispatch[z, t] -= reduce
                        if thermal_dispatch[z, t] < 1e-6:
                            thermal_on[z, t] = 0.0
                        surplus[z] -= reduce
                if surplus[z] <= 0:
                    continue

                # Curtail wind
                wind_dispatched = wind_dispatch[z, t].item()
                if wind_dispatched > 0:
                    spill = min(wind_dispatched, surplus[z].item())
                    wind_dispatch[z, t] -= spill
                    curtailment[z, t] += spill
                    surplus[z] -= spill
                if surplus[z] <= 0:
                    continue

                # Curtail solar
                solar_dispatched = solar_dispatch[z, t].item()
                if solar_dispatched > 0:
                    spill = min(solar_dispatched, surplus[z].item())
                    solar_dispatch[z, t] -= spill
                    curtailment[z, t] += spill
                    surplus[z] -= spill
                if surplus[z] <= 0:
                    continue

                # EBM-guided pumped charge
                if ebm_p_ch[z, t] and pumped_charge[z, t] == 0 and pump_power[z] > 0 and pump_cap[z] > 0:
                    headroom = pump_cap[z].item() - p_level[z].item()
                    if headroom > 0:
                        max_ch = min(pump_power[z].item(), headroom / max(eta_pc, 1e-4) / dt)
                        ch = min(max(max_ch, 0), surplus[z].item())
                        if ch > 0:
                            pumped_charge[z, t] = ch
                            pumped_charging[z, t] = 1.0
                            p_level[z] += ch * dt * eta_pc
                            surplus[z] -= ch
                if surplus[z] <= 0:
                    continue

                # EBM-guided battery charge
                if ebm_b_ch[z, t] and battery_charge[z, t] == 0 and batt_power[z] > 0 and batt_cap[z] > 0:
                    headroom = batt_cap[z].item() - b_soc[z].item()
                    if headroom > 0:
                        max_ch = min(batt_power[z].item(), headroom / max(eta_bc, 1e-4) / dt)
                        ch = min(max(max_ch, 0), surplus[z].item())
                        if ch > 0:
                            battery_charge[z, t] = ch
                            battery_charging[z, t] = 1.0
                            b_soc[z] += ch * dt * eta_bc
                            surplus[z] -= ch
                if surplus[z] <= 0:
                    continue

                # Fallback pumped charge
                if pumped_charge[z, t] == 0 and pump_power[z] > 0 and pump_cap[z] > 0:
                    headroom = pump_cap[z].item() - p_level[z].item()
                    if headroom > 0:
                        max_ch = min(pump_power[z].item(), headroom / max(eta_pc, 1e-4) / dt)
                        ch = min(max(max_ch, 0), surplus[z].item())
                        if ch > 0:
                            pumped_charge[z, t] = ch
                            pumped_charging[z, t] = 1.0
                            p_level[z] += ch * dt * eta_pc
                            surplus[z] -= ch
                if surplus[z] <= 0:
                    continue

                # Fallback battery charge
                if battery_charge[z, t] == 0 and batt_power[z] > 0 and batt_cap[z] > 0:
                    headroom = batt_cap[z].item() - b_soc[z].item()
                    if headroom > 0:
                        max_ch = min(batt_power[z].item(), headroom / max(eta_bc, 1e-4) / dt)
                        ch = min(max(max_ch, 0), surplus[z].item())
                        if ch > 0:
                            battery_charge[z, t] = ch
                            battery_charging[z, t] = 1.0
                            b_soc[z] += ch * dt * eta_bc
                            surplus[z] -= ch
                if surplus[z] <= 0:
                    continue

                # Overgeneration spill
                if surplus[z] > 1e-6:
                    curtailment[z, t] += surplus[z].item()
                    surplus[z] = 0

            # Update thermal tracking for next timestep (Fix #5, #6)
            prev_therm_disp = thermal_dispatch[:, t].clone()

            battery_soc[:, t + 1] = b_soc.clone()
            pumped_level[:, t + 1] = p_level.clone()

        return FeasiblePlan(
            thermal_on=thermal_on,
            nuclear_on=nuclear_on,
            battery_charging=battery_charging,
            battery_discharging=battery_discharging,
            pumped_charging=pumped_charging,
            pumped_discharging=pumped_discharging,
            dr_active=dr_active,
            thermal_dispatch=thermal_dispatch,
            nuclear_dispatch=nuclear_dispatch,
            battery_charge=battery_charge,
            battery_discharge=battery_discharge,
            pumped_charge=pumped_charge,
            pumped_discharge=pumped_discharge,
            demand_response=demand_response,
            solar_dispatch=solar_dispatch,
            wind_dispatch=wind_dispatch,
            hydro_dispatch=hydro_dispatch,
            unserved_energy=unserved_energy,
            curtailment=curtailment,
            net_import=net_import,
            battery_soc=battery_soc,
            pumped_level=pumped_level,
        )


def load_physics_from_scenario(
    scenario_id: str,
    scenarios_dir: str,
    n_timesteps: int = 24,
) -> ScenarioPhysics:
    """
    Load ScenarioPhysics from scenario JSON using MILP scenario_loader.

    Falls back to simplified loader if MILP loader is unavailable.
    """
    # Try dispatch_batch subdirectory first, then root
    json_path = None
    for candidate in [
        os.path.join(scenarios_dir, "dispatch_batch", f"{scenario_id}.json"),
        os.path.join(scenarios_dir, f"{scenario_id}.json"),
    ]:
        if os.path.exists(candidate):
            json_path = candidate
            break

    if json_path is None:
        raise FileNotFoundError(
            f"Scenario JSON not found for {scenario_id} in {scenarios_dir}"
        )

    try:
        from src.milp.scenario_loader import load_scenario_data

        data = load_scenario_data(Path(json_path))
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
        )

    except ImportError:
        # Simplified fallback without MILP loader
        with open(json_path, "r") as f:
            scenario = json.load(f)

        graph = scenario.get("graph", {})
        zones_per_region = graph.get("zones_per_region", [1])
        Z = sum(zones_per_region)
        T = n_timesteps
        n_regions = len(zones_per_region)

        zone_names = []
        zone_to_region = {}
        for r_idx, n_z in enumerate(zones_per_region):
            for z in range(n_z):
                zn = f"R{r_idx+1}Z{z+1}"
                zone_names.append(zn)
                zone_to_region[zn] = f"R{r_idx+1}"

        return ScenarioPhysics(
            n_zones=Z,
            n_timesteps=T,
            n_regions=n_regions,
            zone_names=zone_names,
            zone_to_region=zone_to_region,
            demand=torch.zeros(Z, T),
            solar_available=torch.zeros(Z, T),
            wind_available=torch.zeros(Z, T),
            hydro_ror=torch.zeros(Z, T),
            battery_power_mw=torch.zeros(Z),
            battery_capacity_mwh=torch.zeros(Z),
            battery_initial_soc=torch.ones(Z) * 0.5,
            pumped_power_mw=torch.zeros(Z),
            pumped_capacity_mwh=torch.zeros(Z),
            pumped_initial_soc=torch.ones(Z) * 0.5,
            thermal_capacity_mw=torch.zeros(Z),
            thermal_min_mw=torch.zeros(Z),
            nuclear_capacity_mw=torch.zeros(Z),
            hydro_capacity_mw=torch.zeros(Z),
            hydro_capacity_mwh=torch.zeros(Z),
            hydro_initial=torch.zeros(Z),
            hydro_inflow=torch.zeros(Z, T),
            dr_capacity_mw=torch.zeros(Z),
        )
