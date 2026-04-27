"""
Rolling-Horizon Merit-Order Heuristic (RH-MO).

Stage A of the RH-MO+LP baseline described in the paper. Builds a binary
schedule

    u^heur ∈ {0,1}^{Z, T, 7}

with feature ordering matching the EBM/feasibility-decoder convention
(see :class:`src.ebm.feasibility.HierarchicalFeasibilityDecoder`):

    0: battery charging mode      (b_ch)
    1: battery discharging mode   (b_dis)
    2: pumped-hydro charging      (g_ch)
    3: pumped-hydro discharging   (g_dis)
    4: demand-response activation (delta_DR)
    5: thermal start-up           (v_th)   off->on transition
    6: thermal commitment         (u_th)

The schedule is produced by sweeping forward in time. At each step the
heuristic looks W steps ahead to compute simple stress / scarcity
indicators, ranks resources by merit order, and reserves storage / DR
for upcoming peaks. It does **not** look at MILP labels, GNN embeddings,
or EBM outputs.

Stage B (continuous dispatch reconstruction) is delegated to
:class:`src.milp.lp_worker_two_stage.LPWorkerTwoStage` so the heuristic
is directly comparable to the MILP-GNN-EBM pipeline.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import torch

from src.ebm.feasibility import ScenarioPhysics


# Channel indices (must match FeasiblePlan.to_tensor in src/ebm/feasibility.py)
FEAT_BATTERY_CHARGE = 0
FEAT_BATTERY_DISCHARGE = 1
FEAT_PUMPED_CHARGE = 2
FEAT_PUMPED_DISCHARGE = 3
FEAT_DR = 4
FEAT_THERMAL_SU = 5
FEAT_THERMAL = 6


@dataclass
class HeuristicConfig:
    """Hyper-parameters for the rolling-horizon heuristic."""

    window: int = 6                       # W: look-ahead horizon (steps)
    nuclear_must_run_fraction: float = 0.20
    nuclear_max_fraction: float = 1.00    # fraction of nuclear capacity available
    hydro_reserve_beta: float = 0.5       # fraction of future need to reserve
    discharge_quantile: float = 0.70      # discharge only when RL >= q70 of window
    dr_quantile: float = 0.80             # DR only when RL >= q80 of window
    soc_low_frac: float = 0.10            # do not discharge below this SOC fraction
    soc_high_frac: float = 0.95           # do not charge above this SOC fraction
    storage_min_action_frac: float = 0.05 # ignore tiny charge/discharge moves
    thermal_min_pseudo_on_steps: int = 2  # avoid starting for a single spike
    thermal_commit_threshold_frac: float = 0.20  # commit if expected need > 20% cap


def _to_numpy(x, default_shape=None, fill: float = 0.0) -> np.ndarray:
    if x is None:
        if default_shape is None:
            return np.zeros(0, dtype=np.float32)
        return np.full(default_shape, fill, dtype=np.float32)
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy().astype(np.float32, copy=False)
    return np.asarray(x, dtype=np.float32)


def rolling_horizon_heuristic(
    physics: ScenarioPhysics,
    config: Optional[HeuristicConfig] = None,
) -> torch.Tensor:
    """Build the heuristic binary schedule for one scenario.

    Args:
        physics: ScenarioPhysics for the target scenario, as produced by
            :func:`src.ebm.feasibility.load_physics_from_scenario`.
        config:  Optional HeuristicConfig; sensible defaults otherwise.

    Returns:
        torch.FloatTensor of shape ``[Z, T, 7]`` with values in {0, 1}.
    """
    cfg = config or HeuristicConfig()
    Z, T = int(physics.n_zones), int(physics.n_timesteps)
    dt = float(physics.dt_hours)
    W = max(1, int(cfg.window))

    # ── Time-series inputs ─────────────────────────────────────────────
    demand = _to_numpy(physics.demand, (Z, T))
    solar = _to_numpy(physics.solar_available, (Z, T))
    wind = _to_numpy(physics.wind_available, (Z, T))
    ror = _to_numpy(physics.hydro_ror, (Z, T))
    hydro_inflow = _to_numpy(physics.hydro_inflow, (Z, T))  # MW

    # ── Per-zone scalars ───────────────────────────────────────────────
    nuc_cap = _to_numpy(physics.nuclear_capacity_mw, (Z,))
    nuc_must = cfg.nuclear_must_run_fraction * nuc_cap
    nuc_avail = cfg.nuclear_max_fraction * nuc_cap
    nuc_headroom = np.maximum(nuc_avail - nuc_must, 0.0)

    therm_cap = _to_numpy(physics.thermal_capacity_mw, (Z,))
    therm_min = _to_numpy(physics.thermal_min_mw, (Z,))
    therm_init = _to_numpy(physics.thermal_initial_output, (Z,))

    bat_p = _to_numpy(physics.battery_power_mw, (Z,))
    bat_cap = _to_numpy(physics.battery_capacity_mwh, (Z,))
    bat_soc0_frac = _to_numpy(physics.battery_initial_soc, (Z,), fill=0.5)
    eta_bc = float(physics.battery_eta_charge or physics.battery_efficiency or 0.9)
    eta_bd = float(physics.battery_eta_discharge or physics.battery_efficiency or 0.9)
    bat_ret = _to_numpy(physics.battery_retention_rate, (Z,), fill=1.0)

    phs_p = _to_numpy(physics.pumped_power_mw, (Z,))
    phs_cap = _to_numpy(physics.pumped_capacity_mwh, (Z,))
    phs_soc0_frac = _to_numpy(physics.pumped_initial_soc, (Z,), fill=0.5)
    eta_pc = float(physics.pumped_eta_charge or physics.pumped_efficiency or 0.8)
    eta_pd = float(physics.pumped_eta_discharge or physics.pumped_efficiency or 0.8)
    phs_ret = _to_numpy(physics.pumped_retention_rate, (Z,), fill=1.0)

    hydro_p = _to_numpy(physics.hydro_capacity_mw, (Z,))
    hydro_e_max = _to_numpy(physics.hydro_capacity_mwh, (Z,))
    hydro_e = _to_numpy(physics.hydro_initial, (Z,))

    dr_cap = _to_numpy(physics.dr_capacity_mw, (Z,))
    dr_budget_h = float(physics.dr_max_duration_hours or 0.0)

    imp_cap = float(physics.import_capacity_mw or 0.0)
    anchor = int(physics.import_anchor_zone_idx or 0)

    # ── Derived: residual load after must-run / free resources ────────
    must_run = nuc_must[:, None] + ror + solar + wind        # [Z, T]
    residual = demand - must_run                              # [Z, T]

    # ── State carried forward across t ────────────────────────────────
    soc_b = bat_soc0_frac * bat_cap
    soc_p = phs_soc0_frac * phs_cap
    therm_on_prev = (therm_init > 1e-3).astype(np.float32)
    dr_used_h = np.zeros(Z, dtype=np.float32)
    therm_keep_until = np.full(Z, -1, dtype=np.int32)         # min-on bookkeeping

    u = np.zeros((Z, T, 7), dtype=np.float32)

    eps = 1e-6
    for t in range(T):
        w_end = min(t + W, T)
        rl_w = residual[:, t:w_end]                           # [Z, w]
        rl = residual[:, t]
        rl_max = rl_w.max(axis=1)
        rl_min = rl_w.min(axis=1)
        # Window quantiles per zone
        q_disc = np.quantile(rl_w, cfg.discharge_quantile, axis=1)
        q_dr = np.quantile(rl_w, cfg.dr_quantile, axis=1)

        # Approximate "firm" capacity available across the window for thermal sizing.
        # Used only to decide whether thermal is needed at all.
        firm_flex = (
            nuc_headroom
            + hydro_p
            + np.minimum(bat_p, soc_b * eta_bd / max(dt, eps))
            + np.minimum(phs_p, soc_p * eta_pd / max(dt, eps))
        )
        # Imports only at anchor
        firm_flex_with_imp = firm_flex.copy()
        firm_flex_with_imp[anchor] += imp_cap

        # Future-peak driven SOC target (fraction of capacity)
        # If future stress is high relative to past surplus, target high SOC.
        rl_span = np.maximum(rl_max - rl_min, 1.0)
        future_stress = np.clip((rl_max - rl_min) / rl_span, 0.0, 1.0)
        soc_b_target = bat_cap * np.clip(0.2 + 0.6 * future_stress, 0.1, 0.9)
        soc_p_target = phs_cap * np.clip(0.2 + 0.6 * future_stress, 0.1, 0.9)

        for z in range(Z):
            need = float(rl[z])

            # ── Nuclear modulation headroom (free, no binary tracked) ──
            if need > 0 and nuc_headroom[z] > 0:
                nuc_extra = min(nuc_headroom[z], need)
                need -= nuc_extra

            # ── Reservoir hydro (low cost, but reserve for future peak) ─
            hydro_disp = 0.0
            if hydro_p[z] > 0 and hydro_e_max[z] > 0:
                future_extra = max(0.0, rl_max[z] - rl[z])
                reserve = cfg.hydro_reserve_beta * future_extra * dt
                avail_e = max(0.0, hydro_e[z] - reserve)
                if need > 0:
                    hydro_disp = min(hydro_p[z], need, avail_e / max(dt, eps))
                    hydro_disp = max(hydro_disp, 0.0)
                    need -= hydro_disp
                # Update reservoir energy (inflow + dispatch)
                hydro_e[z] = max(
                    0.0,
                    min(
                        hydro_e_max[z],
                        hydro_e[z] + hydro_inflow[z, t] * dt - hydro_disp * dt,
                    ),
                )

            # ── Storage discharge (battery, then pumped hydro) ──────────
            do_b_ch = do_b_dis = 0
            do_p_ch = do_p_dis = 0

            if need > 0 and bat_p[z] > 0 and bat_cap[z] > 0:
                if rl[z] >= q_disc[z] - eps and soc_b[z] > cfg.soc_low_frac * bat_cap[z]:
                    p_dis = min(
                        bat_p[z],
                        need,
                        soc_b[z] * eta_bd / max(dt, eps),
                    )
                    if p_dis > cfg.storage_min_action_frac * bat_p[z]:
                        do_b_dis = 1
                        soc_b[z] -= p_dis * dt / max(eta_bd, eps)
                        soc_b[z] = max(soc_b[z], 0.0)
                        need -= p_dis

            if need > 0 and phs_p[z] > 0 and phs_cap[z] > 0:
                if rl[z] >= q_disc[z] - eps and soc_p[z] > cfg.soc_low_frac * phs_cap[z]:
                    p_dis = min(
                        phs_p[z],
                        need,
                        soc_p[z] * eta_pd / max(dt, eps),
                    )
                    if p_dis > cfg.storage_min_action_frac * phs_p[z]:
                        do_p_dis = 1
                        soc_p[z] -= p_dis * dt / max(eta_pd, eps)
                        soc_p[z] = max(soc_p[z], 0.0)
                        need -= p_dis

            # ── Imports (anchor zone only) ──────────────────────────────
            if z == anchor and need > 0 and imp_cap > 0:
                imp_used = min(need, imp_cap)
                need -= imp_used

            # ── Thermal commitment with anti-flapping ───────────────────
            therm_on_now = 0
            if therm_cap[z] > 0:
                expected_need = float(rl_max[z])
                window_deficit = max(
                    0.0,
                    expected_need - firm_flex_with_imp[z],
                )
                commit = (
                    window_deficit > cfg.thermal_commit_threshold_frac * therm_cap[z]
                    or need > eps
                )
                if commit:
                    therm_on_now = 1
                    therm_keep_until[z] = max(
                        therm_keep_until[z],
                        t + cfg.thermal_min_pseudo_on_steps - 1,
                    )
                elif therm_on_prev[z] > 0.5 and t <= therm_keep_until[z]:
                    # honour minimum on duration
                    therm_on_now = 1
                # Persistent surplus: allow decommitment
                if rl_max[z] < therm_min[z] and t > therm_keep_until[z]:
                    therm_on_now = 0

            # ── DR (emergency, last flexible lever before unserved) ─────
            do_dr = 0
            if (
                need > eps
                and dr_cap[z] > 0
                and dr_used_h[z] + dt <= dr_budget_h + eps
                and rl[z] >= q_dr[z] - eps
            ):
                do_dr = 1
                dr_used_h[z] += dt
                need -= dr_cap[z]

            # ── Surplus path: charge storage / decommit thermal ─────────
            if rl[z] < 0:
                surplus = -rl[z]
                # Charge only if future stress expected
                if rl_max[z] > 0 and bat_p[z] > 0 and bat_cap[z] > 0:
                    if soc_b[z] < soc_b_target[z] and soc_b[z] < cfg.soc_high_frac * bat_cap[z]:
                        room = (bat_cap[z] - soc_b[z]) / max(dt * eta_bc, eps)
                        p_ch = min(bat_p[z], surplus, room)
                        if p_ch > cfg.storage_min_action_frac * bat_p[z]:
                            do_b_ch = 1
                            soc_b[z] += p_ch * dt * eta_bc
                            soc_b[z] = min(soc_b[z], bat_cap[z])
                            surplus -= p_ch

                if rl_max[z] > 0 and phs_p[z] > 0 and phs_cap[z] > 0:
                    if soc_p[z] < soc_p_target[z] and soc_p[z] < cfg.soc_high_frac * phs_cap[z]:
                        room = (phs_cap[z] - soc_p[z]) / max(dt * eta_pc, eps)
                        p_ch = min(phs_p[z], surplus, room)
                        if p_ch > cfg.storage_min_action_frac * phs_p[z]:
                            do_p_ch = 1
                            soc_p[z] += p_ch * dt * eta_pc
                            soc_p[z] = min(soc_p[z], phs_cap[z])
                            surplus -= p_ch

            # Mutual-exclusion guards
            if do_b_ch and do_b_dis:
                do_b_dis = 0
            if do_p_ch and do_p_dis:
                do_p_dis = 0

            # Self-discharge / retention
            soc_b[z] *= float(bat_ret[z])
            soc_p[z] *= float(phs_ret[z])

            # Record
            u[z, t, FEAT_BATTERY_CHARGE] = do_b_ch
            u[z, t, FEAT_BATTERY_DISCHARGE] = do_b_dis
            u[z, t, FEAT_PUMPED_CHARGE] = do_p_ch
            u[z, t, FEAT_PUMPED_DISCHARGE] = do_p_dis
            u[z, t, FEAT_DR] = do_dr
            u[z, t, FEAT_THERMAL] = float(therm_on_now)
            u[z, t, FEAT_THERMAL_SU] = float(
                therm_on_now == 1 and therm_on_prev[z] < 0.5
            )
            therm_on_prev[z] = float(therm_on_now)

    return torch.from_numpy(u).float()
