from __future__ import annotations

from pathlib import Path
from typing import Dict, List

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from .scenario_loader import load_scenario_data


def _sum_over_zones(detail: Dict[str, List[float]], zones: List[str]) -> List[float]:
    if not detail:
        return []
    horizon = len(next(iter(detail.values())))
    aggregated = [0.0] * horizon
    for z in zones:
        series = detail.get(z, [])
        for idx, value in enumerate(series):
            aggregated[idx] += value
    return aggregated


def _plot_scenario_summary(scenario_path: Path, out_path: Path, detail: Dict[str, object] | None) -> None:
    data = load_scenario_data(scenario_path)
    zones = data.zones
    idx = range(len(zones))

    thermal_cap = [data.thermal_capacity.get(z, 0.0) for z in zones]
    solar_cap = [data.solar_capacity.get(z, 0.0) for z in zones]
    wind_cap = [data.wind_capacity.get(z, 0.0) for z in zones]
    battery_power = [data.battery_power.get(z, 0.0) for z in zones]
    hydro_cap = [data.hydro_res_capacity.get(z, 0.0) for z in zones]

    time_hours = detail["time_hours"] if detail else []
    demand_curves = detail["demand"] if detail else {}

    fig, axes = plt.subplots(2, 1, figsize=(11, 8), sharex=False)

    axes[0].bar(idx, thermal_cap, width=0.6, label="Thermal")
    axes[0].bar(idx, solar_cap, width=0.6, bottom=thermal_cap, label="Solar", color="#FDB462")
    stacked = [thermal_cap[i] + solar_cap[i] for i in range(len(zones))]
    axes[0].bar(idx, wind_cap, width=0.6, bottom=stacked, label="Wind", color="#80B1D3")
    stacked = [stacked[i] + wind_cap[i] for i in range(len(zones))]
    axes[0].bar(idx, battery_power, width=0.6, bottom=stacked, label="Battery power")
    stacked = [stacked[i] + battery_power[i] for i in range(len(zones))]
    axes[0].bar(idx, hydro_cap, width=0.6, bottom=stacked, label="Hydro reservoir")
    axes[0].set_ylabel("Capacity (MW)")
    axes[0].set_xticks(list(idx), zones)
    axes[0].legend(loc="upper right")
    axes[0].set_title("Installed capacities by zone")

    if time_hours:
        for zone in zones:
            axes[1].plot(time_hours, demand_curves.get(zone, []), label=zone)
        axes[1].set_ylabel("Demand (MW)")
        axes[1].set_xlabel("Time (hours)")
        axes[1].legend(loc="upper right")
        axes[1].set_title("Demand profiles")
    else:
        axes[1].text(0.5, 0.5, "No demand detail available", ha="center", va="center")
        axes[1].axis("off")

    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def _plot_dispatch(detail: Dict[str, object], scenario_id: str, out_path: Path) -> None:
    if not detail:
        raise ValueError("Dispatch detail is required for plotting.")

    time_hours: List[float] = detail["time_hours"]
    zones: List[str] = detail["zones"]

    # === Generation dispatch data ===
    thermal = _sum_over_zones(detail["thermal"], zones)
    nuclear = _sum_over_zones(detail["nuclear"], zones)
    solar = _sum_over_zones(detail["solar"], zones)
    wind = _sum_over_zones(detail["wind"], zones)
    hydro_release = _sum_over_zones(detail["hydro_release"], zones)
    hydro_ror = _sum_over_zones(detail["hydro_ror"], zones)
    battery_discharge = _sum_over_zones(detail["battery_discharge"], zones)
    pumped_discharge = _sum_over_zones(detail["pumped_discharge"], zones)
    demand_total = _sum_over_zones(detail["demand"], zones)

    # === Slack & adjustments data ===
    demand_response = _sum_over_zones(detail["demand_response"], zones)
    unserved = _sum_over_zones(detail["unserved"], zones)
    overgen = _sum_over_zones(detail["overgen_spill"], zones)
    solar_spill = _sum_over_zones(detail.get("solar_spill", {}), zones)
    wind_spill = _sum_over_zones(detail.get("wind_spill", {}), zones)
    hydro_spill = _sum_over_zones(detail.get("hydro_spill", {}), zones)

    # === Storage levels data ===
    battery_soc = _sum_over_zones(detail["battery_soc"], zones)
    pumped_level = _sum_over_zones(detail["pumped_level"], zones)
    hydro_level = _sum_over_zones(detail.get("hydro_level", {}), zones)

    # === Binary variables data ===
    thermal_commitment = _sum_over_zones(detail.get("thermal_commitment", {}), zones)
    nuclear_commitment = _sum_over_zones(detail.get("nuclear_commitment", {}), zones)
    thermal_startup = _sum_over_zones(detail.get("thermal_startup", {}), zones)
    nuclear_startup = _sum_over_zones(detail.get("nuclear_startup", {}), zones)
    battery_charge_mode = _sum_over_zones(detail.get("battery_charge_mode", {}), zones)
    pumped_charge_mode = _sum_over_zones(detail.get("pumped_charge_mode", {}), zones)
    dr_active = _sum_over_zones(detail.get("dr_active", {}), zones)
    import_mode = detail.get("import_mode", {}).get("values", [])

    # Generation dispatch components (stacked area)
    components = [
        ("Thermal", thermal, "#d95f02"),
        ("Nuclear", nuclear, "#7570b3"),
        ("Solar", solar, "#FDB462"),
        ("Wind", wind, "#80B1D3"),
        ("Hydro Release", hydro_release, "#66a61e"),
        ("Hydro RoR", hydro_ror, "#a6761d"),
        ("Battery", battery_discharge, "#e7298a"),
        ("Pumped", pumped_discharge, "#e6ab02"),
    ]

    # Create 4 subplots with different heights
    fig, axes = plt.subplots(4, 1, figsize=(12, 14), 
                              gridspec_kw={'height_ratios': [3, 2, 2, 1.5]})

    # === Subplot 1: Generation Dispatch (stacked area) ===
    # Net imports data
    imports = detail.get("net_import", {}).get("values", [])
    exports = detail.get("net_export", {}).get("values", [])
    net_exchange = [imports[i] - exports[i] for i in range(len(imports))] if imports else []
    
    stacks = [comp[1] for comp in components]
    labels = [comp[0] for comp in components]
    colors = [comp[2] for comp in components]
    axes[0].stackplot(time_hours, stacks, labels=labels, colors=colors, alpha=0.85)
    axes[0].plot(time_hours, demand_total, color="black", linewidth=2, label="Demand")
    if net_exchange:
        axes[0].plot(time_hours, net_exchange, color="gray", linestyle="--", linewidth=1.5, label="Net Import")
    axes[0].set_ylabel("Power (MW)")
    axes[0].set_title(f"MILP Oracle - Generation Dispatch", fontsize=12, fontweight='bold')
    axes[0].legend(loc="upper right", ncol=3, fontsize=8)
    axes[0].set_xlim(time_hours[0], time_hours[-1])
    axes[0].grid(True, alpha=0.3)

    # === Subplot 2: Storage Levels ===
    axes[1].plot(time_hours, battery_soc, label="Battery SOC", color="#e7298a", linewidth=2)
    axes[1].plot(time_hours, pumped_level, label="Pumped Storage", color="#e6ab02", linewidth=2)
    if hydro_level and any(v > 0 for v in hydro_level):
        axes[1].plot(time_hours, hydro_level, label="Hydro Reservoir", color="#66a61e", linewidth=2)
    axes[1].set_ylabel("Energy (MWh)")
    axes[1].set_title("MILP Oracle - Storage Levels", fontsize=12, fontweight='bold')
    axes[1].legend(loc="upper right", fontsize=8)
    axes[1].set_xlim(time_hours[0], time_hours[-1])
    axes[1].grid(True, alpha=0.3)

    # === Subplot 3: Slack & Adjustments (stacked area) ===
    slack_components = []
    slack_labels = []
    slack_colors = []
    
    if demand_response and any(v > 0 for v in demand_response):
        slack_components.append(demand_response)
        slack_labels.append("Demand Response")
        slack_colors.append("#66c2a5")
    if unserved and any(v > 0 for v in unserved):
        slack_components.append(unserved)
        slack_labels.append("Unserved")
        slack_colors.append("#fc8d62")
    if solar_spill and any(v > 0 for v in solar_spill):
        slack_components.append(solar_spill)
        slack_labels.append("Solar Spill")
        slack_colors.append("#FDB462")
    if wind_spill and any(v > 0 for v in wind_spill):
        slack_components.append(wind_spill)
        slack_labels.append("Wind Spill")
        slack_colors.append("#80B1D3")
    if hydro_spill and any(v > 0 for v in hydro_spill):
        slack_components.append(hydro_spill)
        slack_labels.append("Hydro Spill")
        slack_colors.append("#8da0cb")
    if overgen and any(v > 0 for v in overgen):
        slack_components.append(overgen)
        slack_labels.append("Overgen Spill")
        slack_colors.append("#e78ac3")

    if slack_components:
        axes[2].stackplot(time_hours, slack_components, labels=slack_labels, 
                          colors=slack_colors, alpha=0.85)
    axes[2].set_ylabel("Energy (MW)")
    axes[2].set_xlabel("Time (hours)")
    axes[2].set_title("MILP Oracle - Slack & Adjustments", fontsize=12, fontweight='bold')
    axes[2].legend(loc="upper right", ncol=2, fontsize=8)
    axes[2].set_xlim(time_hours[0], time_hours[-1])
    axes[2].grid(True, alpha=0.3)

    # === Subplot 4: Binary Commitments (bar chart) ===
    bar_width = 0.8 * (time_hours[1] - time_hours[0]) if len(time_hours) > 1 else 0.5
    
    # Stack binary variables for visualization
    binary_components = []
    binary_labels = []
    binary_colors = []
    
    if thermal_commitment and any(v > 0 for v in thermal_commitment):
        binary_components.append(thermal_commitment)
        binary_labels.append("Thermal ON")
        binary_colors.append("#d95f02")
    if nuclear_commitment and any(v > 0 for v in nuclear_commitment):
        binary_components.append(nuclear_commitment)
        binary_labels.append("Nuclear ON")
        binary_colors.append("#7570b3")
    if thermal_startup and any(v > 0 for v in thermal_startup):
        binary_components.append(thermal_startup)
        binary_labels.append("Thermal Startup")
        binary_colors.append("#e41a1c")
    if nuclear_startup and any(v > 0 for v in nuclear_startup):
        binary_components.append(nuclear_startup)
        binary_labels.append("Nuclear Startup")
        binary_colors.append("#984ea3")
    if battery_charge_mode and any(v > 0 for v in battery_charge_mode):
        binary_components.append(battery_charge_mode)
        binary_labels.append("Battery Charging")
        binary_colors.append("#e7298a")
    if pumped_charge_mode and any(v > 0 for v in pumped_charge_mode):
        binary_components.append(pumped_charge_mode)
        binary_labels.append("Pumped Charging")
        binary_colors.append("#e6ab02")
    if dr_active and any(v > 0 for v in dr_active):
        binary_components.append(dr_active)
        binary_labels.append("DR Active")
        binary_colors.append("#66c2a5")
    if import_mode and any(v > 0 for v in import_mode):
        binary_components.append(import_mode)
        binary_labels.append("Import Mode")
        binary_colors.append("#377eb8")

    # Plot stacked bars for binary variables
    if binary_components:
        bottom = [0.0] * len(time_hours)
        for comp, label, color in zip(binary_components, binary_labels, binary_colors):
            axes[3].bar(time_hours, comp, width=bar_width, bottom=bottom, 
                       label=label, color=color, alpha=0.85)
            bottom = [bottom[i] + comp[i] for i in range(len(time_hours))]
    
    axes[3].set_ylabel("Unit Commitment")
    axes[3].set_xlabel("Time (hours)")
    axes[3].set_title("MILP Oracle - Binary Commitments", fontsize=12, fontweight='bold')
    axes[3].legend(loc="upper right", ncol=4, fontsize=7)
    axes[3].set_xlim(time_hours[0], time_hours[-1])
    axes[3].grid(True, alpha=0.3, axis='y')

    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_scenario_and_dispatch(scenario_path: Path, report: Dict[str, object], out_dir: Path) -> None:
    detail = report.get("detail")
    scenario_id = report.get("scenario_id", "scenario")
    base_name = Path(scenario_path).stem or scenario_id

    summary_path = out_dir / f"{base_name}_scenario.png"
    dispatch_path = out_dir / f"{base_name}_dispatch.png"

    _plot_scenario_summary(Path(scenario_path), summary_path, detail)
    _plot_dispatch(detail, scenario_id, dispatch_path)
