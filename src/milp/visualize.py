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

    thermal = _sum_over_zones(detail["thermal"], zones)
    nuclear = _sum_over_zones(detail["nuclear"], zones)
    solar = _sum_over_zones(detail["solar"], zones)
    wind = _sum_over_zones(detail["wind"], zones)
    hydro_release = _sum_over_zones(detail["hydro_release"], zones)
    hydro_ror = _sum_over_zones(detail["hydro_ror"], zones)
    battery_discharge = _sum_over_zones(detail["battery_discharge"], zones)
    pumped_discharge = _sum_over_zones(detail["pumped_discharge"], zones)
    demand_total = _sum_over_zones(detail["demand"], zones)

    demand_response = _sum_over_zones(detail["demand_response"], zones)
    unserved = _sum_over_zones(detail["unserved"], zones)
    overgen = _sum_over_zones(detail["overgen_spill"], zones)

    battery_soc = _sum_over_zones(detail["battery_soc"], zones)
    pumped_level = _sum_over_zones(detail["pumped_level"], zones)

    imports = detail["net_import"]["values"]
    exports = detail["net_export"]["values"]
    net_exchange = [imports[i] - exports[i] for i in range(len(imports))]

    components = [
        ("Thermal", thermal, "#d95f02"),
        ("Nuclear", nuclear, "#7570b3"),
        ("Solar", solar, "#FDB462"),
        ("Wind", wind, "#80B1D3"),
        ("Hydro release", hydro_release, "#66a61e"),
        ("Hydro RoR", hydro_ror, "#a6761d"),
        ("Battery discharge", battery_discharge, "#e7298a"),
        ("Pumped discharge", pumped_discharge, "#e6ab02"),
    ]

    fig, axes = plt.subplots(3, 1, figsize=(12, 12), sharex=True)

    stacks = [comp[1] for comp in components]
    labels = [comp[0] for comp in components]
    colors = [comp[2] for comp in components]
    axes[0].stackplot(time_hours, stacks, labels=labels, colors=colors)
    axes[0].plot(time_hours, demand_total, color="black", linewidth=1.8, label="Demand")
    axes[0].plot(time_hours, net_exchange, color="gray", linestyle="--", linewidth=1.2, label="Net import")
    axes[0].set_ylabel("Power (MW)")
    axes[0].set_title(f"Dispatch summary for {scenario_id}")
    axes[0].legend(loc="upper right", ncol=2)

    axes[1].plot(time_hours, battery_soc, label="Battery SOC", color="#e7298a")
    axes[1].plot(time_hours, pumped_level, label="Pumped storage", color="#e6ab02")
    axes[1].set_ylabel("Energy (MWh)")
    axes[1].legend(loc="upper right")
    axes[1].set_title("Storage levels")

    axes[2].stackplot(
        time_hours,
        [demand_response, unserved, overgen],
        labels=["Demand response", "Unserved", "Over-generation"],
        colors=["#66c2a5", "#fc8d62", "#8da0cb"],
    )
    axes[2].set_ylabel("Energy (MW)")
    axes[2].set_xlabel("Time (hours)")
    axes[2].legend(loc="upper right")
    axes[2].set_title("Slack and adjustments")

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
