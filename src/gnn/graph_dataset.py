from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

from src.milp.scenario_loader import load_scenario_data, ScenarioData


def _ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def _load_json(path: Path) -> Dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _build_zone_index(zones: List[str]) -> Dict[str, int]:
    return {zone: idx for idx, zone in enumerate(zones)}


def _zone_region_indices(data: ScenarioData, zones: List[str]) -> Tuple[np.ndarray, Dict[str, int]]:
    region_names = [data.region_of_zone.get(zone, "unknown") for zone in zones]
    unique_regions = sorted(set(region_names))
    region_to_index = {region: idx for idx, region in enumerate(unique_regions)}
    zone_to_region = {zone: region_to_index[region_names[i]] for i, zone in enumerate(zones)}
    indices = np.asarray([zone_to_region[zone] for zone in zones], dtype=np.int64)
    return indices, zone_to_region


def _node_static_features(data: ScenarioData, zones: List[str]) -> np.ndarray:
    features: List[List[float]] = []
    periods = data.periods
    for zone in zones:
        thermal_cap = float(data.thermal_capacity.get(zone, 0.0))
        solar_cap = float(data.solar_capacity.get(zone, 0.0))
        wind_cap = float(data.wind_capacity.get(zone, 0.0))
        battery_power = float(data.battery_power.get(zone, 0.0))
        dr_cap = float(max(data.dr_limit.get((zone, t), 0.0) for t in periods)) if periods else 0.0
        nuclear_cap = float(data.nuclear_capacity.get(zone, 0.0))
        hydro_res_cap = float(data.hydro_res_capacity.get(zone, 0.0))
        hydro_ror_avg = float(np.mean([data.hydro_ror_generation.get((zone, t), 0.0) for t in periods])) if periods else 0.0
        pumped_power = float(data.pumped_power.get(zone, 0.0))
        features.append([
            thermal_cap,
            solar_cap,
            wind_cap,
            battery_power,
            dr_cap,
            nuclear_cap,
            hydro_res_cap,
            hydro_ror_avg,
            pumped_power,
        ])
    return np.asarray(features, dtype=np.float32)


def _node_time_features(detail: Dict, zones: List[str]) -> np.ndarray:
    features = []
    for zone in zones:
        demand = detail["demand"][zone]
        solar = detail["solar"][zone]
        wind = detail["wind"][zone]
        renewable = detail["renewable"][zone]
        hydro_release = detail["hydro_release"][zone]
        battery_soc = detail["battery_soc"][zone]
        features.append(np.stack([demand, solar, wind, renewable, hydro_release, battery_soc], axis=1))
    return np.stack(features, axis=1).astype(np.float32)


def _node_labels(detail: Dict, zones: List[str]) -> np.ndarray:
    labels = []
    for zone in zones:
        thermal = detail["thermal"][zone]
        nuclear = detail["nuclear"][zone]
        solar = detail["solar"][zone]
        wind = detail["wind"][zone]
        renewable = detail["renewable"][zone]
        hydro_release = detail["hydro_release"][zone]
        dr = detail["demand_response"][zone]
        unserved = detail["unserved"][zone]
        labels.append(np.stack([thermal, nuclear, solar, wind, renewable, hydro_release, dr, unserved], axis=1))
    return np.stack(labels, axis=1).astype(np.float32)


def _edge_structure(
    detail: Dict,
    data: ScenarioData,
    zones: List[str],
    zone_to_region: Dict[str, int],
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    zone_index = _build_zone_index(zones)
    adjacency: List[List[int]] = []
    capacities: List[float] = []
    flows: List[np.ndarray] = []
    edge_types: List[int] = []
    time_len = len(detail["time_steps"])
    for lid, line in data.lines.items():
        if line.from_zone not in zone_index or line.to_zone not in zone_index:
            continue
        adjacency.append([zone_index[line.from_zone], zone_index[line.to_zone]])
        capacities.append(float(line.capacity_mw))
        flows.append(np.asarray(detail["flows"].get(lid, [0.0] * time_len), dtype=np.float32))
        from_region = zone_to_region.get(line.from_zone)
        to_region = zone_to_region.get(line.to_zone)
        edge_types.append(0 if from_region == to_region else 1)
    if not adjacency:
        return (
            np.zeros((0, 2), dtype=np.int64),
            np.zeros((0,), dtype=np.float32),
            np.zeros((0, time_len), dtype=np.float32),
            np.zeros((0,), dtype=np.int64),
        )
    return (
        np.asarray(adjacency, dtype=np.int64),
        np.asarray(capacities, dtype=np.float32),
        np.stack(flows, axis=0),
        np.asarray(edge_types, dtype=np.int64),
    )


def _duals_to_arrays(report: Dict, zones: List[str]) -> Dict[str, np.ndarray]:
    dual_arrays: Dict[str, np.ndarray] = {}
    zone_index = _build_zone_index(zones)
    time_steps = len(report["detail"]["time_steps"])
    for name, mapping in report.get("lp_duals", {}).items():
        arr = np.zeros((len(zones), time_steps), dtype=np.float32)
        for key, value in mapping.items():
            if isinstance(key, str):
                zone_str, step_str = key.split("|")
                zone_key = zone_str
                step = int(step_str)
            else:
                zone_key, step = key
                zone_key = str(zone_key)
            if zone_key in zone_index:
                arr[zone_index[zone_key], int(step)] = float(value)
        dual_arrays[name] = arr
    return dual_arrays


def build_graph_record(data: ScenarioData, report: Dict) -> Dict[str, np.ndarray]:
    detail = report.get("detail")
    if detail is None:
        raise RuntimeError("Report missing 'detail'; rerun MILP with --save-json")
    zones = detail["zones"]
    record: Dict[str, np.ndarray] = {}
    record["node_static"] = _node_static_features(data, zones)
    record["node_time"] = _node_time_features(detail, zones)
    record["node_labels"] = _node_labels(detail, zones)
    zone_region_index, zone_to_region = _zone_region_indices(data, zones)
    record["zone_region_index"] = zone_region_index
    edge_index, edge_capacity, edge_flows, edge_type = _edge_structure(detail, data, zones, zone_to_region)
    record["edge_index"] = edge_index
    record["edge_type"] = edge_type
    record["edge_capacity"] = edge_capacity
    record["edge_flows"] = edge_flows
    record["time_steps"] = np.asarray(detail["time_steps"], dtype=np.int64)
    record["time_hours"] = np.asarray(detail["time_hours"], dtype=np.float32)
    for name, arr in _duals_to_arrays(report, zones).items():
        record[f"duals_{name}"] = arr
    return record


def save_graph_record(record: Dict[str, np.ndarray], output: Path) -> None:
    _ensure_parent(output)
    np.savez_compressed(output, **record)


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert a scenario/report pair into a graph dataset NPZ.")
    parser.add_argument("scenario", type=Path, help="Scenario JSON path")
    parser.add_argument("report", type=Path, help="Report JSON path (with detail)")
    parser.add_argument("output", type=Path, help="Output NPZ path")
    args = parser.parse_args()

    scenario_data = load_scenario_data(args.scenario)
    report = _load_json(args.report)
    record = build_graph_record(scenario_data, report)
    save_graph_record(record, args.output)
    print(f"Saved graph dataset to {args.output}")


if __name__ == "__main__":
    main()
