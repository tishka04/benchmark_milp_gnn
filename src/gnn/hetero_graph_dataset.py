"""
Heterogeneous multi-level graph dataset builder for MILP scenarios.

Graph Structure:
- Nation node (1): top-level aggregation
- Region nodes (R): geographical/administrative regions  
- Zone nodes (Z): load centers (existing zones)
- Asset nodes (A): individual generators/storage per zone per technology
- Weather nodes (W): weather cells influencing renewable generation

Edge Types:
0. Nation → Region (containment)
1. Region → Zone (containment)
2. Zone → Asset (containment)
3. Weather → Zone (influence on renewables)
4. Weather → Asset (for RES/hydro assets)
5. Zone ↔ Zone (transmission lines - existing)
6. Asset → Asset (temporal state dependencies for storage)
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Union
from collections import defaultdict
from datetime import datetime, timedelta

import numpy as np

from src.milp.scenario_loader import load_scenario_data, ScenarioData


# Node type enumeration
NODE_TYPE_NATION = 0
NODE_TYPE_REGION = 1
NODE_TYPE_ZONE = 2
NODE_TYPE_ASSET = 3
NODE_TYPE_WEATHER = 4

# Edge type enumeration
EDGE_TYPE_NATION_TO_REGION = 0
EDGE_TYPE_REGION_TO_ZONE = 1
EDGE_TYPE_ZONE_TO_ASSET = 2
EDGE_TYPE_WEATHER_TO_ZONE = 3
EDGE_TYPE_WEATHER_TO_ASSET = 4
EDGE_TYPE_TRANSMISSION = 5
EDGE_TYPE_TEMPORAL_STORAGE = 6
EDGE_TYPE_TEMPORAL_SOC = 7
EDGE_TYPE_TEMPORAL_RAMP = 8
EDGE_TYPE_TEMPORAL_DR = 9


def _ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def _load_json(path: Path) -> Dict:
    return json.loads(path.read_text(encoding="utf-8"))


class HeteroGraphBuilder:
    """Builds multi-level heterogeneous graph from MILP scenario."""
    
    def __init__(self, data: ScenarioData, report: Dict):
        self.data = data
        self.report = report
        self.detail = report.get("detail")
        if self.detail is None:
            raise RuntimeError("Report missing 'detail'; rerun MILP with --save-json")
        
        # Node tracking
        self.node_features = []
        self.node_types = []
        self.node_metadata = []  # for debugging/interpretation
        
        # Edge tracking
        self.edges = []
        self.edge_types = []
        self.edge_features = []
        
        # Indexing
        self.nation_idx = None
        self.region_to_idx = {}
        self.zone_to_idx = {}
        self.asset_to_idx = {}
        self.weather_to_idx = {}
        
    def build(self) -> Dict[str, np.ndarray]:
        """Build complete heterogeneous graph."""
        self._build_nation_node()
        self._build_region_nodes()
        self._build_zone_nodes()
        self._build_asset_nodes()
        self._build_weather_nodes()
        
        self._build_hierarchy_edges()
        self._build_transmission_edges()
        self._build_weather_influence_edges()
        self._build_temporal_edges()
        
        return self._to_record()
    
    def _build_nation_node(self):
        """Create single nation-level node."""
        self.nation_idx = 0
        
        # Nation-level features: total capacity, demand, imports
        total_thermal = sum(self.data.thermal_capacity.values())
        total_solar = sum(self.data.solar_capacity.values())
        total_wind = sum(self.data.wind_capacity.values())
        total_nuclear = sum(self.data.nuclear_capacity.values())
        total_hydro = sum(self.data.hydro_res_capacity.values())
        total_battery_power = sum(self.data.battery_power.values())
        total_pumped_power = sum(self.data.pumped_power.values())
        total_demand = sum(self.data.peak_demand.values())
        
        features = [
            total_thermal,
            total_solar,
            total_wind,
            total_nuclear,
            total_hydro,
            total_battery_power,
            total_pumped_power,
            total_demand,
            self.data.import_capacity,
            float(len(self.data.zones)),  # num zones
        ]
        
        self.node_features.append(features)
        self.node_types.append(NODE_TYPE_NATION)
        self.node_metadata.append({"type": "nation", "name": "Nation"})
    
    def _build_region_nodes(self):
        """Create region nodes."""
        regions = sorted(set(self.data.region_of_zone.values()))
        
        for region in regions:
            region_zones = [z for z in self.data.zones if self.data.region_of_zone.get(z) == region]
            
            # Aggregate capacities per region
            thermal = sum(self.data.thermal_capacity.get(z, 0.0) for z in region_zones)
            solar = sum(self.data.solar_capacity.get(z, 0.0) for z in region_zones)
            wind = sum(self.data.wind_capacity.get(z, 0.0) for z in region_zones)
            nuclear = sum(self.data.nuclear_capacity.get(z, 0.0) for z in region_zones)
            hydro = sum(self.data.hydro_res_capacity.get(z, 0.0) for z in region_zones)
            battery = sum(self.data.battery_power.get(z, 0.0) for z in region_zones)
            pumped = sum(self.data.pumped_power.get(z, 0.0) for z in region_zones)
            demand = sum(self.data.peak_demand.get(z, 0.0) for z in region_zones)
            
            # Weather encoding
            weather_profile = self.data.region_weather_profile.get(region, "unknown")
            weather_spread = self.data.region_weather_spread.get(region, 0.0)
            
            features = [
                thermal, solar, wind, nuclear, hydro,
                battery, pumped, demand,
                float(len(region_zones)),
                weather_spread,
            ]
            
            idx = len(self.node_features)
            self.region_to_idx[region] = idx
            self.node_features.append(features)
            self.node_types.append(NODE_TYPE_REGION)
            self.node_metadata.append({"type": "region", "name": region, "weather": weather_profile})
    
    def _build_zone_nodes(self):
        """Create zone nodes (original zones)."""
        zones = self.detail["zones"]
        
        for zone in zones:
            # Zone capacities (same as original flat graph)
            thermal_cap = float(self.data.thermal_capacity.get(zone, 0.0))
            solar_cap = float(self.data.solar_capacity.get(zone, 0.0))
            wind_cap = float(self.data.wind_capacity.get(zone, 0.0))
            nuclear_cap = float(self.data.nuclear_capacity.get(zone, 0.0))
            hydro_cap = float(self.data.hydro_res_capacity.get(zone, 0.0))
            battery_power = float(self.data.battery_power.get(zone, 0.0))
            pumped_power = float(self.data.pumped_power.get(zone, 0.0))
            demand_peak = float(self.data.peak_demand.get(zone, 0.0))
            
            # Time-averaged demand
            periods = self.data.periods
            avg_demand = float(np.mean([self.data.demand.get((zone, t), 0.0) for t in periods]))
            
            # Region info
            region = self.data.region_of_zone.get(zone, "unknown")
            region_idx = self.region_to_idx.get(region, -1)
            
            features = [
                thermal_cap, solar_cap, wind_cap, nuclear_cap,
                hydro_cap, battery_power, pumped_power,
                demand_peak, avg_demand,
                float(region_idx),
            ]
            
            idx = len(self.node_features)
            self.zone_to_idx[zone] = idx
            self.node_features.append(features)
            self.node_types.append(NODE_TYPE_ZONE)
            self.node_metadata.append({"type": "zone", "name": zone, "region": region})
    
    def _build_asset_nodes(self):
        """Create individual asset nodes per technology type per zone."""
        zones = self.detail["zones"]
        
        # Asset types: thermal, solar, wind, nuclear, hydro_res, hydro_ror, battery, pumped, DR
        for zone in zones:
            zone_idx = self.zone_to_idx[zone]
            
            # Thermal assets
            thermal_cap = self.data.thermal_capacity.get(zone, 0.0)
            if thermal_cap > 0:
                self._add_asset_node(
                    zone, zone_idx, "thermal",
                    capacity=thermal_cap,
                    marginal_cost=self.data.thermal_cost.get(zone, 0.0),
                    min_power=self.data.thermal_min_power.get(zone, 0.0),
                    ramp_rate=self.data.thermal_ramp.get(zone, 0.0),
                )
            
            # Solar assets
            solar_cap = self.data.solar_capacity.get(zone, 0.0)
            if solar_cap > 0:
                self._add_asset_node(zone, zone_idx, "solar", capacity=solar_cap)
            
            # Wind assets
            wind_cap = self.data.wind_capacity.get(zone, 0.0)
            if wind_cap > 0:
                self._add_asset_node(zone, zone_idx, "wind", capacity=wind_cap)
            
            # Nuclear assets
            nuclear_cap = self.data.nuclear_capacity.get(zone, 0.0)
            if nuclear_cap > 0:
                self._add_asset_node(
                    zone, zone_idx, "nuclear",
                    capacity=nuclear_cap,
                    marginal_cost=self.data.nuclear_cost.get(zone, 0.0),
                    min_power=self.data.nuclear_min_power.get(zone, 0.0),
                )
            
            # Hydro reservoir
            hydro_cap = self.data.hydro_res_capacity.get(zone, 0.0)
            if hydro_cap > 0:
                self._add_asset_node(
                    zone, zone_idx, "hydro_res",
                    capacity=hydro_cap,
                    energy_capacity=self.data.hydro_res_energy.get(zone, 0.0),
                )
            
            # Run-of-river hydro
            periods = self.data.periods
            hydro_ror_avg = float(np.mean([self.data.hydro_ror_generation.get((zone, t), 0.0) for t in periods])) if periods else 0.0
            if hydro_ror_avg > 0:
                self._add_asset_node(zone, zone_idx, "hydro_ror", capacity=hydro_ror_avg)
            
            # Battery
            battery_power = self.data.battery_power.get(zone, 0.0)
            if battery_power > 0:
                self._add_asset_node(
                    zone, zone_idx, "battery",
                    capacity=battery_power,
                    energy_capacity=self.data.battery_energy.get(zone, 0.0),
                    efficiency=(self.data.battery_eta_charge + self.data.battery_eta_discharge) / 2,
                )
            
            # Pumped hydro
            pumped_power = self.data.pumped_power.get(zone, 0.0)
            if pumped_power > 0:
                self._add_asset_node(
                    zone, zone_idx, "pumped",
                    capacity=pumped_power,
                    energy_capacity=self.data.pumped_energy.get(zone, 0.0),
                    efficiency=(self.data.pumped_eta_charge + self.data.pumped_eta_discharge) / 2,
                )
            
            # Demand response
            dr_max = max((self.data.dr_limit.get((zone, t), 0.0) for t in periods), default=0.0)
            if dr_max > 0:
                self._add_asset_node(zone, zone_idx, "dr", capacity=dr_max)
    
    def _add_asset_node(self, zone: str, zone_idx: int, asset_type: str, **kwargs):
        """Helper to add asset node."""
        # Feature vector: [capacity, marginal_cost, min_power, ramp_rate, energy_cap, efficiency, zone_idx]
        features = [
            kwargs.get("capacity", 0.0),
            kwargs.get("marginal_cost", 0.0),
            kwargs.get("min_power", 0.0),
            kwargs.get("ramp_rate", 0.0),
            kwargs.get("energy_capacity", 0.0),
            kwargs.get("efficiency", 1.0),
            float(zone_idx),
        ]
        
        idx = len(self.node_features)
        asset_key = f"{zone}_{asset_type}"
        self.asset_to_idx[asset_key] = idx
        self.node_features.append(features)
        self.node_types.append(NODE_TYPE_ASSET)
        self.node_metadata.append({
            "type": "asset",
            "name": asset_key,
            "zone": zone,
            "asset_type": asset_type,
        })
    
    def _build_weather_nodes(self):
        """Create weather cell nodes per region."""
        for region, weather_profile in self.data.region_weather_profile.items():
            weather_spread = self.data.region_weather_spread.get(region, 0.0)
            
            # Weather features (could be extended with actual weather time series)
            features = [
                weather_spread,
                float(hash(weather_profile) % 100),  # simple encoding
                0.0, 0.0, 0.0,  # placeholder for weather variables
            ]
            
            idx = len(self.node_features)
            self.weather_to_idx[region] = idx
            self.node_features.append(features)
            self.node_types.append(NODE_TYPE_WEATHER)
            self.node_metadata.append({
                "type": "weather",
                "name": f"Weather_{region}",
                "region": region,
                "profile": weather_profile,
            })
    
    def _build_hierarchy_edges(self):
        """Build containment hierarchy edges."""
        # Nation → Regions
        for region, region_idx in self.region_to_idx.items():
            self.edges.append([self.nation_idx, region_idx])
            self.edge_types.append(EDGE_TYPE_NATION_TO_REGION)
            self.edge_features.append([1.0])  # simple weight
        
        # Regions → Zones
        for zone, zone_idx in self.zone_to_idx.items():
            region = self.data.region_of_zone.get(zone)
            if region and region in self.region_to_idx:
                region_idx = self.region_to_idx[region]
                self.edges.append([region_idx, zone_idx])
                self.edge_types.append(EDGE_TYPE_REGION_TO_ZONE)
                self.edge_features.append([1.0])
        
        # Zones → Assets
        for asset_key, asset_idx in self.asset_to_idx.items():
            zone = asset_key.rsplit("_", 1)[0]
            if zone in self.zone_to_idx:
                zone_idx = self.zone_to_idx[zone]
                self.edges.append([zone_idx, asset_idx])
                self.edge_types.append(EDGE_TYPE_ZONE_TO_ASSET)
                self.edge_features.append([1.0])
    
    def _build_transmission_edges(self):
        """Build transmission line edges (zone-to-zone)."""
        for lid, line in self.data.lines.items():
            if line.from_zone in self.zone_to_idx and line.to_zone in self.zone_to_idx:
                from_idx = self.zone_to_idx[line.from_zone]
                to_idx = self.zone_to_idx[line.to_zone]
                
                # Bidirectional edges with [capacity_mw, distance_km] features
                self.edges.append([from_idx, to_idx])
                self.edge_types.append(EDGE_TYPE_TRANSMISSION)
                self.edge_features.append([line.capacity_mw, line.distance_km])
                
                self.edges.append([to_idx, from_idx])
                self.edge_types.append(EDGE_TYPE_TRANSMISSION)
                self.edge_features.append([line.capacity_mw, line.distance_km])
    
    def _build_weather_influence_edges(self):
        """Build weather → zone and weather → asset edges."""
        for region, weather_idx in self.weather_to_idx.items():
            # Weather → Zones in region
            for zone, zone_idx in self.zone_to_idx.items():
                if self.data.region_of_zone.get(zone) == region:
                    self.edges.append([weather_idx, zone_idx])
                    self.edge_types.append(EDGE_TYPE_WEATHER_TO_ZONE)
                    self.edge_features.append([1.0])
                    
                    # Weather → RES/Hydro assets in this zone
                    for asset_type in ["solar", "wind", "hydro_res", "hydro_ror"]:
                        asset_key = f"{zone}_{asset_type}"
                        if asset_key in self.asset_to_idx:
                            asset_idx = self.asset_to_idx[asset_key]
                            self.edges.append([weather_idx, asset_idx])
                            self.edge_types.append(EDGE_TYPE_WEATHER_TO_ASSET)
                            self.edge_features.append([1.0])
    
    def _build_temporal_edges(self):
        """Build temporal edges for storage assets (state dependencies)."""
        # Connect storage assets to themselves (temporal self-loop)
        for asset_key, asset_idx in self.asset_to_idx.items():
            if any(storage_type in asset_key for storage_type in ["battery", "pumped", "hydro_res"]):
                self.edges.append([asset_idx, asset_idx])
                self.edge_types.append(EDGE_TYPE_TEMPORAL_STORAGE)
                self.edge_features.append([1.0])
    
    def _to_record(self) -> Dict[str, np.ndarray]:
        """Convert to NPZ-compatible record."""
        # Pad node features to same dimension
        max_node_dim = max(len(f) for f in self.node_features)
        padded_features = []
        for f in self.node_features:
            padded = f + [0.0] * (max_node_dim - len(f))
            padded_features.append(padded)
        
        # Pad edge features
        max_edge_dim = max(len(f) for f in self.edge_features) if self.edge_features else 1
        padded_edge_features = []
        for f in self.edge_features:
            padded = f + [0.0] * (max_edge_dim - len(f))
            padded_edge_features.append(padded)
        
        # Extract zone-only features for compatibility with flat graph loader
        zones = sorted(self.zone_to_idx.keys())
        zone_indices = [self.zone_to_idx[z] for z in zones]
        zone_node_features = [padded_features[idx] for idx in zone_indices]
        
        # Build zone-level static and temporal features compatible with dataset loader
        flat_compat = self._build_flat_compatibility(zones)
        
        record = {
            # Heterogeneous graph fields
            "node_features": np.array(padded_features, dtype=np.float32),
            "node_types": np.array(self.node_types, dtype=np.int64),
            "edge_index": np.array(self.edges, dtype=np.int64).T,  # [2, num_edges]
            "edge_types": np.array(self.edge_types, dtype=np.int64),
            "edge_features": np.array(padded_edge_features, dtype=np.float32),
            "zone_node_indices": np.array(zone_indices, dtype=np.int64),
            
            # Flat graph compatibility fields (for existing dataset loader)
            **flat_compat,
        }
        
        return record
    
    def _build_flat_compatibility(self, zones: List[str]) -> Dict[str, np.ndarray]:
        """Build flat graph compatibility fields for existing dataset loader."""
        detail = self.detail
        periods = self.data.periods
        anchor_zone = getattr(self.data, "import_anchor_zone", None)
        
        # node_static: [N_zones, static_dim]
        node_static = []
        for zone in zones:
            thermal_cap = float(self.data.thermal_capacity.get(zone, 0.0))
            solar_cap = float(self.data.solar_capacity.get(zone, 0.0))
            wind_cap = float(self.data.wind_capacity.get(zone, 0.0))
            battery_power = float(self.data.battery_power.get(zone, 0.0))
            dr_cap = float(max((self.data.dr_limit.get((zone, t), 0.0) for t in periods), default=0.0))
            nuclear_cap = float(self.data.nuclear_capacity.get(zone, 0.0))
            hydro_res_cap = float(self.data.hydro_res_capacity.get(zone, 0.0))
            hydro_ror_avg = float(np.mean([self.data.hydro_ror_generation.get((zone, t), 0.0) for t in periods])) if periods else 0.0
            pumped_power = float(self.data.pumped_power.get(zone, 0.0))
            battery_energy = float(self.data.battery_energy.get(zone, 0.0))
            battery_initial_frac = self.data.battery_initial.get(zone, 0.0) / battery_energy if battery_energy > 1e-6 else 0.0
            battery_final_min_frac = self.data.battery_final_min.get(zone, 0.0) / battery_energy if battery_energy > 1e-6 else 0.0
            battery_final_max_frac = self.data.battery_final_max.get(zone, 0.0) / battery_energy if battery_energy > 1e-6 else 0.0
            battery_retention = float(self.data.battery_retention.get(zone, 1.0))
            battery_cycle_cost = float(self.data.battery_cycle_cost.get(zone, 0.0))
            pumped_energy = float(self.data.pumped_energy.get(zone, 0.0))
            pumped_initial_frac = self.data.pumped_initial.get(zone, 0.0) / pumped_energy if pumped_energy > 1e-6 else 0.0
            pumped_final_min_frac = self.data.pumped_final_min.get(zone, 0.0) / pumped_energy if pumped_energy > 1e-6 else 0.0
            pumped_final_max_frac = self.data.pumped_final_max.get(zone, 0.0) / pumped_energy if pumped_energy > 1e-6 else 0.0
            pumped_retention = float(self.data.pumped_retention.get(zone, 1.0))
            pumped_cycle_cost = float(self.data.pumped_cycle_cost.get(zone, 0.0))
            import_cap = float(self.data.import_capacity) if zone == anchor_zone else 0.0
            
            node_static.append([
                thermal_cap, solar_cap, wind_cap, battery_power, dr_cap,
                nuclear_cap, hydro_res_cap, hydro_ror_avg, pumped_power,
                battery_energy, battery_initial_frac, battery_final_min_frac, battery_final_max_frac,
                battery_retention, battery_cycle_cost, pumped_energy, pumped_initial_frac,
                pumped_final_min_frac, pumped_final_max_frac, pumped_retention,
                pumped_cycle_cost, import_cap,
            ])
        
        # node_time: [T, N_zones, temporal_dim]
        node_time = []
        net_import_series = detail.get("net_import", {}).get("values")
        net_export_series = detail.get("net_export", {}).get("values")
        net_import_arr = np.asarray(net_import_series, dtype=np.float32) if net_import_series is not None else None
        net_export_arr = np.asarray(net_export_series, dtype=np.float32) if net_export_series is not None else None
        
        for zone in zones:
            demand = detail["demand"][zone]
            solar = detail["solar"][zone]
            wind = detail["wind"][zone]
            hydro_release = detail["hydro_release"][zone]
            hydro_ror = detail.get("hydro_ror", {}).get(zone, [0.0] * len(demand))
            battery_soc = detail["battery_soc"][zone]
            pumped_level = detail["pumped_level"][zone]
            battery_charge = detail["battery_charge"][zone]
            battery_discharge = detail["battery_discharge"][zone]
            pumped_charge = detail["pumped_charge"][zone]
            pumped_discharge = detail["pumped_discharge"][zone]
            demand_response = detail["demand_response"][zone]
            unserved = detail["unserved"][zone]
            solar_spill = detail.get("solar_spill", {}).get(zone, [0.0] * len(demand))
            wind_spill = detail.get("wind_spill", {}).get(zone, [0.0] * len(demand))
            hydro_spill = detail["hydro_spill"][zone]
            
            is_anchor = anchor_zone is not None and zone == anchor_zone
            net_import = net_import_arr if net_import_arr is not None and is_anchor else np.zeros_like(demand, dtype=np.float32)
            net_export = net_export_arr if net_export_arr is not None and is_anchor else np.zeros_like(demand, dtype=np.float32)
            
            node_time.append(np.stack([
                demand, solar, wind, hydro_release, hydro_ror, battery_soc, pumped_level,
                battery_charge, battery_discharge, pumped_charge, pumped_discharge,
                demand_response, unserved, solar_spill, wind_spill, hydro_spill,
                net_import, net_export,
            ], axis=1))
        
        # node_labels: [T, N_zones, label_dim]
        node_labels = []
        net_import_dispatch = detail.get("net_import") or {}
        for zone in zones:
            thermal = detail["thermal"][zone]
            nuclear = detail["nuclear"][zone]
            solar = detail["solar"][zone]
            wind = detail["wind"][zone]
            hydro_release = detail["hydro_release"][zone]
            hydro_ror = detail.get("hydro_ror", {}).get(zone, [0.0] * len(thermal))
            dr = detail["demand_response"][zone]
            battery_charge = detail["battery_charge"][zone]
            battery_discharge = detail["battery_discharge"][zone]
            pumped_charge = detail["pumped_charge"][zone]
            pumped_discharge = detail["pumped_discharge"][zone]
            net_import = net_import_dispatch.get(zone, [0.0] * len(thermal)) if zone == anchor_zone else [0.0] * len(thermal)
            unserved = detail["unserved"][zone]
            
            node_labels.append(np.stack([
                thermal, nuclear, solar, wind, hydro_release, hydro_ror, dr,
                battery_charge, battery_discharge, pumped_charge, pumped_discharge,
                net_import, unserved,
            ], axis=1))
        
        # Extract transmission edges only (edge_type == EDGE_TYPE_TRANSMISSION)
        # Need to remap edge indices from hetero graph (full node set) to flat graph (zone nodes only)
        zone_idx_to_flat = {self.zone_to_idx[z]: i for i, z in enumerate(zones)}
        
        transmission_edges = []
        for i in range(len(self.edge_types)):
            if self.edge_types[i] == EDGE_TYPE_TRANSMISSION:
                src, dst = self.edges[i]
                # Only include if both endpoints are zones
                if src in zone_idx_to_flat and dst in zone_idx_to_flat:
                    flat_src = zone_idx_to_flat[src]
                    flat_dst = zone_idx_to_flat[dst]
                    capacity = self.edge_features[i][0] if self.edge_features[i] else 0.0
                    transmission_edges.append((flat_src, flat_dst, capacity))
        
        edge_index_trans = np.array([[e[0], e[1]] for e in transmission_edges], dtype=np.int64) if transmission_edges else np.zeros((0, 2), dtype=np.int64)
        edge_capacity = np.array([e[2] for e in transmission_edges], dtype=np.float32) if transmission_edges else np.zeros(0, dtype=np.float32)
        # edge_flows would need to be extracted from time-series data - for now use zeros
        time_len = len(detail["time_steps"])
        edge_flows = np.zeros((len(transmission_edges), time_len), dtype=np.float32)
        
        # Region indices
        region_indices = []
        for zone in zones:
            region = self.data.region_of_zone.get(zone, "unknown")
            # Find region index
            reg_idx = list(self.region_to_idx.keys()).index(region) if region in self.region_to_idx else 0
            region_indices.append(reg_idx)
        
        # Edge types for transmission edges (0 = intra-region, 1 = inter-region)
        edge_type_flat = np.zeros(len(transmission_edges), dtype=np.int64)
        for i, (src, dst, _) in enumerate(transmission_edges):
            if src < len(region_indices) and dst < len(region_indices):
                edge_type_flat[i] = 0 if region_indices[src] == region_indices[dst] else 1
        
        return {
            "node_static": np.array(node_static, dtype=np.float32),
            "node_time": np.stack(node_time, axis=1).astype(np.float32),  # [T, N, D]
            "node_labels": np.stack(node_labels, axis=1).astype(np.float32),  # [T, N, D]
            "zone_region_index": np.array(region_indices, dtype=np.int64),
            "edge_index": edge_index_trans,  # [2, num_edges] transmission edges only
            "edge_capacity": edge_capacity,
            "edge_flows": edge_flows,
            "edge_type": edge_type_flat,
            "time_steps": np.array(detail["time_steps"], dtype=np.int64),
            "time_hours": np.array(detail["time_hours"], dtype=np.float32),
        }


# ============================================================================
# Temporal Graph Helper Functions
# ============================================================================

def _extract_time_index(scenario_data: ScenarioData, report: Dict) -> Tuple[int, List]:
    """Extract time steps from scenario data."""
    detail = report.get("detail", {})
    time_steps = detail.get("time_steps", [])
    time_hours = detail.get("time_hours", [])
    
    T = len(time_steps)
    if T == 0:
        raise ValueError("No time steps found in report detail")
    
    # Create time index (can be enhanced with actual timestamps)
    time_index = [f"t={t}" for t in range(T)]
    return T, time_index


def _make_time_encoding(time_index: List, T: int, method: str = "sinusoidal") -> np.ndarray:
    """
    Create time encoding features.
    
    Args:
        time_index: List of time identifiers
        T: Number of time steps
        method: 'sinusoidal' or 'cyclic-hod' (hour-of-day)
    
    Returns:
        [T, Ft] time encoding array
    """
    if method == "sinusoidal":
        # Simple positional encoding
        positions = np.arange(T, dtype=np.float32)
        dim = 4  # 2 frequencies
        encoding = np.zeros((T, dim), dtype=np.float32)
        
        for i in range(dim // 2):
            div_term = 10000 ** (2 * i / dim)
            encoding[:, 2*i] = np.sin(positions / div_term)
            encoding[:, 2*i + 1] = np.cos(positions / div_term)
        
        return encoding
    
    elif method == "cyclic-hod":
        # Hour-of-day cyclic encoding (assumes 30-min or 1-hour intervals)
        # For 48 periods = 24h with 30-min intervals
        # For 24 periods = 24h with 1-hour intervals
        hours = np.arange(T, dtype=np.float32) * (24.0 / T)
        encoding = np.zeros((T, 4), dtype=np.float32)
        encoding[:, 0] = np.sin(2 * np.pi * hours / 24)
        encoding[:, 1] = np.cos(2 * np.pi * hours / 24)
        # Day of week (placeholder - would need actual date info)
        encoding[:, 2] = 0.0  # sin(day_of_week)
        encoding[:, 3] = 1.0  # cos(day_of_week)
        
        return encoding
    
    else:
        raise ValueError(f"Unknown time encoding method: {method}")


def _windows(T: int, window: Optional[int], stride: int) -> List[int]:
    """Generate window start indices for sequence mode."""
    if window is None or window >= T:
        return [0]
    
    starts = []
    for t0 in range(0, T - window + 1, stride):
        starts.append(t0)
    return starts


def _extract_layers_nodes(scenario_data: ScenarioData) -> Tuple[List, List, Dict]:
    """Extract layer structure and nodes from scenario."""
    zones = sorted(scenario_data.zones)
    regions = sorted(set(scenario_data.region_of_zone.values()))
    
    layers = {
        "nation": ["Nation"],
        "regions": regions,
        "zones": zones,
    }
    
    # All nodes (flattened)
    nodes = ["Nation"] + regions + zones
    
    # Node types mapping
    node_types = {}
    node_types["Nation"] = NODE_TYPE_NATION
    for r in regions:
        node_types[r] = NODE_TYPE_REGION
    for z in zones:
        node_types[z] = NODE_TYPE_ZONE
    
    return layers, nodes, node_types


def _build_single_hetero_snapshot(
    scenario_data: ScenarioData,
    report: Dict,
    time_index_t: str,
    time_enc: np.ndarray,
) -> Dict:
    """Build a single heterogeneous graph snapshot at time t."""
    # Use the existing builder but for a single timestep
    builder = HeteroGraphBuilder(scenario_data, report)
    record = builder.build()
    
    # Augment node features with time encoding
    node_features = record["node_features"]
    N = node_features.shape[0]
    time_enc_broadcast = np.tile(time_enc, (N, 1))
    
    record["node_features"] = np.concatenate([node_features, time_enc_broadcast], axis=1)
    record["time_index"] = time_index_t
    
    return record


def _make_node_ids(nodes: List[str], T: int) -> List[str]:
    """Generate node IDs for supra-graph: (node_name, t)."""
    node_ids = []
    for t in range(T):
        for node in nodes:
            node_ids.append(f"{node}#t={t}")
    return node_ids


def _stack_time_features(
    node_features_all: np.ndarray,  # [N, F]
    time_encoding: np.ndarray,      # [T, Ft]
    T: int,
) -> np.ndarray:
    """
    Stack node features over time with time encoding.
    
    Args:
        node_features_all: [N, F] static + time-varying features (already combined)
        time_encoding: [T, Ft] time encoding
        T: Number of time steps
    
    Returns:
        [N*T, F+Ft] stacked features
    """
    N, F = node_features_all.shape
    Ft = time_encoding.shape[1]
    
    X = np.zeros((N * T, F + Ft), dtype=np.float32)
    
    for t in range(T):
        idx_start = t * N
        idx_end = (t + 1) * N
        X[idx_start:idx_end, :F] = node_features_all
        X[idx_start:idx_end, F:] = time_encoding[t]
    
    return X


def _tile_node_types(node_types: np.ndarray, T: int) -> np.ndarray:
    """Replicate node types for each time step."""
    return np.tile(node_types, T)


def _repeat_edges_over_time(
    edges: List[Tuple],  # [(src, dst, etype, attr), ...]
    edge_types: np.ndarray,
    edge_features: np.ndarray,
    N: int,
    T: int,
) -> Tuple[List, List, List]:
    """
    Repeat spatial edges for each time step.
    
    Returns:
        edges_expanded, edge_types_expanded, edge_features_expanded
    """
    E = len(edges)
    edges_expanded = []
    edge_types_expanded = []
    edge_features_expanded = []
    
    for t in range(T):
        offset = t * N
        for i in range(E):
            src, dst = edges[i]
            edges_expanded.append([src + offset, dst + offset])
            edge_types_expanded.append(edge_types[i])
            edge_features_expanded.append(edge_features[i])
    
    return edges_expanded, edge_types_expanded, edge_features_expanded


def _build_soc_edges(
    asset_to_idx: Dict[str, int],
    N: int,
    T: int,
) -> Tuple[List, List, List]:
    """Build SOC (state-of-charge) temporal edges for storage assets."""
    edges = []
    edge_types = []
    edge_attrs = []
    
    # Find storage assets (battery, pumped, hydro_res)
    storage_assets = [
        (key, idx) for key, idx in asset_to_idx.items()
        if any(s in key for s in ["battery", "pumped", "hydro_res"])
    ]
    
    for key, base_idx in storage_assets:
        # Connect t -> t+1 for storage continuity
        for t in range(T - 1):
            src = base_idx + t * N
            dst = base_idx + (t + 1) * N
            edges.append([src, dst])
            edge_types.append(EDGE_TYPE_TEMPORAL_SOC)
            edge_attrs.append([1.0])  # Could add retention rate here
    
    return edges, edge_types, edge_attrs


def _build_ramp_edges(
    asset_to_idx: Dict[str, int],
    scenario_data: ScenarioData,
    N: int,
    T: int,
) -> Tuple[List, List, List]:
    """Build ramping constraint edges for thermal/nuclear generators."""
    edges = []
    edge_types = []
    edge_attrs = []
    
    # Find thermal and nuclear assets
    for key, base_idx in asset_to_idx.items():
        if "thermal" in key or "nuclear" in key:
            zone = key.rsplit("_", 1)[0]
            ramp_rate = scenario_data.thermal_ramp.get(zone, 0.0) if "thermal" in key else 0.0
            
            # Connect t -> t+1 for ramp constraints
            for t in range(T - 1):
                src = base_idx + t * N
                dst = base_idx + (t + 1) * N
                edges.append([src, dst])
                edge_types.append(EDGE_TYPE_TEMPORAL_RAMP)
                edge_attrs.append([ramp_rate])
    
    return edges, edge_types, edge_attrs


def _build_dr_edges(
    asset_to_idx: Dict[str, int],
    scenario_data: ScenarioData,
    N: int,
    T: int,
) -> Tuple[List, List, List]:
    """Build demand response cooldown edges."""
    edges = []
    edge_types = []
    edge_attrs = []
    
    # Find DR assets
    dr_assets = [(key, idx) for key, idx in asset_to_idx.items() if "dr" in key]
    
    for key, base_idx in dr_assets:
        # DR cooldown: connect multiple future time steps
        cooldown = 2  # periods (could be parameterized)
        for t in range(T - cooldown):
            src = base_idx + t * N
            for dt in range(1, cooldown + 1):
                if t + dt < T:
                    dst = base_idx + (t + dt) * N
                    edges.append([src, dst])
                    edge_types.append(EDGE_TYPE_TEMPORAL_DR)
                    edge_attrs.append([float(dt)])
    
    return edges, edge_types, edge_attrs


def _pack_edges(
    edges: List,
    edge_types: List,
    edge_attrs: List,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Pack edges into numpy arrays."""
    if not edges:
        return (
            np.zeros((2, 0), dtype=np.int64),
            np.zeros(0, dtype=np.int64),
            np.zeros((0, 1), dtype=np.float32),
        )
    
    edge_index = np.array(edges, dtype=np.int64).T  # [2, E]
    edge_types_arr = np.array(edge_types, dtype=np.int64)
    
    # Pad edge attributes to same dimension
    if edge_attrs:
        max_dim = max(len(a) for a in edge_attrs)
        padded_attrs = []
        for attr in edge_attrs:
            padded = list(attr) + [0.0] * (max_dim - len(attr))
            padded_attrs.append(padded)
        edge_attr = np.array(padded_attrs, dtype=np.float32)
    else:
        edge_attr = np.zeros((len(edges), 1), dtype=np.float32)
    
    return edge_index, edge_types_arr, edge_attr


def build_hetero_temporal_record(
    scenario_data: ScenarioData,
    report: Dict,
    *,
    mode: str = "supra",
    time_window: Optional[int] = None,
    stride: int = 1,
    temporal_edges: Tuple[str, ...] = ("soc", "ramp", "dr"),
    time_encoding: str = "sinusoidal",
    target_horizon: int = 0,
) -> Union[Dict, List[Dict]]:
    """
    Build a temporal heterogeneous multi-layer grid graph.
    
    Args:
        scenario_data: Parsed scenario
        report: MILP report with per-timestep outputs
        mode: "sequence" (list of graphs) or "supra" (single time-expanded graph)
        time_window: Number of steps per graph (for sequence mode)
        stride: Sliding window stride
        temporal_edges: Types of temporal edges to add ("soc", "ramp", "dr")
        time_encoding: Method for time encoding ("sinusoidal", "cyclic-hod")
        target_horizon: Prediction horizon (0 = same-step labels)
    
    Returns:
        Single dict (supra) or list of dicts (sequence)
    """
    # Build base heterogeneous graph to get structure
    base_builder = HeteroGraphBuilder(scenario_data, report)
    base_record = base_builder.build()
    
    # Extract time information
    detail = report.get("detail", {})
    time_steps = detail.get("time_steps", [])
    T = len(time_steps)
    
    if T == 0:
        raise ValueError("No time steps found in report")
    
    time_index = [f"t={t}" for t in range(T)]
    
    # Build time encoding
    TE = _make_time_encoding(time_index, T, method=time_encoding)
    
    if mode == "sequence":
        # Sequence mode: list of graphs, one per time window
        graphs = []
        windows = _windows(T, time_window, stride)
        
        for t0 in windows:
            t_end = min(t0 + (time_window or 1), T)
            for t in range(t0, t_end):
                # Create a snapshot at time t
                g = _build_single_hetero_snapshot(
                    scenario_data,
                    report,
                    time_index[t],
                    TE[t],
                )
                g["time_step"] = t
                graphs.append(g)
        
        return graphs
    
    elif mode == "supra":
        # Supra-graph mode: single large time-expanded graph
        N = len(base_record["node_types"])
        
        # Stack node features with time encoding
        node_features = base_record["node_features"]  # [N, F]
        X = _stack_time_features(node_features, TE, T)  # [N*T, F+Ft]
        
        # Expand node types
        node_types_expanded = _tile_node_types(base_record["node_types"], T)
        
        # Generate node IDs
        node_metadata = base_builder.node_metadata
        node_names = [meta.get("name", f"node_{i}") for i, meta in enumerate(node_metadata)]
        node_ids = _make_node_ids(node_names, T)
        
        # Expand spatial edges over time
        # Note: base hetero graph returns edge_index as [E, 2], not [2, E]
        base_edges = base_record["edge_index"].tolist()  # [E, 2] - list of [src, dst] pairs
        base_edge_types = base_record["edge_types"]
        base_edge_features = base_record["edge_features"]
        
        E_spatial, ET_spatial, EF_spatial = _repeat_edges_over_time(
            base_edges,
            base_edge_types,
            base_edge_features,
            N,
            T,
        )
        
        # Build temporal edges
        E_temporal = []
        ET_temporal = []
        EF_temporal = []
        
        if "soc" in temporal_edges:
            e, et, ef = _build_soc_edges(base_builder.asset_to_idx, N, T)
            E_temporal.extend(e)
            ET_temporal.extend(et)
            EF_temporal.extend(ef)
        
        if "ramp" in temporal_edges:
            e, et, ef = _build_ramp_edges(base_builder.asset_to_idx, scenario_data, N, T)
            E_temporal.extend(e)
            ET_temporal.extend(et)
            EF_temporal.extend(ef)
        
        if "dr" in temporal_edges:
            e, et, ef = _build_dr_edges(base_builder.asset_to_idx, scenario_data, N, T)
            E_temporal.extend(e)
            ET_temporal.extend(et)
            EF_temporal.extend(ef)
        
        # Combine spatial and temporal edges
        all_edges = E_spatial + E_temporal
        all_edge_types = ET_spatial + ET_temporal
        all_edge_features = EF_spatial + EF_temporal
        
        edge_index, edge_types, edge_attr = _pack_edges(
            all_edges,
            all_edge_types,
            all_edge_features,
        )
        
        return {
            "graph_type": "hetero_multi_layer_temporal_supra",
            "node_features": X,
            "node_types": node_types_expanded,
            "node_ids": node_ids,
            "time_index": time_index,
            "time_steps": time_steps,
            "edge_index": edge_index,
            "edge_types": edge_types,
            "edge_features": edge_attr,
            "meta": {
                "N_base": N,
                "T": T,
                "temporal_edges": list(temporal_edges),
                "time_encoding": time_encoding,
                "target_horizon": target_horizon,
                "schema_version": "2.0-temporal",
            },
            # Keep flat compatibility fields from base record
            "node_static": base_record.get("node_static"),
            "node_time": base_record.get("node_time"),
            "node_labels": base_record.get("node_labels"),
            "zone_region_index": base_record.get("zone_region_index"),
        }
    
    else:
        raise ValueError(f"Unknown temporal mode: {mode}")


def build_hetero_graph_record(data: ScenarioData, report: Dict) -> Dict[str, np.ndarray]:
    """Main entry point to build heterogeneous graph."""
    builder = HeteroGraphBuilder(data, report)
    return builder.build()


def save_graph_record(record: Dict[str, np.ndarray], output: Path) -> None:
    """Save graph to NPZ."""
    _ensure_parent(output)
    np.savez_compressed(output, **record)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Convert MILP scenario to heterogeneous multi-level graph."
    )
    parser.add_argument("scenario", type=Path, help="Scenario JSON path")
    parser.add_argument("report", type=Path, help="Report JSON path (with detail)")
    parser.add_argument("output", type=Path, help="Output NPZ path")
    args = parser.parse_args()

    scenario_data = load_scenario_data(args.scenario)
    report = _load_json(args.report)
    record = build_hetero_graph_record(scenario_data, report)
    save_graph_record(record, args.output)
    
    print(f"Saved heterogeneous graph to {args.output}")
    print(f"  Nodes: {len(record['node_types'])} ({dict(zip(*np.unique(record['node_types'], return_counts=True)))})")
    print(f"  Edges: {record['edge_index'].shape[1]} ({dict(zip(*np.unique(record['edge_types'], return_counts=True)))})")


if __name__ == "__main__":
    main()
