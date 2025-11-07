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
from typing import Dict, List, Tuple
from collections import defaultdict

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
                
                # Bidirectional edges
                self.edges.append([from_idx, to_idx])
                self.edge_types.append(EDGE_TYPE_TRANSMISSION)
                self.edge_features.append([line.capacity_mw])
                
                self.edges.append([to_idx, from_idx])
                self.edge_types.append(EDGE_TYPE_TRANSMISSION)
                self.edge_features.append([line.capacity_mw])
    
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
