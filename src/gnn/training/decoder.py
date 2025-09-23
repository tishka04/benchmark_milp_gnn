from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable

import torch

from src.gnn.data.temporal import GraphBatch
from src.gnn.training.config import DecoderConfig


THERMAL_IDX = 0
NUCLEAR_IDX = 1
SOLAR_IDX = 2
WIND_IDX = 3
HYDRO_RELEASE_IDX = 4
HYDRO_ROR_IDX = 5
DEMAND_RESPONSE_IDX = 6
BATTERY_CHARGE_IDX = 7
BATTERY_DISCHARGE_IDX = 8
PUMPED_CHARGE_IDX = 9
PUMPED_DISCHARGE_IDX = 10
NET_IMPORT_IDX = 11
UNSERVED_IDX = 12

@dataclass
class DecoderState:
    raw: torch.Tensor
    decoded: torch.Tensor


class FeasibilityDecoder:
    def __init__(self, config: DecoderConfig) -> None:
        self.config = config
        self._iterations = max(1, int(config.balance_iterations))
        self._dual_scale = float(config.dual_adjustment_scale)

    def __call__(self, batch: GraphBatch, predictions: torch.Tensor) -> torch.Tensor:
        return self.decode(batch, predictions)

    def decode(self, batch: GraphBatch, predictions: torch.Tensor) -> torch.Tensor:
        decoded = predictions.clone()
        if self.config.blend_dual_fraction > 0.0 and hasattr(batch, "target"):
            decoded = torch.lerp(decoded, batch.target, self.config.blend_dual_fraction)
        if self.config.enforce_nonneg:
            net_import_original = decoded[:, NET_IMPORT_IDX].clone()
            decoded = decoded.clamp_min(0.0)
            decoded[:, NET_IMPORT_IDX] = net_import_original
        bounds = self._capacity_bounds(batch)
        if self.config.dual_keys:
            decoded = self._apply_dual_adjustments(batch, decoded, bounds)
        for _ in range(self._iterations):
            if self.config.respect_capacity:
                decoded = self._apply_capacity_bounds(batch, decoded, bounds)
            decoded = self._balance_supply(batch, decoded, bounds)
            if self.config.enforce_nonneg:
                decoded = decoded.clamp_min(0.0)
        if self.config.respect_capacity:
            decoded = self._apply_capacity_bounds(batch, decoded, bounds)
        decoded = self._final_trim(batch, decoded)
        if self.config.enforce_nonneg:
            net_import_original = decoded[:, NET_IMPORT_IDX].clone()
            decoded = decoded.clamp_min(0.0)
            decoded[:, NET_IMPORT_IDX] = net_import_original
        return decoded

    def _apply_dual_adjustments(self, batch: GraphBatch, decoded: torch.Tensor, bounds: Dict[int, torch.Tensor]) -> torch.Tensor:
        for key in self.config.dual_keys or []:
            dual = batch.duals.get(key)
            if dual is None:
                continue
            scale = float(self.config.residual_tolerance) * self._dual_scale
            if scale <= 0.0:
                continue
            shortage = torch.clamp(dual, min=0.0) * scale
            if shortage.any():
                decoded[:, UNSERVED_IDX] = torch.minimum(
                    decoded[:, UNSERVED_IDX] + shortage,
                    bounds.get(UNSERVED_IDX, torch.full_like(decoded[:, UNSERVED_IDX], float("inf"))),
                )
            oversupply = torch.clamp(-dual, min=0.0) * scale
            if oversupply.any():
                remaining = oversupply.clone()
                reduction = torch.minimum(decoded[:, THERMAL_IDX], remaining)
                decoded[:, THERMAL_IDX] = decoded[:, THERMAL_IDX] - reduction
                remaining = remaining - reduction
                if remaining.any():
                    red2 = torch.minimum(decoded[:, SOLAR_IDX], remaining)
                    decoded[:, SOLAR_IDX] = decoded[:, SOLAR_IDX] - red2
                    remaining = remaining - red2
                if remaining.any():
                    red3 = torch.minimum(decoded[:, WIND_IDX], remaining)
                    decoded[:, WIND_IDX] = decoded[:, WIND_IDX] - red3
                    remaining = remaining - red3
                if remaining.any():
                    red4 = torch.minimum(decoded[:, HYDRO_RELEASE_IDX], remaining)
                    decoded[:, HYDRO_RELEASE_IDX] = decoded[:, HYDRO_RELEASE_IDX] - red4
                    remaining = remaining - red4
                if remaining.any():
                    red5 = torch.minimum(decoded[:, HYDRO_ROR_IDX], remaining)
                    decoded[:, HYDRO_ROR_IDX] = decoded[:, HYDRO_ROR_IDX] - red5
                    remaining = remaining - red5
                if remaining.any():
                    red6 = torch.minimum(decoded[:, BATTERY_DISCHARGE_IDX], remaining)
                    decoded[:, BATTERY_DISCHARGE_IDX] = decoded[:, BATTERY_DISCHARGE_IDX] - red6
                    remaining = remaining - red6
                if remaining.any():
                    red7 = torch.minimum(decoded[:, PUMPED_DISCHARGE_IDX], remaining)
                    decoded[:, PUMPED_DISCHARGE_IDX] = decoded[:, PUMPED_DISCHARGE_IDX] - red7
                    remaining = remaining - red7
                if remaining.any():
                    net_import = decoded[:, NET_IMPORT_IDX]
                    positive_import = torch.clamp(net_import, min=0.0)
                    red8 = torch.minimum(positive_import, remaining)
                    decoded[:, NET_IMPORT_IDX] = net_import - red8
        return decoded

    def _apply_capacity_bounds(
        self,
        batch: GraphBatch,
        decoded: torch.Tensor,
        bounds: Dict[int, torch.Tensor] | None = None,
    ) -> torch.Tensor:
        if bounds is None:
            bounds = self._capacity_bounds(batch)
        for idx, upper in bounds.items():
            if idx == NET_IMPORT_IDX:
                decoded[:, idx] = torch.clamp(decoded[:, idx], min=-upper, max=upper)
            else:
                decoded[:, idx] = torch.minimum(decoded[:, idx], upper)
        return decoded

    def _capacity_bounds(self, batch: GraphBatch) -> Dict[int, torch.Tensor]:
        node_static = batch.node_static
        node_time = batch.node_time
        demand = node_time[:, 0]
        solar_avail = torch.minimum(node_static[:, 1], torch.relu(node_time[:, 1]))
        wind_avail = torch.minimum(node_static[:, 2], torch.relu(node_time[:, 2]))
        hydro_release_avail = torch.minimum(node_static[:, 6], torch.relu(node_time[:, 3]))
        hydro_ror_avail = torch.relu(node_time[:, 4])
        dr_limit = torch.minimum(node_static[:, 4], demand)
        bounds: Dict[int, torch.Tensor] = {
            THERMAL_IDX: torch.relu(node_static[:, 0]),
            NUCLEAR_IDX: torch.relu(node_static[:, 5]),
            SOLAR_IDX: solar_avail,
            WIND_IDX: wind_avail,
            HYDRO_RELEASE_IDX: hydro_release_avail,
            HYDRO_ROR_IDX: hydro_ror_avail,
            DEMAND_RESPONSE_IDX: torch.relu(dr_limit),
            UNSERVED_IDX: torch.relu(demand),
        }
        battery_power_cap = torch.relu(node_static[:, 3])
        pumped_power_cap = torch.relu(node_static[:, 8])
        bounds[BATTERY_CHARGE_IDX] = battery_power_cap
        bounds[BATTERY_DISCHARGE_IDX] = battery_power_cap
        bounds[PUMPED_CHARGE_IDX] = pumped_power_cap
        bounds[PUMPED_DISCHARGE_IDX] = pumped_power_cap
        if node_static.size(1) > 21:
            bounds[NET_IMPORT_IDX] = torch.relu(node_static[:, 21])
        else:
            bounds[NET_IMPORT_IDX] = torch.zeros_like(demand)
        return bounds

    def _balance_supply(
        self,
        batch: GraphBatch,
        decoded: torch.Tensor,
        bounds: Dict[int, torch.Tensor],
    ) -> torch.Tensor:
        demand = batch.node_time[:, 0]
        thermal = decoded[:, THERMAL_IDX]
        nuclear = decoded[:, NUCLEAR_IDX]
        solar = decoded[:, SOLAR_IDX]
        wind = decoded[:, WIND_IDX]
        hydro_release = decoded[:, HYDRO_RELEASE_IDX]
        hydro_ror = decoded[:, HYDRO_ROR_IDX]
        dr = decoded[:, DEMAND_RESPONSE_IDX]
        battery_charge = decoded[:, BATTERY_CHARGE_IDX]
        battery_discharge = decoded[:, BATTERY_DISCHARGE_IDX]
        pumped_charge = decoded[:, PUMPED_CHARGE_IDX]
        pumped_discharge = decoded[:, PUMPED_DISCHARGE_IDX]
        net_import = decoded[:, NET_IMPORT_IDX]
        unserved = decoded[:, UNSERVED_IDX]

        supply = (
            thermal
            + nuclear
            + solar
            + wind
            + hydro_release
            + hydro_ror
            + dr
            + battery_discharge
            + pumped_discharge
            + net_import
        )
        demand_side = demand + battery_charge + pumped_charge
        balance = demand_side - supply

        shortage = torch.relu(balance)
        unserved = torch.clamp(unserved + shortage, min=0.0)
        unserved = torch.minimum(unserved, demand)
        balance = balance - shortage

        overgen = torch.relu(-balance)
        if overgen.any():
            remaining = overgen.clone()
            reduction = torch.minimum(thermal, remaining)
            thermal = thermal - reduction
            remaining = remaining - reduction

            reduction = torch.minimum(solar, remaining)
            solar = solar - reduction
            remaining = remaining - reduction

            reduction = torch.minimum(wind, remaining)
            wind = wind - reduction
            remaining = remaining - reduction

            reduction = torch.minimum(hydro_release, remaining)
            hydro_release = hydro_release - reduction
            remaining = remaining - reduction

            reduction = torch.minimum(hydro_ror, remaining)
            hydro_ror = hydro_ror - reduction
            remaining = remaining - reduction

            reduction = torch.minimum(battery_discharge, remaining)
            battery_discharge = battery_discharge - reduction
            remaining = remaining - reduction

            reduction = torch.minimum(pumped_discharge, remaining)
            pumped_discharge = pumped_discharge - reduction
            remaining = remaining - reduction

            if remaining.any():
                positive_import = torch.clamp(net_import, min=0.0)
                reduction = torch.minimum(positive_import, remaining)
                net_import = net_import - reduction

        decoded[:, THERMAL_IDX] = thermal
        decoded[:, NUCLEAR_IDX] = nuclear
        decoded[:, SOLAR_IDX] = solar
        decoded[:, WIND_IDX] = wind
        decoded[:, HYDRO_RELEASE_IDX] = hydro_release
        decoded[:, HYDRO_ROR_IDX] = hydro_ror
        decoded[:, DEMAND_RESPONSE_IDX] = dr
        decoded[:, BATTERY_CHARGE_IDX] = torch.maximum(battery_charge, torch.zeros_like(battery_charge))
        decoded[:, BATTERY_DISCHARGE_IDX] = torch.maximum(battery_discharge, torch.zeros_like(battery_discharge))
        decoded[:, PUMPED_CHARGE_IDX] = torch.maximum(pumped_charge, torch.zeros_like(pumped_charge))
        decoded[:, PUMPED_DISCHARGE_IDX] = torch.maximum(pumped_discharge, torch.zeros_like(pumped_discharge))
        decoded[:, NET_IMPORT_IDX] = net_import
        decoded[:, UNSERVED_IDX] = unserved

        dr_limit = bounds.get(DEMAND_RESPONSE_IDX)
        if dr_limit is not None:
            decoded[:, DEMAND_RESPONSE_IDX] = torch.minimum(decoded[:, DEMAND_RESPONSE_IDX], dr_limit)
        decoded[:, UNSERVED_IDX] = torch.minimum(decoded[:, UNSERVED_IDX], demand)

        return decoded

    def _final_trim(self, batch: GraphBatch, decoded: torch.Tensor) -> torch.Tensor:
        demand = batch.node_time[:, 0]
        supply = (
            decoded[:, THERMAL_IDX]
            + decoded[:, NUCLEAR_IDX]
            + decoded[:, SOLAR_IDX]
            + decoded[:, WIND_IDX]
            + decoded[:, HYDRO_RELEASE_IDX]
            + decoded[:, HYDRO_ROR_IDX]
            + decoded[:, DEMAND_RESPONSE_IDX]
            + decoded[:, BATTERY_DISCHARGE_IDX]
            + decoded[:, PUMPED_DISCHARGE_IDX]
            + decoded[:, NET_IMPORT_IDX]
        )
        demand_side = demand + decoded[:, BATTERY_CHARGE_IDX] + decoded[:, PUMPED_CHARGE_IDX]
        residual = demand_side - (supply + decoded[:, UNSERVED_IDX])
        tolerance = float(self.config.residual_tolerance)
        needs_more = residual > tolerance
        if needs_more.any():
            addition = torch.zeros_like(residual)
            addition[needs_more] = residual[needs_more]
            unserved = torch.minimum(decoded[:, UNSERVED_IDX] + addition, demand)
            decoded[:, UNSERVED_IDX] = torch.clamp(unserved, min=0.0)
        negative_residual = residual < -tolerance
        if negative_residual.any():
            correction = torch.zeros_like(residual)
            correction[negative_residual] = residual[negative_residual]
            thermal = decoded[:, THERMAL_IDX] + correction
            decoded[:, THERMAL_IDX] = torch.clamp(thermal, min=0.0)
        if self.config.enforce_nonneg:
            net_import_original = decoded[:, NET_IMPORT_IDX].clone()
            decoded = decoded.clamp_min(0.0)
            decoded[:, NET_IMPORT_IDX] = net_import_original
        return decoded


def build_decoder(config: DecoderConfig) -> FeasibilityDecoder:
    if config.name not in {"feasibility", "feasibility_decoder"}:
        raise ValueError(f"Unsupported decoder '{config.name}'")
    return FeasibilityDecoder(config)
