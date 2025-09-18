from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable

import torch

from src.gnn.data.temporal import GraphBatch
from src.gnn.training.config import DecoderConfig


@dataclass
class DecoderState:
    raw: torch.Tensor
    decoded: torch.Tensor


class FeasibilityDecoder:
    def __init__(self, config: DecoderConfig) -> None:
        self.config = config

    def __call__(self, batch: GraphBatch, predictions: torch.Tensor) -> torch.Tensor:
        return self.decode(batch, predictions)

    def decode(self, batch: GraphBatch, predictions: torch.Tensor) -> torch.Tensor:
        decoded = predictions.clone()
        if self.config.blend_dual_fraction > 0.0 and hasattr(batch, "target"):
            decoded = torch.lerp(decoded, batch.target, self.config.blend_dual_fraction)
        if self.config.enforce_nonneg:
            decoded = decoded.clamp_min(0.0)
        if self.config.dual_keys:
            decoded = self._apply_dual_adjustments(batch, decoded)
        if self.config.respect_capacity:
            decoded = self._apply_capacity_bounds(batch, decoded)
        decoded = self._enforce_renewable_link(decoded)
        decoded = self._balance_supply(batch, decoded)
        if self.config.enforce_nonneg:
            decoded = decoded.clamp_min(0.0)
        decoded = self._final_trim(batch, decoded)
        return decoded

    def _apply_dual_adjustments(self, batch: GraphBatch, decoded: torch.Tensor) -> torch.Tensor:
        for key in self.config.dual_keys or []:
            dual = batch.duals.get(key)
            if dual is None:
                continue
            decoded[:, 7] = decoded[:, 7] + torch.relu(dual) * float(self.config.residual_tolerance)
        return decoded

    def _apply_capacity_bounds(self, batch: GraphBatch, decoded: torch.Tensor) -> torch.Tensor:
        bounds = self._capacity_bounds(batch)
        for idx, upper in bounds.items():
            decoded[:, idx] = torch.minimum(decoded[:, idx], upper)
        return decoded

    def _capacity_bounds(self, batch: GraphBatch) -> Dict[int, torch.Tensor]:
        node_static = batch.node_static
        node_time = batch.node_time
        demand = node_time[:, 0]
        solar_avail = torch.minimum(node_static[:, 1], torch.relu(node_time[:, 1]))
        wind_avail = torch.minimum(node_static[:, 2], torch.relu(node_time[:, 2]))
        renewable_avail = torch.minimum(node_static[:, 1] + node_static[:, 2], torch.relu(node_time[:, 3]))
        hydro_avail = torch.minimum(node_static[:, 6], torch.relu(node_time[:, 4]))
        dr_limit = torch.minimum(node_static[:, 4], demand)
        bounds: Dict[int, torch.Tensor] = {
            0: torch.relu(node_static[:, 0]),
            1: torch.relu(node_static[:, 5]),
            2: solar_avail,
            3: wind_avail,
            4: renewable_avail,
            5: hydro_avail,
            6: torch.relu(dr_limit),
            7: torch.relu(demand),
        }
        return bounds

    def _enforce_renewable_link(self, decoded: torch.Tensor) -> torch.Tensor:
        solar = decoded[:, 2]
        wind = decoded[:, 3]
        decoded[:, 4] = solar + wind
        return decoded

    def _balance_supply(self, batch: GraphBatch, decoded: torch.Tensor) -> torch.Tensor:
        demand = batch.node_time[:, 0]
        thermal = decoded[:, 0]
        nuclear = decoded[:, 1]
        solar = decoded[:, 2]
        wind = decoded[:, 3]
        renewable = decoded[:, 4]
        hydro = decoded[:, 5]
        dr = decoded[:, 6]
        unserved = decoded[:, 7]

        supply = thermal + nuclear + renewable + hydro + dr
        balance = demand - supply

        # Address shortages first via unserved energy
        shortage = torch.relu(balance)
        unserved = torch.clamp(unserved + shortage, min=0.0)
        unserved = torch.minimum(unserved, demand)
        balance = balance - shortage

        # Over-generation: roll back thermal -> renewable -> hydro
        overgen = torch.relu(-balance)
        if overgen.any():
            reduction = torch.minimum(thermal, overgen)
            thermal = thermal - reduction
            overgen = overgen - reduction

            renewable_prev = renewable.clone()
            reduction = torch.minimum(renewable, overgen)
            renewable = renewable - reduction
            overgen = overgen - reduction
            scale = torch.ones_like(renewable)
            mask = renewable_prev > 0
            scale[mask] = renewable[mask] / renewable_prev[mask]
            solar = solar * scale
            wind = wind * scale

            reduction = torch.minimum(hydro, overgen)
            hydro = hydro - reduction
            overgen = overgen - reduction

        decoded[:, 0] = thermal
        decoded[:, 1] = nuclear
        decoded[:, 2] = solar
        decoded[:, 3] = wind
        decoded[:, 4] = renewable
        decoded[:, 5] = hydro
        decoded[:, 6] = dr
        decoded[:, 7] = unserved

        return decoded

    def _final_trim(self, batch: GraphBatch, decoded: torch.Tensor) -> torch.Tensor:
        demand = batch.node_time[:, 0]
        supply = decoded[:, 0] + decoded[:, 1] + decoded[:, 4] + decoded[:, 5] + decoded[:, 6]
        residual = demand - supply
        tolerance = float(self.config.residual_tolerance)
        needs_more = residual > tolerance
        if needs_more.any():
            addition = torch.zeros_like(residual)
            addition[needs_more] = residual[needs_more]
            unserved = torch.minimum(decoded[:, 7] + addition, demand)
            decoded[:, 7] = torch.clamp(unserved, min=0.0)
        negative_residual = residual < -tolerance
        if negative_residual.any():
            correction = torch.zeros_like(residual)
            correction[negative_residual] = residual[negative_residual]
            thermal = decoded[:, 0] + correction
            decoded[:, 0] = torch.clamp(thermal, min=0.0)
        return decoded.clamp_min(0.0)


def build_decoder(config: DecoderConfig) -> FeasibilityDecoder:
    if config.name not in {"feasibility", "feasibility_decoder"}:
        raise ValueError(f"Unsupported decoder '{config.name}'")
    return FeasibilityDecoder(config)
