from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional

import torch

from src.gnn.data.temporal import GraphBatch


@dataclass
class MetricResult:
    value: float
    details: Dict[str, float]


_COMPONENT_ORDER = [
    "thermal",
    "nuclear",
    "solar",
    "wind",
    "hydro_release",
    "hydro_ror",
    "demand_response",
    "battery_charge",
    "battery_discharge",
    "pumped_charge",
    "pumped_discharge",
    "net_import",
    "unserved",
]


def _component_indices(components: Optional[Iterable[str]]) -> List[int]:
    if components is None:
        return list(range(len(_COMPONENT_ORDER)))
    mapping = {name: idx for idx, name in enumerate(_COMPONENT_ORDER)}
    indices = []
    for name in components:
        idx = mapping.get(name)
        if idx is None:
            raise ValueError(f"Unknown component '{name}'. Known: {_COMPONENT_ORDER}")
        indices.append(idx)
    return indices


def compute_dispatch_error(
    pred: torch.Tensor,
    target: torch.Tensor,
    *,
    demand: torch.Tensor,
    components: Optional[Iterable[str]] = None,
    reduction: str = "mean",
) -> MetricResult:
    idx = _component_indices(components)
    diff = torch.abs(pred[:, idx] - target[:, idx])
    if reduction == "mean":
        mae = diff.mean().item()
    elif reduction == "sum":
        mae = diff.sum().item()
    else:
        raise ValueError(f"Unsupported reduction '{reduction}'")
    denom = demand.abs().mean().item() + 1e-6
    normalized = diff.mean().item() / (denom + 1e-6)
    rmse = torch.sqrt((diff ** 2).mean()).item()
    return MetricResult(
        value=mae,
        details={
            "mae": mae,
            "rmse": rmse,
            "normalized_mae": normalized,
        },
    )


def compute_cost_gap(
    pred: torch.Tensor,
    node_batch: torch.LongTensor,
    metadata: List[Dict[str, object]],
    *,
    weights: Optional[Dict[str, float]] = None,
) -> MetricResult:
    device = pred.device
    if weights is None:
        weights = {
            "thermal": 55.0,
            "nuclear": 18.0,
            "solar": 5.0,
            "wind": 7.0,
            "hydro_release": 15.0,
            "hydro_ror": 10.0,
            "demand_response": 80.0,
            "battery_charge": 0.0,
            "battery_discharge": 20.0,
            "pumped_charge": 0.0,
            "pumped_discharge": 20.0,
            "net_import": 45.0,
            "unserved": 500.0,
        }
    weight_vec = torch.tensor([weights.get(name, 0.0) for name in _COMPONENT_ORDER], device=device)
    node_cost = (pred * weight_vec).sum(dim=1)
    num_graphs = len(metadata)
    batch_cost = torch.zeros(num_graphs, device=device)
    batch_cost.index_add_(0, node_batch, node_cost)
    objectives = torch.tensor(
        [float(meta.get("objective", float("nan"))) for meta in metadata],
        device=device,
    )
    denom = objectives.abs().clamp(min=1.0)
    gaps = (batch_cost - objectives) / denom
    mean_gap = gaps.mean().item()
    median_gap = gaps.median().item()
    max_gap = gaps.abs().max().item()
    return MetricResult(
        value=mean_gap,
        details={
            "mean_gap": mean_gap,
            "median_gap": median_gap,
            "max_gap": max_gap,
        },
    )


def compute_constraint_violation_rate(
    pred: torch.Tensor,
    batch: GraphBatch,
    *,
    tolerance: float = 1e-3,
) -> MetricResult:
    demand = batch.node_time[:, 0]
    idx = {name: index for index, name in enumerate(_COMPONENT_ORDER)}

    thermal = pred[:, idx["thermal"]]
    nuclear = pred[:, idx["nuclear"]]
    solar = pred[:, idx["solar"]]
    wind = pred[:, idx["wind"]]
    hydro_release = pred[:, idx["hydro_release"]]
    hydro_ror = pred[:, idx["hydro_ror"]]
    demand_response = pred[:, idx["demand_response"]]
    battery_charge = pred[:, idx["battery_charge"]]
    battery_discharge = pred[:, idx["battery_discharge"]]
    pumped_charge = pred[:, idx["pumped_charge"]]
    pumped_discharge = pred[:, idx["pumped_discharge"]]
    net_import = pred[:, idx["net_import"]]
    unserved = pred[:, idx["unserved"]]

    supply = (
        thermal
        + nuclear
        + solar
        + wind
        + hydro_release
        + hydro_ror
        + demand_response
        + battery_discharge
        + pumped_discharge
        + net_import
    )
    demand_side = demand + battery_charge + pumped_charge
    served = supply + unserved
    shortage = demand_side - served
    violation = shortage.abs() > tolerance
    rate = violation.float().mean().item()
    per_graph = torch.zeros(batch.batch_size, device=pred.device)
    per_graph.index_add_(0, batch.node_batch, violation.float())
    node_counts = torch.zeros(batch.batch_size, device=pred.device)
    node_counts.index_add_(0, batch.node_batch, torch.ones_like(violation, dtype=torch.float32))
    per_graph_rate = (per_graph / node_counts.clamp(min=1.0)).cpu().tolist()
    return MetricResult(
        value=rate,
        details={
            "mean_rate": rate,
            "max_rate": max(per_graph_rate) if per_graph_rate else 0.0,
        },
    )


class MetricSuite:
    def __init__(
        self,
        *,
        dispatch_cfg: Optional[Dict[str, object]] = None,
        cost_cfg: Optional[Dict[str, object]] = None,
        violation_cfg: Optional[Dict[str, object]] = None,
    ) -> None:
        self.dispatch_cfg = dispatch_cfg or {}
        self.cost_cfg = cost_cfg or {}
        self.violation_cfg = violation_cfg or {}

    def evaluate(
        self,
        batch: GraphBatch,
        predictions: torch.Tensor,
        target: torch.Tensor,
    ) -> Dict[str, MetricResult]:
        demand = batch.node_time[:, 0]
        dispatch = compute_dispatch_error(
            predictions,
            target,
            demand=demand,
            components=self.dispatch_cfg.get("components"),
            reduction=str(self.dispatch_cfg.get("reduction", "mean")),
        )
        cost = compute_cost_gap(
            predictions,
            batch.node_batch,
            batch.metadata,
            weights=self.cost_cfg.get("weights"),
        )
        violation = compute_constraint_violation_rate(
            predictions,
            batch,
            tolerance=float(self.violation_cfg.get("tolerance", 1e-3)),
        )
        return {
            "dispatch_error": dispatch,
            "cost_gap": cost,
            "violation_rate": violation,
        }
