from __future__ import annotations

from typing import Callable, Dict

import torch.nn as nn

from src.gnn.data.temporal import GraphTemporalDataset
from src.gnn.training.config import ModelConfig
from .gat import GATModel
from .gcn import GCNModel
from .graphsage import GraphSAGEModel


_MODEL_BUILDERS: Dict[str, Callable[[ModelConfig, GraphTemporalDataset], nn.Module]] = {}


def _register(name: str):
    def decorator(func: Callable[[ModelConfig, GraphTemporalDataset], nn.Module]) -> Callable[[ModelConfig, GraphTemporalDataset], nn.Module]:
        _MODEL_BUILDERS[name.lower()] = func
        return func

    return decorator


@_register("gcn")
def _build_gcn(cfg: ModelConfig, dataset: GraphTemporalDataset) -> nn.Module:
    output_dim = cfg.output_dim or dataset.target_dim
    return GCNModel(
        input_dim=dataset.input_dim,
        hidden_dims=cfg.hidden_dims,
        output_dim=output_dim,
        activation=cfg.activation,
        dropout=cfg.dropout,
        typed_message_passing=cfg.typed_message_passing,
        num_edge_types=dataset.edge_type_cardinality,
        node_type_cardinality=dataset.node_type_cardinality,
        type_embedding_dim=cfg.type_embedding_dim,
    )


@_register("graphsage")
@_register("sage")
def _build_graphsage(cfg: ModelConfig, dataset: GraphTemporalDataset) -> nn.Module:
    output_dim = cfg.output_dim or dataset.target_dim
    return GraphSAGEModel(
        input_dim=dataset.input_dim,
        hidden_dims=cfg.hidden_dims,
        output_dim=output_dim,
        activation=cfg.activation,
        dropout=cfg.dropout,
        aggregator=cfg.aggregator,
        typed_message_passing=cfg.typed_message_passing,
        num_edge_types=dataset.edge_type_cardinality,
        node_type_cardinality=dataset.node_type_cardinality,
        type_embedding_dim=cfg.type_embedding_dim,
    )


@_register("gat")
def _build_gat(cfg: ModelConfig, dataset: GraphTemporalDataset) -> nn.Module:
    output_dim = cfg.output_dim or dataset.target_dim
    return GATModel(
        input_dim=dataset.input_dim,
        hidden_dims=cfg.hidden_dims,
        output_dim=output_dim,
        heads=cfg.heads,
        activation=cfg.activation,
        dropout=cfg.dropout,
        attn_dropout=cfg.attn_dropout,
        typed_message_passing=cfg.typed_message_passing,
        num_edge_types=dataset.edge_type_cardinality,
        node_type_cardinality=dataset.node_type_cardinality,
        type_embedding_dim=cfg.type_embedding_dim,
    )


def build_model(cfg: ModelConfig, dataset: GraphTemporalDataset) -> nn.Module:
    builder = _MODEL_BUILDERS.get(cfg.name.lower())
    if builder is None:
        raise ValueError(f"Unknown model '{cfg.name}'. Available: {sorted(_MODEL_BUILDERS)}")
    return builder(cfg, dataset)
