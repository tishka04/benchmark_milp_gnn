from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn

from src.gnn.data.temporal import GraphBatch
from . import utils


def _agg_reduce(
    x: torch.Tensor,
    edge_index: torch.LongTensor,
    *,
    num_nodes: int,
    mode: str,
) -> torch.Tensor:
    src, dst = edge_index
    aggregated = utils.scatter_add(x[src], dst, num_nodes)
    if mode == "mean":
        deg = utils.degree(dst, num_nodes, x.device).clamp(min=1.0).unsqueeze(1)
        aggregated = aggregated / deg
    elif mode == "sum":
        pass
    else:
        raise ValueError(f"Unsupported aggregator '{mode}' for GraphSAGE")
    return aggregated


class GraphSAGELayer(nn.Module):
    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        *,
        activation: str = "relu",
        dropout: float = 0.0,
        aggregator: str = "mean",
        typed_edge_types: int = 0,
    ) -> None:
        super().__init__()
        self.out_dim = out_dim
        self.aggregator = aggregator
        self.self_linear = nn.Linear(in_dim, out_dim, bias=False)
        self.neigh_linear = (
            nn.Linear(in_dim, out_dim, bias=False)
            if typed_edge_types == 0
            else None
        )
        self.type_linears = (
            nn.ModuleList(nn.Linear(in_dim, out_dim, bias=False) for _ in range(typed_edge_types))
            if typed_edge_types > 0
            else None
        )
        self.bias = nn.Parameter(torch.zeros(out_dim))
        self.activation = utils.get_activation(activation)
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.LongTensor,
        *,
        num_nodes: int,
        edge_type: Optional[torch.LongTensor] = None,
    ) -> torch.Tensor:
        edge_index_bidirectional, edge_type_bidirectional = utils.make_bidirectional(edge_index, edge_type=edge_type)
        self_term = self.self_linear(x)
        if self.type_linears is None:
            neigh = _agg_reduce(x, edge_index_bidirectional, num_nodes=num_nodes, mode=self.aggregator)
            neigh = self.neigh_linear(neigh)
        else:
            if edge_type_bidirectional is None:
                raise RuntimeError("edge_type tensor required for typed GraphSAGE layer")
            neigh = torch.zeros(num_nodes, self.out_dim, device=x.device, dtype=x.dtype)
            for type_idx, linear in enumerate(self.type_linears):
                mask = edge_type_bidirectional == type_idx
                if not mask.any():
                    continue
                sub_edges = edge_index_bidirectional[:, mask]
                agg = _agg_reduce(x, sub_edges, num_nodes=num_nodes, mode=self.aggregator)
                neigh = neigh + linear(agg)
        out = self_term + neigh + self.bias
        out = self.activation(out)
        out = self.dropout(out)
        return out


class GraphSAGEModel(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dims: list[int],
        output_dim: int,
        *,
        activation: str = "relu",
        dropout: float = 0.0,
        aggregator: str = "mean",
        typed_message_passing: bool = False,
        num_edge_types: int = 0,
        node_type_cardinality: int = 4,
        type_embedding_dim: int = 16,
    ) -> None:
        super().__init__()
        if not hidden_dims:
            raise ValueError("GraphSAGEModel requires at least one hidden dimension")
        self.typed = typed_message_passing
        self.node_type_embedding = (
            nn.Embedding(node_type_cardinality, type_embedding_dim)
            if typed_message_passing and type_embedding_dim > 0
            else None
        )
        in_dim = input_dim + (type_embedding_dim if self.node_type_embedding is not None else 0)
        typed_edge_types = max(num_edge_types, 0)
        if self.typed:
            typed_edge_types = max(typed_edge_types, 1)
        dims = [in_dim] + hidden_dims
        layers = [
            GraphSAGELayer(
                dims[idx],
                dims[idx + 1],
                activation=activation,
                dropout=dropout,
                aggregator=aggregator,
                typed_edge_types=typed_edge_types if self.typed else 0,
            )
            for idx in range(len(hidden_dims))
        ]
        self.layers = nn.ModuleList(layers)
        self.head = nn.Sequential(
            nn.Linear(hidden_dims[-1], hidden_dims[-1]),
            utils.get_activation(activation),
            nn.Dropout(dropout) if dropout > 0 else nn.Identity(),
            nn.Linear(hidden_dims[-1], output_dim),
        )

    def forward(self, batch: GraphBatch) -> torch.Tensor:
        x = batch.node_input
        edge_index = batch.edge_index
        edge_type = batch.edge_type if self.typed else None
        if self.node_type_embedding is not None:
            x = torch.cat([x, self.node_type_embedding(batch.node_type)], dim=1)
        for layer in self.layers:
            x = layer(x, edge_index, num_nodes=x.size(0), edge_type=edge_type)
        return self.head(x)
