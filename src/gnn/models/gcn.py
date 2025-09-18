from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn

from src.gnn.data.temporal import GraphBatch
from . import utils


class GCNLayer(nn.Module):
    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        *,
        activation: str = "relu",
        dropout: float = 0.0,
        typed_edge_types: int = 0,
    ) -> None:
        super().__init__()
        self.out_dim = out_dim
        self.typed_edge_types = typed_edge_types
        self.loop_type_index: Optional[int] = typed_edge_types - 1 if typed_edge_types > 0 else None
        self.linear = nn.Linear(in_dim, out_dim, bias=False)
        self.type_linears = (
            nn.ModuleList(nn.Linear(in_dim, out_dim, bias=False) for _ in range(typed_edge_types))
            if typed_edge_types > 0
            else None
        )
        self.bias = nn.Parameter(torch.zeros(out_dim))
        self.activation = utils.get_activation(activation)
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

    def _propagate(
        self,
        x: torch.Tensor,
        edge_index: torch.LongTensor,
        *,
        num_nodes: int,
    ) -> torch.Tensor:
        norm = utils.normalise_adjacency(edge_index, num_nodes)
        src, dst = edge_index
        messages = x[src] * norm.unsqueeze(-1)
        return utils.scatter_add(messages, dst, num_nodes)

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.LongTensor,
        *,
        num_nodes: int,
        edge_type: Optional[torch.LongTensor] = None,
    ) -> torch.Tensor:
        edge_index_aug, edge_type_aug = utils.make_bidirectional(edge_index, edge_type=edge_type)
        edge_index_aug, edge_type_aug = utils.add_self_loops(
            edge_index_aug,
            num_nodes,
            edge_type=edge_type_aug,
            self_loop_type=self.loop_type_index,
        )
        if self.type_linears is None:
            aggregated = self._propagate(x, edge_index_aug, num_nodes=num_nodes)
            out = self.linear(aggregated)
        else:
            if edge_type_aug is None:
                raise RuntimeError("edge_type tensor required for typed GCN layer")
            out = torch.zeros(num_nodes, self.out_dim, device=x.device, dtype=x.dtype)
            for type_idx, linear in enumerate(self.type_linears):
                mask = edge_type_aug == type_idx
                if not mask.any():
                    continue
                sub_edges = edge_index_aug[:, mask]
                aggregated = self._propagate(x, sub_edges, num_nodes=num_nodes)
                out = out + linear(aggregated)
        out = out + self.bias
        out = self.activation(out)
        out = self.dropout(out)
        return out


class GCNModel(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dims: list[int],
        output_dim: int,
        *,
        activation: str = "relu",
        dropout: float = 0.0,
        typed_message_passing: bool = False,
        num_edge_types: int = 0,
        node_type_cardinality: int = 4,
        type_embedding_dim: int = 16,
    ) -> None:
        super().__init__()
        if not hidden_dims:
            raise ValueError("GCNModel requires at least one hidden dimension")
        self.typed = typed_message_passing
        self.node_type_embedding = (
            nn.Embedding(node_type_cardinality, type_embedding_dim)
            if typed_message_passing and type_embedding_dim > 0
            else None
        )
        in_dim = input_dim + (type_embedding_dim if self.node_type_embedding is not None else 0)
        layers: list[nn.Module] = []
        dims = [in_dim] + hidden_dims
        typed_edge_types = max(num_edge_types, 0)
        if self.typed:
            typed_edge_types = max(typed_edge_types, 1) + 1  # reserve an extra type id for self-loops
        for idx in range(len(hidden_dims)):
            layers.append(
                GCNLayer(
                    dims[idx],
                    dims[idx + 1],
                    activation=activation,
                    dropout=dropout,
                    typed_edge_types=typed_edge_types if self.typed else 0,
                )
            )
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
