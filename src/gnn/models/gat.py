from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.gnn.data.temporal import GraphBatch
from . import utils


def _segment_softmax(scores: torch.Tensor, dst: torch.LongTensor, num_nodes: int) -> torch.Tensor:
    if scores.numel() == 0:
        return scores
    order = torch.argsort(dst)
    dst_sorted = dst[order]
    scores_sorted = scores[order]
    result = torch.zeros_like(scores_sorted)
    total = scores_sorted.size(0)
    start = 0
    while start < total:
        node = int(dst_sorted[start].item())
        end = start + 1
        while end < total and int(dst_sorted[end].item()) == node:
            end += 1
        segment = scores_sorted[start:end]
        segment = segment - segment.max(dim=0, keepdim=True).values
        exp_segment = segment.exp()
        denom = exp_segment.sum(dim=0, keepdim=True)
        result[start:end] = exp_segment / denom.clamp(min=1e-12)
        start = end
    softmax = torch.zeros_like(scores)
    softmax[order] = result
    return softmax


class GATLayer(nn.Module):
    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        *,
        heads: int = 4,
        dropout: float = 0.0,
        activation: str = "elu",
        attn_dropout: float = 0.0,
        typed_edge_types: int = 0,
    ) -> None:
        super().__init__()
        self.heads = heads
        self.out_dim = out_dim
        self.loop_type_index: Optional[int] = typed_edge_types - 1 if typed_edge_types > 0 else None
        self.linear = nn.Linear(in_dim, heads * out_dim, bias=False)
        self.attn_src = nn.Parameter(torch.empty(heads, out_dim))
        self.attn_dst = nn.Parameter(torch.empty(heads, out_dim))
        nn.init.xavier_uniform_(self.attn_src)
        nn.init.xavier_uniform_(self.attn_dst)
        self.edge_type_bias = (
            nn.Parameter(torch.zeros(typed_edge_types, heads))
            if typed_edge_types > 0
            else None
        )
        self.bias = nn.Parameter(torch.zeros(heads * out_dim))
        self.activation = utils.get_activation(activation)
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self.attn_dropout = attn_dropout

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
        src, dst = edge_index_aug
        proj = self.linear(x).view(num_nodes, self.heads, self.out_dim)
        src_proj = proj[src]
        dst_proj = proj[dst]
        scores = (src_proj * self.attn_src).sum(dim=-1) + (dst_proj * self.attn_dst).sum(dim=-1)
        if self.edge_type_bias is not None and edge_type_aug is not None:
            scores = scores + self.edge_type_bias[edge_type_aug]
        scores = F.leaky_relu(scores, negative_slope=0.2)
        attn = _segment_softmax(scores, dst, num_nodes)
        attn = F.dropout(attn, p=self.attn_dropout, training=self.training)
        messages = src_proj * attn.unsqueeze(-1)
        out = torch.zeros(num_nodes, self.heads, self.out_dim, device=x.device, dtype=x.dtype)
        out.index_add_(0, dst, messages)
        out = out.reshape(num_nodes, self.heads * self.out_dim)
        out = out + self.bias
        out = self.activation(out)
        out = self.dropout(out)
        return out


class GATModel(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dims: list[int],
        output_dim: int,
        *,
        heads: int = 4,
        activation: str = "elu",
        dropout: float = 0.0,
        attn_dropout: float = 0.1,
        typed_message_passing: bool = False,
        num_edge_types: int = 0,
        node_type_cardinality: int = 4,
        type_embedding_dim: int = 16,
    ) -> None:
        super().__init__()
        if not hidden_dims:
            raise ValueError("GATModel requires at least one hidden dimension")
        self.typed = typed_message_passing
        self.node_type_embedding = (
            nn.Embedding(node_type_cardinality, type_embedding_dim)
            if typed_message_passing and type_embedding_dim > 0
            else None
        )
        in_dim = input_dim + (type_embedding_dim if self.node_type_embedding is not None else 0)
        typed_edge_types = max(num_edge_types, 0)
        if self.typed:
            typed_edge_types = max(typed_edge_types, 1) + 1
        dims = [in_dim] + hidden_dims
        layers: list[nn.Module] = []
        current_dim = dims[0]
        for hidden in hidden_dims:
            layer = GATLayer(
                current_dim,
                hidden,
                heads=heads,
                dropout=dropout,
                activation=activation,
                attn_dropout=attn_dropout,
                typed_edge_types=typed_edge_types if self.typed else 0,
            )
            layers.append(layer)
            current_dim = hidden * heads
        self.layers = nn.ModuleList(layers)
        self.final_linear = nn.Linear(current_dim, output_dim)

    def forward(self, batch: GraphBatch) -> torch.Tensor:
        x = batch.node_input
        edge_index = batch.edge_index
        edge_type = batch.edge_type if self.typed else None
        if self.node_type_embedding is not None:
            x = torch.cat([x, self.node_type_embedding(batch.node_type)], dim=1)
        num_nodes = x.size(0)
        for layer in self.layers:
            x = layer(x, edge_index, num_nodes=num_nodes, edge_type=edge_type)
        return self.final_linear(x)
