from __future__ import annotations

from typing import Optional, Tuple

import torch
import torch.nn as nn


def get_activation(name: str) -> nn.Module:
    name = name.lower()
    if name == "relu":
        return nn.ReLU()
    if name == "gelu":
        return nn.GELU()
    if name in {"elu", "elu+"}:
        return nn.ELU()
    if name == "leaky_relu":
        return nn.LeakyReLU(negative_slope=0.01)
    if name == "tanh":
        return nn.Tanh()
    if name == "identity":
        return nn.Identity()
    raise ValueError(f"Unsupported activation '{name}'")


def add_self_loops(
    edge_index: torch.LongTensor,
    num_nodes: int,
    *,
    edge_type: Optional[torch.LongTensor] = None,
    self_loop_type: Optional[int] = None,
) -> Tuple[torch.LongTensor, Optional[torch.LongTensor]]:
    device = edge_index.device
    loop_index = torch.arange(num_nodes, device=device)
    loops = torch.stack([loop_index, loop_index], dim=0)
    edge_index = torch.cat([edge_index, loops], dim=1)
    if edge_type is not None:
        if self_loop_type is None:
            loop_type_val = int(edge_type.max().item()) + 1 if edge_type.numel() else 0
        else:
            loop_type_val = self_loop_type
        loop_types = torch.full((num_nodes,), loop_type_val, dtype=edge_type.dtype, device=device)
        edge_type = torch.cat([edge_type, loop_types], dim=0)
    return edge_index, edge_type


def make_bidirectional(
    edge_index: torch.LongTensor,
    *,
    edge_type: Optional[torch.LongTensor] = None,
) -> Tuple[torch.LongTensor, Optional[torch.LongTensor]]:
    src, dst = edge_index
    rev_edges = torch.stack([dst, src], dim=0)
    edge_index = torch.cat([edge_index, rev_edges], dim=1)
    if edge_type is not None:
        edge_type = torch.cat([edge_type, edge_type], dim=0)
    return edge_index, edge_type


def scatter_add(src: torch.Tensor, index: torch.LongTensor, dim_size: int) -> torch.Tensor:
    out = torch.zeros(dim_size, src.size(1), device=src.device, dtype=src.dtype)
    out.index_add_(0, index, src)
    return out


def degree(index: torch.LongTensor, num_nodes: int, device: torch.device) -> torch.Tensor:
    ones = torch.ones(index.size(0), device=device, dtype=torch.float32)
    deg = torch.zeros(num_nodes, device=device, dtype=torch.float32)
    deg.index_add_(0, index, ones)
    return deg


def normalise_adjacency(
    edge_index: torch.LongTensor,
    num_nodes: int,
    edge_weight: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    if edge_weight is None:
        edge_weight = torch.ones(edge_index.size(1), device=edge_index.device, dtype=torch.float32)
    src, dst = edge_index
    deg = torch.zeros(num_nodes, device=edge_index.device, dtype=edge_weight.dtype)
    deg.index_add_(0, dst, edge_weight)
    deg_inv_sqrt = torch.pow(deg.clamp(min=1e-12), -0.5)
    return edge_weight * deg_inv_sqrt[src] * deg_inv_sqrt[dst]
