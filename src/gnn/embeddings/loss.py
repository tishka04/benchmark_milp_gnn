"""
Contrastive multi-lag InfoNCE loss for temporal node embeddings.
"""

import torch
import torch.nn.functional as F


def contrastive_loss_multilag(
    embeddings: torch.Tensor,
    lags=(1, 4, 8),
    neg_sample_ratio: float = 0.10,
    max_nodes: int = 160,
    temperature: float = 0.20,
    logits_clamp: float = 50.0,
    eps: float = 1e-8,
) -> torch.Tensor:
    """
    Multi-lag InfoNCE loss for temporal node embeddings.

    Args:
        embeddings: [N, T, D] tensor or dict with key ``'assets'``.
        lags: positive time offsets used to build positive pairs.
        neg_sample_ratio: ``n_neg = max(128, int(M * neg_sample_ratio))``.
        max_nodes: subsample nodes for stability / speed.
        temperature: softmax temperature.
        logits_clamp: clamp logits to ``[-c, c]`` for numerical safety.
        eps: epsilon for safe L2 normalisation.

    Returns:
        Scalar loss tensor.
    """
    # Unwrap dict output from HierarchicalTemporalEncoder
    if isinstance(embeddings, dict):
        if "assets" not in embeddings:
            raise KeyError("Expected embeddings['assets'] for dict input.")
        embeddings = embeddings["assets"]

    if embeddings.dim() != 3:
        raise ValueError(
            f"Expected embeddings [N, T, D], got {tuple(embeddings.shape)}"
        )

    N, T, D = embeddings.shape
    device = embeddings.device

    # Subsample nodes
    if N > max_nodes:
        idx = torch.randperm(N, device=device)[:max_nodes]
        embeddings = embeddings[idx]
        N = max_nodes

    # L2 normalise
    emb = embeddings / embeddings.norm(dim=-1, keepdim=True).clamp_min(eps)

    # Build anchor / positive pairs across lags
    anchor_chunks, pos_chunks = [], []
    valid_lags = [lag for lag in lags if 0 < lag < T]

    if len(valid_lags) == 0:
        return torch.zeros([], device=device, dtype=embeddings.dtype)

    for lag in valid_lags:
        a = emb[:, : T - lag, :].reshape(-1, D)
        p = emb[:, lag:, :].reshape(-1, D)
        anchor_chunks.append(a)
        pos_chunks.append(p)

    anchors = torch.cat(anchor_chunks, dim=0)   # [M, D]
    positives = torch.cat(pos_chunks, dim=0)     # [M, D]
    M = anchors.size(0)

    # Negatives from pooled (node, time) embeddings
    pool = emb.reshape(N * T, D)
    n_neg = max(128, int(M * neg_sample_ratio))
    neg_idx = torch.randint(0, N * T, (n_neg,), device=device)
    negatives = pool[neg_idx]  # [n_neg, D]

    # Logits
    pos_logits = (anchors * positives).sum(dim=-1, keepdim=True) / temperature
    neg_logits = (anchors @ negatives.t()) / temperature
    logits = torch.cat([pos_logits, neg_logits], dim=1)

    if logits_clamp is not None:
        logits = logits.clamp(-logits_clamp, logits_clamp)

    labels = torch.zeros(M, dtype=torch.long, device=device)
    return F.cross_entropy(logits, labels)
