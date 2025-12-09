"""
Energy-Based Model (EBM) for UC/DR/Storage Configuration Learning

This module implements an EBM that learns the energy landscape of feasible
binary decisions (Unit Commitment, Demand Response, Storage) conditioned on
graph embeddings from the Hierarchical Temporal Encoder.
"""

from .model import EnergyModel, StructuredEnergyModel
from .sampler import GibbsSampler, SGLDSampler
from .trainer import EBMTrainer
from .dataset import MILPBinaryDataset
from .metrics import EBMMetrics
from .embedding_loader import EmbeddingLoader, GoogleDriveEmbeddingLoader

# Optional THRML samplers (require thrml package)
try:
    from .thrml_sampler import ThrmlConditionalSampler, HybridThrmlGibbsSampler
    THRML_AVAILABLE = True
except ImportError:
    THRML_AVAILABLE = False
    ThrmlConditionalSampler = None
    HybridThrmlGibbsSampler = None

__all__ = [
    'EnergyModel',
    'StructuredEnergyModel',
    'GibbsSampler',
    'SGLDSampler',
    'EBMTrainer',
    'MILPBinaryDataset',
    'EBMMetrics',
    'EmbeddingLoader',
    'GoogleDriveEmbeddingLoader',
    'ThrmlConditionalSampler',
    'HybridThrmlGibbsSampler',
    'THRML_AVAILABLE',
]
