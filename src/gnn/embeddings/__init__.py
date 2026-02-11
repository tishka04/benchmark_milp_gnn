"""
Hierarchical Temporal Encoder training and embedding generation for v3 scenarios.

Modules:
    config   - Training configuration dataclass
    dataset  - TemporalGraphDataset with hierarchy extraction
    loss     - Contrastive multi-lag InfoNCE loss
    train    - Training and validation loops
    generate - Embedding generation for all scenarios
"""

from src.gnn.embeddings.config import TrainingConfig
from src.gnn.embeddings.dataset import TemporalGraphDatasetV3, load_dataset_v3
from src.gnn.embeddings.loss import contrastive_loss_multilag
from src.gnn.embeddings.train import train_epoch, validate_epoch
from src.gnn.embeddings.generate import generate_embeddings
