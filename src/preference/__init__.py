# ==============================================================================
# PREFERENCE-BASED ENERGY LEARNING MODULE
# ==============================================================================
# Implementation of conditional preference-based EBM training with:
# - HTE embeddings for scenario conditioning
# - EBM for learning energy landscape over decisions
# - Langevin sampler for candidate generation
# - LP Worker for economic oracle feedback
# - Margin ranking loss for preference learning
# ==============================================================================

from .data_models import (
    ScenarioData,
    DecisionVector,
    CandidateResult,
    PreferencePair,
    TrainingBatch,
)
from .ebm import (
    ConditionalEBM,
    ConditionalEBMWithGRU,
    ConditionalEBMWithTemporalTransformer,
    ConditionalEBMWithZoneAttention,
    build_ebm,
)
from .sampler import LangevinSampler
from .decoder import HierarchicalDecoder
from .dataset import PreferenceDataset, PreferenceDataLoader
from .embedding_processor import (
    ZonalEmbeddingProcessor,
    TemporalZonalDataset,
    temporal_collate_fn,
    load_zone_embeddings,
    EmbeddingConfig,
)
from .trainer import PreferenceTrainer
from .losses import MarginRankingLoss, WeightedMarginRankingLoss
from .training_strategy import (
    TwoPhaseTrainer,
    TwoPhaseConfig,
    PhaseConfig,
    TrainingPhase,
    CostProxy,
    UncertaintyEstimator,
    AdaptiveCandidateScheduler,
    create_two_phase_trainer,
)
from .conditioning import (
    HConditioner,
    DecisionFeatureExtractor,
    FeatureBasedCostProxy,
    ConditionedEBMWrapper,
    FeatureConfig,
)
from .lp_oracle import (
    LPOracle,
    PreferenceLPOracle,
    CachedLPOracle,
    LPOracleConfig,
    OracleResult,
    OracleStage,
    create_lp_oracle,
)

__all__ = [
    # Data models
    "ScenarioData",
    "DecisionVector", 
    "CandidateResult",
    "PreferencePair",
    "TrainingBatch",
    # Models
    "ConditionalEBM",
    "ConditionalEBMWithGRU",
    "ConditionalEBMWithTemporalTransformer",
    "ConditionalEBMWithZoneAttention",
    "build_ebm",
    "LangevinSampler",
    "HierarchicalDecoder",
    # Dataset
    "PreferenceDataset",
    "PreferenceDataLoader",
    # Embedding processing
    "ZonalEmbeddingProcessor",
    "TemporalZonalDataset",
    "temporal_collate_fn",
    "load_zone_embeddings",
    "EmbeddingConfig",
    # Training
    "PreferenceTrainer",
    "MarginRankingLoss",
    "WeightedMarginRankingLoss",
    # Two-phase training
    "TwoPhaseTrainer",
    "TwoPhaseConfig",
    "PhaseConfig",
    "TrainingPhase",
    "CostProxy",
    "UncertaintyEstimator",
    "AdaptiveCandidateScheduler",
    "create_two_phase_trainer",
    # Conditioning
    "HConditioner",
    "DecisionFeatureExtractor",
    "FeatureBasedCostProxy",
    "ConditionedEBMWrapper",
    "FeatureConfig",
    # LP Oracle
    "LPOracle",
    "PreferenceLPOracle",
    "CachedLPOracle",
    "LPOracleConfig",
    "OracleResult",
    "OracleStage",
    "create_lp_oracle",
]
