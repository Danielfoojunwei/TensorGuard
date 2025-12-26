"""
TensorGuard: Privacy-Preserving VLA Fine-Tuning for Humanoid Robotics

Plug-and-play SDK for Robotics System Integrators to fine-tune
Vision-Language-Action models without exposing proprietary data.

Production-Grade Features:
- Operating envelope enforcement
- Canonical UpdatePackage format
- Separate privacy/training controls
- Enterprise key management
- Evaluation gating
- IL/RL pipeline separation
- SRE observability
- Resilient aggregation
"""

__version__ = "1.1.0"
__author__ = "Dynamical.ai"

from .edge_client import EdgeClient, create_client
from .structures import ShieldConfig, Demonstration, SubmissionReceipt, ClientStatus
from .adapters import VLAAdapter

from .n2he import (
    N2HEContext,
    N2HEParams,
    N2HE_128,
    N2HE_192,
    LWECiphertext,
)

# Production components
from .production import (
    # Operating envelope
    OperatingEnvelope,
    PEFTStrategy,
    # Update package
    UpdatePackage,
    ModelTargetMap,
    TrainingMetadata,
    SafetyStatistics,
    ObjectiveType,
    # Policy profiles
    DPPolicyProfile,
    EncryptionPolicyProfile,
    TrainingPolicyProfile,
    # Key management
    KeyManagementSystem,
    KeyMetadata,
    # Evaluation gating
    EvaluationGate,
    SafetyThresholds,
    EvaluationMetrics,
    # Training pipeline
    TrainingPipeline,
    TrainingStage,
    StageConfig,
    # Observability
    ObservabilityCollector,
    RoundLatencyBreakdown,
    CompressionMetrics,
    ModelQualityMetrics,
    # Aggregation
    ResilientAggregator,
    ClientContribution,
    # Utilities
    print_production_status,
)

__all__ = [
    # Edge Client
    "EdgeClient",
    "ShieldConfig",
    "VLAAdapter",
    "Demonstration",
    "SubmissionReceipt",
    "ClientStatus",
    "create_client",
    # N2HE
    "N2HEContext",
    "N2HEParams",
    "N2HE_128",
    "N2HE_192",
    "LWECiphertext",
    # Production - Operating Envelope
    "OperatingEnvelope",
    "PEFTStrategy",
    # Production - Update Package
    "UpdatePackage",
    "ModelTargetMap",
    "TrainingMetadata",
    "SafetyStatistics",
    "ObjectiveType",
    # Production - Policy Profiles
    "DPPolicyProfile",
    "EncryptionPolicyProfile",
    "TrainingPolicyProfile",
    # Production - Key Management
    "KeyManagementSystem",
    "KeyMetadata",
    # Production - Evaluation Gating
    "EvaluationGate",
    "SafetyThresholds",
    "EvaluationMetrics",
    # Production - Training Pipeline
    "TrainingPipeline",
    "TrainingStage",
    "StageConfig",
    # Production - Observability
    "ObservabilityCollector",
    "RoundLatencyBreakdown",
    "CompressionMetrics",
    "ModelQualityMetrics",
    # Production - Aggregation
    "ResilientAggregator",
    "ClientContribution",
    # Production - Utilities
    "print_production_status",
]
