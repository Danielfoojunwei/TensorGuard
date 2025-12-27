import numpy as np
import time
import hashlib
from typing import List, Optional, Dict, Any, Tuple
from datetime import datetime

from ..api.schemas import ShieldConfig, Demonstration, SubmissionReceipt, ClientStatus
from ..core.adapters import VLAAdapter
from ..core.crypto import N2HEEncryptor, CryptographyError
from ..core.pipeline import GradientClipper, SemanticSparsifier, APHECompressor, QualityMonitor
from ..utils.logging import get_logger
from ..utils.exceptions import ValidationError, CommunicationError
from ..utils.config import settings

# Production components
from .production import (
    OperatingEnvelope,
    UpdatePackage,
    ModelTargetMap,
    TrainingMetadata,
    SafetyStatistics,
    ObjectiveType,
    DPPolicyProfile,
    EncryptionPolicyProfile,
    TrainingPolicyProfile,
    ObservabilityCollector,
    RoundLatencyBreakdown,
    CompressionMetrics,
    ModelQualityMetrics,
)

logger = get_logger(__name__)

# Flower compatibility
try:
    import flwr as fl
except ImportError:
    class fl:
        class client:
            class NumPyClient: pass

class EdgeClient(fl.client.NumPyClient if 'fl' in globals() else object):
    """Main client for TensorGuard edge deployment with production optimizations."""
    
    def __init__(
        self, 
        config: Optional[ShieldConfig] = None,
        operating_envelope: Optional[OperatingEnvelope] = None,
        dp_profile: Optional[DPPolicyProfile] = None,
        encryption_profile: Optional[EncryptionPolicyProfile] = None,
        training_profile: Optional[TrainingPolicyProfile] = None,
        enable_observability: bool = True
    ):
        self.config = config or ShieldConfig(
            model_type="pi0", 
            key_path=settings.KEY_PATH,
            security_level=settings.SECURITY_LEVEL
        )
        
        # Production components
        self.operating_envelope = operating_envelope or OperatingEnvelope()
        self.dp_profile = dp_profile or DPPolicyProfile(
            profile_name="default",
            clipping_norm=self.config.max_gradient_norm,
            epsilon_budget=self.config.dp_epsilon
        )
        self.observability = ObservabilityCollector() if enable_observability else None
        
        # Initialize pipeline components
        self._clipper = GradientClipper(self.dp_profile.clipping_norm)
        self._sparsifier = SemanticSparsifier(self.config.sparsity)
        self._compressor = APHECompressor(self.config.compression_ratio)
        self._encryptor = N2HEEncryptor(self.config.key_path, self.config.security_level)
        self._quality_monitor = QualityMonitor()
        
        self._adapter: Optional[VLAAdapter] = None
        self._current_round_demos: List[Demonstration] = []
        self._privacy_budget_used = 0.0
        self._total_submissions = 0
        self._error_memory: Dict[str, np.ndarray] = {}
        self._current_round = 0

        logger.info(f"EdgeClient initialized for {self.config.model_type}")

    def set_adapter(self, adapter: VLAAdapter) -> None:
        """Configure the VLA adapter for gradient computation."""
        self._adapter = adapter
        logger.info(f"VLA Adapter configured: {type(adapter).__name__}")

    def add_demonstration(self, demo: Demonstration):
        """Buffer a demonstration for processing."""
        self._current_round_demos.append(demo)

    def get_status(self) -> ClientStatus:
        """Return current status metrics."""
        return ClientStatus(
            pending_submissions=len(self._current_round_demos),
            total_submissions=self._total_submissions,
            privacy_budget_remaining=max(0.0, self.config.dp_epsilon - self._privacy_budget_used),
            last_model_version="v1.1.0",
            connection_status="online"
        )

    def process_round(self) -> Optional[bytes]:
        """Compute, privacy-process, and encrypt gradients for all buffered demos."""
        if not self._current_round_demos:
            return None
        
        if not self._adapter:
            raise ValidationError("VLA Adapter not configured")

        self._current_round += 1
        logger.info(f"Processing training round {self._current_round} with {len(self._current_round_demos)} demos")
        
        latency = RoundLatencyBreakdown()
        train_start = time.time()

        # 1. MoI Gradient Computation
        combined_grads = {}
        for demo in self._current_round_demos:
            experts = self._adapter.compute_expert_gradients(demo)
            for exp_name, grads in experts.items():
                weight = {"visual": 1.0, "language": 0.8, "auxiliary": 1.2}.get(exp_name, 1.0)
                for k, v in grads.items():
                    combined_grads[k] = combined_grads.get(k, 0) + v * weight
        
        demo_count = len(self._current_round_demos)
        self._current_round_demos = []
        latency.train_ms = (time.time() - train_start) * 1000

        # 2. Privacy Pipeline with Error Feedback
        # a) Add Residuals
        for k, v in self._error_memory.items():
            if k in combined_grads: combined_grads[k] += v

        # b) Clip
        clipped = self._clipper.clip(combined_grads)
        
        # c) Sparsify
        sparse = self._sparsifier.sparsify(clipped)
        
        # d) Update residuals
        self._error_memory = {k: clipped[k] - sparse[k] for k in clipped if k in sparse}

        # 3. Aggressive Compression & Encryption
        compress_start = time.time()
        pixel_data = self._compressor.compress(sparse)
        latency.compress_ms = (time.time() - compress_start) * 1000
        
        # Quality Check
        check = self._compressor.decompress(pixel_data)
        quality_mse = self._quality_monitor.check_quality(sparse, check)
        
        encrypt_start = time.time()
        encrypted = self._encryptor.encrypt(pixel_data)
        latency.encrypt_ms = (time.time() - encrypt_start) * 1000

        # Stats for UpdatePackage
        original_size = sum(g.nbytes for g in sparse.values())
        grad_norms = [np.linalg.norm(v) for v in combined_grads.values()]
        
        # DP Budget Check
        epsilon_per_round = self.config.dp_epsilon / 100
        if not self.dp_profile.consume_epsilon(epsilon_per_round):
            logger.critical("DP budget exhausted!")
            return None

        # Create UpdatePackage
        update_package = UpdatePackage(
            client_id=hashlib.sha256(str(id(self)).encode()).hexdigest()[:16],
            target_map=ModelTargetMap(
                module_names=list(combined_grads.keys()),
                adapter_ids=self.operating_envelope.trainable_modules,
                tensor_shapes={k: v.shape for k, v in combined_grads.items()}
            ),
            delta_tensors={"encrypted": encrypted},
            compression_metadata={
                "ratio": self.config.compression_ratio,
                "sparsity": self.config.sparsity,
                "size": len(encrypted)
            },
            training_meta=TrainingMetadata(
                steps=demo_count,
                learning_rate=1e-4,
                objective_type=ObjectiveType.IMITATION_LEARNING,
                num_demonstrations=demo_count,
                training_duration_seconds=latency.train_ms / 1000
            ),
            safety_stats=SafetyStatistics(
                kl_divergence=quality_mse,
                grad_norm_mean=float(np.mean(grad_norms)),
                grad_norm_max=float(np.max(grad_norms)),
                dp_epsilon_consumed=self.dp_profile.epsilon_consumed
            )
        )

        # Observability
        if self.observability:
            self.observability.record_latency(latency, self._current_round)
            self.observability.record_compression(
                CompressionMetrics(original_size, len(pixel_data), original_size / max(len(pixel_data), 1)),
                self._current_round
            )
            self.observability.record_quality(
                ModelQualityMetrics(success_rate=1.0, kl_divergence=quality_mse, update_norm=float(np.max(grad_norms))),
                self._current_round
            )

        self._total_submissions += 1
        self._privacy_budget_used = self.dp_profile.epsilon_consumed
        
        return update_package.serialize()

    # Flower overrides
    def get_parameters(self, config: Dict[str, Any]) -> List[np.ndarray]:
        return []

    def fit(self, parameters: List[np.ndarray], config: Dict[str, Any]) -> Tuple[List[np.ndarray], int, Dict[str, Any]]:
        package_bytes = self.process_round()
        if package_bytes:
            return [np.frombuffer(package_bytes, dtype=np.uint8)], 1, {"round": self._current_round}
        return [], 0, {"error": "no_data"}

def create_client(model_type: str = "pi0", **kwargs) -> EdgeClient:
    """Factory for EdgeClient."""
    config = ShieldConfig(model_type=model_type, **kwargs)
    return EdgeClient(config)
