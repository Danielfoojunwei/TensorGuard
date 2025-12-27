"""
TensorGuard Edge Client - Core Implementation
"""

import numpy as np
import time
from typing import List, Optional, Dict, Any, Tuple
from datetime import datetime

from ..api.schemas import ShieldConfig, Demonstration, SubmissionReceipt, ClientStatus
from ..core.adapters import VLAAdapter
from ..core.crypto import N2HEEncryptor, CryptographyError
from ..core.pipeline import GradientClipper, SemanticSparsifier, APHECompressor, QualityMonitor
from ..utils.logging import get_logger
from ..utils.exceptions import ValidationError, CommunicationError
from ..utils.config import settings

logger = get_logger(__name__)

class EdgeClient:
    """Main client for TensorGuard edge deployment."""
    
    def __init__(self, config: Optional[ShieldConfig] = None):
        self.config = config or ShieldConfig(
            model_type="pi0", 
            key_path=settings.KEY_PATH,
            security_level=settings.SECURITY_LEVEL
        )
        
        # Initialize pipeline
        self._clipper = GradientClipper(self.config.max_gradient_norm)
        self._sparsifier = SemanticSparsifier(self.config.sparsity)
        self._compressor = APHECompressor(self.config.compression_ratio)
        self._encryptor = N2HEEncryptor(self.config.key_path, self.config.security_level)
        self._quality_monitor = QualityMonitor()
        
        self._adapter: Optional[VLAAdapter] = None
        self._current_round_demos: List[Demonstration] = []
        self._privacy_budget_used = 0.0
        self._total_submissions = 0
        self._error_memory: Dict[str, np.ndarray] = {}

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

        logger.info(f"Processing training round with {len(self._current_round_demos)} demos")
        
        # 1. MoI Gradient Computation
        combined_grads = {}
        for demo in self._current_round_demos:
            experts = self._adapter.compute_expert_gradients(demo)
            for exp_name, grads in experts.items():
                weight = {"visual": 1.0, "language": 0.8, "auxiliary": 1.2}.get(exp_name, 1.0)
                for k, v in grads.items():
                    combined_grads[k] = combined_grads.get(k, 0) + v * weight
        
        self._current_round_demos = []

        # 2. Privacy Pipeline with Error Feedback
        for k, v in self._error_memory.items():
            if k in combined_grads: combined_grads[k] += v

        clipped = self._clipper.clip(combined_grads)
        sparse = self._sparsifier.sparsify(clipped)
        
        # Update residuals
        self._error_memory = {k: clipped[k] - sparse[k] for k in clipped if k in sparse}

        # 3. Aggressive Compression & Encryption
        pixel_data = self._compressor.compress(sparse)
        
        # Quality Check
        check = self._compressor.decompress(pixel_data)
        self._quality_monitor.check_quality(sparse, check)
        
        encrypted = self._encryptor.encrypt(pixel_data)
        self._total_submissions += 1
        self._privacy_budget_used += self.config.dp_epsilon / 100
        
        return encrypted

def create_client(model_type: str = "pi0", **kwargs) -> EdgeClient:
    """Factory for EdgeClient."""
    config = ShieldConfig(model_type=model_type, **kwargs)
    return EdgeClient(config)
