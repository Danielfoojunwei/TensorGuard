"""
TensorGuard API Schemas and Data Structures
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import List, Optional, Dict, Any
import numpy as np
import time

@dataclass
class ShieldConfig:
    """Configuration for TensorGuard Edge Client."""
    model_type: str = "pi0"
    key_path: str = "keys/node_key.pem"
    sparsity: float = 0.01
    compression_ratio: int = 32
    max_gradient_norm: float = 1.0
    dp_epsilon: float = 1.0
    security_level: int = 128
    cloud_endpoint: str = "https://api.tensor-crate.ai"
    
@dataclass
class Demonstration:
    """Robot demonstration data."""
    observations: List[np.ndarray]
    actions: List[np.ndarray]
    task_id: Optional[str] = None
    episode_id: Optional[str] = None
    timestamp: float = field(default_factory=time.time)
    contains_pii: bool = False
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class SubmissionReceipt:
    """Receipt for encrypted gradient submission."""
    submission_id: str
    encrypted_size_bytes: int
    compression_achieved: float
    estimated_aggregation: datetime
    privacy_budget_used: float

@dataclass
class ClientStatus:
    """Current state of the Edge client."""
    pending_submissions: int
    total_submissions: int
    privacy_budget_remaining: float
    last_model_version: str
    connection_status: str
