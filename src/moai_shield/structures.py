"""
APHE-Shield Data Types
"""

import numpy as np
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any
from datetime import datetime
import time

@dataclass
class ShieldConfig:
    """Configuration for APHE-Shield edge client."""
    
    # Model Configuration
    model_type: str = "pi0"               # pi0 | openvla | rt2 | custom
    model_path: Optional[str] = None      # Path to local model weights
    
    # Security
    key_path: str = ""                    # Path to encryption key (PEM)
    security_level: int = 128             # 128 | 192 (post-quantum bits)
    
    # Network
    cloud_endpoint: str = "https://api.aphe-shield.ai"
    use_tor: bool = False                 # Enable Tor routing
    timeout_seconds: int = 30
    
    # Performance
    compression_ratio: int = 32           # 8-128 (higher = smaller)
    sparsity: float = 0.01                # Top-K ratio (0.01 = top 1%)
    batch_size: int = 8                   # Demos before upload
    
    # Privacy
    dp_epsilon: float = 1.0               # Differential privacy budget
    dp_delta: float = 1e-5                # DP delta parameter
    max_gradient_norm: float = 1.0        # Gradient clipping norm


@dataclass
class Demonstration:
    """Single demonstration for fine-tuning."""
    
    observations: List[np.ndarray]        # Camera frames [T, H, W, C]
    actions: List[np.ndarray]             # Robot actions [T, action_dim]
    task_id: Optional[str] = None
    episode_id: Optional[str] = None
    timestamp: Optional[float] = field(default_factory=time.time)
    contains_pii: bool = False
    
    def __post_init__(self):
        if len(self.observations) != len(self.actions):
            raise ValueError("Observations and actions must have same length")


@dataclass 
class SubmissionReceipt:
    """Receipt for submitted encrypted update."""
    
    submission_id: str
    encrypted_size_bytes: int
    compression_achieved: float
    estimated_aggregation: datetime
    privacy_budget_used: float


@dataclass
class ClientStatus:
    """Current status of the edge client."""
    
    pending_submissions: int
    total_submissions: int
    privacy_budget_remaining: float
    last_model_version: str
    connection_status: str
