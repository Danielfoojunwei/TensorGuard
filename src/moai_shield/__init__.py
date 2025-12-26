"""
APHE-Shield: Privacy-Preserving VLA Fine-Tuning for Humanoid Robotics

Plug-and-play SDK for Robotics System Integrators to fine-tune
Vision-Language-Action models without exposing proprietary data.
"""

__version__ = "1.0.0"
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
]
