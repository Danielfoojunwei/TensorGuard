"""
TensorGuard: Privacy-Preserving VLA Fine-Tuning SDK
"""

__version__ = "1.1.0"
__author__ = "Daniel Foo"

from .core.client import EdgeClient, create_client
from .api.schemas import ShieldConfig, Demonstration, SubmissionReceipt, ClientStatus
from .core.adapters import VLAAdapter
from .utils.config import settings

__all__ = [
    "EdgeClient",
    "create_client",
    "ShieldConfig",
    "Demonstration",
    "SubmissionReceipt",
    "ClientStatus",
    "VLAAdapter",
    "settings",
]
