"""
APHE-Shield VLA Adapters
"""

import logging
import numpy as np
from typing import Dict, Any, Callable
from .structures import Demonstration

logger = logging.getLogger(__name__)

class VLAAdapter:
    """Base adapter for VLA models."""
    
    def __init__(self, model: Any, gradient_fn: Callable, apply_fn: Callable):
        self.model = model
        self._gradient_fn = gradient_fn
        self._apply_fn = apply_fn
    
    def compute_gradients(self, demo: Demonstration) -> Dict[str, np.ndarray]:
        """Compute gradients from demonstration."""
        return self._gradient_fn(self.model, demo)
    
    def apply_update(self, gradients: Dict[str, np.ndarray]) -> None:
        """Apply gradient update to model."""
        self._apply_fn(self.model, gradients)
    
    def compute_expert_gradients(self, demo: Demonstration) -> Dict[str, Dict[str, np.ndarray]]:
        """
        Compute gradients split by 'Expert' category:
        - visual: Vision encoder updates
        - language: Language backbone updates
        - auxiliary: Adapter/Connector updates
        """
        # Default implementation splits by layer name heuristics
        all_grads = self.compute_gradients(demo)
        experts = {"visual": {}, "language": {}, "auxiliary": {}}
        
        for k, v in all_grads.items():
            kl = k.lower()
            if any(x in kl for x in ['vision', 'encoder', 'patch']):
                experts["visual"][k] = v
            elif any(x in kl for x in ['llm', 'language', 'decoder']):
                experts["language"][k] = v
            else:
                experts["auxiliary"][k] = v
        return experts
    
    @classmethod
    def from_pi0(cls, model_path: str) -> "VLAAdapter":
        """Create adapter for Pi0 VLA."""
        from .adapters.pi0_adapter import load_pi0, pi0_gradient, pi0_apply
        model = load_pi0(model_path)
        return cls(model, pi0_gradient, pi0_apply)
    
    @classmethod
    def from_openvla(cls, model_path: str) -> "VLAAdapter":
        """Create adapter for OpenVLA."""
        from .adapters.openvla_adapter import load_openvla, openvla_gradient, openvla_apply
        model = load_openvla(model_path)
        return cls(model, openvla_gradient, openvla_apply)
    
    @classmethod
    def from_rt2(cls, model_path: str) -> "VLAAdapter":
        """Create adapter for RT-2."""
        from .adapters.rt2_adapter import load_rt2, rt2_gradient, rt2_apply
        model = load_rt2(model_path)
        return cls(model, rt2_gradient, rt2_apply)
    
    @classmethod
    def from_custom(cls, model: Any, gradient_fn: Callable, apply_fn: Callable) -> "VLAAdapter":
        """Create adapter for custom VLA architecture."""
        return cls(model, gradient_fn, apply_fn)
