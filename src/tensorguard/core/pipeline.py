"""
TensorGuard Privacy Pipeline Components
Implementing Differential Privacy, Sparsification, and Compression.
"""

import numpy as np
import gzip
import pickle
from typing import Dict, Any, Optional
from ..utils.logging import get_logger
from ..utils.exceptions import QualityWarning, ValidationError

logger = get_logger(__name__)

class GradientClipper:
    """Clips gradients to bounded norm for differential privacy."""
    def __init__(self, max_norm: float = 1.0):
        self.max_norm = max_norm
    
    def clip(self, gradients: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        total_norm = np.sqrt(sum(np.sum(g**2) for g in gradients.values()))
        clip_coef = min(self.max_norm / (total_norm + 1e-6), 1.0)
        return {k: v * clip_coef for k, v in gradients.items()}

class ExpertGater:
    """
    Expert-Driven Gating (v2.0).
    Based on instruction relevance (IOSP) instead of raw magnitude.
    Addresses parameter interference in heterogeneous robot fleets.
    """
    def __init__(self, gate_threshold: float = 0.1):
        self.gate_threshold = gate_threshold

    def gate(self, expert_grads: Dict[str, Dict[str, np.ndarray]], gate_weights: Dict[str, float]) -> Dict[str, np.ndarray]:
        combined = {}
        if not expert_grads or not gate_weights:
            return combined
            
        for expert, grads in expert_grads.items():
            weight = gate_weights.get(expert, 0.0)
            if weight > self.gate_threshold:
                for k, v in grads.items():
                    combined[k] = combined.get(k, 0) + v
        return combined

class ThresholdSparsifier:
    """
    Threshold-based Sparsification.
    Maintains O(c) error accumulation (Canini et al., 2021), 
    avoiding the O(c^4) blowup of Top-K.
    """
    def __init__(self, threshold: float = 0.001):
        self.threshold = threshold
    
    def sparsify(self, gradients: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        result = {}
        for name, grad in gradients.items():
            # Keep only gradients exceeding the learnable/fixed threshold
            mask = np.abs(grad) > self.threshold
            sparse = np.zeros_like(grad)
            sparse[mask] = grad[mask]
            result[name] = sparse
        return result

class LegacyMagnitudeSparsifier:
    """
    [LEGACY] Top-K Sparsification.
    Deprecated in v2.0 due to O(c^4) error accumulation and index-leakage risks.
    """
    def __init__(self, k_ratio: float = 0.01):
        self.k_ratio = k_ratio
        self.attn_boost = 2.0
    
    def sparsify(self, gradients: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        # Implementation remains for backward compatibility
        result = {}
        for name, grad in gradients.items():
            flat = grad.flatten()
            is_critical = any(key in name.lower() for key in ['query', 'key', 'value', 'attn', 'attention'])
            local_ratio = self.k_ratio * (self.attn_boost if is_critical else 0.5)
            k = max(1, int(len(flat) * local_ratio))
            top_k_idx = np.argpartition(np.abs(flat), -k)[-k:]
            sparse = np.zeros_like(flat)
            sparse[top_k_idx] = flat[top_k_idx]
            result[name] = sparse.reshape(grad.shape)
        return result

class APHECompressor:
    """Neural compression using quantization."""
    def __init__(self, compression_ratio: int = 32):
        self.compression_ratio = compression_ratio
        self.bits = max(2, 32 // compression_ratio)
    
    def compress(self, gradients: Dict[str, np.ndarray]) -> bytes:
        compressed_data = {}
        for name, grad in gradients.items():
            v_min, v_max = grad.min(), grad.max()
            if v_max - v_min > 1e-8:
                normalized = (grad - v_min) / (v_max - v_min)
                levels = 2 ** self.bits
                quantized = np.round(normalized * (levels - 1)).astype(np.uint8)
            else:
                quantized = np.zeros_like(grad, dtype=np.uint8)
                v_min = v_max = 0.0
            
            compressed_data[name] = {'q': quantized, 'min': float(v_min), 'max': float(v_max), 'shape': grad.shape}
        return gzip.compress(pickle.dumps(compressed_data))
    
    def decompress(self, data: bytes) -> Dict[str, np.ndarray]:
        compressed_data = pickle.loads(gzip.decompress(data))
        result = {}
        levels = 2 ** self.bits
        for name, payload in compressed_data.items():
            dequantized = payload['q'].astype(np.float32) / (levels - 1)
            result[name] = (dequantized * (payload['max'] - payload['min']) + payload['min']).reshape(payload['shape'])
        return result

class QualityMonitor:
    """Monitors reconstruction integrity."""
    def __init__(self, mse_threshold: float = 0.05):
        self.mse_threshold = mse_threshold
        
    def check_quality(self, original: Dict[str, np.ndarray], reconstructed: Dict[str, np.ndarray]) -> float:
        max_mse = 0.0
        for k in original:
            orig_norm = original[k] / (np.max(np.abs(original[k])) + 1e-9)
            recon_norm = reconstructed[k] / (np.max(np.abs(reconstructed[k])) + 1e-9)
            mse = np.mean((orig_norm - recon_norm) ** 2)
            max_mse = max(max_mse, mse)
            
        if max_mse > self.mse_threshold:
            logger.info(f"High reconstruction error (MSE={max_mse:.4f})", extra={"mse": max_mse})
        return max_mse
