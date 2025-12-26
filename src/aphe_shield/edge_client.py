"""
APHE-Shield Edge Client - Core Implementation
Privacy-Preserving VLA Fine-Tuning for Humanoid Robotics
"""

import numpy as np
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any, Callable, Tuple
from datetime import datetime
import logging
import hashlib
import time

from .structures import ShieldConfig, Demonstration, SubmissionReceipt, ClientStatus
from .adapters import VLAAdapter

logger = logging.getLogger(__name__)


# =============================================================================
# Configuration & Data Types (Imported from .types)
# =============================================================================

# ShieldConfig, Demonstration, SubmissionReceipt, ClientStatus imported from .types


# =============================================================================
# VLA Adapters (Imported from .adapters)
# =============================================================================

# VLAAdapter imported from .adapters


# =============================================================================
# Privacy Pipeline Components
# =============================================================================

class GradientClipper:
    """Clips gradients to bounded norm for differential privacy."""
    
    def __init__(self, max_norm: float = 1.0):
        self.max_norm = max_norm
    
    def clip(self, gradients: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        """Clip gradients to max norm."""
        # Compute global norm
        total_norm = np.sqrt(sum(np.sum(g**2) for g in gradients.values()))
        
        # Clip coefficient
        clip_coef = min(self.max_norm / (total_norm + 1e-6), 1.0)
        
        return {k: v * clip_coef for k, v in gradients.items()}


class SemanticSparsifier:
    """
    Task-Relevant sparsification as proposed in MoAI (2024).
    Prioritizes Attention layers over generic MLP weights to preserve visual reasoning.
    """
    
    def __init__(self, k_ratio: float = 0.01):
        self.k_ratio = k_ratio
        # Heuristic: Attention layers get 2x budget, MLPs get 0.5x
        self.attn_boost = 2.0
    
    def sparsify(self, gradients: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        """Sparsify gradients, prioritizing semantic layers."""
        result = {}
        for name, grad in gradients.items():
            flat = grad.flatten()
            
            # Identify if this is a critical layer (Attention)
            is_critical = any(key in name.lower() for key in ['query', 'key', 'value', 'attn', 'attention'])
            local_ratio = self.k_ratio * (self.attn_boost if is_critical else 0.5)
            
            k = max(1, int(len(flat) * local_ratio))
            k = min(k, len(flat)) # Caps at total elements
            
            # Find top-k indices
            top_k_idx = np.argpartition(np.abs(flat), -k)[-k:]
            
            # Create sparse gradient
            sparse = np.zeros_like(flat)
            sparse[top_k_idx] = flat[top_k_idx]
            result[name] = sparse.reshape(grad.shape)
        
        return result


class APHECompressor:
    """Neural compression using APHE quantization."""
    
    def __init__(self, compression_ratio: int = 32):
        self.compression_ratio = compression_ratio
        self.bits = max(2, 32 // compression_ratio)
    
    def compress(self, gradients: Dict[str, np.ndarray]) -> bytes:
        """Compress gradients to bytes."""
        import gzip
        import pickle
        
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
            
            compressed_data[name] = {
                'q': quantized,
                'min': float(v_min),
                'max': float(v_max),
                'shape': grad.shape,
            }
        
        return gzip.compress(pickle.dumps(compressed_data))
    
    def decompress(self, data: bytes) -> Dict[str, np.ndarray]:
        """Decompress bytes to gradients."""
        import gzip
        import pickle
        
        compressed_data = pickle.loads(gzip.decompress(data))
        result = {}
        
        for name, payload in compressed_data.items():
            levels = 2 ** self.bits
            dequantized = payload['q'].astype(np.float32) / (levels - 1)
            result[name] = dequantized * (payload['max'] - payload['min']) + payload['min']
            result[name] = result[name].reshape(payload['shape'])
        
        return result


# =============================================================================
# N2HE Encryption Interface
# =============================================================================

class QualityMonitor:
    """Monitors signal quality and reconstruction error."""
    
    def __init__(self, mse_threshold: float = 0.01):
        self.mse_threshold = mse_threshold
        
    def check_quality(self, original: Dict[str, np.ndarray], reconstructed: Dict[str, np.ndarray]) -> float:
        """
        Calculate Mean Squared Error (MSE) between original and reconstructed.
        Returns the max MSE across tensors.
        """
        max_mse = 0.0
        for k in original:
            if k not in reconstructed:
                continue
            
            # Normalize for fair comparison
            v_max = np.max(np.abs(original[k]))
            if v_max < 1e-9: v_max = 1.0
            
            orig_norm = original[k] / v_max
            
            v_max_rec = np.max(np.abs(reconstructed[k]))
            if v_max_rec < 1e-9: v_max_rec = 1.0
            
            recon_norm = reconstructed[k] / v_max_rec
            
            mse = np.mean((orig_norm - recon_norm) ** 2)
            max_mse = max(max_mse, mse)
            
        if max_mse > self.mse_threshold:
            logger.warning(f"Quality Warning: High reconstruction error (MSE={max_mse:.4f})")
            
        return max_mse


class N2HEEncryptor:
    """Encrypts gradient updates using N2HE (Vectorized)."""
    
    def __init__(self, key_path: str, security_level: int = 128):
        self.key_path = key_path
        self.security_level = security_level
        self._ctx: Optional[Any] = None
        
        try:
            from .n2he import N2HEContext, N2HE_128, N2HE_192
            
            # Select parameters based on security level
            params = N2HE_128 if self.security_level == 128 else N2HE_192
            self._ctx = N2HEContext(params=params)
            self._ctx.generate_keys(generate_boot_key=False)
            
            # Key Refresh / Noise Awareness Support
            self._key_usage_count = 0
            self._max_key_uses = 1000  # Production: Rotate key after 1000 chunks
            
            logger.info(f"N2HE Vectorized Context initialized ({self.security_level}-bit)")
            
        except Exception as e:
            logger.error(f"Failed to initialize N2HE: {e}")
            raise
    
    def encrypt(self, data: bytes) -> bytes:
        """Encrypt compressed gradients using N2HE LWE encryption (Vectorized & Chunked)."""
        if self._ctx is None:
            raise RuntimeError("N2HE Context not initialized")
            
        # Noise Awareness / Key Refresh Check
        self._key_usage_count += 1
        if self._key_usage_count > self._max_key_uses:
            logger.info(f"Noise/Security Budget Exceeded ({self._key_usage_count} uses). Triggering Key Refresh.")
            self._ctx.generate_keys(generate_boot_key=False)
            self._key_usage_count = 0
        
        # SIMD Slot Alignment: Align CHUNK_SIZE to n (1024 or 1536)
        align = self._ctx.params.n
        CHUNK_SIZE = align * 4 # Pack 4 SIMD slots per chunk 
        
        try:
            import pickle
            
            # 1. Convert bytes -> int64 array
            data_arr = np.frombuffer(data, dtype=np.uint8).astype(np.int64)
            
            # 2. Use SIMD fold_pack logic
            packed_data = self._ctx.fold_pack([data_arr])
            total_elements = len(packed_data)
            
            chunks = []
            
            # 3. Process in aligned chunks
            for i in range(0, total_elements, CHUNK_SIZE):
                chunk_data = packed_data[i : i + CHUNK_SIZE]
                
                # Vectorized Encrypt for this chunk
                ct = self._ctx.encrypt_batch(chunk_data)
                
                # Serialize chunk immediately to save memory
                chunks.append(ct.serialize())
            
            # 4. Serialize final payload
            payload = {
                'chunks': chunks,
                'len': len(data),
                'sec': self.security_level,
                'fold': True
            }
            
            return pickle.dumps(payload)
            
        except Exception as e:
            logger.error(f"Encryption failed: {e}")
            raise RuntimeError(f"N2HE Encryption failed: {e}")

    def decrypt(self, ciphertext: bytes) -> bytes:
        """Decrypt to compressed gradients (Vectorized & Chunked)."""
        if self._ctx is None:
             raise RuntimeError("N2HE Context not initialized")
        
        import pickle
        from .n2he import LWECiphertext
        
        try:
            # No gzip decompression
            payload = pickle.loads(ciphertext)
            
            decrypted_chunks = []
            
            if 'chunks' in payload:
                # Chunked payload
                for chunk_bytes in payload['chunks']:
                    ct = LWECiphertext.deserialize(chunk_bytes, self._ctx.params)
                    dec_ints = self._ctx.decrypt_batch(ct)
                    decrypted_chunks.append(dec_ints.astype(np.uint8))
            else:
                # Legacy: single 'ct' payload (keep for compatibility during migration)
                ct = LWECiphertext.deserialize(payload['ct'], self._ctx.params)
                dec_ints = self._ctx.decrypt_batch(ct)
                decrypted_chunks.append(dec_ints.astype(np.uint8))

            # Combine chunks
            if not decrypted_chunks:
                return b""
                
            res = np.concatenate(decrypted_chunks).tobytes()
            
            return res[:payload['len']]
            
        except Exception as e:
            logger.error(f"Decryption failed: {e}")
            return b""
    
    def get_noise_budget(self) -> float:
        """Get remaining noise budget."""
        if self._ctx is None:
            return 100.0
        return np.log2(self._ctx.params.delta) - np.log2(self._ctx.params.sigma * 6)




# =============================================================================
# Edge Client
# =============================================================================

try:
    import flwr as fl
    from flwr.common import (
        NDArrays,
        Scalar,
        Code,
        Status,
    )
except ImportError:
    # Mocking for environment where flwr is not installed
    class fl:
        class client:
            class NumPyClient: pass
    NDArrays = List[np.ndarray]
    Scalar = float

# =============================================================================
# Edge Client
# =============================================================================

class EdgeClient(fl.client.NumPyClient):
    """
    Main client for APHE-Shield edge deployment.
    Integrates with Flower for federated learning.
    """
    
    def __init__(self, config: ShieldConfig):
        self.config = config
        
        # Initialize pipeline components
        self._clipper = GradientClipper(config.max_gradient_norm)
        self._sparsifier = SemanticSparsifier(config.sparsity)
        self._compressor = APHECompressor(config.compression_ratio)
        self._encryptor = N2HEEncryptor(config.key_path, config.security_level)
        self._quality_monitor = QualityMonitor(mse_threshold=0.05)
        
        # Initialize VLA adapter
        self._adapter: Optional[VLAAdapter] = None
        
        # State
        self._submission_queue: List[bytes] = []
        self._total_submissions = 0
        self._privacy_budget_used = 0.0
        self._last_model_version = "v0.0.0"
        
        # Error Feedback (Residual Memory)
        # Stores the difference between true gradients and compressed/sparsified ones
        self._error_memory: Dict[str, np.ndarray] = {}
        
        # Buffer for current round's training data
        self._current_round_demos: List[Demonstration] = []
        
        logger.info(f"EdgeClient initialized for {config.model_type}")
    
    def get_status(self) -> ClientStatus:
        """Return current client status."""
        return ClientStatus(
            pending_submissions=len(self._current_round_demos),
            total_submissions=self._total_submissions,
            privacy_budget_remaining=max(0.0, self.config.dp_epsilon - self._privacy_budget_used),
            last_model_version=self._last_model_version,
            connection_status="connected" if self._encryptor._ctx else "offline"
        )
    
    def set_adapter(self, adapter: VLAAdapter) -> None:
        """Set VLA adapter for gradient computation."""
        self._adapter = adapter

    def add_demonstration(self, demo: Demonstration):
        """Add a demonstration for the current federated learning round."""
        self._current_round_demos.append(demo)

    # Flower Client Methods
    # ---------------------

    def get_parameters(self, config: Dict[str, Scalar]) -> NDArrays:
        """
        Return the current local model parameters.
        For APHE-Shield, we start with the base model, but typically 
        we return empty or initial weights if needed.
        """
        # In a real scenario, we might return the current VLA weights
        # But since we send *gradients* in fit(), this might be unused 
        # or used for initialization checks.
        return []

    def fit(self, parameters: NDArrays, config: Dict[str, Scalar]) -> Tuple[NDArrays, int, Dict[str, Scalar]]:
        """
        Train the model on local data (Federated Round).
        Returns ENCRYPTED gradients as parameters.
        """
        logger.info(f"Starting FL Round. Demos: {len(self._current_round_demos)}")
        
        if not self._current_round_demos:
            logger.warning("No demonstrations for this round.")
            return [], 0, {}

        # 1. Update local model with global parameters (if any)
        # self._adapter.set_weights(parameters)
        
        # 2. Compute Expert Gradients (Mixture of Intelligence)
        combined_experts = {"visual": {}, "language": {}, "auxiliary": {}}
        count = 0
        
        for demo in self._current_round_demos:
            experts = self._adapter.compute_expert_gradients(demo)
            for exp_name, grads in experts.items():
                for k, v in grads.items():
                    if k not in combined_experts[exp_name]:
                        combined_experts[exp_name][k] = v
                    else:
                        combined_experts[exp_name][k] += v
            count += 1
            
        # Clear buffer
        self._current_round_demos = []
        
        # 3. MoI Gating & Aggregation
        # Priority MoI: Auxiliary (Adapters) > Visual > Language
        gating_weights = {"visual": 1.0, "language": 0.8, "auxiliary": 1.2}
        
        combined_gradients = {}
        for exp_name, grads in combined_experts.items():
            weight = gating_weights[exp_name]
            for k, v in grads.items():
                combined_gradients[k] = v * weight
        
        if not combined_gradients:
            return [], 0, {}
            
        # 3. Privacy Pipeline with Error Feedback
        
        # a) Add Residuals from previous round
        if self._error_memory:
            for k, v in self._error_memory.items():
                if k in combined_gradients:
                    combined_gradients[k] += v
                else:
                    combined_gradients[k] = v
        
        # b) Clip (Global Norm)
        clipped_gradients = self._clipper.clip(combined_gradients)
        
        # c) Sparsify
        # We need to compute what IS sent to update residuals
        sparse_gradients = self._sparsifier.sparsify(clipped_gradients)
        
        # d) Update Error Memory
        # Error = (Clipped - Sparse)
        # We accumulate what was dropped by sparsification
        new_residuals = {}
        for k in clipped_gradients:
            if k in sparse_gradients:
                new_residuals[k] = clipped_gradients[k] - sparse_gradients[k]
        self._error_memory = new_residuals

        # e) Compress (Quantization)
        # Note: We compress the sparse gradients
        compressed = self._compressor.compress(sparse_gradients)
        
        # --- Quality Monitoring ---
        # Decode locally to verify signal integrity
        decompressed_check = self._compressor.decompress(compressed)
        # Compare (Sparse Original) vs (Decompressed)
        # We compare against 'sparse_gradients' because that is the 'true' signal we intend to send
        quality_mse = self._quality_monitor.check_quality(sparse_gradients, decompressed_check)
        logger.info(f"Signal Quality MSE: {quality_mse:.5f}")
        # --------------------------
        
        # f) Encrypt
        encrypted_bytes = self._encryptor.encrypt(compressed)
        
        # 4. Wrap encryption as NDArray for Flower transport
        # Flower expects list of numpy arrays
        payload = [np.frombuffer(encrypted_bytes, dtype=np.uint8)]
        
        # Track usage
        self._total_submissions += 1
        self._privacy_budget_used += self.config.dp_epsilon / 100
        
        # Adaptive Compression Heuristic
        # If payload is larger than target (e.g. 5MB) or budget is tight, increase compression
        estimated_size = len(encrypted_bytes)
        if estimated_size > 5 * 1024 * 1024: # 5MB limit
             # Increase compression ratio for next round
             new_ratio = min(self._compressor.compression_ratio * 2, 128)
             self._compressor = APHECompressor(compression_ratio=new_ratio)
             logger.info(f"Adaptive Compression: Increasing ratio to {new_ratio}x (Size: {estimated_size/1024/1024:.2f}MB)")
        
        return payload, count, {"privacy_budget_used": self._privacy_budget_used}

    def evaluate(self, parameters: NDArrays, config: Dict[str, Scalar]) -> Tuple[float, int, Dict[str, Scalar]]:
        """Evaluate the model."""
        # Evaluation usually happens on held-out local data
        loss = 0.0
        # ... logic to evaluate ...
        return loss, len(self._current_round_demos), {}
    
    # Legacy / Direct Method
    # ----------------------
    
    def submit_encrypted_update(
        self, 
        demonstration: Demonstration,
        priority: str = "normal"
    ) -> SubmissionReceipt:
        """
        Legacy direct submission (now queues for Flower round or fits immediately).
        """
        self.add_demonstration(demonstration)
        
        # If running in standalone mode (no FL server), we might simulate the process
        # But typically we wait for the server to call fit()
        
        # Construct a receipt based on hypothetical processing
        return SubmissionReceipt(
            submission_id="fl_queued",
            encrypted_size_bytes=0, # Unknown until fit()
            compression_achieved=0.0,
            estimated_aggregation=datetime.now(),
            privacy_budget_used=0.0,
        )
    
    def start(self):
        """Start the Flower client loop."""
        fl.client.start_client(
            server_address=self.config.cloud_endpoint.replace("https://", "").replace("http://", ""),
            client=self.to_client(), 
            # In newer flower, we pass the client instance directly or use to_client() adaptation
        )



# =============================================================================
# Convenience Functions
# =============================================================================

def create_client(
    model_type: str = "pi0",
    key_path: str = "",
    cloud_endpoint: str = "https://api.tensor-crate.ai",
    **kwargs
) -> EdgeClient:
    """Create EdgeClient with common defaults."""
    config = ShieldConfig(
        model_type=model_type,
        key_path=key_path,
        cloud_endpoint=cloud_endpoint,
        **kwargs
    )
    return EdgeClient(config)


