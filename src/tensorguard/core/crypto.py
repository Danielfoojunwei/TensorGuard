"""
TensorGuard Cryptography Module (N2HE)

Integrated into TensorGuard for privacy-preserving VLA fine-tuning.
Based on HintSight Technology's N2HE-hexl library.
"""

import numpy as np
import struct
import logging
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any, Union

from ..utils.config import settings
from ..utils.logging import get_logger
from ..utils.exceptions import CryptographyError

logger = get_logger(__name__)

@dataclass
class N2HEParams:
    """N2HE cryptographic parameters."""
    n: int = settings.LATTICE_DIMENSION
    q: int = 2**32 if settings.SECURITY_LEVEL == 128 else 2**48
    sigma: float = 3.2
    t: int = settings.PLAINTEXT_MODULUS
    security_bits: int = settings.SECURITY_LEVEL
    
    @property
    def delta(self) -> int:
        return self.q // self.t

@dataclass
class LWECiphertext:
    """LWE Ciphertext structure."""
    a: np.ndarray
    b: Union[int, np.ndarray]
    params: N2HEParams = field(default_factory=N2HEParams)
    noise_budget: float = 0.0
    
    def __post_init__(self):
        if self.noise_budget == 0.0:
            self.noise_budget = np.log2(self.params.delta) - np.log2(self.params.sigma * 6)
    
    @property
    def is_batch(self) -> bool:
        return isinstance(self.b, np.ndarray) and self.b.ndim > 0

    def serialize(self) -> bytes:
        """Fast binary serialization."""
        k, n = self.a.shape if self.a.ndim == 2 else (1, self.a.shape[0])
        flags = 0x01 if self.is_batch else 0x00
        header = struct.pack('<4sII B', b'LWE1', k, n, flags)
        a_bytes = self.a.astype(np.int64).tobytes()
        b_bytes = self.b.astype(np.int64).tobytes() if self.is_batch else struct.pack('<q', int(self.b))
        return header + a_bytes + b_bytes

    @classmethod
    def deserialize(cls, data: bytes, params: Optional[N2HEParams] = None) -> 'LWECiphertext':
        """Fast binary deserialization."""
        try:
            magic, k, n, flags = struct.unpack('<4sII B', data[:13])
            if magic != b'LWE1':
                raise CryptographyError("Invalid LWE Ciphertext MAGIC")
                
            params = params or N2HEParams(n=n)
            offset = 13
            a_size = k * n * 8
            a_arr = np.frombuffer(data[offset : offset + a_size], dtype=np.int64)
            if k > 1: a_arr = a_arr.reshape(k, n)
            offset += a_size
            
            if flags & 0x01:
                b_val = np.frombuffer(data[offset : offset + k * 8], dtype=np.int64)
            else:
                b_val = struct.unpack('<q', data[offset : offset + 8])[0]
                
            return cls(a=a_arr, b=b_val, params=params)
        except Exception as e:
            raise CryptographyError(f"Deserialization failed: {e}")

class N2HEContext:
    """N2HE Encryption Context and Operations."""
    def __init__(self, params: Optional[N2HEParams] = None):
        self.params = params or N2HEParams()
        self.lwe_key: Optional[np.ndarray] = None
        self.stats = {'encryptions': 0, 'decryptions': 0}

    def generate_keys(self):
        """Generate secret key."""
        self.lwe_key = np.random.choice([-1, 0, 1], size=self.params.n).astype(np.int64)
        logger.debug("N2HE Keys generated")

    def encrypt_batch(self, messages: np.ndarray) -> LWECiphertext:
        """Vectorized encryption."""
        if self.lwe_key is None: self.generate_keys()
        
        k = messages.shape[0]
        n, q, t = self.params.n, self.params.q, self.params.t
        sigma, delta = self.params.sigma, self.params.delta
        
        m_vec = messages.astype(np.int64) % t
        A = np.random.randint(0, q, size=(k, n), dtype=np.int64)
        E = np.round(np.random.normal(0, sigma, size=k)).astype(np.int64)
        B = (np.dot(A, self.lwe_key) + E + delta * m_vec) % q
        
        self.stats['encryptions'] += k
        return LWECiphertext(a=A, b=B, params=self.params)

    def decrypt_batch(self, ct: LWECiphertext) -> np.ndarray:
        """Vectorized decryption."""
        if self.lwe_key is None: raise CryptographyError("Keys not generated")
        
        A, B = (ct.a.reshape(1, -1), np.array([ct.b])) if not ct.is_batch else (ct.a, ct.b)
        q, delta, t = self.params.q, self.params.delta, self.params.t
        
        m_scaled = (B - np.dot(A, self.lwe_key)) % q
        m_scaled[m_scaled > (q // 2)] -= q
        m = np.round(m_scaled / delta).astype(np.int64) % t
        
        self.stats['decryptions'] += len(m)
        return m

    def fold_pack(self, messages: List[np.ndarray]) -> np.ndarray:
        """SIMD-style tensor packing."""
        flat = np.concatenate([m.flatten() for m in messages])
        pad = (self.params.n - (len(flat) % self.params.n)) % self.params.n
        return np.pad(flat, (0, pad)) if pad > 0 else flat

class N2HEEncryptor:
    """Professional wrapper for N2HE encryption with chunking and key rotation."""
    def __init__(self, key_path: str, security_level: int = 128):
        self.params = N2HEParams(security_bits=security_level)
        self._ctx = N2HEContext(self.params)
        self._usage_count = 0
        self._max_uses = settings.MAX_KEY_USES
        
    def encrypt(self, data: bytes) -> bytes:
        """Encrypt binary data with SIMD folding and chunking."""
        import pickle
        self._usage_count += 1
        if self._usage_count > self._max_uses:
            self._ctx.generate_keys()
            self._usage_count = 0
            
        data_arr = np.frombuffer(data, dtype=np.uint8).astype(np.int64)
        packed = self._ctx.fold_pack([data_arr])
        
        # Chunking for lattice alignment
        chunk_size = self.params.n * 4
        chunks = [self._ctx.encrypt_batch(packed[i : i + chunk_size]).serialize() 
                 for i in range(0, len(packed), chunk_size)]
        
        return pickle.dumps({'chunks': chunks, 'len': len(data)})

    def decrypt(self, ciphertext: bytes) -> bytes:
        """Decrypt chunked ciphertext."""
        import pickle
        payload = pickle.loads(ciphertext)
        dec_chunks = [self._ctx.decrypt_batch(LWECiphertext.deserialize(c, self.params)).astype(np.uint8) 
                     for c in payload['chunks']]
        return np.concatenate(dec_chunks).tobytes()[:payload['len']]
