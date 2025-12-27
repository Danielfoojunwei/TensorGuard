import pytest
import numpy as np
from tensorguard.core.crypto import N2HEContext, LWECiphertext

def test_crypto_key_generation():
    ctx = N2HEContext()
    ctx.generate_keys()
    assert ctx.lwe_key is not None
    assert len(ctx.lwe_key) == ctx.params.n

def test_crypto_encryption_decryption():
    ctx = N2HEContext()
    ctx.generate_keys()
    
    messages = np.array([1, 2, 3, 4, 42], dtype=np.int64)
    ct = ctx.encrypt_batch(messages)
    
    assert ct.is_batch
    decoded = ctx.decrypt_batch(ct)
    
    np.testing.assert_array_equal(messages, decoded)

def test_ciphertext_serialization():
    ctx = N2HEContext()
    ctx.generate_keys()
    
    messages = np.array([100, 200, 300], dtype=np.int64)
    ct = ctx.encrypt_batch(messages)
    
    data = ct.serialize()
    assert isinstance(data, bytes)
    
    ct_new = LWECiphertext.deserialize(data, params=ctx.params)
    decoded = ctx.decrypt_batch(ct_new)
    
    np.testing.assert_array_equal(messages, decoded)
