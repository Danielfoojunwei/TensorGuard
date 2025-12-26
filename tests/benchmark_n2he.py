
import time
import numpy as np
import sys
import os

# Force unbuffered output
sys.stdout.reconfigure(line_buffering=True)

# Add src to path
sys.path.append(os.path.join(os.getcwd(), "src"))

from moai_shield.n2he import N2HEContext, N2HE_128
from moai_shield.edge_client import N2HEEncryptor

def benchmark():
    print("Initializing N2HE Context (128-bit)...")
    ctx = N2HEContext(params=N2HE_128)
    ctx.generate_keys(generate_boot_key=False)
    
    # payload size: 100KB
    size_bytes = 100 * 1024
    data = np.random.randint(0, 256, size=size_bytes, dtype=np.uint8)
    
    print(f"Benchmarking encryption of {size_bytes/1024:.2f} KB...")
    
    # 1. Direct Context Batch Encryption
    start = time.time()
    ct = ctx.encrypt_batch(data.astype(np.int64))
    enc_time = time.time() - start
    
    print(f"Context.encrypt_batch time: {enc_time:.4f}s")
    print(f"Throughput: {size_bytes / enc_time / 1024 / 1024:.2f} MB/s")
    
    # 2. Decryption
    start = time.time()
    dec = ctx.decrypt_batch(ct)
    dec_time = time.time() - start
    print(f"Context.decrypt_batch time: {dec_time:.4f}s")
    
    # Verify correctness
    is_correct = np.allclose(data, dec.astype(np.uint8))
    print(f"Correctness: {'PASS' if is_correct else 'FAIL'}")
    
    # 3. N2HEEncryptor Wrapper
    print("\nTesting N2HEEncryptor Wrapper...")
    encryptor = N2HEEncryptor(key_path="test", security_level=128)
    # The wrapper re-initializes context, might take a moment
    
    data_bytes = data.tobytes()
    start = time.time()
    enc_bytes = encryptor.encrypt(data_bytes)
    wrapper_enc_time = time.time() - start
    print(f"Wrapper Encrypt time: {wrapper_enc_time:.4f}s")
    
    start = time.time()
    dec_bytes = encryptor.decrypt(enc_bytes)
    wrapper_dec_time = time.time() - start
    print(f"Wrapper Decrypt time: {wrapper_dec_time:.4f}s")
    
    is_correct_wrapper = (data_bytes == dec_bytes)
    print(f"Wrapper Correctness: {'PASS' if is_correct_wrapper else 'FAIL'}")

if __name__ == "__main__":
    benchmark()
