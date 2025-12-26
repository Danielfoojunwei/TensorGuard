
import sys
import os
import numpy as np
import time

sys.path.append(os.path.join(os.getcwd(), "src"))

from moai_shield.n2he import N2HEContext, N2HE_128

def sanity():
    print("Starting sanity check...")
    try:
        ctx = N2HEContext(params=N2HE_128)
        ctx.generate_keys(generate_boot_key=False)
        print("Context initialized.")
        
        # data = np.array([1, 2, 3, 4, 5], dtype=np.int64)
        data = np.random.randint(0, 256, size=10240, dtype=np.int64)
        print(f"Encrypting {len(data)} items (10KB)...")
        
        start = time.time()
        ct = ctx.encrypt_batch(data)
        print(f"Encryption done in {time.time() - start:.4f}s")
        
        start = time.time()
        dec = ctx.decrypt_batch(ct)
        print(f"Decryption done in {time.time() - start:.4f}s")
        
        print(f"Decrypted: {dec}")
        assert np.all(dec == data)
        print("Sanity PASS")
        
    except Exception as e:
        print(f"Sanity FAIL: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    sanity()
