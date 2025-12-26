
import logging
import numpy as np
from tensorguard.edge_client import QualityMonitor, N2HEEncryptor, EdgeClient, create_client

def test_quality_monitor():
    print("\n--- Testing Quality Monitor ---")
    qm = QualityMonitor(mse_threshold=0.001)
    
    # Clean signal
    original = {"w": np.array([0.1, 0.5, 0.9])}
    
    # 1. Good reconstruction
    recon_good = {"w": np.array([0.11, 0.49, 0.91])}
    mse = qm.check_quality(original, recon_good)
    print(f"Good MSE: {mse:.6f}")
    assert mse < 0.01
    
    # 2. Bad reconstruction
    recon_bad = {"w": np.array([0.9, 0.1, 0.0])}
    mse_bad = qm.check_quality(original, recon_bad)
    print(f"Bad MSE: {mse_bad:.6f}")
    assert mse_bad > 0.01

def test_key_refresh():
    print("\n--- Testing Key Refresh Logic ---")
    # Setup encryptor with low max_uses to trigger refresh quickly
    enc = N2HEEncryptor(key_path="mock", security_level=128)
    enc._max_key_uses = 2 # Refresh after 2 uses
    
    # Capture initial key state (mock check)
    if enc._ctx:
        initial_key = enc._ctx.lwe_key.copy() if enc._ctx.lwe_key is not None else None
    else:
        print("Skipping real key check (mock mode)")
        return

    print("Use 1")
    enc.encrypt(b"data1")
    print("Use 2")
    enc.encrypt(b"data2")
    
    # Next use should trigger refresh
    print("Use 3 (Should Refresh)")
    enc.encrypt(b"data3")
    
    new_key = enc._ctx.lwe_key
    
    # Verify key changed
    assert not np.array_equal(initial_key, new_key)
    print("PASS: Key was refreshed successfully.")

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    try:
        test_quality_monitor()
        test_key_refresh()
        print("\nALL TESTS PASSED")
    except Exception as e:
        print(f"\nTEST FAILED: {e}")
        exit(1)
