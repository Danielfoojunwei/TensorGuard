
import numpy as np
import pytest
from moai_shield.edge_client import TopKSparsifier

def test_top_k_sparsification():
    """Verify Top-K sparsifier zeroes out 99% of values."""
    print("Testing Top-K Sparsification...")
    
    # 1. Create dense gradients (10,000 parameters)
    # Use distinct values to verify we keep the largest MAGNITUDES
    grads = {
        "layer1": np.linspace(-100, 100, 10000).reshape(100, 100)
    }
    
    # 2. Initialize Sparsifier (Keep 1%)
    k_ratio = 0.01
    sparsifier = TopKSparsifier(k_ratio=k_ratio)
    
    # 3. Sparsify
    sparse_grads = sparsifier.sparsify(grads)
    
    # 4. Verify
    flat_res = sparse_grads["layer1"].flatten()
    non_zeros = np.count_nonzero(flat_res)
    expected_non_zeros = int(10000 * k_ratio)
    
    print(f"Total Params: 10000")
    print(f"Non-Zeros: {non_zeros} (Expected: {expected_non_zeros})")
    print(f"Sparsity: {1.0 - non_zeros/10000:.4%}")
    
    # Check simple count
    assert non_zeros == expected_non_zeros
    
    # Check value retention (largest magnitudes should remain)
    # The original data included 100 and -100.
    # Top 1% of 10000 is 100 items.
    # The linspace covers [-100, ..., 100].
    # Largest magnitudes are near -100 and 100.
    assert np.max(flat_res) > 99.0
    assert np.min(flat_res) < -99.0
    
    # Check that near-zero values were zeroed
    # Midpoint of linspace is 0.
    mid_idx = 5000
    # Original value at mid was near 0. Result should be exactly 0.
    assert flat_res[mid_idx] == 0.0
    
    print("PASS: Top-K Sparsification works correctly.")

if __name__ == "__main__":
    test_top_k_sparsification()
