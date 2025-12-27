import pytest
import numpy as np
from tensorguard.core.pipeline import GradientClipper, SemanticSparsifier, APHECompressor

def test_gradient_clipper():
    clipper = GradientClipper(max_norm=1.0)
    grads = {"w1": np.array([10.0, 10.0]), "w2": np.array([10.0, 10.0])}
    
    clipped = clipper.clip(grads)
    total_norm = np.sqrt(sum(np.sum(g**2) for g in clipped.values()))
    
    assert total_norm <= 1.0001 # Floating point tolerance

def test_semantic_sparsifier():
    sparsifier = SemanticSparsifier(k_ratio=0.1)
    grads = {
        "attention_query": np.random.randn(100),
        "mlp_layer": np.random.randn(100)
    }
    
    sparse = sparsifier.sparsify(grads)
    
    # Check that attention layer has more elements non-zero than mlp (due to boost)
    attn_nz = np.count_nonzero(sparse["attention_query"])
    mlp_nz = np.count_nonzero(sparse["mlp_layer"])
    
    assert attn_nz > mlp_nz

def test_aphe_compressor():
    compressor = APHECompressor(compression_ratio=32)
    grads = {"w1": np.random.randn(10, 10).astype(np.float32)}
    
    compressed = compressor.compress(grads)
    assert isinstance(compressed, bytes)
    
    decompressed = compressor.decompress(compressed)
    assert "w1" in decompressed
    assert decompressed["w1"].shape == (10, 10)
