"""
TensorGuard v2.0 Integrity Stress Test
Verifies end-to-end data routes and reassembly.
"""

import numpy as np
import time
from typing import List, Tuple
from ..core.client import EdgeClient
from ..core.adapters import MoEAdapter
from ..server.aggregator import ExpertDrivenStrategy
from ..api.schemas import ShieldConfig, Demonstration
from ..utils.logging import get_logger

logger = get_logger("stress_test")

def run_integrity_test():
    logger.info("Starting V2.0 Integrity Stress Test")
    
    # 1. Setup Client
    config = ShieldConfig(model_type="pi0", dp_epsilon=1.0)
    client = EdgeClient(config, cid="stresser_1")
    adapter = MoEAdapter()
    client.set_adapter(adapter)
    
    # 2. Add Demo with specific task for MoI check
    demo = Demonstration(
        observations=[np.zeros((64,64,3))],
        actions=[np.zeros(7)],
        task_id="analyze depth and lighting shadows" # Should trigger visual_aux
    )
    client.add_demonstration(demo)
    
    # 3. Process Round (Client -> Bytes)
    logger.info("Client: Processing round...")
    package_bytes = client.process_round()
    assert package_bytes is not None, "process_round returned None"
    logger.info(f"Client produced package: {len(package_bytes)} bytes")
    
    # 4. Simulate Flower/Network Chunking
    # EdgeClient.fit returns List[np.ndarray] (chunks)
    chunk_size = 100 * 1024 # 100KB chunks
    payload_array = np.frombuffer(package_bytes, dtype=np.uint8)
    chunks = []
    for i in range(0, len(payload_array), chunk_size):
        chunks.append(payload_array[i:i + chunk_size].copy())
    
    logger.info(f"Simulated {len(chunks)} network chunks")
    
    # 5. Setup Server Strategy
    strategy = ExpertDrivenStrategy(quorum_threshold=1) # Single client quorum for test
    
    # Mock FitRes
    from flwr.common import FitRes, Parameters, Status, Code, ndarrays_to_parameters
    class MockClientProxy:
        def __init__(self, cid): self.cid = cid
    
    proxy = MockClientProxy("stresser_1")
    fit_res = FitRes(
        status=Status(Code.OK, "Success"),
        parameters=ndarrays_to_parameters(chunks),
        num_examples=1,
        metrics={}
    )
    
    # 6. Server: Aggregate (Bytes -> UpdatePackage -> EDA)
    logger.info("Server: Reassembling and aggregating...")
    parameters, metrics = strategy.aggregate_fit(1, [(proxy, fit_res)], [])
    
    # 7. Verifications
    assert parameters is not None, "Server failed to aggregate"
    assert "expert_weights" in metrics, "Server failed to produce expert weights"
    
    weights = metrics["expert_weights"]
    logger.info(f"Final Expert Weights: {weights}")
    
    # Check if 'visual_aux' is high as expected
    assert weights.get("visual_aux", 0) > 0.2, f"Expert routing failed: {weights}"
    
    logger.info("INTEGRITY TEST PASSED")

if __name__ == "__main__":
    run_integrity_test()
