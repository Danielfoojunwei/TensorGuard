"""
TensorGuard v2.0 Outlier Resistance Test
Verifies that malicious or out-of-distribution updates are handled gracefully.
"""

import numpy as np
import hashlib
from ..core.production import UpdatePackage, SafetyStatistics, ClientContribution, ModelTargetMap
from ..server.aggregator import ExpertDrivenStrategy
from datetime import datetime
from ..utils.logging import get_logger

logger = get_logger("outlier_test")

def run_outlier_test():
    logger.info("Starting V2.0 Outlier Resistance Test")
    
    # 1. Setup Strategy
    strategy = ExpertDrivenStrategy(quorum_threshold=3)
    strategy.aggregator.start_round()
    
    # 2. Add 2 Healthy Clients
    for i in range(2):
        pkg = UpdatePackage(
            client_id=f"healthy_{i}",
            safety_stats=SafetyStatistics(grad_norm_max=1.0) # Normal norm
        )
        strategy.aggregator.add_contribution(ClientContribution(
            client_id=f"healthy_{i}",
            update_package=pkg,
            received_at=datetime.utcnow()
        ))
    
    # 3. Add 1 Malicious Outlier (Adversary)
    poisoned_pkg = UpdatePackage(
        client_id="attacker_1",
        safety_stats=SafetyStatistics(grad_norm_max=100.0) # Extreme norm
    )
    strategy.aggregator.add_contribution(ClientContribution(
        client_id="attacker_1",
        update_package=poisoned_pkg,
        received_at=datetime.utcnow()
    ))
    
    # 4. Run Detection
    logger.info("Running MAD Outlier Detection...")
    outliers = strategy.aggregator.detect_outliers()
    weights = strategy.aggregator.get_aggregation_weights()
    
    logger.info(f"Detected Outliers: {outliers}")
    logger.info(f"Aggregation Weights: {weights}")
    
    # 5. Verifications
    assert "attacker_1" in outliers, "Failed to detect outlier"
    assert weights["attacker_1"] < weights["healthy_0"], "Outlier should have lower/zero weight"
    
    # 6. Simulate Health Scoring
    # If a client is an outlier, we should down-weight them future rounds
    strategy.aggregator.update_client_health("attacker_1", 0.1)
    
    logger.info("OUTLIER RESISTANCE TEST PASSED")

if __name__ == "__main__":
    run_outlier_test()
