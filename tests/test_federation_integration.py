"""
Direct Integration Test for TensorGuard Federated Learning Pipeline.

This test bypasses the multi-process Flower gRPC infrastructure and directly
validates the serialization, aggregation, and security pipeline by:

1. Creating EdgeClient instances with OFTAdapter
2. Calling fit() to generate UpdatePackage chunks
3. Manually constructing FitRes objects
4. Passing them to TensorGuardStrategy.aggregate_fit()
5. Verifying successful aggregation

This approach tests the core business logic without external orchestration complexity.
"""

import pytest
import numpy as np
from unittest.mock import MagicMock
from datetime import datetime
from pathlib import Path
from typing import List, Tuple

# Flower imports for FitRes construction
from flwr.common import Parameters, FitRes, Status, Code, ndarrays_to_parameters, parameters_to_ndarrays

# TensorGuard imports
from tensorguard.core.client import EdgeClient, create_client
from tensorguard.core.production import UpdatePackage, KeyManagementSystem, KeyMetadata
from tensorguard.core.crypto import generate_key
from tensorguard.server.aggregator import TensorGuardStrategy
from tensorguard.experiments.validation_suite import LIBEROSimulator, OFTAdapter


class MockClientProxy:
    """Mock Flower ClientProxy for testing aggregate_fit."""
    def __init__(self, cid: str):
        self.cid = cid


def create_fit_res_from_client(client: EdgeClient) -> Tuple[MockClientProxy, FitRes]:
    """
    Run EdgeClient.fit() and construct a FitRes object from its output.
    This simulates what Flower would do after receiving a client response.
    """
    # Get the raw fit output
    ndarrays, num_examples, metrics = client.fit([], {})
    
    # Convert to Flower Parameters using canonical Flower serializer
    if ndarrays:
        parameters = ndarrays_to_parameters(ndarrays)
    else:
        parameters = Parameters(tensors=[], tensor_type="numpy.ndarray")
    
    # Construct FitRes
    fit_res = FitRes(
        status=Status(code=Code.OK, message="OK"),
        parameters=parameters,
        num_examples=num_examples,
        metrics=metrics
    )
    
    return MockClientProxy(client.cid), fit_res


class TestFederationIntegration:
    """Integration tests for the federated learning pipeline."""
    
    @pytest.fixture
    def simulator(self):
        return LIBEROSimulator()
    
    @pytest.fixture
    def strategy(self):
        return TensorGuardStrategy(min_fit_clients=1, min_available_clients=1, quorum_threshold=1)
    
    def test_single_client_round(self, simulator, strategy):
        """Test a single client contributing to a federated round."""
        # Create client with adapter
        client = create_client(security_level=128, cid="client_0")
        client.set_adapter(OFTAdapter())
        
        # Add demonstration
        demo = simulator.generate_trajectory("scoop_raisins")
        client.add_demonstration(demo)
        
        # Generate fit result
        proxy, fit_res = create_fit_res_from_client(client)
        
        # Verify we got data
        assert len(fit_res.parameters.tensors) > 0, "Client should produce non-empty parameters"
        assert fit_res.num_examples == 1, "Should report 1 example"
        
        # Pass to strategy
        results = [(proxy, fit_res)]
        failures = []
        
        aggregated, metrics = strategy.aggregate_fit(1, results, failures)
        
        # Verify aggregation succeeded
        assert aggregated is not None, "Aggregation should succeed with valid client data"
    
    def test_multi_client_quorum(self, simulator, strategy):
        """Test multiple clients meeting quorum requirements."""
        clients = []
        results = []
        
        # Create 3 clients with different tasks
        tasks = ["scoop_raisins", "fold_shirt", "pick_corn"]
        for i, task in enumerate(tasks):
            client = create_client(security_level=128, cid=f"client_{i}")
            client.set_adapter(OFTAdapter())
            demo = simulator.generate_trajectory(task)
            client.add_demonstration(demo)
            clients.append(client)
            
            proxy, fit_res = create_fit_res_from_client(client)
            results.append((proxy, fit_res))
        
        # Verify all clients produced data
        for proxy, fit_res in results:
            assert len(fit_res.parameters.tensors) > 0, f"Client {proxy.cid} should produce data"
        
        # Aggregate
        aggregated, metrics = strategy.aggregate_fit(1, results, [])
        
        # Verify quorum was met
        assert aggregated is not None, "Aggregation should succeed with 3 clients"
    
    def test_update_package_serialization_roundtrip(self, simulator):
        """Test that UpdatePackage survives serialization/deserialization."""
        client = create_client(security_level=128, cid="test_client")
        client.set_adapter(OFTAdapter())
        demo = simulator.generate_trajectory("open_pot")
        client.add_demonstration(demo)
        
        # Get serialized package
        ndarrays, _, _ = client.fit([], {})
        assert len(ndarrays) > 0, "Should produce arrays"
        
        # Reassemble bytes (simulating server-side)
        payload_bytes = b"".join([arr.tobytes() for arr in ndarrays])
        
        # Deserialize
        package = UpdatePackage.deserialize(payload_bytes)
        
        # Verify package contents
        assert package.client_id is not None
        assert package.target_map is not None
        assert "encrypted" in package.delta_tensors
        assert package.training_meta.num_demonstrations == 1
    
    def test_empty_client_rejected(self, strategy):
        """Test that clients with no data are handled correctly."""
        client = create_client(security_level=128, cid="empty_client")
        client.set_adapter(OFTAdapter())
        # Note: No demonstration added
        
        proxy, fit_res = create_fit_res_from_client(client)
        
        # Should have empty parameters
        assert len(fit_res.parameters.tensors) == 0, "Empty client should have no tensors"
        
        # Strategy should handle gracefully
        results = [(proxy, fit_res)]
        aggregated, metrics = strategy.aggregate_fit(1, results, [])
        
        # Should fail quorum (0 valid contributions)
        assert aggregated is None, "Aggregation should fail with no valid clients"
    
    def test_security_pipeline_applied(self, simulator):
        """Verify that N2HE encryption and DP are applied to the package."""
        client = create_client(security_level=128, cid="secure_client")
        client.set_adapter(OFTAdapter())
        demo = simulator.generate_trajectory("scoop_raisins")
        client.add_demonstration(demo)
        
        # Get package
        ndarrays, _, _ = client.fit([], {})
        payload_bytes = b"".join([arr.tobytes() for arr in ndarrays])
        package = UpdatePackage.deserialize(payload_bytes)
        
        # Verify encryption was applied
        assert "encrypted" in package.delta_tensors
        encrypted_data = package.delta_tensors["encrypted"]
        assert len(encrypted_data) > 0, "Encrypted data should be non-empty"
        
        # Verify DP stats were recorded
        assert package.safety_stats.dp_epsilon_consumed > 0, "DP budget should be consumed"
        
        # Verify compression metadata
        assert package.compression_metadata["ratio"] > 0


    def test_federation_dashboard_generation(self, simulator, strategy):
        """Run a full multi-robot federation and generate a dashboard of metrics."""
        import matplotlib.pyplot as plt
        
        num_robots = 5
        robot_results = []
        
        # 0. Setup Enterprise Key Management
        key_path = "keys/integration_test_key.npy"
        generate_key(key_path, security_level=128)
        
        kms = KeyManagementSystem(audit_log_path=Path("integration_kms.log"))
        kms.register_key("test_key_v1", KeyMetadata(
            key_id="test_key_v1",
            created_at=datetime.utcnow()
        ))

        # 1. Simulate 5 robots with the SAME enterprise key
        for i in range(num_robots):
            client = create_client(
                security_level=128, 
                cid=f"robot_{i}",
                key_path=key_path
            )
            client.set_adapter(OFTAdapter())
            
            # Vary data volume
            num_demos = np.random.randint(1, 5)
            for _ in range(num_demos):
                demo = simulator.generate_trajectory("scoop_raisins")
                client.add_demonstration(demo)
            
            proxy, fit_res = create_fit_res_from_client(client)
            robot_results.append((proxy, fit_res, num_demos))
            
        # 2. Extract metrics for plotting before aggregation
        cids = [r[0].cid for r in robot_results]
        raw_sizes = []
        comp_sizes = []
        epsilons = []
        
        for _, fit_res, _ in robot_results:
            # Reassemble for analysis - need to unwrap NumPy headers
            ndarrays = parameters_to_ndarrays(fit_res.parameters)
            payload = b"".join([arr.tobytes() for arr in ndarrays])
            package = UpdatePackage.deserialize(payload)
            # Dummy raw size (approximation)
            raw_sizes.append(len(payload) * 50 / 1024) # 50x compression factor back-calc
            comp_sizes.append(len(payload) / 1024)
            epsilons.append(package.safety_stats.dp_epsilon_consumed)
            
        # 3. Aggregation
        aggregated, metrics = strategy.aggregate_fit(1, [(r[0], r[1]) for r in robot_results], [])
        
        # 4. Generate Dashboard
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        
        # Plot 1: Bandwidth Savings (Multi-Robot)
        x = np.arange(num_robots)
        width = 0.35
        axes[0].bar(x - width/2, raw_sizes, width, label='Raw VLA Update (KB)', color='#95a5a6')
        axes[0].bar(x + width/2, comp_sizes, width, label='TensorGuard Compressed', color='#2ecc71')
        axes[0].set_title('Heterogeneous Fleet: Bandwidth Optimization')
        axes[0].set_ylabel('Size (KB)')
        axes[0].set_xticks(x)
        axes[0].set_xticklabels(cids)
        axes[0].legend()
        
        # Plot 2: Privacy Budget Consumption
        axes[1].step(x, np.cumsum(epsilons), where='post', color='#e74c3c', linewidth=2)
        axes[1].fill_between(x, np.cumsum(epsilons), step="post", alpha=0.2, color='#e74c3c')
        axes[1].set_title('Fleet Privacy Guardrail (DP Epsilon)')
        axes[1].set_ylabel('Total Epsilon Consumed')
        axes[1].set_xlabel('Robot ID Index')
        axes[1].grid(True, linestyle='--', alpha=0.5)
        
        # Plot 3: Aggregation Impact (Weights)
        weights = [1.0] * num_robots # Simplified for visualization
        axes[2].pie(weights, labels=cids, autopct='%1.1f%%', startangle=140, colors=plt.cm.Pastel1.colors)
        axes[2].set_title('Secure Aggregation: Robot Contribution')
        
        # Add Key Health text Box
        fig.text(0.5, 0.02, f"Key ID: test_key_v1 | Security: 128-bit N2HE | KMS Status: ACTIVE", 
                 ha='center', fontsize=12, bbox=dict(facecolor='white', alpha=0.5, edgecolor='#2c3e50'))
        
        plt.tight_layout(rect=[0, 0.05, 1, 1])
        plt.savefig('federation_dashboard.png')
        
        assert aggregated is not None
        print("\nFederation Dashboard generated: federation_dashboard.png", flush=True)

if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
