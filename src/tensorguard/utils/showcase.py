"""
TensorGuard Showcase & Simulation Utilities (v2.0 FedMoE)
"""

import time
import threading
import numpy as np
import os
from typing import Optional

from ..core.client import EdgeClient
from ..core.adapters import VLAAdapter, MoEAdapter
from ..api.schemas import ShieldConfig, Demonstration
from ..server.dashboard import run_dashboard as _run_server_dashboard
from ..utils.logging import get_logger
from ..utils.config import settings

logger = get_logger("showcase")

class ShowcaseMoEAdapter(MoEAdapter):
    """v2.0 Mock adapter for dashboard visualization with Task Heterogeneity."""
    def __init__(self):
        super().__init__()
        self._tasks = [
            "pick up the blue geometric block",
            "analyze texture and shadows of the object",
            "follow verbal instructions to find goal intent",
            "apply force to maintain gripper contact"
        ]
        self._task_idx = 0

    def get_next_task(self) -> str:
        t = self._tasks[self._task_idx]
        self._task_idx = (self._task_idx + 1) % len(self._tasks)
        return t

    def compute_expert_gradients(self, demo: Demonstration):
        """Mock gradient computation simulating specialized experts."""
        # Simulate some meaningful delay for 'computation'
        time.sleep(0.05)
        
        # Return mock gradients structure consistent with MoE
        return {
            "visual": {"vision_encoder.weight": np.random.randn(10, 10).astype(np.float32)},
            "language": {"llm_backbone.attn_q.weight": np.random.randn(10, 10).astype(np.float32)},
            "auxiliary": {"adapter_connector.weight": np.random.randn(10, 10).astype(np.float32)}
        }

def start_showcase_simulation(client: EdgeClient):
    """Runs a background loop submitting mock demonstrations to simulate a live fleet."""
    def simulation_loop():
        logger.info("Starting FedMoE Showcase Simulation Loop")
        adapter = client._adapter
        
        # Ensure we have the right adapter type for the showcase
        if not isinstance(adapter, ShowcaseMoEAdapter):
            logger.warning("Client adapter is not ShowcaseMoEAdapter, simulation might be generic.")
            
        while True:
            current_task = "generic_task"
            if isinstance(adapter, ShowcaseMoEAdapter):
                current_task = adapter.get_next_task()
                
            logger.info(f"Simulating Task Context: {current_task}")
            
            # Simulate a VLA observation (RGB image)
            demo = Demonstration(
                observations=[np.random.normal(0, 1, (224, 224, 3)).astype(np.float32)],
                actions=[np.zeros(7)],
                task_id=current_task
            )
            
            client.add_demonstration(demo)
            try:
                # Process the round: Compute Gradients -> Clip -> Sparsify -> Encrypt -> Upload
                client.process_round()
            except Exception as e:
                logger.error(f"Showcase simulation step failed: {e}")
            
            # Sleep to allow dashboard to update visible metrics
            time.sleep(3.0) 
            
    thread = threading.Thread(target=simulation_loop, daemon=True)
    thread.start()
    return thread

def run_showcase_dashboard(port: Optional[int] = None):
    """
    Entry point for 'tensorguard dashboard'.
    Sets up a mocked Pi0 client environment and launches the visualization server.
    """
    port = port or settings.DASHBOARD_PORT
    
    # Configure a simulation client with reasonable defaults
    config = ShieldConfig(
        model_type="Showcase (Pi0)",
        key_path=settings.KEY_PATH,
        dp_epsilon=settings.DP_EPSILON,
        compression_ratio=settings.DEFAULT_COMPRESSION
    )
    
    client = EdgeClient(config)
    client.set_adapter(ShowcaseMoEAdapter())
    
    # Start the simulation loop in background
    start_showcase_simulation(client)
    
    # Hand over to the actual server implementation
    _run_server_dashboard(port=port, client=client)
