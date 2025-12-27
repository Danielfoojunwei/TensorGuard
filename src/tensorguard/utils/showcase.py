"""
TensorGuard Showcase \u0026 Simulation Utilities
"""

import time
import threading
import numpy as np
from typing import Optional

from ..core.client import EdgeClient
from ..core.adapters import VLAAdapter
from ..api.schemas import ShieldConfig, Demonstration
from ..server.dashboard import run_dashboard
from ..utils.logging import get_logger
from ..utils.config import settings

logger = get_logger("showcase")

class ShowcaseAdapter(VLAAdapter):
    """Mock adapter for dashboard visualization."""
    def __init__(self):
        super().__init__(None, self._mock_gradient, lambda m, g: None)
        
    def _mock_gradient(self, model, demo):
        time.sleep(0.1)
        return {
            "vision_encoder.weight": np.random.randn(10, 10).astype(np.float32),
            "llm_backbone.attn_q.weight": np.random.randn(10, 10).astype(np.float32),
            "adapter_connector.weight": np.random.randn(10, 10).astype(np.float32)
        }

    def compute_expert_gradients(self, demo):
        all_grads = self.compute_gradients(demo)
        return {
            "visual": {"vision_encoder.weight": all_grads["vision_encoder.weight"]},
            "language": {"llm_backbone.attn_q.weight": all_grads["llm_backbone.attn_q.weight"]},
            "auxiliary": {"adapter_connector.weight": all_grads["adapter_connector.weight"]}
        }

def start_showcase_simulation(client: EdgeClient):
    """Runs a background loop submitting mock demonstrations."""
    def simulation_loop():
        logger.info("Starting Showcase Simulation Loop")
        while True:
            demo = Demonstration(
                observations=[np.zeros((224,224,3))],
                actions=[np.zeros(7)]
            )
            client.add_demonstration(demo)
            try:
                client.process_round()
            except Exception as e:
                logger.error(f"Simulation step failed: {e}")
            time.sleep(2.0)
            
    thread = threading.Thread(target=simulation_loop, daemon=True)
    thread.start()
    return thread

def run_showcase_dashboard(port: Optional[int] = None):
    """Initializes a showcase client and starts the dashboard."""
    port = port or settings.DASHBOARD_PORT
    
    config = ShieldConfig(
        model_type="Showcase (Pi0)",
        key_path=settings.KEY_PATH,
        dp_epsilon=settings.DP_EPSILON,
        compression_ratio=settings.DEFAULT_COMPRESSION
    )
    
    client = EdgeClient(config)
    client.set_adapter(ShowcaseAdapter())
    
    # Start background simulation
    start_showcase_simulation(client)
    
    # Run dashboard server
    run_dashboard(port=port, client=client)
