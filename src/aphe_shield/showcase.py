"""
APHE-Shield Showcase & Simulation Utilities
"""

import time
import threading
import numpy as np
import logging
from .edge_client import EdgeClient, ShieldConfig
from .adapters import VLAAdapter
from .structures import Demonstration

logger = logging.getLogger("aphe_shield.showcase")

class ShowcaseAdapter(VLAAdapter):
    """Mock adapter for dashboard visualization."""
    def __init__(self):
        super().__init__(None, self._mock_gradient, self._mock_apply)
        
    def _mock_gradient(self, model, demo):
        # Simulate gradient computation latency
        time.sleep(0.1)
        return {
            "vision_encoder.weight": np.random.randn(10, 10).astype(np.float32),
            "llm_backbone.attn_q.weight": np.random.randn(10, 10).astype(np.float32),
            "adapter_connector.weight": np.random.randn(10, 10).astype(np.float32)
        }
        
    def _mock_apply(self, model, grads):
        pass

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
            # Simulate a robot demonstration
            demo = Demonstration(
                observations=[np.zeros((224,224,3))],
                actions=[np.zeros(7)]
            )
            client.add_demonstration(demo)
            client.fit([], {})
            time.sleep(2.0)
            
    sim_thread = threading.Thread(target=simulation_loop, daemon=True)
    sim_thread.start()
    return sim_thread

def run_showcase_dashboard(port: int = 8000):
    """Initializes a showcase client and starts the dashboard."""
    from .dashboard_server import run_server
    
    config = ShieldConfig(
        model_type="Showcase (Pi0)",
        key_path="demo_key.pem",
        dp_epsilon=5.0,
        compression_ratio=32
    )
    client = EdgeClient(config)
    client.set_adapter(ShowcaseAdapter())
    
    # Start background simulation
    start_showcase_simulation(client)
    
    # Run dashboard server
    run_server(port=port, client_instance=client)
