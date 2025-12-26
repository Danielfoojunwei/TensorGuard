import http.server
import socketserver
import json
import threading
import time
import os
import sys

# Ensure src is in python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import webbrowser
from typing import Optional
from moai_shield import EdgeClient, ShieldConfig, VLAAdapter, Demonstration, create_client
import numpy as np
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("dashboard")

# Global Client Instance
client: Optional[EdgeClient] = None
is_running = False

# Mock VLA Adapter for Showcase
class ShowcaseAdapter(VLAAdapter):
    def __init__(self):
        pass
    def compute_gradients(self, demo):
        # Simulate gradient computation latency
        time.sleep(0.1)
        # Return fake gradients
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
    def apply_update(self, grads):
        pass

def init_client():
    global client
    config = ShieldConfig(
        model_type="Showcase (Pi0)",
        key_path="demo_key.pem",
        dp_epsilon=5.0, # Higher budget for demo
        compression_ratio=32
    )
    client = EdgeClient(config)
    client.set_adapter(ShowcaseAdapter())

# Simulation Loop
def simulation_loop():
    global is_running
    while True:
        if is_running and client:
            # Simulate a robot demonstration
            demo = Demonstration(
                observations=[np.zeros((224,224,3))],
                actions=[np.zeros(7)]
            )
            # Trigger Federated Learning Step (Encryption Pipeline)
            # This runs: Gradient -> Clip -> Sparsify -> Compress -> Encrypt
            client.add_demonstration(demo)
            client.fit([], {}) # Returns encrypted payload
            
            # Sleep to simulate real-time behavior
            time.sleep(2.0)
        else:
            time.sleep(0.5)

# API Handler
class DashboardHandler(http.server.SimpleHTTPRequestHandler):
    def do_GET(self):
        global is_running
        
        if self.path == "/api/status":
            self.send_response(200)
            self.send_header('Content-type', 'application/json')
            self.end_headers()
            
            status = client.get_status() if client else {}
            response = {
                "running": is_running,
                "submissions": status.total_submissions if client else 0,
                "privacy_budget": f"{status.privacy_budget_remaining:.2f}" if client else "0.00",
                "budget_percent": int((status.privacy_budget_remaining / 5.0) * 100) if client else 0,
                "connection": status.connection_status if client else "offline",
                "security": "128-bit Post-Quantum (N2HE)",
                "simd": True,
                "experts": {
                    "visual": 1.0,
                    "language": 0.8,
                    "auxiliary": 1.2
                }
            }
            self.wfile.write(json.dumps(response).encode())
            return
        
        if self.path == "/api/start":
            is_running = True
            logger.info("Simulation STARTED")
            self.send_response(200)
            self.end_headers()
            return

        if self.path == "/api/stop":
            is_running = False
            logger.info("Simulation STOPPED")
            self.send_response(200)
            self.end_headers()
            return

        # Serve static files from dashboard directory
        if self.path == "/" or self.path == "":
            self.path = "/index.html"
            
        return http.server.SimpleHTTPRequestHandler.do_GET(self)

def run_server(port=8000):
    # Initialize SDK
    init_client()
    
    # Start Simulation Thread
    sim_thread = threading.Thread(target=simulation_loop, daemon=True)
    sim_thread.start()
    
    # Change directory to serve static files correctly
    web_dir = os.path.join(os.path.dirname(__file__), "dashboard")
    os.chdir(web_dir)
    
    logger.info(f"Starting APHE-Shield Showcase on port {port}")
    with socketserver.TCPServer(("", port), DashboardHandler) as httpd:
        print(f"Serving at http://localhost:{port}")
        # Automatically open browser
        # webbrowser.open(f"http://localhost:{port}")
        httpd.serve_forever()

if __name__ == "__main__":
    run_server()
