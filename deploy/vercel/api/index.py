from http.server import BaseHTTPRequestHandler
import json
import random
from datetime import datetime

class handler(BaseHTTPRequestHandler):
    def do_GET(self):
        self.send_response(200)
        self.send_header('Content-type', 'application/json')
        self.end_headers()
        
        # Mock status for public demo tracking
        # On Vercel, this simulates the local EdgeClient status
        data = {
            "running": True,
            "submissions": 125,
            "privacy_budget": "4.12",
            "budget_percent": 82,
            "connection": "connected",
            "security": "128-bit Post-Quantum (N2HE)",
            "key_path": "keys/demo_key.npy",
            "key_exists": True,
            "simd": True, 
            "experts": {"visual": 1.0, "language": 0.8, "auxiliary": 1.2},
            "telemetry": {
                "latency_train": 120.5 + random.uniform(-5, 5),
                "latency_compress": 15.2 + random.uniform(-2, 2),
                "latency_encrypt": 18.4 + random.uniform(-2, 2),
                "compression_ratio": 32.0,
                "quality_mse": 0.0012 + random.uniform(-0.0001, 0.0001),
                "bandwidth_saved_mb": 15.2
            },
            "audit": [
                {"timestamp": datetime.utcnow().isoformat(), "event": "SESSION_STARTED", "key_id": "demo_v1"},
                {"timestamp": datetime.utcnow().isoformat(), "event": "KEY_ROTATED", "key_id": "demo_v1"}
            ]
        }
        
        self.wfile.write(json.dumps(data).encode())
        return
