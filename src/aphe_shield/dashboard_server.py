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
from aphe_shield import EdgeClient, ShieldConfig, VLAAdapter, Demonstration, create_client
import numpy as np
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("dashboard")

# Global Client Instance
client: Optional[EdgeClient] = None
is_running = False

def init_dashboard(provided_client: EdgeClient):
    global client
    client = provided_client

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

def run_server(port=8000, client_instance: Optional[EdgeClient] = None):
    if client_instance:
        init_dashboard(client_instance)
    
    # Change directory to serve static files correctly
    web_dir = os.path.join(os.path.dirname(__file__), "dashboard")
    os.chdir(web_dir)
    
    logger.info(f"Starting APHE-Shield Dashboard on port {port}")
    with socketserver.TCPServer(("", port), DashboardHandler) as httpd:
        print(f"Serving at http://localhost:{port}")
        httpd.serve_forever()

if __name__ == "__main__":
    run_server()
