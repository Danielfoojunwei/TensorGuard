"""
TensorGuard Dashboard API Server
"""

import http.server
import socketserver
import json
import os
from typing import Optional

from ..core.client import EdgeClient
from ..utils.logging import get_logger
from ..utils.config import settings

logger = get_logger("dashboard")

class DashboardHandler(http.server.SimpleHTTPRequestHandler):
    """API and Static File Handler for TensorGuard Dashboard."""
    
    client_instance: Optional[EdgeClient] = None
    simulation_active: bool = False

    def do_GET(self):
        if self.path == "/api/status":
            self._send_json(self._get_system_status())
        elif self.path == "/api/start":
            DashboardHandler.simulation_active = True
            logger.info("Showcase simulation started")
            self._send_json({"status": "started"})
        elif self.path == "/api/stop":
            DashboardHandler.simulation_active = False
            logger.info("Showcase simulation stopped")
            self._send_json({"status": "stopped"})
        else:
            # Serve static files from the 'dashboard' subdirectory
            super().do_GET()

    def _send_json(self, data: dict):
        self.send_response(200)
        self.send_header('Content-type', 'application/json')
        self.end_headers()
        self.wfile.write(json.dumps(data).encode())

    def _get_system_status(self) -> dict:
        client = DashboardHandler.client_instance
        status = client.get_status() if client else None
        
        return {
            "running": DashboardHandler.simulation_active,
            "submissions": status.total_submissions if status else 0,
            "privacy_budget": f"{status.privacy_budget_remaining:.2f}" if status else "0.00",
            "budget_percent": int((status.privacy_budget_remaining / settings.DP_EPSILON) * 100) if status else 0,
            "connection": status.connection_status if status else "online",
            "security": f"{settings.SECURITY_LEVEL}-bit Post-Quantum (N2HE)",
            "simd": True, 
            "experts": {"visual": 1.0, "language": 0.8, "auxiliary": 1.2}
        }

def run_dashboard(port: Optional[int] = None, client: Optional[EdgeClient] = None):
    """Start the dashboard server."""
    port = port or settings.DASHBOARD_PORT
    DashboardHandler.client_instance = client
    
    # Path to static assets
    base_dir = os.path.dirname(__file__)
    web_dir = os.path.join(base_dir, "dashboard")
    
    # Change OS directory to the web assets folder for SimpleHTTPRequestHandler
    os.chdir(web_dir)
    
    logger.info(f"Dashboard serving from {web_dir}")
    logger.info(f"Access at http://localhost:{port}")
    
    with socketserver.TCPServer(("", port), DashboardHandler) as httpd:
        httpd.serve_forever()
