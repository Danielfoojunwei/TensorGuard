"""
APHE-Shield CLI
Unified entry point for Server, Dashboard, and Tools.
"""

import argparse
import sys
import logging
from .server import start_server
from .dashboard_server import run_server as run_dashboard
from .edge_client import create_client, ShieldConfig

def main():
    parser = argparse.ArgumentParser(description="APHE-Shield Command Line Interface")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")
    
    # helper: start-server
    server_parser = subparsers.add_parser("start-server", help="Start the Federated Learning Aggregation Server")
    server_parser.add_argument("--port", type=int, default=8080, help="Server port (default: 8080)")
    
    # helper: dashboard
    dash_parser = subparsers.add_parser("dashboard", help="Start the White-Label Showcase Dashboard")
    dash_parser.add_argument("--port", type=int, default=8000, help="Dashboard port (default: 8000)")
    
    # helper: client (placeholder)
    client_parser = subparsers.add_parser("client", help="Run a standalone Edge Client")
    client_parser.add_argument("--endpoint", type=str, default="localhost:8080", help="FL Server endpoint")
    
    args = parser.parse_args()
    
    logging.basicConfig(level=logging.INFO)
    
    if args.command == "start-server":
        print(f"Starting Aggregation Server on port {args.port}...")
        start_server(port=args.port)
        
    elif args.command == "dashboard":
        print(f"Starting Showcase Dashboard on port {args.port}...")
        run_dashboard(port=args.port)
        
    elif args.command == "client":
        print("Starting Edge Client...")
        # In a real CLI, we'd load config from file
        config = ShieldConfig(cloud_endpoint=args.endpoint)
        client = create_client(config)
        client.start()
        
    else:
        parser.print_help()
        sys.exit(1)

if __name__ == "__main__":
    main()
