"""
TensorGuard Empirical Federated Simulation (Multi-Process).
Tests real gRPC transport, serialization, and aggregation across isolated processes.
"""

import flwr as fl
import numpy as np
import time
import argparse
import subprocess
import sys
import os
from typing import Dict, List, Tuple, Optional

from ..core.client import EdgeClient, create_client
from ..server.aggregator import TensorGuardStrategy
from ..experiments.validation_suite import LIBEROSimulator, OFTAdapter
from ..utils.logging import get_logger

logger = get_logger(__name__)

# Constants
SERVER_ADDRESS = "127.0.0.1:8082"
NUM_CLIENTS = 1
NUM_ROUNDS = 2

def run_server():
    """Run the secure aggregation server."""
    logger.info("Starting TensorGuard Secure Aggregation Server...")
    strategy = TensorGuardStrategy(
        quorum_threshold=NUM_CLIENTS,
        min_fit_clients=NUM_CLIENTS,
        min_available_clients=NUM_CLIENTS,
    )
    
    fl.server.start_server(
        server_address=SERVER_ADDRESS,
        config=fl.server.ServerConfig(num_rounds=NUM_ROUNDS),
        strategy=strategy,
        grpc_max_message_length=100 * 1024 * 1024, # 100MB
    )

def run_client(cid: str):
    """Run a secured robot edge client."""
    # Use real EdgeClient with cid
    from .validation_suite import OFTAdapter
    edge_client = create_client(security_level=128, cid=cid)
    edge_client.set_adapter(OFTAdapter())
    
    # Add one sample demonstration to trigger fit()
    from .validation_suite import LIBEROSimulator
    sim = LIBEROSimulator()
    demo = sim.generate_trajectory("scoop_raisins")
    edge_client.add_demonstration(demo)
    
    fl.client.start_numpy_client(
        server_address=SERVER_ADDRESS,
        client=edge_client,
        grpc_max_message_length=100 * 1024 * 1024
    )

def orchestrate():
    """Launch server and clients as separate processes."""
    print(f"Orchestrating Empirical Federation with {NUM_CLIENTS} processes...")
    
    processes = []
    python_exe = sys.executable
    
    # Ensure src is in PYTHONPATH
    env = os.environ.copy()
    env["PYTHONPATH"] = os.path.abspath("src")
    env["PYTHONUNBUFFERED"] = "1"
    
    # 1. Start Server
    print("Launching Server process...")
    server_proc = subprocess.Popen([
        python_exe, "-m", "tensorguard.experiments.federated_sim", "--server"
    ], env=env)
    processes.append(server_proc)
    
    # Give server more time
    time.sleep(10)
    
    # 2. Start Clients
    for i in range(NUM_CLIENTS):
        print(f"Launching Client {i} process...")
        cp = subprocess.Popen([
            python_exe, "-m", "tensorguard.experiments.federated_sim", "--client", str(i)
        ], env=env)
        processes.append(cp)
        time.sleep(1)

    # 3. Monitor
    print("Monitoring processes...")
    try:
        while processes:
            for p in list(processes):
                if p.poll() is not None:
                    print(f"Process {p.pid} finished with code {p.returncode}")
                    processes.remove(p)
            time.sleep(1)
    except KeyboardInterrupt:
        print("Terminating...")
        for p in processes:
            p.terminate()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--server", action="store_true")
    parser.add_argument("--client", type=str)
    parser.add_argument("--orchestrate", action="store_true", default=True)
    
    args, unknown = parser.parse_known_args()
    
    # Ensure src is in PYTHONPATH for subprocesses
    if "PYTHONPATH" not in os.environ:
        os.environ["PYTHONPATH"] = "src"

    if args.server:
        run_server()
    elif args.client:
        run_client(args.client)
    else:
        orchestrate()
