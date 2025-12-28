"""
TensorGuard CLI Entry Point
"""

import click
import logging
import sys
from .utils.config import settings
from .utils.logging import get_logger

logger = get_logger("cli")

@click.group()
def main():
    """TensorGuard: Secure VLA Fine-Tuning SDK"""
    pass

@main.command()
@click.option('--port', default=settings.DEFAULT_PORT, help='Aggregation server port')
def server(port):
    """Launch the Homomorphic Aggregation Server."""
    from .server.aggregator import start_server
    start_server(port)

@main.command()
@click.option('--port', default=settings.DASHBOARD_PORT, help='Dashboard port')
def dashboard(port):
    """Launch the TensorGuard Developer Dashboard."""
    from .utils.showcase import run_showcase_dashboard
    run_showcase_dashboard(port)

@main.command()
def info():
    """Display system and security information."""
    from . import __version__
    click.echo(f"TensorGuard SDK v{__version__}")
    click.echo(f"Security: {settings.SECURITY_LEVEL}-bit Post-Quantum (N2HE)")
    click.echo(f"Lattice Dimension: {settings.LATTICE_DIMENSION}")
    click.echo(f"Cloud Endpoint: {settings.CLOUD_ENDPOINT}")

if __name__ == "__main__":
    main()
