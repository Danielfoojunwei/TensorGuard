from pydantic_settings import BaseSettings, SettingsConfigDict
from typing import Optional

class TensorGuardSettings(BaseSettings):
    """
    TensorGuard System settings.
    Loads from environment variables with TENSORGUARD_ prefix.
    """
    model_config = SettingsConfigDict(env_prefix='TENSORGUARD_', env_file='.env', env_file_encoding='utf-8')

    # General
    ENVIRONMENT: str = "production"
    LOG_LEVEL: str = "INFO"

    # Crypto Parameters
    SECURITY_LEVEL: int = 128
    MAX_KEY_USES: int = 1000
    LATTICE_DIMENSION: int = 1024
    PLAINTEXT_MODULUS: int = 65536

    # Privacy Pipeline
    DP_EPSILON: float = 1.0
    DEFAULT_SPARSITY: float = 0.01
    DEFAULT_COMPRESSION: int = 32
    MAX_GRADIENT_NORM: float = 1.0

    # Networking
    CLOUD_ENDPOINT: str = "https://api.tensor-crate.ai"
    DEFAULT_PORT: int = 8080
    DASHBOARD_PORT: int = 8000

    # Paths
    KEY_PATH: str = "keys/enterprise_key.npy"

settings = TensorGuardSettings()
