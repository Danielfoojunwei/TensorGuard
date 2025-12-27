class TensorGuardError(Exception):
    """Base exception for all TensorGuard errors."""
    pass

class CryptographyError(TensorGuardError):
    """Raised when encryption or decryption fails."""
    pass

class ConfigurationError(TensorGuardError):
    """Raised when the system is misconfigured."""
    pass

class CommunicationError(TensorGuardError):
    """Raised when networking/aggregation fails."""
    pass

class ValidationError(TensorGuardError):
    """Raised when data validation fails."""
    pass

class QualityWarning(UserWarning):
    """Issued when gradient quality falls below threshold."""
    pass
