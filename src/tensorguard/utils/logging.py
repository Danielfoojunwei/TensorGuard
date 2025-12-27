import logging
import sys
import json
from datetime import datetime
from typing import Any, Dict

class StructuredFormatter(logging.Formatter):
    """JSON formatter for structured logging."""
    def format(self, record: logging.LogRecord) -> str:
        log_entry = {
            "timestamp": datetime.fromtimestamp(record.created).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        if record.exc_info:
            log_entry["exception"] = self.formatException(record.exc_info)
        
        # Add extra attributes if provided
        if hasattr(record, "extra_data"):
            log_entry.update(record.extra_data)
            
        return json.dumps(log_entry)

def get_logger(name: str) -> logging.Logger:
    """Returns a structured logger instance."""
    logger = logging.getLogger(name)
    
    # Avoid duplicate handlers
    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        
        # In production, use JSON. In dev, use standard format.
        # For simplicity in this refactor, we provide both options.
        from .config import settings
        
        if settings.ENVIRONMENT == "production":
            handler.setFormatter(StructuredFormatter())
        else:
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            
        logger.addHandler(handler)
        logger.setLevel(settings.LOG_LEVEL)
        
    return logger

# Global default logger
logger = get_logger("tensorguard")
