"""
Logging Configuration Module
"""
import logging
import logging.handlers
from pathlib import Path
from typing import Dict, Any

from .config import get_settings

# Project root directory (parent of llm_score)
PROJECT_ROOT = Path(__file__).resolve().parents[3]  # repo root


def setup_logging() -> Dict[str, Any]:
    """Setup application logging configuration"""
    settings = get_settings()
    
    # Log files are stored in the logs directory under project root
    log_file_path = PROJECT_ROOT / ".runtime" / "logs" / "llm_score.log"
    log_file_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Log format
    log_format = (
        "%(asctime)s - %(name)s - %(levelname)s - "
        "%(filename)s:%(lineno)d - %(funcName)s - %(message)s"
    )
    
    # Configuration dictionary
    logging_config = {
        "version": 1,
        "disable_existing_loggers": False,
        "formatters": {
            "default": {
                "format": log_format,
                "datefmt": "%Y-%m-%d %H:%M:%S",
            },
        },
        "handlers": {
            "console": {
                "class": "logging.StreamHandler",
                "level": settings.log_level,
                "formatter": "default",
                "stream": "ext://sys.stdout",
            },
            "file": {
                "class": "logging.handlers.RotatingFileHandler",
                "level": settings.log_level,
                "formatter": "default",
                "filename": str(log_file_path),
                "maxBytes": 10485760,  # 10MB
                "backupCount": 5,
                "encoding": "utf8",
            },
        },
        "loggers": {
            "": {
                "level": settings.log_level,
                "handlers": ["console", "file"],
                "propagate": False,
            },
        },
    }
    
    # Apply configuration
    import logging.config
    logging.config.dictConfig(logging_config)
    
    return logging_config


def get_logger(name: str) -> logging.Logger:
    """Get logger with specified name"""
    return logging.getLogger(name)
