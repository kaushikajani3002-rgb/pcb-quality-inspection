import logging
import os
from pathlib import Path

def setup_logger(name: str = "PCB_AOI") -> logging.Logger:
    """
    Configures a logger that outputs structured logs to logs/inspection.log
    and console output. Keeps track of date, time, messages, and level.
    """
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)

    # Prevent double registration of handlers if already initialized
    if logger.handlers:
        return logger

    # Format for logging outputs
    log_format = logging.Formatter(
        '[%(asctime)s] [%(levelname)s] [%(filename)s:%(lineno)d] - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )

    # Console Handler
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(log_format)
    logger.addHandler(console_handler)

    # File Handler - write to logs/inspection.log
    project_root = Path(__file__).resolve().parent.parent
    log_dir = project_root / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    log_file = log_dir / "inspection.log"

    try:
        file_handler = logging.FileHandler(log_file, encoding="utf-8")
        file_handler.setFormatter(log_format)
        logger.addHandler(file_handler)
    except Exception as e:
        # Fallback to standard console stream if logging folder is read-only
        logger.warning(f"Could not create file log handler at {log_file} due to: {e}")

    return logger

# Globally available logger instance
logger = setup_logger()
