"""
Centralized logging configuration
Creates log files with detailed step-by-step information
"""
import logging
import os
from datetime import datetime
from pathlib import Path

# Create logs directory if it doesn't exist
LOG_DIR = Path("logs")
LOG_DIR.mkdir(exist_ok=True)

# Create log file with timestamp
LOG_FILE = LOG_DIR / f"rag_app_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"

def setup_logging(log_level=logging.INFO):
    """
    Setup centralized logging configuration
    Logs to both console and file
    
    Args:
        log_level: Logging level (default: INFO)
    """
    # Create formatters
    detailed_formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    simple_formatter = logging.Formatter(
        '%(asctime)s - %(levelname)s - %(message)s',
        datefmt='%H:%M:%S'
    )
    
    # Root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(log_level)
    
    # Remove existing handlers to avoid duplicates
    root_logger.handlers = []
    
    # File handler - detailed logs
    file_handler = logging.FileHandler(LOG_FILE, encoding='utf-8')
    file_handler.setLevel(logging.DEBUG)  # Log everything to file
    file_handler.setFormatter(detailed_formatter)
    root_logger.addHandler(file_handler)
    
    # Console handler - simpler format
    console_handler = logging.StreamHandler()
    console_handler.setLevel(log_level)
    console_handler.setFormatter(simple_formatter)
    root_logger.addHandler(console_handler)
    
    # Log initialization
    root_logger.info(f"=" * 80)
    root_logger.info(f"Logging initialized - Log file: {LOG_FILE}")
    root_logger.info(f"=" * 80)
    
    return root_logger

def get_logger(name: str) -> logging.Logger:
    """
    Get a logger instance for a module
    
    Args:
        name: Module name (usually __name__)
    
    Returns:
        Logger instance
    """
    return logging.getLogger(name)

# Initialize logging on import
setup_logging()

