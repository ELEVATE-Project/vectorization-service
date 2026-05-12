"""
Test logger configuration for capturing test execution results
"""
import logging
import os
from datetime import datetime
from logging.handlers import RotatingFileHandler

# Create logger directory if it doesn't exist
LOGGER_DIR = os.path.dirname(os.path.abspath(__file__))
LOG_FILE = os.path.join(LOGGER_DIR, "test_results.log")

def setup_test_logger(name: str = "test_logger") -> logging.Logger:
    """
    Set up a logger for test execution with file and console handlers
    
    Args:
        name: Logger name
        
    Returns:
        Configured logger instance
    """
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)
    
    # Remove existing handlers to avoid duplicates
    logger.handlers.clear()
    
    # Create formatters
    detailed_formatter = logging.Formatter(
        fmt='%(asctime)s - %(name)s - %(levelname)s - %(funcName)s:%(lineno)d - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    simple_formatter = logging.Formatter(
        fmt='%(asctime)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # File handler with rotation (max 10MB, keep 5 backup files)
    file_handler = RotatingFileHandler(
        LOG_FILE,
        maxBytes=10 * 1024 * 1024,  # 10MB
        backupCount=5,
        encoding='utf-8'
    )
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(detailed_formatter)
    
    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(simple_formatter)
    
    # Add handlers to logger
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    
    return logger

def log_test_start(logger: logging.Logger, test_name: str):
    """Log the start of a test"""
    logger.info("=" * 80)
    logger.info(f"STARTING TEST: {test_name}")
    logger.info("=" * 80)

def log_test_end(logger: logging.Logger, test_name: str, status: str):
    """Log the end of a test with status"""
    logger.info("-" * 80)
    logger.info(f"TEST {status.upper()}: {test_name}")
    logger.info("-" * 80)
    logger.info("")

def log_test_summary(logger: logging.Logger, total: int, passed: int, failed: int, skipped: int):
    """Log test execution summary"""
    logger.info("=" * 80)
    logger.info("TEST EXECUTION SUMMARY")
    logger.info("=" * 80)
    logger.info(f"Total Tests: {total}")
    logger.info(f"Passed: {passed}")
    logger.info(f"Failed: {failed}")
    logger.info(f"Skipped: {skipped}")
    logger.info(f"Success Rate: {(passed/total*100) if total > 0 else 0:.2f}%")
    logger.info("=" * 80)

# Create default test logger
test_logger = setup_test_logger()
