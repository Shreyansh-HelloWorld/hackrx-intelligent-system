# src/utils/logger.py
# HackRx 6.0 - Centralized Logging Module

import logging
import sys
import os
from datetime import datetime
from typing import Optional
from pathlib import Path

from .config import get_settings


class ColoredFormatter(logging.Formatter):
    """Custom formatter with color coding for different log levels"""
    
    # ANSI color codes
    COLORS = {
        'DEBUG': '\033[36m',      # Cyan
        'INFO': '\033[32m',       # Green
        'WARNING': '\033[33m',    # Yellow
        'ERROR': '\033[31m',      # Red
        'CRITICAL': '\033[35m',   # Magenta
        'RESET': '\033[0m'        # Reset
    }
    
    def format(self, record):
        # Add color to the level name
        if record.levelname in self.COLORS:
            record.levelname = (
                f"{self.COLORS[record.levelname]}"
                f"{record.levelname}"
                f"{self.COLORS['RESET']}"
            )
        return super().format(record)


class HackRxLogger:
    """
    Centralized logger for the HackRx system with file and console outputs
    """
    
    def __init__(self):
        self.settings = get_settings()
        self.loggers = {}
        self._setup_log_directory()
    
    def _setup_log_directory(self):
        """Create logs directory if it doesn't exist"""
        log_dir = Path(self.settings.logs_dir)
        log_dir.mkdir(exist_ok=True)
    
    def get_logger(self, name: str) -> logging.Logger:
        """
        Get or create a logger with the specified name
        
        Args:
            name: Logger name (usually module name)
        
        Returns:
            Configured logger instance
        """
        if name in self.loggers:
            return self.loggers[name]
        
        logger = logging.getLogger(name)
        logger.setLevel(getattr(logging, self.settings.log_level.upper()))
        
        # Remove existing handlers to avoid duplicates
        logger.handlers.clear()
        
        # Console handler with colors
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(logging.INFO)
        
        console_formatter = ColoredFormatter(
            fmt='%(asctime)s | %(levelname)s | %(name)s | %(message)s',
            datefmt='%H:%M:%S'
        )
        console_handler.setFormatter(console_formatter)
        logger.addHandler(console_handler)
        
        # File handler (no colors)
        if not self.settings.debug:  # Only in production/deployment
            timestamp = datetime.now().strftime("%Y%m%d")
            log_file = Path(self.settings.logs_dir) / f"hackrx_{timestamp}.log"
            
            file_handler = logging.FileHandler(log_file)
            file_handler.setLevel(logging.DEBUG)
            
            file_formatter = logging.Formatter(
                fmt='%(asctime)s | %(levelname)s | %(name)s | %(funcName)s:%(lineno)d | %(message)s',
                datefmt='%Y-%m-%d %H:%M:%S'
            )
            file_handler.setFormatter(file_formatter)
            logger.addHandler(file_handler)
        
        # Prevent propagation to root logger
        logger.propagate = False
        
        self.loggers[name] = logger
        return logger


# Global logger instance
_logger_instance = None


def get_logger(name: Optional[str] = None) -> logging.Logger:
    """
    Get a logger instance for the specified module
    
    Args:
        name: Module name (defaults to caller's module)
    
    Returns:
        Configured logger instance
    """
    global _logger_instance
    
    if _logger_instance is None:
        _logger_instance = HackRxLogger()
    
    if name is None:
        # Try to get the caller's module name
        import inspect
        frame = inspect.currentframe().f_back
        name = frame.f_globals.get('__name__', 'hackrx')
    
    return _logger_instance.get_logger(name)


def log_api_request(endpoint: str, method: str, status_code: int, 
                   response_time: float, user_agent: Optional[str] = None):
    """
    Log API request details
    
    Args:
        endpoint: API endpoint called
        method: HTTP method
        status_code: Response status code
        response_time: Response time in seconds
        user_agent: Client user agent
    """
    logger = get_logger("api")
    
    log_message = (
        f"{method} {endpoint} | "
        f"Status: {status_code} | "
        f"Time: {response_time:.3f}s"
    )
    
    if user_agent:
        log_message += f" | UA: {user_agent}"
    
    if status_code >= 400:
        logger.error(log_message)
    elif status_code >= 300:
        logger.warning(log_message)
    else:
        logger.info(log_message)


def log_llm_usage(provider: str, model: str, input_tokens: int, 
                  output_tokens: int, response_time: float, success: bool):
    """
    Log LLM API usage for monitoring and cost tracking
    
    Args:
        provider: LLM provider (gemini, groq)
        model: Model name
        input_tokens: Number of input tokens
        output_tokens: Number of output tokens
        response_time: Response time in seconds
        success: Whether the request was successful
    """
    logger = get_logger("llm")
    
    status = "SUCCESS" if success else "FAILED"
    total_tokens = input_tokens + output_tokens
    
    log_message = (
        f"{provider.upper()} | {model} | "
        f"Tokens: {input_tokens}→{output_tokens} ({total_tokens}) | "
        f"Time: {response_time:.3f}s | {status}"
    )
    
    if success:
        logger.info(log_message)
    else:
        logger.error(log_message)


def log_performance_metric(operation: str, duration: float, 
                          items_processed: Optional[int] = None):
    """
    Log performance metrics for system monitoring
    
    Args:
        operation: Name of the operation
        duration: Time taken in seconds
        items_processed: Number of items processed (if applicable)
    """
    logger = get_logger("performance")
    
    log_message = f"{operation} | Duration: {duration:.3f}s"
    
    if items_processed:
        rate = items_processed / duration if duration > 0 else 0
        log_message += f" | Items: {items_processed} | Rate: {rate:.2f}/s"
    
    logger.info(log_message)


# Initialize logging on module import
def setup_logging():
    """Initialize the logging system"""
    try:
        # Create logs directory
        settings = get_settings()
        os.makedirs(settings.logs_dir, exist_ok=True)
        
        # Get root logger to initialize the system
        get_logger("hackrx.system").info("🚀 HackRx logging system initialized")
        
    except Exception as e:
        print(f"❌ Failed to initialize logging: {e}")


# Auto-initialize when module is imported
setup_logging()