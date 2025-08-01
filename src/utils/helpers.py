# src/utils/helpers.py
# HackRx 6.0 - Helper Utilities Module

import asyncio
import hashlib
import re
import time
from typing import Any, Dict, List, Optional, Union
from urllib.parse import urlparse
import aiofiles
import httpx
from pathlib import Path

from .logger import get_logger

logger = get_logger(__name__)


def generate_hash(text: str) -> str:
    """
    Generate a consistent hash for text content
    
    Args:
        text: Input text to hash
    
    Returns:
        SHA-256 hash as hexadecimal string
    """
    return hashlib.sha256(text.encode('utf-8')).hexdigest()[:16]


def clean_text(text: str) -> str:
    """
    Clean and normalize text content
    
    Args:
        text: Raw text to clean
    
    Returns:
        Cleaned text
    """
    if not text:
        return ""
    
    # Remove excessive whitespace
    text = re.sub(r'\s+', ' ', text)
    
    # Remove special characters but keep basic punctuation
    text = re.sub(r'[^\w\s\.\,\?\!\;\:\-\(\)]', '', text)
    
    # Remove extra periods and spaces
    text = re.sub(r'\.{2,}', '.', text)
    text = re.sub(r'\s+', ' ', text)
    
    return text.strip()


def extract_numbers_from_text(text: str) -> List[float]:
    """
    Extract all numbers from text
    
    Args:
        text: Text to search for numbers
    
    Returns:
        List of numbers found in text
    """
    pattern = r'\b\d+(?:\.\d+)?\b'
    matches = re.findall(pattern, text)
    return [float(match) for match in matches]


def format_response_time(seconds: float) -> str:
    """
    Format response time for logging
    
    Args:
        seconds: Time in seconds
    
    Returns:
        Formatted time string
    """
    if seconds < 1:
        return f"{seconds*1000:.0f}ms"
    else:
        return f"{seconds:.2f}s"


def validate_url(url: str) -> bool:
    """
    Validate if a string is a proper URL
    
    Args:
        url: URL string to validate
    
    Returns:
        True if valid URL, False otherwise
    """
    try:
        result = urlparse(url)
        return all([result.scheme, result.netloc])
    except Exception:
        return False


def get_file_extension(url: str) -> Optional[str]:
    """
    Extract file extension from URL
    
    Args:
        url: URL to extract extension from
    
    Returns:
        File extension (without dot) or None
    """
    try:
        path = urlparse(url).path
        return Path(path).suffix.lower().lstrip('.')
    except Exception:
        return None


async def download_file(url: str, max_size_mb: int = 50) -> bytes:
    """
    Download file from URL with size validation
    
    Args:
        url: URL to download from
        max_size_mb: Maximum file size in MB
    
    Returns:
        File content as bytes
    
    Raises:
        ValueError: If file is too large or download fails
        httpx.HTTPError: If HTTP request fails
    """
    max_size_bytes = max_size_mb * 1024 * 1024
    
    async with httpx.AsyncClient() as client:
        logger.info(f"Downloading file from: {url}")
        
        try:
            response = await client.get(url, timeout=30.0)
            response.raise_for_status()
            
            # Check content length
            content_length = response.headers.get('content-length')
            if content_length and int(content_length) > max_size_bytes:
                raise ValueError(f"File too large: {content_length} bytes (max: {max_size_bytes})")
            
            content = response.content
            
            if len(content) > max_size_bytes:
                raise ValueError(f"File too large: {len(content)} bytes (max: {max_size_bytes})")
            
            logger.info(f"Downloaded {len(content)} bytes successfully")
            return content
            
        except httpx.HTTPError as e:
            logger.error(f"HTTP error downloading file: {e}")
            raise
        except Exception as e:
            logger.error(f"Error downloading file: {e}")
            raise ValueError(f"Download failed: {e}")


class RateLimiter:
    """
    Simple rate limiter for API calls
    """
    
    def __init__(self, max_calls: int, time_window: int = 60):
        """
        Initialize rate limiter
        
        Args:
            max_calls: Maximum calls allowed in time window
            time_window: Time window in seconds
        """
        self.max_calls = max_calls
        self.time_window = time_window
        self.calls = []
    
    async def wait_if_needed(self):
        """
        Wait if rate limit would be exceeded
        """
        now = time.time()
        
        # Remove old calls outside the time window
        self.calls = [call_time for call_time in self.calls 
                     if now - call_time < self.time_window]
        
        # Check if we need to wait
        if len(self.calls) >= self.max_calls:
            oldest_call = min(self.calls)
            wait_time = self.time_window - (now - oldest_call)
            
            if wait_time > 0:
                logger.info(f"Rate limit reached, waiting {wait_time:.2f}s")
                await asyncio.sleep(wait_time)
        
        # Record this call
        self.calls.append(now)


class TokenCounter:
    """
    Simple token counter for text (approximation)
    """
    
    @staticmethod
    def count_tokens(text: str) -> int:
        """
        Approximate token count for text
        
        Args:
            text: Text to count tokens for
        
        Returns:
            Approximate token count
        """
        if not text:
            return 0
        
        # Simple approximation: ~4 characters per token
        # This is rough but works for our purposes
        return len(text) // 4
    
    @staticmethod
    def truncate_to_tokens(text: str, max_tokens: int) -> str:
        """
        Truncate text to approximate token limit
        
        Args:
            text: Text to truncate
            max_tokens: Maximum tokens allowed
        
        Returns:
            Truncated text
        """
        if not text:
            return ""
        
        max_chars = max_tokens * 4  # Approximate
        if len(text) <= max_chars:
            return text
        
        # Truncate at word boundary
        truncated = text[:max_chars]
        last_space = truncated.rfind(' ')
        
        if last_space > max_chars * 0.8:  # Don't cut too much
            truncated = truncated[:last_space]
        
        return truncated + "..."


def create_prompt_template(template: str, **kwargs) -> str:
    """
    Create a prompt from template with variable substitution
    
    Args:
        template: Template string with {variable} placeholders
        **kwargs: Variables to substitute
    
    Returns:
        Formatted prompt string
    """
    try:
        return template.format(**kwargs)
    except KeyError as e:
        logger.error(f"Missing template variable: {e}")
        return template
    except Exception as e:
        logger.error(f"Template formatting error: {e}")
        return template


def extract_json_from_text(text: str) -> Optional[Dict[str, Any]]:
    """
    Extract JSON object from text that might contain other content
    
    Args:
        text: Text that might contain JSON
    
    Returns:
        Parsed JSON object or None if not found
    """
    import json
    
    # Try to find JSON block in text
    json_patterns = [
        r'\{.*\}',  # Simple JSON object
        r'\[.*\]',  # JSON array
    ]
    
    for pattern in json_patterns:
        matches = re.findall(pattern, text, re.DOTALL)
        for match in matches:
            try:
                return json.loads(match)
            except json.JSONDecodeError:
                continue
    
    # Try parsing the entire text
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    
    return None


def measure_execution_time(func):
    """
    Decorator to measure function execution time
    
    Args:
        func: Function to measure
    
    Returns:
        Decorated function that logs execution time
    """
    def wrapper(*args, **kwargs):
        start_time = time.time()
        try:
            result = func(*args, **kwargs)
            execution_time = time.time() - start_time
            logger.info(f"{func.__name__} executed in {format_response_time(execution_time)}")
            return result
        except Exception as e:
            execution_time = time.time() - start_time
            logger.error(f"{func.__name__} failed after {format_response_time(execution_time)}: {e}")
            raise
    
    return wrapper


async def measure_async_execution_time(func, *args, **kwargs):
    """
    Measure execution time for async functions
    
    Args:
        func: Async function to measure
        *args: Function arguments
        **kwargs: Function keyword arguments
    
    Returns:
        Tuple of (result, execution_time)
    """
    start_time = time.time()
    try:
        result = await func(*args, **kwargs)
        execution_time = time.time() - start_time
        return result, execution_time
    except Exception as e:
        execution_time = time.time() - start_time
        logger.error(f"{func.__name__} failed after {format_response_time(execution_time)}: {e}")
        raise