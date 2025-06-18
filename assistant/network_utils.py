"""
Network utilities for Personal Assistant CLI.

This module provides network error handling, retry logic, and connectivity
checking for API calls to various model providers and services.
"""

import time
import asyncio
import logging
import requests
import socket
from typing import Dict, Any, List, Optional, Callable, Union, Tuple
from functools import wraps
from contextlib import contextmanager

logger = logging.getLogger(__name__)

class NetworkError(Exception):
    """Base exception for network-related errors."""
    pass

class ConnectivityError(NetworkError):
    """Exception for network connectivity issues."""
    pass

class RetryableError(NetworkError):
    """Exception for errors that should be retried."""
    pass

class NonRetryableError(NetworkError):
    """Exception for errors that should not be retried."""
    pass

class NetworkErrorHandler:
    """
    Handles network error detection, classification, and retry logic.
    """
    
    # Default retry configuration
    DEFAULT_MAX_RETRIES = 3
    DEFAULT_BASE_DELAY = 1.0
    DEFAULT_MAX_DELAY = 30.0
    DEFAULT_BACKOFF_MULTIPLIER = 2.0
    DEFAULT_TIMEOUT = 30.0
    
    # Retryable HTTP status codes
    RETRYABLE_STATUS_CODES = {429, 500, 502, 503, 504}
    
    # Retryable exception types
    RETRYABLE_EXCEPTIONS = (
        requests.exceptions.ConnectionError,
        requests.exceptions.Timeout,
        requests.exceptions.ReadTimeout,
        requests.exceptions.ConnectTimeout,
        socket.timeout,
        socket.gaierror,
        ConnectionResetError,
        ConnectionAbortedError,
    )
    
    def __init__(self, 
                 max_retries: int = DEFAULT_MAX_RETRIES,
                 base_delay: float = DEFAULT_BASE_DELAY,
                 max_delay: float = DEFAULT_MAX_DELAY,
                 backoff_multiplier: float = DEFAULT_BACKOFF_MULTIPLIER,
                 timeout: float = DEFAULT_TIMEOUT):
        """
        Initialize network error handler.
        
        Args:
            max_retries: Maximum number of retry attempts.
            base_delay: Base delay between retries in seconds.
            max_delay: Maximum delay between retries in seconds.
            backoff_multiplier: Multiplier for exponential backoff.
            timeout: Default timeout for network operations.
        """
        self.max_retries = max_retries
        self.base_delay = base_delay
        self.max_delay = max_delay
        self.backoff_multiplier = backoff_multiplier
        self.timeout = timeout
    
    def is_retryable_error(self, error: Exception) -> bool:
        """
        Determine if an error is retryable.
        
        Args:
            error: The exception to check.
            
        Returns:
            True if the error should be retried, False otherwise.
        """
        # Check for retryable exception types
        if isinstance(error, self.RETRYABLE_EXCEPTIONS):
            return True
        
        # Check for HTTP status codes
        if hasattr(error, 'response') and error.response is not None:
            status_code = getattr(error.response, 'status_code', None)
            if status_code in self.RETRYABLE_STATUS_CODES:
                return True
        
        # Check for specific error messages
        error_str = str(error).lower()
        retryable_messages = [
            'connection error',
            'timeout',
            'network is unreachable',
            'name resolution failed',
            'connection refused',
            'connection reset',
            'rate limit',
            'too many requests',
            'service unavailable',
            'bad gateway',
            'gateway timeout'
        ]
        
        for message in retryable_messages:
            if message in error_str:
                return True
        
        return False
    
    def calculate_delay(self, attempt: int) -> float:
        """
        Calculate delay for the given retry attempt.
        
        Args:
            attempt: Current attempt number (0-based).
            
        Returns:
            Delay in seconds.
        """
        delay = self.base_delay * (self.backoff_multiplier ** attempt)
        return min(delay, self.max_delay)
    
    def get_error_details(self, error: Exception) -> Dict[str, Any]:
        """
        Extract detailed information from an error.
        
        Args:
            error: The exception to analyze.
            
        Returns:
            Dictionary with error details.
        """
        details = {
            'type': type(error).__name__,
            'message': str(error),
            'retryable': self.is_retryable_error(error)
        }
        
        # Add HTTP-specific details
        if hasattr(error, 'response') and error.response is not None:
            details['status_code'] = getattr(error.response, 'status_code', None)
            details['reason'] = getattr(error.response, 'reason', None)
        
        # Add request details if available
        if hasattr(error, 'request') and error.request is not None:
            details['url'] = getattr(error.request, 'url', None)
            details['method'] = getattr(error.request, 'method', None)
        
        return details
    
    def with_retry(self, func: Callable, *args, **kwargs) -> Any:
        """
        Execute a function with retry logic.
        
        Args:
            func: Function to execute.
            *args: Positional arguments for the function.
            **kwargs: Keyword arguments for the function.
            
        Returns:
            Result of the function execution.
            
        Raises:
            NetworkError: If all retry attempts fail.
        """
        last_error = None
        
        for attempt in range(self.max_retries + 1):
            try:
                logger.debug(f"Executing {func.__name__} (attempt {attempt + 1}/{self.max_retries + 1})")
                return func(*args, **kwargs)
                
            except Exception as e:
                last_error = e
                error_details = self.get_error_details(e)
                
                logger.warning(f"Attempt {attempt + 1} failed: {error_details['message']}")
                
                # Check if we should retry
                if attempt >= self.max_retries:
                    logger.error(f"All {self.max_retries + 1} attempts failed")
                    break
                
                if not error_details['retryable']:
                    logger.error(f"Non-retryable error: {error_details['message']}")
                    raise NonRetryableError(f"Non-retryable error: {e}")
                
                # Calculate and apply delay
                delay = self.calculate_delay(attempt)
                logger.info(f"Retrying in {delay:.2f} seconds...")
                time.sleep(delay)
        
        # All attempts failed
        if last_error:
            error_details = self.get_error_details(last_error)
            if error_details['retryable']:
                raise RetryableError(f"Failed after {self.max_retries + 1} attempts: {last_error}")
            else:
                raise NonRetryableError(f"Non-retryable error: {last_error}")
        
        raise NetworkError("Unknown error occurred during retry execution")
    
    async def with_retry_async(self, func: Callable, *args, **kwargs) -> Any:
        """
        Execute an async function with retry logic.
        
        Args:
            func: Async function to execute.
            *args: Positional arguments for the function.
            **kwargs: Keyword arguments for the function.
            
        Returns:
            Result of the function execution.
            
        Raises:
            NetworkError: If all retry attempts fail.
        """
        last_error = None
        
        for attempt in range(self.max_retries + 1):
            try:
                logger.debug(f"Executing {func.__name__} async (attempt {attempt + 1}/{self.max_retries + 1})")
                return await func(*args, **kwargs)
                
            except Exception as e:
                last_error = e
                error_details = self.get_error_details(e)
                
                logger.warning(f"Async attempt {attempt + 1} failed: {error_details['message']}")
                
                # Check if we should retry
                if attempt >= self.max_retries:
                    logger.error(f"All {self.max_retries + 1} async attempts failed")
                    break
                
                if not error_details['retryable']:
                    logger.error(f"Non-retryable async error: {error_details['message']}")
                    raise NonRetryableError(f"Non-retryable error: {e}")
                
                # Calculate and apply delay
                delay = self.calculate_delay(attempt)
                logger.info(f"Retrying async in {delay:.2f} seconds...")
                await asyncio.sleep(delay)
        
        # All attempts failed
        if last_error:
            error_details = self.get_error_details(last_error)
            if error_details['retryable']:
                raise RetryableError(f"Failed after {self.max_retries + 1} attempts: {last_error}")
            else:
                raise NonRetryableError(f"Non-retryable error: {last_error}")
        
        raise NetworkError("Unknown error occurred during async retry execution")

def with_network_retry(max_retries: int = 3, 
                      base_delay: float = 1.0,
                      max_delay: float = 30.0,
                      backoff_multiplier: float = 2.0):
    """
    Decorator for adding network retry logic to functions.
    
    Args:
        max_retries: Maximum number of retry attempts.
        base_delay: Base delay between retries in seconds.
        max_delay: Maximum delay between retries in seconds.
        backoff_multiplier: Multiplier for exponential backoff.
        
    Returns:
        Decorated function with retry logic.
    """
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            handler = NetworkErrorHandler(
                max_retries=max_retries,
                base_delay=base_delay,
                max_delay=max_delay,
                backoff_multiplier=backoff_multiplier
            )
            return handler.with_retry(func, *args, **kwargs)
        return wrapper
    return decorator

def with_network_retry_async(max_retries: int = 3,
                           base_delay: float = 1.0, 
                           max_delay: float = 30.0,
                           backoff_multiplier: float = 2.0):
    """
    Decorator for adding network retry logic to async functions.
    
    Args:
        max_retries: Maximum number of retry attempts.
        base_delay: Base delay between retries in seconds.
        max_delay: Maximum delay between retries in seconds.
        backoff_multiplier: Multiplier for exponential backoff.
        
    Returns:
        Decorated async function with retry logic.
    """
    def decorator(func):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            handler = NetworkErrorHandler(
                max_retries=max_retries,
                base_delay=base_delay,
                max_delay=max_delay,
                backoff_multiplier=backoff_multiplier
            )
            return await handler.with_retry_async(func, *args, **kwargs)
        return wrapper
    return decorator

class ConnectivityChecker:
    """
    Checks network connectivity to various services.
    """
    
    def __init__(self, timeout: float = 5.0):
        """
        Initialize connectivity checker.
        
        Args:
            timeout: Timeout for connectivity checks in seconds.
        """
        self.timeout = timeout
    
    def check_internet_connectivity(self) -> bool:
        """
        Check basic internet connectivity.
        
        Returns:
            True if internet is available, False otherwise.
        """
        test_hosts = [
            ('8.8.8.8', 53),      # Google DNS
            ('1.1.1.1', 53),      # Cloudflare DNS
            ('208.67.222.222', 53) # OpenDNS
        ]
        
        for host, port in test_hosts:
            try:
                socket.create_connection((host, port), timeout=self.timeout).close()
                logger.debug(f"Internet connectivity confirmed via {host}:{port}")
                return True
            except (socket.timeout, socket.error):
                continue
        
        logger.warning("No internet connectivity detected")
        return False
    
    def check_provider_connectivity(self, provider: str) -> Dict[str, Any]:
        """
        Check connectivity to specific AI provider endpoints.
        
        Args:
            provider: Provider name (anthropic, openai, bedrock, ollama).
            
        Returns:
            Dictionary with connectivity status and details.
        """
        result = {
            'provider': provider,
            'reachable': False,
            'response_time': None,
            'error': None
        }
        
        # Provider endpoint mappings
        endpoints = {
            'anthropic': 'https://api.anthropic.com',
            'openai': 'https://api.openai.com',
            'bedrock': 'https://bedrock-runtime.us-west-2.amazonaws.com',
            'ollama': 'http://localhost:11434'
        }
        
        if provider not in endpoints:
            result['error'] = f"Unknown provider: {provider}"
            return result
        
        try:
            start_time = time.time()
            response = requests.head(endpoints[provider], timeout=self.timeout)
            response_time = time.time() - start_time
            
            result['reachable'] = response.status_code < 500
            result['response_time'] = response_time
            result['status_code'] = response.status_code
            
            logger.debug(f"Provider {provider} connectivity: {result['reachable']} "
                        f"(status: {response.status_code}, time: {response_time:.2f}s)")
            
        except requests.exceptions.RequestException as e:
            result['error'] = str(e)
            logger.warning(f"Provider {provider} connectivity check failed: {e}")
        
        return result
    
    def check_all_providers(self) -> Dict[str, Dict[str, Any]]:
        """
        Check connectivity to all supported providers.
        
        Returns:
            Dictionary mapping provider names to connectivity results.
        """
        providers = ['anthropic', 'openai', 'bedrock', 'ollama']
        results = {}
        
        for provider in providers:
            results[provider] = self.check_provider_connectivity(provider)
        
        return results

# Global instances for convenience
default_error_handler = NetworkErrorHandler()
default_connectivity_checker = ConnectivityChecker()

# Convenience functions
def check_internet() -> bool:
    """Check basic internet connectivity."""
    return default_connectivity_checker.check_internet_connectivity()

def check_provider(provider: str) -> Dict[str, Any]:
    """Check connectivity to a specific provider."""
    return default_connectivity_checker.check_provider_connectivity(provider)

def check_all_providers() -> Dict[str, Dict[str, Any]]:
    """Check connectivity to all providers."""
    return default_connectivity_checker.check_all_providers() 