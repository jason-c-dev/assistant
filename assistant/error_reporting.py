"""
Error reporting and logging utilities for Personal Assistant CLI.

This module provides comprehensive error logging, user-friendly error reporting,
and centralized error handling for the entire assistant application.
"""

import os
import sys
import json
import logging
import traceback
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List, Optional, Union, Tuple
from contextlib import contextmanager

# Error type mappings for user-friendly messages
ERROR_TYPE_MESSAGES = {
    'ConfigError': "Configuration issue",
    'YAMLSyntaxError': "Configuration file syntax error",
    'ConfigValidationError': "Configuration validation failed",
    'APIKeyValidationError': "API key validation failed",
    'ProviderError': "AI provider setup issue",
    'MemoryClientError': "Memory system issue",
    'MemoryServerStartupError': "Memory server startup failed",
    'MemoryServerValidationError': "Memory server validation failed",
    'NetworkError': "Network connectivity issue",
    'RetryableError': "Temporary network issue",
    'NonRetryableError': "Permanent network issue",
    'AgentError': "AI agent execution issue",
    'FileNotFoundError': "File not found",
    'PermissionError': "Permission denied",
    'ConnectionError': "Network connection failed",
    'TimeoutError': "Operation timed out",
}

class ErrorReporter:
    """
    Centralized error reporting and logging system.
    
    Provides both detailed logging for debugging and user-friendly
    error messages with actionable suggestions.
    """
    
    def __init__(self, 
                 log_dir: Optional[str] = None,
                 enable_file_logging: bool = True,
                 verbose: bool = False):
        """
        Initialize error reporter.
        
        Args:
            log_dir: Directory for log files (default: ~/.assistant/logs)
            enable_file_logging: Whether to log to files
            verbose: Enable verbose logging
        """
        self.verbose = verbose
        self.enable_file_logging = enable_file_logging
        
        # Setup log directory
        if log_dir is None:
            log_dir = os.path.expanduser("~/.assistant/logs")
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        # Setup log files before logger (logger needs session_log_file)
        self.error_log_file = self.log_dir / "errors.log"
        self.session_log_file = self.log_dir / f"session_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
        
        # Setup loggers
        self.logger = self._setup_logger()
        
        # Error statistics
        self.error_counts = {}
        self.session_errors = []
    
    def _setup_logger(self) -> logging.Logger:
        """Setup comprehensive logging configuration."""
        logger = logging.getLogger('assistant.error_reporter')
        logger.setLevel(logging.DEBUG if self.verbose else logging.INFO)
        
        # Prevent duplicate handlers
        if logger.handlers:
            return logger
        
        # Console handler for user-facing messages
        console_handler = logging.StreamHandler(sys.stderr)
        console_handler.setLevel(logging.WARNING)
        console_formatter = logging.Formatter('%(levelname)s: %(message)s')
        console_handler.setFormatter(console_formatter)
        logger.addHandler(console_handler)
        
        # File handler for detailed logs
        if self.enable_file_logging:
            file_handler = logging.FileHandler(self.session_log_file, encoding='utf-8')
            file_handler.setLevel(logging.DEBUG)
            file_formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(filename)s:%(lineno)d - %(message)s'
            )
            file_handler.setFormatter(file_formatter)
            logger.addHandler(file_handler)
        
        return logger
    
    def report_error(self, 
                    error: Exception,
                    context: Optional[Dict[str, Any]] = None,
                    user_action: Optional[str] = None,
                    show_traceback: bool = None) -> str:
        """
        Report an error with comprehensive logging and user-friendly messaging.
        
        Args:
            error: The exception to report
            context: Additional context information
            user_action: Description of what the user was trying to do
            show_traceback: Whether to show traceback to user (default: verbose mode)
            
        Returns:
            User-friendly error message
        """
        if show_traceback is None:
            show_traceback = self.verbose
        
        # Generate error details
        error_details = self._extract_error_details(error, context, user_action)
        
        # Log detailed error information
        self._log_error_details(error_details)
        
        # Track error statistics
        self._track_error(error_details)
        
        # Generate user-friendly message
        user_message = self._generate_user_message(error_details, show_traceback)
        
        return user_message
    
    def _extract_error_details(self, 
                              error: Exception,
                              context: Optional[Dict[str, Any]] = None,
                              user_action: Optional[str] = None) -> Dict[str, Any]:
        """Extract comprehensive details from an error."""
        error_type = type(error).__name__
        error_message = str(error)
        
        details = {
            'timestamp': datetime.now().isoformat(),
            'error_type': error_type,
            'error_message': error_message,
            'user_action': user_action,
            'context': context or {},
            'traceback': traceback.format_exc(),
            'traceback_lines': traceback.format_tb(error.__traceback__),
            'file_location': None,
            'line_number': None,
            'function_name': None,
        }
        
        # Extract location information
        if error.__traceback__:
            tb = error.__traceback__
            while tb.tb_next:  # Get the last frame
                tb = tb.tb_next
            
            details['file_location'] = tb.tb_frame.f_code.co_filename
            details['line_number'] = tb.tb_lineno
            details['function_name'] = tb.tb_frame.f_code.co_name
        
        # Add error-specific details
        if hasattr(error, 'response'):
            details['http_status'] = getattr(error.response, 'status_code', None)
            details['http_reason'] = getattr(error.response, 'reason', None)
        
        if hasattr(error, '__cause__') and error.__cause__:
            details['root_cause'] = str(error.__cause__)
        
        return details
    
    def _log_error_details(self, error_details: Dict[str, Any]) -> None:
        """Log detailed error information."""
        error_type = error_details['error_type']
        error_message = error_details['error_message']
        user_action = error_details.get('user_action', 'Unknown action')
        
        # Log to application logger
        log_message = f"ERROR: {error_type} during '{user_action}': {error_message}"
        self.logger.error(log_message)
        
        if self.verbose:
            self.logger.debug(f"Error details: {json.dumps(error_details, indent=2, default=str)}")
        
        # Log to error file
        if self.enable_file_logging:
            self._append_to_error_log(error_details)
    
    def _append_to_error_log(self, error_details: Dict[str, Any]) -> None:
        """Append error details to the error log file."""
        try:
            with open(self.error_log_file, 'a', encoding='utf-8') as f:
                json.dump(error_details, f, default=str, ensure_ascii=False)
                f.write('\n')
        except Exception as e:
            self.logger.warning(f"Failed to write to error log: {e}")
    
    def _track_error(self, error_details: Dict[str, Any]) -> None:
        """Track error statistics for analysis."""
        error_type = error_details['error_type']
        self.error_counts[error_type] = self.error_counts.get(error_type, 0) + 1
        self.session_errors.append(error_details)
    
    def _generate_user_message(self, 
                              error_details: Dict[str, Any],
                              show_traceback: bool = False) -> str:
        """Generate user-friendly error message with suggestions."""
        error_type = error_details['error_type']
        error_message = error_details['error_message']
        user_action = error_details.get('user_action', 'performing an operation')
        
        # Start with friendly error type description
        friendly_type = ERROR_TYPE_MESSAGES.get(error_type, "An error occurred")
        
        message = f"❌ {friendly_type} while {user_action}\n\n"
        
        # Add specific error message if it's user-friendly
        if self._is_user_friendly_message(error_message):
            message += f"Details: {error_message}\n\n"
        
        # Add context-specific suggestions
        suggestions = self._get_error_suggestions(error_details)
        if suggestions:
            message += "💡 Suggestions:\n"
            for suggestion in suggestions:
                message += f"  • {suggestion}\n"
            message += "\n"
        
        # Add troubleshooting information
        troubleshooting = self._get_troubleshooting_info(error_details)
        if troubleshooting:
            message += "🔧 Troubleshooting:\n"
            for info in troubleshooting:
                message += f"  • {info}\n"
            message += "\n"
        
        # Add traceback in verbose mode
        if show_traceback and error_details.get('traceback'):
            message += "📋 Technical Details:\n"
            message += f"```\n{error_details['traceback']}\n```\n\n"
        
        # Add log file information
        if self.enable_file_logging:
            message += f"📝 Detailed logs saved to: {self.session_log_file}\n"
        
        return message.strip()
    
    def _is_user_friendly_message(self, message: str) -> bool:
        """Check if an error message is user-friendly."""
        unfriendly_patterns = [
            'Traceback',
            'File "',
            'line ',
            'Exception:',
            'Error:',
            'TypeError:',
            'ValueError:',
            'AttributeError:',
            'ImportError:',
        ]
        
        return not any(pattern in message for pattern in unfriendly_patterns)
    
    def _get_error_suggestions(self, error_details: Dict[str, Any]) -> List[str]:
        """Get specific suggestions based on error type and context."""
        error_type = error_details['error_type']
        error_message = error_details['error_message'].lower()
        suggestions = []
        
        # Configuration-related suggestions
        if 'config' in error_type.lower():
            suggestions.extend([
                "Check your configuration file syntax and structure",
                "Ensure all required environment variables are set",
                "Copy from the default config.yaml as a reference",
                "Use 'assistant --help' to see configuration options"
            ])
        
        # API key suggestions
        if 'api' in error_message or 'key' in error_message:
            suggestions.extend([
                "Verify your API key is correct and has necessary permissions",
                "Check that environment variables are properly set",
                "Ensure your API key hasn't expired",
                "Try using a different provider with '--provider' flag"
            ])
        
        # Network-related suggestions
        if any(term in error_message for term in ['network', 'connection', 'timeout']):
            suggestions.extend([
                "Check your internet connection",
                "Try again in a few moments (may be temporary)",
                "Consider using a different AI provider",
                "Check if you're behind a firewall or proxy"
            ])
        
        # Memory-related suggestions
        if 'memory' in error_type.lower():
            suggestions.extend([
                "Ensure the memory server dependencies are installed",
                "Check that Python is available in your PATH",
                "Try resetting memory with '--reset-memory'",
                "Verify file permissions in ~/.assistant/ directory"
            ])
        
        # File-related suggestions
        if 'file' in error_message or error_type in ['FileNotFoundError', 'PermissionError']:
            suggestions.extend([
                "Check that all required files exist",
                "Verify you have read/write permissions",
                "Ensure the file path is correct",
                "Try running with elevated permissions if necessary"
            ])
        
        # General suggestions for any error
        suggestions.extend([
            "Run with '--verbose' flag for more detailed information",
            "Check the documentation for troubleshooting guides",
            "Try the operation again (some errors are temporary)"
        ])
        
        return suggestions[:5]  # Limit to 5 most relevant suggestions
    
    def _get_troubleshooting_info(self, error_details: Dict[str, Any]) -> List[str]:
        """Get troubleshooting information based on error details."""
        troubleshooting = []
        error_type = error_details['error_type']
        
        # Add error code/type information
        troubleshooting.append(f"Error code: {error_type}")
        
        # Add file location if available
        if error_details.get('file_location') and error_details.get('line_number'):
            file_name = os.path.basename(error_details['file_location'])
            troubleshooting.append(f"Location: {file_name}:{error_details['line_number']}")
        
        # Add timestamp
        troubleshooting.append(f"Time: {error_details['timestamp']}")
        
        # Add session info
        troubleshooting.append(f"Session log: {self.session_log_file.name}")
        
        return troubleshooting
    
    def get_error_summary(self) -> Dict[str, Any]:
        """Get summary of errors encountered in this session."""
        return {
            'total_errors': len(self.session_errors),
            'error_types': dict(self.error_counts),
            'most_common_error': max(self.error_counts, key=self.error_counts.get) if self.error_counts else None,
            'session_log': str(self.session_log_file),
            'error_log': str(self.error_log_file)
        }
    
    def clear_session_errors(self) -> None:
        """Clear session error tracking."""
        self.session_errors.clear()
        self.error_counts.clear()

@contextmanager
def error_context(reporter: ErrorReporter, 
                 user_action: str,
                 context: Optional[Dict[str, Any]] = None,
                 raise_on_error: bool = True):
    """
    Context manager for automatic error reporting.
    
    Args:
        reporter: ErrorReporter instance
        user_action: Description of what the user is trying to do
        context: Additional context information
        raise_on_error: Whether to re-raise the error after reporting
        
    Example:
        with error_context(reporter, "loading configuration"):
            config = load_config()
    """
    try:
        yield
    except Exception as e:
        user_message = reporter.report_error(e, context, user_action)
        print(user_message, file=sys.stderr)
        
        if raise_on_error:
            raise

# Global error reporter instance
_global_reporter = None

def get_error_reporter(verbose: bool = False) -> ErrorReporter:
    """Get the global error reporter instance."""
    global _global_reporter
    if _global_reporter is None:
        _global_reporter = ErrorReporter(verbose=verbose)
    return _global_reporter

def report_error(error: Exception,
                context: Optional[Dict[str, Any]] = None,
                user_action: Optional[str] = None,
                verbose: bool = False) -> str:
    """Convenience function for reporting errors."""
    reporter = get_error_reporter(verbose)
    return reporter.report_error(error, context, user_action)

def setup_global_error_handling(verbose: bool = False) -> None:
    """Setup global error handling for uncaught exceptions."""
    def handle_exception(exc_type, exc_value, exc_traceback):
        if issubclass(exc_type, KeyboardInterrupt):
            # Allow keyboard interrupts to exit normally
            sys.__excepthook__(exc_type, exc_value, exc_traceback)
            return
        
        # Report the error
        reporter = get_error_reporter(verbose)
        user_message = reporter.report_error(
            exc_value, 
            context={'uncaught': True},
            user_action="running the assistant"
        )
        
        print(f"\n{user_message}", file=sys.stderr)
        print("\nThe assistant encountered an unexpected error and will exit.", file=sys.stderr)
    
    sys.excepthook = handle_exception 