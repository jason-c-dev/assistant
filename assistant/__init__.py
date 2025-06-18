"""
Personal Assistant CLI - A memory-enabled AI assistant using AWS Strands Agent SDK.

This package provides a command-line interface for an AI assistant that maintains
persistent memory across conversations using the Model Context Protocol (MCP).
"""

__version__ = "0.1.0"
__author__ = "Assistant Development Team"
__description__ = "A memory-enabled AI assistant CLI using AWS Strands Agent SDK and MCP"
__license__ = "MIT"

# Package level imports for easy access
from assistant.cli import main
from assistant.config import load_config
from assistant.providers import detect_provider, setup_model
from assistant.agent import create_agent

__all__ = [
    "main",
    "load_config", 
    "detect_provider",
    "setup_model",
    "create_agent",
    "__version__",
] 