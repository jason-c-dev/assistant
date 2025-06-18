"""
MCP Memory Server Integration for Personal Assistant CLI.

This module handles integration with the Python MCP memory server using Strands'
MCPClient with stdio transport and proper lifecycle management.
"""

import os
import json
import time
import asyncio
import logging
import subprocess
import signal
from pathlib import Path
from typing import Dict, Any, List, Optional, Union
from contextlib import asynccontextmanager

# Import Strands MCP integration components
try:
    from mcp import stdio_client, StdioServerParameters
    from strands.tools.mcp import MCPClient
    MCP_AVAILABLE = True
except ImportError:
    MCP_AVAILABLE = False
    # Define placeholder classes for development
    class MCPClient:
        def __init__(self, *args, **kwargs):
            pass
    
    def stdio_client(*args, **kwargs):
        pass
    
    class StdioServerParameters:
        def __init__(self, *args, **kwargs):
            pass

logger = logging.getLogger(__name__)

class MemoryClientError(Exception):
    """Custom exception for memory client-related errors."""
    pass

class MemoryServerStartupError(MemoryClientError):
    """Specific exception for memory server startup failures."""
    pass

class MemoryServerValidationError(MemoryClientError):
    """Specific exception for memory server validation failures."""
    pass

class MemoryClient:
    """
    Manages MCP memory server integration with lifecycle management and operations.
    
    This class follows Strands patterns for MCP integration with stdio transport
    and provides high-level interfaces for memory operations with comprehensive
    error handling and graceful fallbacks.
    """
    
    DEFAULT_STARTUP_TIMEOUT = 10  # seconds
    DEFAULT_RETRY_ATTEMPTS = 3
    
    def __init__(self, memory_config: Dict[str, Any]):
        """
        Initialize MemoryClient with configuration.
        
        Args:
            memory_config: Memory server configuration from config.yaml
        """
        self.memory_config = memory_config
        self.memory_file_path = self._resolve_memory_file_path()
        self.mcp_client = None
        self._tools_cache = None
        self._fallback_mode = False
        self._validation_cache = None
        
        if not MCP_AVAILABLE:
            logger.warning("MCP components not available - memory integration will not work properly")
    
    def _resolve_memory_file_path(self) -> str:
        """
        Resolve the memory file path from configuration with environment variable support.
        
        Returns:
            Resolved absolute path for memory storage.
        """
        # Get path from memory server config
        memory_server_config = self.memory_config
        memory_file_path = memory_server_config.get('env', {}).get('MEMORY_FILE_PATH', '~/.assistant/memory.json')
        
        # Expand user home directory
        expanded_path = os.path.expanduser(memory_file_path)
        
        # Ensure directory exists
        memory_dir = os.path.dirname(expanded_path)
        os.makedirs(memory_dir, exist_ok=True)
        
        logger.info(f"Memory file path resolved to: {expanded_path}")
        return expanded_path
    
    def validate_server_dependencies(self) -> Dict[str, Any]:
        """
        Validate that memory server dependencies are available.
        
        Returns:
            Dictionary with validation results and detailed error information.
        """
        if self._validation_cache is not None:
            return self._validation_cache
        
        validation_result = {
            'valid': True,
            'errors': [],
            'warnings': [],
            'suggestions': [],
            'details': {}
        }
        
        try:
            # Check if command exists
            command = self.memory_config.get('command', 'python')
            args = self.memory_config.get('args', ['./memory_server/mcp_memory_server.py'])
            
            # Validate command executable
            try:
                result = subprocess.run([command, '--version'], 
                                      capture_output=True, text=True, timeout=5)
                validation_result['details']['command_available'] = True
                validation_result['details']['command_version'] = result.stdout.strip()
            except (subprocess.TimeoutExpired, subprocess.CalledProcessError, FileNotFoundError) as e:
                validation_result['valid'] = False
                validation_result['errors'].append(f"Command '{command}' not available or not working")
                validation_result['suggestions'].append(f"Ensure '{command}' is installed and in PATH")
                validation_result['details']['command_available'] = False
                validation_result['details']['command_error'] = str(e)
            
            # Validate memory server script exists
            if args and len(args) > 0:
                server_script = args[0]
                server_path = os.path.expanduser(server_script)
                
                if os.path.exists(server_path):
                    validation_result['details']['server_script_exists'] = True
                    validation_result['details']['server_script_path'] = server_path
                else:
                    validation_result['valid'] = False
                    validation_result['errors'].append(f"Memory server script not found: {server_path}")
                    validation_result['suggestions'].append("Ensure memory server is properly installed")
                    validation_result['suggestions'].append("Run 'git submodule update --init --recursive' if using git submodule")
                    validation_result['details']['server_script_exists'] = False
            
            # Check memory server dependencies (if script exists)
            if validation_result['details'].get('server_script_exists', False):
                try:
                    # Try to check if server script can be executed
                    test_result = subprocess.run([command, server_path, '--help'], 
                                               capture_output=True, text=True, timeout=10)
                    validation_result['details']['server_executable'] = True
                except subprocess.TimeoutExpired:
                    validation_result['warnings'].append("Memory server script check timed out")
                    validation_result['details']['server_executable'] = 'timeout'
                except subprocess.CalledProcessError as e:
                    validation_result['warnings'].append(f"Memory server script execution failed: {e}")
                    validation_result['details']['server_executable'] = False
                except Exception as e:
                    validation_result['warnings'].append(f"Could not validate server script: {e}")
                    validation_result['details']['server_executable'] = 'unknown'
            
            # Validate memory file directory is writable
            memory_dir = os.path.dirname(self.memory_file_path)
            if os.access(memory_dir, os.W_OK):
                validation_result['details']['memory_dir_writable'] = True
            else:
                validation_result['valid'] = False
                validation_result['errors'].append(f"Memory directory not writable: {memory_dir}")
                validation_result['suggestions'].append(f"Ensure write permissions for directory: {memory_dir}")
                validation_result['details']['memory_dir_writable'] = False
            
        except Exception as e:
            validation_result['valid'] = False
            validation_result['errors'].append(f"Validation failed with error: {e}")
            validation_result['details']['validation_error'] = str(e)
        
        self._validation_cache = validation_result
        return validation_result
    
    def create_mcp_client(self, retry_attempts: int = None, startup_timeout: int = None) -> MCPClient:
        """
        Create MCPClient with comprehensive error handling and graceful fallbacks.
        
        Args:
            retry_attempts: Number of retry attempts for server startup (default: 3)
            startup_timeout: Timeout in seconds for server startup (default: 10)
        
        Returns:
            Configured MCPClient instance.
            
        Raises:
            MemoryServerStartupError: If server startup fails after all retries.
            MemoryServerValidationError: If server dependencies are not available.
        """
        retry_attempts = retry_attempts or self.DEFAULT_RETRY_ATTEMPTS
        startup_timeout = startup_timeout or self.DEFAULT_STARTUP_TIMEOUT
        
        # First validate dependencies
        validation = self.validate_server_dependencies()
        if not validation['valid']:
            error_msg = "Memory server validation failed:\n"
            for error in validation['errors']:
                error_msg += f"  ❌ {error}\n"
            if validation['suggestions']:
                error_msg += "Suggestions:\n"
                for suggestion in validation['suggestions']:
                    error_msg += f"  💡 {suggestion}\n"
            
            # Check if we can enable fallback mode
            if self._can_enable_fallback():
                logger.warning("Memory server validation failed, enabling fallback mode")
                self._fallback_mode = True
                return self._create_fallback_client()
            else:
                raise MemoryServerValidationError(error_msg.strip())
        
        # Attempt to create client with retries
        last_error = None
        for attempt in range(retry_attempts):
            try:
                logger.info(f"Creating MCP client (attempt {attempt + 1}/{retry_attempts})")
                
                # Get command and args from config
                command = self.memory_config.get('command', 'python')
                args = self.memory_config.get('args', ['./memory_server/mcp_memory_server.py'])
                
                # Prepare environment variables
                env = dict(os.environ)  # Start with current environment
                config_env = self.memory_config.get('env', {})
                
                # Add memory file path to environment
                env['MEMORY_FILE_PATH'] = self.memory_file_path
                
                # Add any additional env vars from config
                for key, value in config_env.items():
                    if key != 'MEMORY_FILE_PATH':  # Don't override our resolved path
                        env[key] = str(value)
                
                logger.debug(f"Creating MCP client with command: {command}, args: {args}")
                logger.debug(f"Memory file path in environment: {env.get('MEMORY_FILE_PATH')}")
                
                # Create MCPClient following Strands patterns
                mcp_client = MCPClient(
                    lambda: stdio_client(StdioServerParameters(
                        command=command,
                        args=args,
                        env=env
                    ))
                )
                
                # Test the client connection with timeout
                if self._test_client_connection(mcp_client, startup_timeout):
                    self.mcp_client = mcp_client
                    logger.info("MCP client created successfully")
                    return mcp_client
                else:
                    raise MemoryServerStartupError(f"Server connection test failed (timeout: {startup_timeout}s)")
                    
            except Exception as e:
                last_error = e
                logger.warning(f"MCP client creation attempt {attempt + 1} failed: {e}")
                
                if attempt < retry_attempts - 1:
                    wait_time = 2 ** attempt  # Exponential backoff
                    logger.info(f"Retrying in {wait_time} seconds...")
                    time.sleep(wait_time)
        
        # All attempts failed
        error_msg = f"Failed to create MCP client after {retry_attempts} attempts. Last error: {last_error}"
        
        # Check if we can enable fallback mode
        if self._can_enable_fallback():
            logger.warning("MCP client creation failed, enabling fallback mode")
            self._fallback_mode = True
            return self._create_fallback_client()
        else:
            raise MemoryServerStartupError(error_msg)
    
    def _test_client_connection(self, client: MCPClient, timeout: int) -> bool:
        """
        Test if the MCP client connection is working.
        
        Args:
            client: MCPClient instance to test.
            timeout: Timeout in seconds for the test.
            
        Returns:
            True if connection test succeeds, False otherwise.
        """
        try:
            # Use asyncio.wait_for to implement timeout
            async def test_connection():
                async with client as test_client:
                    await test_client.list_tools()
                    return True
            
            # Run the test with timeout
            result = asyncio.wait_for(test_connection(), timeout=timeout)
            asyncio.run(result)
            return True
            
        except asyncio.TimeoutError:
            logger.warning(f"Client connection test timed out after {timeout} seconds")
            return False
        except Exception as e:
            logger.warning(f"Client connection test failed: {e}")
            return False
    
    def _can_enable_fallback(self) -> bool:
        """
        Check if fallback mode can be enabled.
        
        Returns:
            True if fallback mode is possible, False otherwise.
        """
        # Fallback is possible if memory file directory is writable
        memory_dir = os.path.dirname(self.memory_file_path)
        return os.access(memory_dir, os.W_OK)
    
    def _create_fallback_client(self) -> 'FallbackMemoryClient':
        """
        Create a fallback memory client that provides basic functionality.
        
        Returns:
            FallbackMemoryClient instance.
        """
        return FallbackMemoryClient(self.memory_file_path)
    
    def is_fallback_mode(self) -> bool:
        """
        Check if client is running in fallback mode.
        
        Returns:
            True if in fallback mode, False otherwise.
        """
        return self._fallback_mode
    
    async def list_tools_async(self) -> List[Dict[str, Any]]:
        """
        List available tools from the memory server asynchronously.
        
        Returns:
            List of tool definitions.
            
        Raises:
            MemoryClientError: If tools listing fails.
        """
        if self._fallback_mode:
            logger.info("Using fallback mode for tool listing")
            return self._get_fallback_tools()
        
        if not self.mcp_client:
            raise MemoryClientError("MCP client not created. Call create_mcp_client() first.")
        
        try:
            # Use async context manager for proper lifecycle
            async with self.mcp_client as client:
                tools = await client.list_tools()
                logger.info(f"Retrieved {len(tools)} tools from memory server")
                return tools
                
        except Exception as e:
            logger.warning(f"Failed to list tools from MCP server: {e}")
            
            # Try to enable fallback mode if not already enabled
            if self._can_enable_fallback():
                logger.info("Enabling fallback mode due to tools listing failure")
                self._fallback_mode = True
                return self._get_fallback_tools()
            else:
                raise MemoryClientError(f"Failed to list tools: {e}")
    
    def list_tools_sync(self) -> List[Dict[str, Any]]:
        """
        List available tools from the memory server synchronously.
        
        Returns:
            List of tool definitions.
            
        Raises:
            MemoryClientError: If tools listing fails.
        """
        if self._fallback_mode:
            logger.info("Using fallback mode for tool listing (sync)")
            return self._get_fallback_tools()
        
        if self._tools_cache is not None and not self._fallback_mode:
            return self._tools_cache
        
        if not self.mcp_client:
            raise MemoryClientError("MCP client not created. Call create_mcp_client() first.")
        
        try:
            # For synchronous access, we use the sync version if available
            if hasattr(self.mcp_client, 'list_tools_sync'):
                tools = self.mcp_client.list_tools_sync()
            else:
                # Fallback to running async in a new event loop
                tools = asyncio.run(self.list_tools_async())
            
            self._tools_cache = tools
            logger.info(f"Retrieved {len(tools)} tools from memory server (sync)")
            return tools
            
        except Exception as e:
            logger.warning(f"Failed to list tools synchronously: {e}")
            
            # Try to enable fallback mode if not already enabled
            if self._can_enable_fallback():
                logger.info("Enabling fallback mode due to sync tools listing failure")
                self._fallback_mode = True
                return self._get_fallback_tools()
            else:
                raise MemoryClientError(f"Failed to list tools synchronously: {e}")
    
    def _get_fallback_tools(self) -> List[Dict[str, Any]]:
        """
        Get a basic set of fallback tools when MCP server is not available.
        
        Returns:
            List of basic memory tool definitions.
        """
        return [
            {
                "name": "read_memory_fallback",
                "description": "Read memory data from local file (fallback mode)",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": "Search query for memory"
                        }
                    }
                }
            },
            {
                "name": "write_memory_fallback",
                "description": "Write memory data to local file (fallback mode)", 
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "content": {
                            "type": "string",
                            "description": "Content to store in memory"
                        }
                    }
                }
            }
        ]
    
    @asynccontextmanager
    async def get_client_context(self):
        """
        Async context manager for MCP client lifecycle management.
        
        Yields:
            MCPClient instance for use within context.
            
        Example:
            async with memory_client.get_client_context() as client:
                tools = await client.list_tools()
        """
        if not self.mcp_client:
            raise MemoryClientError("MCP client not created. Call create_mcp_client() first.")
        
        async with self.mcp_client as client:
            yield client
    
    def export_memory(self, export_path: Optional[str] = None) -> str:
        """
        Export memory data to JSON file.
        
        Args:
            export_path: Optional custom export path. If None, creates timestamped export.
            
        Returns:
            Path to the exported file.
            
        Raises:
            MemoryClientError: If export fails.
        """
        try:
            # Determine export path
            if export_path is None:
                from datetime import datetime
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                export_path = f"memory_export_{timestamp}.json"
            
            export_path = os.path.expanduser(export_path)
            
            # Check if memory file exists
            if not os.path.exists(self.memory_file_path):
                logger.warning(f"Memory file does not exist: {self.memory_file_path}")
                # Create empty export
                with open(export_path, 'w', encoding='utf-8') as f:
                    json.dump([], f, indent=2)
                return export_path
            
            # Copy memory file to export location
            import shutil
            shutil.copy2(self.memory_file_path, export_path)
            
            logger.info(f"Memory exported to: {export_path}")
            return export_path
            
        except Exception as e:
            raise MemoryClientError(f"Failed to export memory: {e}")
    
    def import_memory(self, import_path: str, backup: bool = True) -> None:
        """
        Import memory data from JSON file.
        
        Args:
            import_path: Path to the JSON file to import.
            backup: Whether to backup existing memory before import.
            
        Raises:
            MemoryClientError: If import fails.
        """
        try:
            import_path = os.path.expanduser(import_path)
            
            # Validate import file exists
            if not os.path.exists(import_path):
                raise MemoryClientError(f"Import file does not exist: {import_path}")
            
            # Validate JSON format
            with open(import_path, 'r', encoding='utf-8') as f:
                try:
                    json.load(f)
                except json.JSONDecodeError as e:
                    raise MemoryClientError(f"Invalid JSON in import file: {e}")
            
            # Backup existing memory if requested
            if backup and os.path.exists(self.memory_file_path):
                backup_path = self.export_memory(f"{self.memory_file_path}.backup")
                logger.info(f"Existing memory backed up to: {backup_path}")
            
            # Import memory
            import shutil
            shutil.copy2(import_path, self.memory_file_path)
            
            logger.info(f"Memory imported from: {import_path}")
            
        except Exception as e:
            raise MemoryClientError(f"Failed to import memory: {e}")
    
    def reset_memory(self, confirm: bool = False) -> None:
        """
        Reset memory by deleting the memory file.
        
        Args:
            confirm: Must be True to actually perform the reset.
            
        Raises:
            MemoryClientError: If reset fails or not confirmed.
        """
        if not confirm:
            raise MemoryClientError("Memory reset requires explicit confirmation. Set confirm=True.")
        
        try:
            if os.path.exists(self.memory_file_path):
                # Create a backup before deletion
                backup_path = self.export_memory(f"{self.memory_file_path}.pre_reset_backup")
                logger.info(f"Memory backed up before reset to: {backup_path}")
                
                # Delete memory file
                os.remove(self.memory_file_path)
                logger.info(f"Memory file deleted: {self.memory_file_path}")
            else:
                logger.info(f"Memory file does not exist, nothing to reset: {self.memory_file_path}")
                
        except Exception as e:
            raise MemoryClientError(f"Failed to reset memory: {e}")
    
    def get_memory_stats(self) -> Dict[str, Any]:
        """
        Get statistics about the memory file.
        
        Returns:
            Dictionary with memory statistics.
        """
        stats = {
            'memory_file_path': self.memory_file_path,
            'exists': False,
            'size_bytes': 0,
            'size_mb': 0.0,
            'entry_count': 0,
            'last_modified': None
        }
        
        try:
            if os.path.exists(self.memory_file_path):
                stats['exists'] = True
                
                # Get file size
                size_bytes = os.path.getsize(self.memory_file_path)
                stats['size_bytes'] = size_bytes
                stats['size_mb'] = round(size_bytes / (1024 * 1024), 2)
                
                # Get modification time
                import datetime
                mtime = os.path.getmtime(self.memory_file_path)
                stats['last_modified'] = datetime.datetime.fromtimestamp(mtime).isoformat()
                
                # Count entries (JSONL format - one entry per line)
                try:
                    with open(self.memory_file_path, 'r', encoding='utf-8') as f:
                        stats['entry_count'] = sum(1 for line in f if line.strip())
                except Exception:
                    stats['entry_count'] = 'unknown'
            
        except Exception as e:
            logger.debug(f"Error getting memory stats: {e}")
        
        return stats
    
    def validate_memory_file(self) -> Dict[str, Any]:
        """
        Validate the memory file format and structure.
        
        Returns:
            Dictionary with validation results.
        """
        validation = {
            'valid': False,
            'errors': [],
            'warnings': [],
            'format': 'unknown'
        }
        
        try:
            if not os.path.exists(self.memory_file_path):
                validation['errors'].append('Memory file does not exist')
                return validation
            
            # Try to parse as JSONL
            with open(self.memory_file_path, 'r', encoding='utf-8') as f:
                line_count = 0
                for line_num, line in enumerate(f, 1):
                    line = line.strip()
                    if not line:
                        continue
                    
                    line_count += 1
                    try:
                        json.loads(line)
                    except json.JSONDecodeError as e:
                        validation['errors'].append(f"Invalid JSON on line {line_num}: {e}")
                        continue
                
                if not validation['errors']:
                    validation['valid'] = True
                    validation['format'] = 'jsonl'
                    if line_count == 0:
                        validation['warnings'].append('Memory file is empty')
            
        except Exception as e:
            validation['errors'].append(f"Error validating memory file: {e}")
        
        return validation

# Factory function for easy creation
def create_memory_client(memory_config: Dict[str, Any]) -> MemoryClient:
    """
    Create and setup a MemoryClient instance.
    
    Args:
        memory_config: Memory server configuration.
        
    Returns:
        Configured MemoryClient instance.
    """
    client = MemoryClient(memory_config)
    client.create_mcp_client()
    return client

# Context manager for easy usage
@asynccontextmanager
async def memory_client_context(memory_config: Dict[str, Any]):
    """
    Async context manager for memory client lifecycle.
    
    Args:
        memory_config: Memory server configuration.
        
    Yields:
        Tuple of (MemoryClient, MCPClient) for use within context.
        
    Example:
        async with memory_client_context(config) as (memory_client, mcp_client):
            tools = await mcp_client.list_tools()
    """
    memory_client = create_memory_client(memory_config)
    async with memory_client.get_client_context() as mcp_client:
        yield memory_client, mcp_client 

class FallbackMemoryClient:
    """
    Fallback memory client that provides basic memory functionality when MCP server is not available.
    
    This client provides file-based memory operations as a graceful fallback when the full
    MCP memory server cannot be started.
    """
    
    def __init__(self, memory_file_path: str):
        """
        Initialize fallback memory client.
        
        Args:
            memory_file_path: Path to the memory storage file.
        """
        self.memory_file_path = memory_file_path
        self.logger = logging.getLogger(f"{__name__}.FallbackMemoryClient")
        
        # Ensure memory file exists
        if not os.path.exists(self.memory_file_path):
            self._create_empty_memory_file()
    
    def _create_empty_memory_file(self):
        """Create an empty memory file with basic structure."""
        try:
            empty_memory = {
                "entities": [],
                "relations": [],
                "observations": [],
                "metadata": {
                    "created": time.time(),
                    "fallback_mode": True,
                    "version": "1.0"
                }
            }
            
            with open(self.memory_file_path, 'w', encoding='utf-8') as f:
                json.dump(empty_memory, f, indent=2)
                
            self.logger.info(f"Created empty memory file: {self.memory_file_path}")
            
        except Exception as e:
            self.logger.error(f"Failed to create empty memory file: {e}")
    
    def read_memory(self, query: str = None) -> Dict[str, Any]:
        """
        Read memory data from the file.
        
        Args:
            query: Optional search query (basic text matching).
            
        Returns:
            Dictionary containing memory data.
        """
        try:
            if not os.path.exists(self.memory_file_path):
                return {"entities": [], "relations": [], "observations": []}
            
            with open(self.memory_file_path, 'r', encoding='utf-8') as f:
                memory_data = json.load(f)
            
            # Basic query filtering if provided
            if query and query.strip():
                filtered_data = {"entities": [], "relations": [], "observations": []}
                query_lower = query.lower()
                
                # Filter entities
                for entity in memory_data.get("entities", []):
                    if query_lower in str(entity).lower():
                        filtered_data["entities"].append(entity)
                
                # Filter observations  
                for obs in memory_data.get("observations", []):
                    if query_lower in str(obs).lower():
                        filtered_data["observations"].append(obs)
                
                # Filter relations
                for rel in memory_data.get("relations", []):
                    if query_lower in str(rel).lower():
                        filtered_data["relations"].append(rel)
                
                return filtered_data
            
            return memory_data
            
        except Exception as e:
            self.logger.error(f"Failed to read memory: {e}")
            return {"entities": [], "relations": [], "observations": []}
    
    def write_memory(self, content: str) -> bool:
        """
        Write content to memory file.
        
        Args:
            content: Content to store (will be added as an observation).
            
        Returns:
            True if successful, False otherwise.
        """
        try:
            # Read existing memory
            memory_data = self.read_memory()
            
            # Add new observation
            observation = {
                "content": content,
                "timestamp": time.time(),
                "source": "fallback_client"
            }
            
            if "observations" not in memory_data:
                memory_data["observations"] = []
            
            memory_data["observations"].append(observation)
            
            # Update metadata
            if "metadata" not in memory_data:
                memory_data["metadata"] = {}
            
            memory_data["metadata"]["last_updated"] = time.time()
            memory_data["metadata"]["fallback_mode"] = True
            
            # Write back to file
            with open(self.memory_file_path, 'w', encoding='utf-8') as f:
                json.dump(memory_data, f, indent=2)
            
            self.logger.info(f"Added observation to memory: {content[:100]}...")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to write memory: {e}")
            return False
    
    def list_tools_sync(self) -> List[Dict[str, Any]]:
        """
        Get fallback tool definitions.
        
        Returns:
            List of fallback tool definitions.
        """
        return [
            {
                "name": "read_memory_fallback",
                "description": "Read memory data from local file (fallback mode)",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": "Search query for memory"
                        }
                    }
                }
            },
            {
                "name": "write_memory_fallback",
                "description": "Write memory data to local file (fallback mode)", 
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "content": {
                            "type": "string",
                            "description": "Content to store in memory"
                        }
                    }
                }
            }
        ] 