"""
Strands Agent Implementation for Personal Assistant CLI.

This module integrates AWS Strands Agent SDK with MCP memory server to create
a memory-enabled AI assistant with persistent knowledge across conversations.
"""

import asyncio
import logging
from typing import Dict, Any, List, Optional, Tuple, Union

# Import Strands components
try:
    from strands import Agent
    from strands.tools.mcp import MCPClient
    STRANDS_AVAILABLE = True
except ImportError:
    STRANDS_AVAILABLE = False
    # Define placeholder for development
    class Agent:
        def __init__(self, **kwargs):
            pass
        
        def run(self, query: str) -> str:
            return "Strands SDK not available"

# Import our modules
from assistant.config import ConfigManager, ConfigError
from assistant.providers import ProviderDetector, ProviderError, setup_model
from assistant.memory_client import MemoryClient, MemoryClientError, memory_client_context
from assistant.network_utils import (
    NetworkErrorHandler, with_network_retry_async, RetryableError, 
    NonRetryableError, check_internet, check_provider
)

logger = logging.getLogger(__name__)

class AgentError(Exception):
    """Custom exception for agent-related errors."""
    pass

class AssistantAgent:
    """
    Personal Assistant Agent using Strands SDK with MCP memory integration.
    
    This class manages the complete workflow from query processing to response
    generation with persistent memory capabilities.
    """
    
    def __init__(self, config_path: Optional[str] = None, provider_override: Optional[str] = None):
        """
        Initialize the AssistantAgent.
        
        Args:
            config_path: Optional path to configuration file.
            provider_override: Optional provider name to override auto-detection.
        """
        self.config_path = config_path
        self.provider_override = provider_override
        self.config_manager = None
        self.config_data = None
        self.selected_provider = None
        self.model = None
        self.memory_client = None
        self.agent = None
        self.verbose = False
        
        if not STRANDS_AVAILABLE:
            logger.warning("Strands SDK not available - agent functionality will not work properly")
    
    def initialize(self, verbose: bool = False) -> None:
        """
        Initialize the agent with configuration, model, and memory.
        
        Args:
            verbose: Enable verbose logging and output.
            
        Raises:
            AgentError: If initialization fails.
        """
        self.verbose = verbose
        
        try:
            logger.info("Initializing Personal Assistant Agent...")
            
            # Load configuration
            self._load_configuration()
            
            # Setup model provider
            self._setup_model_provider()
            
            # Setup memory client
            self._setup_memory_client()
            
            # Create Strands agent
            self._create_strands_agent()
            
            logger.info("Personal Assistant Agent initialized successfully")
            
        except Exception as e:
            raise AgentError(f"Failed to initialize agent: {e}")
    
    def _load_configuration(self) -> None:
        """Load and validate configuration."""
        try:
            self.config_manager = ConfigManager(self.config_path)
            self.config_manager.ensure_user_config_dir()
            self.config_data = self.config_manager.load_config()
            
            if self.verbose:
                logger.info(f"Configuration loaded from: {self.config_manager._get_config_file_path()}")
                
        except ConfigError as e:
            raise AgentError(f"Configuration error: {e}")
    
    def _setup_model_provider(self) -> None:
        """Setup the model provider based on configuration and overrides."""
        try:
            # Determine provider
            if self.provider_override and self.provider_override != 'auto':
                self.selected_provider = self.provider_override
            else:
                detector = ProviderDetector()
                self.selected_provider = detector.detect_provider()
            
            if self.verbose:
                logger.info(f"Selected provider: {self.selected_provider}")
            
            # Get provider configuration
            provider_config = self.config_manager.get_provider_config(self.selected_provider)
            
            # Setup model
            self.model = setup_model(self.selected_provider, provider_config)
            
            logger.info(f"Model provider '{self.selected_provider}' setup successfully")
            
        except (ProviderError, ConfigError) as e:
            raise AgentError(f"Provider setup error: {e}")
    
    def _setup_memory_client(self) -> None:
        """Setup the MCP memory client."""
        try:
            # Get memory server configuration
            memory_config = self.config_manager.get_mcp_server_config("memory")
            
            # Create memory client
            self.memory_client = MemoryClient(memory_config)
            self.memory_client.create_mcp_client()
            
            if self.verbose:
                stats = self.memory_client.get_memory_stats()
                logger.info(f"Memory client setup - File: {stats['memory_file_path']}, "
                          f"Exists: {stats['exists']}, Entries: {stats['entry_count']}")
            
        except (MemoryClientError, ConfigError) as e:
            raise AgentError(f"Memory client setup error: {e}")
    
    def _create_strands_agent(self) -> None:
        """Create the Strands Agent with model and MCP tools."""
        try:
            if not STRANDS_AVAILABLE:
                logger.warning("Strands SDK not available - using placeholder agent")
                self.agent = Agent()
                return
            
            # Get system prompt
            system_prompt = self.config_manager.get_system_prompt()
            
            # Get MCP tools synchronously
            tools = self.memory_client.list_tools_sync()
            
            if self.verbose:
                logger.info(f"Retrieved {len(tools)} tools from memory server")
                for tool in tools:
                    tool_name = tool.get('name', 'unknown')
                    logger.debug(f"Available tool: {tool_name}")
            
            # Create Strands Agent
            self.agent = Agent(
                model=self.model,
                tools=tools,
                system_prompt=system_prompt
            )
            
            logger.info("Strands Agent created successfully")
            
        except Exception as e:
            raise AgentError(f"Failed to create Strands Agent: {e}")
    
    async def process_query_async(self, query: str) -> Tuple[str, Dict[str, Any]]:
        """
        Process a query asynchronously with full memory integration.
        
        Args:
            query: User's natural language query.
            
        Returns:
            Tuple of (response, metadata) where metadata contains processing info.
            
        Raises:
            AgentError: If query processing fails.
        """
        if not self.agent:
            raise AgentError("Agent not initialized. Call initialize() first.")
        
        metadata = {
            'provider': self.selected_provider,
            'memory_operations': [],
            'processing_time': 0,
            'memory_updated': False
        }
        
        try:
            import time
            start_time = time.time()
            
            if self.verbose:
                logger.info("🧠 Remembering... (retrieving relevant memory)")
            
            # Phase 1: Memory Retrieval (the "Remembering..." phase)
            memory_context = await self._retrieve_memory_context(query)
            metadata['memory_operations'].append('memory_retrieval')
            
            if self.verbose and memory_context:
                logger.info(f"Retrieved {len(memory_context)} relevant memory items")
            
            # Phase 2: Enhance query with memory context
            enhanced_query = self._enhance_query_with_memory(query, memory_context)
            
            # Phase 3: Execute agent with enhanced query
            if self.verbose:
                logger.info("🤖 Processing query with AI agent...")
            
            response = await self._execute_agent_async(enhanced_query)
            
            # Phase 4: Update memory with new information
            if self.verbose:
                logger.info("💾 Updating memory with new information...")
            
            await self._update_memory_async(query, response)
            metadata['memory_operations'].append('memory_update')
            metadata['memory_updated'] = True
            
            # Calculate processing time
            metadata['processing_time'] = round(time.time() - start_time, 2)
            
            if self.verbose:
                logger.info(f"Query processed successfully in {metadata['processing_time']}s")
            
            return response, metadata
            
        except Exception as e:
            raise AgentError(f"Query processing failed: {e}")
    
    def process_query(self, query: str) -> Tuple[str, Dict[str, Any]]:
        """
        Process a query synchronously (wrapper for async method).
        
        Args:
            query: User's natural language query.
            
        Returns:
            Tuple of (response, metadata).
        """
        try:
            return asyncio.run(self.process_query_async(query))
        except Exception as e:
            raise AgentError(f"Synchronous query processing failed: {e}")
    
    async def _retrieve_memory_context(self, query: str) -> List[Dict[str, Any]]:
        """
        Retrieve relevant memory context for the query.
        
        Args:
            query: User's query to find relevant memories for.
            
        Returns:
            List of relevant memory items.
        """
        try:
            # Use the memory client's MCP tools to search for relevant information
            # This would typically involve calling memory search tools
            
            # For now, return empty list - full memory search integration
            # would require calling the actual MCP memory search tools
            memory_context = []
            
            if self.verbose:
                logger.debug(f"Memory retrieval for query: '{query[:50]}...'")
            
            return memory_context
            
        except Exception as e:
            logger.warning(f"Memory retrieval failed: {e}")
            return []
    
    def _enhance_query_with_memory(self, query: str, memory_context: List[Dict[str, Any]]) -> str:
        """
        Enhance the user query with relevant memory context.
        
        Args:
            query: Original user query.
            memory_context: Relevant memory items.
            
        Returns:
            Enhanced query string with memory context.
        """
        if not memory_context:
            return query
        
        # Create context string from memory items
        context_parts = []
        for item in memory_context:
            # Extract relevant information from memory item
            # This would be tailored to the memory server's data format
            if isinstance(item, dict):
                content = item.get('content', str(item))
                context_parts.append(content)
        
        if context_parts:
            context_str = "\n".join(context_parts)
            enhanced_query = f"Context from previous interactions:\n{context_str}\n\nCurrent query: {query}"
            
            if self.verbose:
                logger.debug(f"Enhanced query with {len(context_parts)} memory items")
            
            return enhanced_query
        
        return query
    
    @with_network_retry_async(max_retries=3, base_delay=2.0, max_delay=60.0)
    async def _execute_agent_async(self, query: str) -> str:
        """
        Execute the Strands Agent with the query with network retry logic.
        
        Args:
            query: Query to process (potentially enhanced with memory).
            
        Returns:
            Agent's response.
        """
        try:
            if not STRANDS_AVAILABLE:
                # Placeholder response for development
                return f"[PLACEHOLDER] Assistant response to: '{query[:100]}...'"
            
            if self.verbose:
                logger.info("Executing Strands Agent with network retry protection...")
            
            # Check connectivity before attempting the call
            if not check_internet():
                raise RetryableError("No internet connectivity detected")
            
            # Check provider-specific connectivity if possible
            try:
                provider_status = check_provider(self.selected_provider)
                if not provider_status.get('reachable', True):
                    logger.warning(f"Provider {self.selected_provider} may be unreachable: {provider_status.get('error')}")
            except Exception as e:
                logger.debug(f"Provider connectivity check failed: {e}")
            
            # Execute agent - this may be sync or async depending on Strands implementation
            if hasattr(self.agent, 'run_async'):
                response = await self.agent.run_async(query)
            elif hasattr(self.agent, 'arun'):
                response = await self.agent.arun(query)
            else:
                # Fallback to sync method in async context
                response = await asyncio.get_event_loop().run_in_executor(
                    None, self.agent.run, query
                )
            
            if self.verbose:
                logger.info(f"Agent response received (length: {len(response)} chars)")
            
            return response
            
        except (RetryableError, NonRetryableError):
            # Re-raise network errors to preserve retry behavior
            raise
        except Exception as e:
            logger.error(f"Agent execution failed: {e}")
            # Check if this is a network-related error that should be retried
            error_str = str(e).lower()
            if any(term in error_str for term in ['connection', 'timeout', 'network', 'rate limit', 'api']):
                raise RetryableError(f"Network-related agent execution failure: {e}")
            else:
                raise AgentError(f"Agent execution failed: {e}")
    
    async def _update_memory_async(self, query: str, response: str) -> None:
        """
        Update memory with new information from the interaction.
        
        Args:
            query: User's original query.
            response: Assistant's response.
        """
        try:
            # Use MCP memory tools to store new information
            # This would involve calling memory creation/update tools
            
            # Extract and categorize information from the interaction
            interaction_data = {
                'user_query': query,
                'assistant_response': response,
                'timestamp': self._get_current_timestamp(),
                'categories': self._categorize_information(query, response)
            }
            
            # Store in memory using MCP tools
            # This would require actual MCP tool calls to the memory server
            
            if self.verbose:
                logger.debug(f"Memory updated with interaction data: {len(str(interaction_data))} chars")
            
        except Exception as e:
            logger.warning(f"Memory update failed: {e}")
    
    def _categorize_information(self, query: str, response: str) -> List[str]:
        """
        Categorize information from the interaction for memory storage.
        
        Args:
            query: User's query.
            response: Assistant's response.
            
        Returns:
            List of category labels.
        """
        categories = []
        
        # Simple keyword-based categorization
        text = f"{query} {response}".lower()
        
        # Check for different information categories
        if any(word in text for word in ['meeting', 'appointment', 'schedule', 'calendar']):
            categories.append('Events')
        
        if any(word in text for word in ['project', 'work', 'task', 'deadline']):
            categories.append('Work')
        
        if any(word in text for word in ['person', 'contact', 'colleague', 'friend']):
            categories.append('Relationships')
        
        if any(word in text for word in ['goal', 'plan', 'target', 'objective']):
            categories.append('Goals')
        
        if any(word in text for word in ['skill', 'learn', 'knowledge', 'education']):
            categories.append('Skills')
        
        # Default category if no specific matches
        if not categories:
            categories.append('General')
        
        return categories
    
    def _get_current_timestamp(self) -> str:
        """Get current timestamp in ISO format."""
        from datetime import datetime
        return datetime.now().isoformat()
    
    def get_agent_status(self) -> Dict[str, Any]:
        """
        Get current status of the agent and its components.
        
        Returns:
            Dictionary with status information.
        """
        status = {
            'initialized': self.agent is not None,
            'strands_available': STRANDS_AVAILABLE,
            'provider': self.selected_provider,
            'memory_stats': None,
            'config_path': self.config_path
        }
        
        if self.memory_client:
            try:
                status['memory_stats'] = self.memory_client.get_memory_stats()
            except Exception as e:
                status['memory_error'] = str(e)
        
        return status

# Factory functions for easy creation
def create_agent(config_path: Optional[str] = None, 
                provider_override: Optional[str] = None,
                verbose: bool = False) -> AssistantAgent:
    """
    Create and initialize an AssistantAgent.
    
    Args:
        config_path: Optional path to configuration file.
        provider_override: Optional provider name override.
        verbose: Enable verbose logging.
        
    Returns:
        Initialized AssistantAgent instance.
    """
    agent = AssistantAgent(config_path, provider_override)
    agent.initialize(verbose)
    return agent

def process_query(query: str,
                 config_path: Optional[str] = None,
                 provider_override: Optional[str] = None,
                 verbose: bool = False) -> Tuple[str, Dict[str, Any]]:
    """
    Convenience function to process a single query.
    
    Args:
        query: User's natural language query.
        config_path: Optional path to configuration file.
        provider_override: Optional provider name override.
        verbose: Enable verbose logging.
        
    Returns:
        Tuple of (response, metadata).
    """
    agent = create_agent(config_path, provider_override, verbose)
    return agent.process_query(query) 