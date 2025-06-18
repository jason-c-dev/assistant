"""
Unit tests for agent integration and MCP operations.

Tests the assistant.agent module with various scenarios including
agent initialization, query processing, memory operations, and error handling.
"""

import os
import tempfile
import pytest
import json
from pathlib import Path
from unittest.mock import patch, MagicMock, AsyncMock

from assistant.agent import AssistantAgent
from assistant.config import ConfigError
from assistant.memory_client import MemoryClientError


class TestAssistantAgentInitialization:
    """Test AssistantAgent initialization scenarios."""
    
    @patch('assistant.agent.ConfigManager')
    @patch('assistant.agent.ProviderDetector')
    @patch('assistant.agent.MCPMemoryClient')
    def test_agent_initialization_default_config(self, mock_memory_client, mock_provider, mock_config):
        """Test agent initialization with default configuration."""
        # Mock configuration
        mock_config_instance = MagicMock()
        mock_config.return_value = mock_config_instance
        mock_config_instance.load_config.return_value = {
            'default_provider': 'auto',
            'providers': {'anthropic': {'api_key': 'test', 'model_id': 'claude-3'}},
            'system_prompt': 'Test prompt',
            'mcp_servers': {'memory': {'command': 'python', 'args': ['test.py']}}
        }
        
        # Mock provider detector
        mock_detector = MagicMock()
        mock_provider.return_value = mock_detector
        mock_detector.detect_provider.return_value = 'anthropic'
        mock_model = MagicMock()
        mock_detector.setup_model.return_value = mock_model
        
        # Mock memory client
        mock_memory_instance = MagicMock()
        mock_memory_client.return_value = mock_memory_instance
        
        # Initialize agent
        agent = AssistantAgent()
        
        # Verify initialization calls
        mock_config.assert_called_once()
        mock_config_instance.load_config.assert_called_once()
        mock_provider.assert_called_once()
        mock_detector.detect_provider.assert_called_once()
        mock_memory_client.assert_called_once()
    
    @patch('assistant.agent.ConfigManager')
    def test_agent_initialization_custom_config_path(self, mock_config):
        """Test agent initialization with custom configuration path."""
        custom_path = '/custom/config.yaml'
        
        mock_config_instance = MagicMock()
        mock_config.return_value = mock_config_instance
        mock_config_instance.load_config.return_value = {
            'providers': {'ollama': {'base_url': 'localhost', 'model_id': 'llama3.3'}},
            'system_prompt': 'Test prompt',
            'mcp_servers': {'memory': {'command': 'python', 'args': ['test.py']}}
        }
        
        with patch('assistant.agent.ProviderDetector'), \
             patch('assistant.agent.MCPMemoryClient'):
            
            agent = AssistantAgent(config_path=custom_path)
            
            mock_config.assert_called_once_with(custom_path)
    
    @patch('assistant.agent.ConfigManager')
    def test_agent_initialization_provider_override(self, mock_config):
        """Test agent initialization with provider override."""
        mock_config_instance = MagicMock()
        mock_config.return_value = mock_config_instance
        mock_config_instance.load_config.return_value = {
            'providers': {
                'anthropic': {'api_key': 'test1', 'model_id': 'claude-3'},
                'openai': {'api_key': 'test2', 'model_id': 'gpt-4'}
            },
            'system_prompt': 'Test prompt',
            'mcp_servers': {'memory': {'command': 'python', 'args': ['test.py']}}
        }
        
        with patch('assistant.agent.ProviderDetector') as mock_provider, \
             patch('assistant.agent.MCPMemoryClient'):
            
            mock_detector = MagicMock()
            mock_provider.return_value = mock_detector
            mock_model = MagicMock()
            mock_detector.setup_model.return_value = mock_model
            
            agent = AssistantAgent(provider_override='openai')
            
            # Verify provider override was used
            mock_detector.setup_model.assert_called_once_with('openai', mock_config_instance.load_config.return_value['providers']['openai'])
    
    @patch('assistant.agent.ConfigManager')
    def test_agent_initialization_config_error(self, mock_config):
        """Test agent initialization with configuration error."""
        mock_config_instance = MagicMock()
        mock_config.return_value = mock_config_instance
        mock_config_instance.load_config.side_effect = ConfigError("Invalid config")
        
        with pytest.raises(ConfigError):
            AssistantAgent()
    
    @patch('assistant.agent.ConfigManager')
    @patch('assistant.agent.ProviderDetector')
    def test_agent_initialization_provider_error(self, mock_provider, mock_config):
        """Test agent initialization with provider setup error."""
        # Mock configuration
        mock_config_instance = MagicMock()
        mock_config.return_value = mock_config_instance
        mock_config_instance.load_config.return_value = {
            'providers': {'anthropic': {'api_key': 'invalid', 'model_id': 'claude-3'}},
            'system_prompt': 'Test prompt',
            'mcp_servers': {'memory': {'command': 'python', 'args': ['test.py']}}
        }
        
        # Mock provider detector
        mock_detector = MagicMock()
        mock_provider.return_value = mock_detector
        mock_detector.detect_provider.return_value = 'anthropic'
        mock_detector.setup_model.side_effect = Exception("Provider setup failed")
        
        with pytest.raises(Exception) as exc_info:
            AssistantAgent()
        
        assert "Provider setup failed" in str(exc_info.value)


class TestAssistantAgentQueryProcessing:
    """Test query processing functionality."""
    
    @pytest.fixture
    def mock_agent(self):
        """Create a mock agent for testing."""
        with patch('assistant.agent.ConfigManager'), \
             patch('assistant.agent.ProviderDetector'), \
             patch('assistant.agent.MCPMemoryClient') as mock_memory, \
             patch('assistant.agent.Agent') as mock_strands_agent:
            
            # Setup mock agent
            mock_agent_instance = MagicMock()
            mock_strands_agent.return_value = mock_agent_instance
            
            # Setup mock memory client
            mock_memory_instance = MagicMock()
            mock_memory.return_value = mock_memory_instance
            mock_memory_instance.search_memories.return_value = []
            mock_memory_instance.add_memory.return_value = True
            
            agent = AssistantAgent()
            agent.strands_agent = mock_agent_instance
            agent.memory_client = mock_memory_instance
            
            yield agent
    
    def test_process_query_basic(self, mock_agent):
        """Test basic query processing."""
        mock_agent.strands_agent.run.return_value = "Test response"
        
        response = mock_agent.process_query("What is the weather?")
        
        assert response == "Test response"
        mock_agent.strands_agent.run.assert_called_once()
        mock_agent.memory_client.search_memories.assert_called_once_with("What is the weather?")
    
    def test_process_query_with_memory_retrieval(self, mock_agent):
        """Test query processing with memory retrieval."""
        # Mock memory search results
        memory_results = [
            {"content": "User likes sunny weather"},
            {"content": "User lives in San Francisco"}
        ]
        mock_agent.memory_client.search_memories.return_value = memory_results
        mock_agent.strands_agent.run.return_value = "Based on your preferences, today looks sunny!"
        
        response = mock_agent.process_query("How's the weather today?")
        
        assert "Based on your preferences" in response
        mock_agent.memory_client.search_memories.assert_called_once_with("How's the weather today?")
        
        # Verify agent was called with enhanced query including memory context
        call_args = mock_agent.strands_agent.run.call_args[0][0]
        assert "User likes sunny weather" in call_args
        assert "User lives in San Francisco" in call_args
    
    def test_process_query_memory_update(self, mock_agent):
        """Test query processing updates memory with new information."""
        mock_agent.strands_agent.run.return_value = "I'll remember that you're working on a Python project."
        
        response = mock_agent.process_query("I'm starting a new Python project called WebApp")
        
        # Verify memory was updated
        mock_agent.memory_client.add_memory.assert_called_once()
        add_memory_call = mock_agent.memory_client.add_memory.call_args[0][0]
        assert "Python project" in add_memory_call
        assert "WebApp" in add_memory_call
    
    def test_process_query_verbose_mode(self, mock_agent):
        """Test query processing in verbose mode."""
        mock_agent.verbose = True
        mock_agent.strands_agent.run.return_value = "Test response"
        
        response = mock_agent.process_query("Test query")
        
        # In verbose mode, should still work the same but provide detailed output
        assert response == "Test response"
        mock_agent.memory_client.search_memories.assert_called_once()
    
    def test_process_query_agent_error(self, mock_agent):
        """Test query processing with agent error."""
        mock_agent.strands_agent.run.side_effect = Exception("Agent processing failed")
        
        with pytest.raises(Exception) as exc_info:
            mock_agent.process_query("Test query")
        
        assert "Agent processing failed" in str(exc_info.value)
    
    def test_process_query_memory_error_graceful_fallback(self, mock_agent):
        """Test query processing with memory error falls back gracefully."""
        # Memory search fails but agent should still work
        mock_agent.memory_client.search_memories.side_effect = MemoryClientError("Memory search failed")
        mock_agent.strands_agent.run.return_value = "Response without memory context"
        
        response = mock_agent.process_query("Test query")
        
        # Should still get response despite memory error
        assert response == "Response without memory context"
        mock_agent.strands_agent.run.assert_called_once()


class TestAssistantAgentMemoryOperations:
    """Test memory management operations."""
    
    @pytest.fixture
    def mock_agent(self):
        """Create a mock agent for testing."""
        with patch('assistant.agent.ConfigManager'), \
             patch('assistant.agent.ProviderDetector'), \
             patch('assistant.agent.MCPMemoryClient') as mock_memory:
            
            mock_memory_instance = MagicMock()
            mock_memory.return_value = mock_memory_instance
            
            agent = AssistantAgent()
            agent.memory_client = mock_memory_instance
            
            yield agent
    
    def test_reset_memory_success(self, mock_agent):
        """Test successful memory reset."""
        mock_agent.memory_client.reset_memory.return_value = True
        
        result = mock_agent.reset_memory()
        
        assert result is True
        mock_agent.memory_client.reset_memory.assert_called_once()
    
    def test_reset_memory_error(self, mock_agent):
        """Test memory reset with error."""
        mock_agent.memory_client.reset_memory.side_effect = MemoryClientError("Reset failed")
        
        with pytest.raises(MemoryClientError):
            mock_agent.reset_memory()
    
    def test_export_memory_success(self, mock_agent):
        """Test successful memory export."""
        test_memory_data = {
            "entities": [{"id": 1, "name": "John"}],
            "relations": [],
            "observations": ["John is a developer"]
        }
        mock_agent.memory_client.export_memory.return_value = test_memory_data
        
        result = mock_agent.export_memory()
        
        assert result == test_memory_data
        mock_agent.memory_client.export_memory.assert_called_once()
    
    def test_export_memory_error(self, mock_agent):
        """Test memory export with error."""
        mock_agent.memory_client.export_memory.side_effect = MemoryClientError("Export failed")
        
        with pytest.raises(MemoryClientError):
            mock_agent.export_memory()
    
    def test_import_memory_success(self, mock_agent):
        """Test successful memory import."""
        test_memory_data = {
            "entities": [{"id": 1, "name": "Jane"}],
            "relations": [],
            "observations": ["Jane is a designer"]
        }
        mock_agent.memory_client.import_memory.return_value = True
        
        result = mock_agent.import_memory(test_memory_data)
        
        assert result is True
        mock_agent.memory_client.import_memory.assert_called_once_with(test_memory_data)
    
    def test_import_memory_error(self, mock_agent):
        """Test memory import with error."""
        test_memory_data = {"invalid": "data"}
        mock_agent.memory_client.import_memory.side_effect = MemoryClientError("Import failed")
        
        with pytest.raises(MemoryClientError):
            mock_agent.import_memory(test_memory_data)


class TestAssistantAgentIntegration:
    """Integration tests for assistant agent."""
    
    @patch('assistant.agent.ConfigManager')
    @patch('assistant.agent.ProviderDetector')
    @patch('assistant.agent.MCPMemoryClient')
    @patch('assistant.agent.Agent')
    def test_full_workflow_integration(self, mock_strands_agent, mock_memory, mock_provider, mock_config):
        """Test full workflow from initialization to query processing."""
        # Setup configuration
        mock_config_instance = MagicMock()
        mock_config.return_value = mock_config_instance
        mock_config_instance.load_config.return_value = {
            'default_provider': 'anthropic',
            'providers': {'anthropic': {'api_key': 'sk-ant-test123', 'model_id': 'claude-3'}},
            'system_prompt': 'You are a helpful assistant.',
            'mcp_servers': {'memory': {'command': 'python', 'args': ['test.py']}}
        }
        
        # Setup provider
        mock_detector = MagicMock()
        mock_provider.return_value = mock_detector
        mock_model = MagicMock()
        mock_detector.setup_model.return_value = mock_model
        
        # Setup memory client
        mock_memory_instance = MagicMock()
        mock_memory.return_value = mock_memory_instance
        mock_memory_instance.search_memories.return_value = []
        mock_memory_instance.add_memory.return_value = True
        
        # Setup Strands agent
        mock_agent_instance = MagicMock()
        mock_strands_agent.return_value = mock_agent_instance
        mock_agent_instance.run.return_value = "Hello! I'm ready to help."
        
        # Initialize and use agent
        agent = AssistantAgent(provider_override='anthropic')
        response = agent.process_query("Hello, can you help me?")
        
        # Verify full workflow
        assert response == "Hello! I'm ready to help."
        mock_config_instance.load_config.assert_called_once()
        mock_detector.setup_model.assert_called_once()
        mock_memory_instance.search_memories.assert_called_once()
        mock_agent_instance.run.assert_called_once()
    
    @patch('assistant.agent.ConfigManager')
    @patch('assistant.agent.ProviderDetector') 
    @patch('assistant.agent.MCPMemoryClient')
    def test_memory_client_fallback_integration(self, mock_memory, mock_provider, mock_config):
        """Test integration with memory client fallback mode."""
        # Setup configuration
        mock_config_instance = MagicMock()
        mock_config.return_value = mock_config_instance
        mock_config_instance.load_config.return_value = {
            'providers': {'ollama': {'base_url': 'localhost', 'model_id': 'llama3.3'}},
            'system_prompt': 'Test prompt',
            'mcp_servers': {'memory': {'command': 'python', 'args': ['test.py']}}
        }
        
        # Setup provider
        mock_detector = MagicMock()
        mock_provider.return_value = mock_detector
        mock_detector.detect_provider.return_value = 'ollama'
        mock_model = MagicMock()
        mock_detector.setup_model.return_value = mock_model
        
        # Setup memory client in fallback mode
        mock_memory_instance = MagicMock()
        mock_memory.return_value = mock_memory_instance
        mock_memory_instance.is_fallback_mode = True
        mock_memory_instance.search_memories.return_value = []
        
        with patch('assistant.agent.Agent') as mock_strands_agent:
            mock_agent_instance = MagicMock()
            mock_strands_agent.return_value = mock_agent_instance
            mock_agent_instance.run.return_value = "Response in fallback mode"
            
            agent = AssistantAgent()
            response = agent.process_query("Test query")
            
            assert response == "Response in fallback mode"
            # Should still work even in fallback mode
            mock_memory_instance.search_memories.assert_called_once()


class TestAssistantAgentAsync:
    """Test async functionality of assistant agent."""
    
    @pytest.fixture
    def mock_agent(self):
        """Create a mock agent for async testing."""
        with patch('assistant.agent.ConfigManager'), \
             patch('assistant.agent.ProviderDetector'), \
             patch('assistant.agent.MCPMemoryClient') as mock_memory:
            
            mock_memory_instance = MagicMock()
            mock_memory.return_value = mock_memory_instance
            
            agent = AssistantAgent()
            agent.memory_client = mock_memory_instance
            
            yield agent
    
    @pytest.mark.asyncio
    async def test_process_query_async(self, mock_agent):
        """Test asynchronous query processing."""
        # Mock async agent run
        with patch('assistant.agent.Agent') as mock_strands_agent:
            mock_agent_instance = MagicMock()
            mock_strands_agent.return_value = mock_agent_instance
            mock_agent.strands_agent = mock_agent_instance
            
            # Setup async run method
            async_run = AsyncMock(return_value="Async response")
            mock_agent_instance.run_async = async_run
            
            # Mock memory operations
            mock_agent.memory_client.search_memories.return_value = []
            mock_agent.memory_client.add_memory.return_value = True
            
            # Test async processing
            response = await mock_agent.process_query_async("Async test query")
            
            assert response == "Async response"
            async_run.assert_called_once()


if __name__ == '__main__':
    pytest.main([__file__]) 