"""
Unit tests for CLI interface and command handling.

Tests the assistant.cli module with various command-line arguments,
options, and error conditions.
"""

import os
import tempfile
import pytest
import json
from pathlib import Path
from unittest.mock import patch, MagicMock, mock_open
from click.testing import CliRunner

from assistant.cli import cli, main
from assistant.config import ConfigError
from assistant.error_reporting import ErrorReporter


class TestCLIBasics:
    """Test basic CLI functionality and argument parsing."""
    
    def setup_method(self):
        """Set up test environment before each test."""
        self.runner = CliRunner()
    
    def test_cli_help(self):
        """Test CLI help display."""
        result = self.runner.invoke(cli, ['--help'])
        
        assert result.exit_code == 0
        assert 'Personal Assistant CLI' in result.output
        assert '--config' in result.output
        assert '--provider' in result.output
        assert '--verbose' in result.output
        assert '--reset-memory' in result.output
        assert '--export-memory' in result.output
        assert '--import-memory' in result.output
    
    def test_cli_version(self):
        """Test CLI version display."""
        result = self.runner.invoke(cli, ['--version'])
        
        assert result.exit_code == 0
        assert 'version' in result.output.lower()
    
    def test_cli_with_query(self):
        """Test CLI with basic query."""
        with patch('assistant.cli.AssistantAgent') as mock_agent_class:
            mock_agent = MagicMock()
            mock_agent_class.return_value = mock_agent
            mock_agent.process_query.return_value = "Test response"
            
            result = self.runner.invoke(cli, ['What is the weather?'])
            
            assert result.exit_code == 0
            mock_agent.process_query.assert_called_once_with('What is the weather?')
    
    def test_cli_no_query(self):
        """Test CLI without query argument."""
        result = self.runner.invoke(cli, [])
        
        assert result.exit_code != 0
        assert 'Missing argument' in result.output or 'Usage:' in result.output
    
    def test_cli_empty_query(self):
        """Test CLI with empty query."""
        result = self.runner.invoke(cli, [''])
        
        assert result.exit_code != 0
        assert 'empty' in result.output.lower()


class TestCLIConfigurationOptions:
    """Test CLI configuration and provider options."""
    
    def setup_method(self):
        """Set up test environment before each test."""
        self.runner = CliRunner()
    
    def test_cli_custom_config_path(self):
        """Test CLI with custom configuration file path."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as config_file:
            yaml_content = """
providers:
  anthropic:
    api_key: test
    model_id: claude-3
mcp_servers:
  memory:
    command: python
    args: ['test.py']
    required: true
system_prompt: Test prompt
"""
            config_file.write(yaml_content)
            config_file.flush()
            
            with patch('assistant.cli.AssistantAgent') as mock_agent_class:
                mock_agent = MagicMock()
                mock_agent_class.return_value = mock_agent
                mock_agent.process_query.return_value = "Test response"
                
                result = self.runner.invoke(cli, [
                    '--config', config_file.name,
                    'Test query'
                ])
                
                assert result.exit_code == 0
                mock_agent.process_query.assert_called_once()
        
        # Clean up
        os.unlink(config_file.name)
    
    def test_cli_provider_override(self):
        """Test CLI with provider override."""
        with patch('assistant.cli.AssistantAgent') as mock_agent_class:
            mock_agent = MagicMock()
            mock_agent_class.return_value = mock_agent
            mock_agent.process_query.return_value = "Test response"
            
            result = self.runner.invoke(cli, [
                '--provider', 'anthropic',
                'Test query'
            ])
            
            assert result.exit_code == 0
            # Verify that provider override was passed to agent initialization
            mock_agent_class.assert_called_once()
            call_kwargs = mock_agent_class.call_args[1]
            assert call_kwargs.get('provider_override') == 'anthropic'
    
    def test_cli_verbose_mode(self):
        """Test CLI with verbose mode enabled."""
        with patch('assistant.cli.AssistantAgent') as mock_agent_class:
            mock_agent = MagicMock()
            mock_agent_class.return_value = mock_agent
            mock_agent.process_query.return_value = "Test response"
            
            result = self.runner.invoke(cli, [
                '--verbose',
                'Test query'
            ])
            
            assert result.exit_code == 0
            # Verify verbose mode was enabled
            mock_agent_class.assert_called_once()
            call_kwargs = mock_agent_class.call_args[1]
            assert call_kwargs.get('verbose') is True


class TestCLIMemoryManagement:
    """Test CLI memory management commands."""
    
    def setup_method(self):
        """Set up test environment before each test."""
        self.runner = CliRunner()
    
    def test_reset_memory_with_confirmation(self):
        """Test memory reset with user confirmation."""
        with patch('assistant.cli.AssistantAgent') as mock_agent_class:
            mock_agent = MagicMock()
            mock_agent_class.return_value = mock_agent
            mock_agent.reset_memory.return_value = True
            
            # Simulate user confirming with 'y'
            result = self.runner.invoke(cli, ['--reset-memory'], input='y\n')
            
            assert result.exit_code == 0
            assert 'Memory has been reset successfully' in result.output
            mock_agent.reset_memory.assert_called_once()
    
    def test_reset_memory_cancelled(self):
        """Test memory reset cancelled by user."""
        with patch('assistant.cli.AssistantAgent') as mock_agent_class:
            mock_agent = MagicMock()
            mock_agent_class.return_value = mock_agent
            
            # Simulate user cancelling with 'n'
            result = self.runner.invoke(cli, ['--reset-memory'], input='n\n')
            
            assert result.exit_code == 0
            assert 'Memory reset cancelled' in result.output
            mock_agent.reset_memory.assert_not_called()
    
    def test_export_memory_success(self):
        """Test successful memory export."""
        test_memory_data = {
            "entities": [{"id": 1, "name": "John", "type": "person"}],
            "relations": [],
            "observations": ["John is a developer"]
        }
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as export_file:
            export_path = export_file.name
        
        with patch('assistant.cli.AssistantAgent') as mock_agent_class:
            mock_agent = MagicMock()
            mock_agent_class.return_value = mock_agent
            mock_agent.export_memory.return_value = test_memory_data
            
            result = self.runner.invoke(cli, ['--export-memory', export_path])
            
            assert result.exit_code == 0
            assert f'Memory exported to {export_path}' in result.output
            mock_agent.export_memory.assert_called_once()
            
            # Verify file was created with correct content
            with open(export_path, 'r') as f:
                exported_data = json.load(f)
                assert exported_data == test_memory_data
        
        # Clean up
        os.unlink(export_path)


if __name__ == '__main__':
    pytest.main([__file__]) 