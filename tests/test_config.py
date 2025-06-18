"""
Unit tests for configuration management and provider detection.

Tests the assistant.config and assistant.providers modules with various
configuration scenarios, error conditions, and environment setups.
"""

import os
import tempfile
import pytest
import yaml
from pathlib import Path
from unittest.mock import patch, MagicMock

from assistant.config import (
    ConfigManager, ConfigError, YAMLSyntaxError, ConfigValidationError
)
from assistant.providers import (
    ProviderDetector, ProviderError, APIKeyValidationError
)


class TestConfigManager:
    """Test cases for ConfigManager class."""
    
    @pytest.fixture
    def temp_config_dir(self):
        """Create a temporary directory for config tests."""
        with tempfile.TemporaryDirectory() as temp_dir:
            yield temp_dir
    
    @pytest.fixture
    def valid_config_data(self):
        """Sample valid configuration data."""
        return {
            'default_provider': 'auto',
            'providers': {
                'anthropic': {
                    'api_key': '${ANTHROPIC_API_KEY}',
                    'model_id': 'claude-3-7-sonnet-20250219',
                    'temperature': 0.7
                },
                'openai': {
                    'api_key': '${OPENAI_API_KEY}',
                    'model_id': 'gpt-4',
                    'temperature': 0.7
                }
            },
            'mcp_servers': {
                'memory': {
                    'command': 'python',
                    'args': ['./memory_server/mcp_memory_server.py'],
                    'env': {
                        'MEMORY_FILE_PATH': '~/.assistant/memory.json'
                    },
                    'required': True
                }
            },
            'system_prompt': 'You are a helpful assistant.',
            'cli': {
                'verbose': False,
                'log_level': 'INFO'
            }
        }
    
    def test_load_valid_config(self, temp_config_dir, valid_config_data):
        """Test loading a valid configuration file."""
        config_file = Path(temp_config_dir) / 'config.yaml'
        with open(config_file, 'w') as f:
            yaml.dump(valid_config_data, f)
        
        config_manager = ConfigManager(str(config_file))
        loaded_config = config_manager.load_config()
        
        assert loaded_config['default_provider'] == 'auto'
        assert 'anthropic' in loaded_config['providers']
        assert 'memory' in loaded_config['mcp_servers']
        assert loaded_config['system_prompt'] == 'You are a helpful assistant.'
    
    def test_load_config_file_not_found(self):
        """Test handling of missing configuration file."""
        config_manager = ConfigManager('/nonexistent/path/config.yaml')
        
        with pytest.raises(ConfigError) as exc_info:
            config_manager.load_config()
        
        assert 'Configuration file not found' in str(exc_info.value)
        assert 'Suggestions:' in str(exc_info.value)
    
    def test_load_config_empty_file(self, temp_config_dir):
        """Test handling of empty configuration file."""
        config_file = Path(temp_config_dir) / 'config.yaml'
        config_file.write_text('')
        
        config_manager = ConfigManager(str(config_file))
        
        with pytest.raises(ConfigError) as exc_info:
            config_manager.load_config()
        
        assert 'empty' in str(exc_info.value)
    
    def test_load_config_invalid_yaml(self, temp_config_dir):
        """Test handling of invalid YAML syntax."""
        config_file = Path(temp_config_dir) / 'config.yaml'
        # Invalid YAML with missing colon
        invalid_yaml = """
providers
  anthropic:
    api_key: test
"""
        config_file.write_text(invalid_yaml)
        
        config_manager = ConfigManager(str(config_file))
        
        with pytest.raises(YAMLSyntaxError) as exc_info:
            config_manager.load_config()
        
        error_msg = str(exc_info.value)
        assert 'YAML syntax error' in error_msg
        assert 'Line' in error_msg
        assert 'Column' in error_msg
        assert 'Common YAML syntax issues' in error_msg
    
    def test_config_validation_missing_sections(self, temp_config_dir):
        """Test validation with missing required sections."""
        config_file = Path(temp_config_dir) / 'config.yaml'
        incomplete_config = {'providers': {'anthropic': {'api_key': 'test'}}}
        
        with open(config_file, 'w') as f:
            yaml.dump(incomplete_config, f)
        
        config_manager = ConfigManager(str(config_file))
        
        with pytest.raises(ConfigValidationError) as exc_info:
            config_manager.load_config()
        
        error_msg = str(exc_info.value)
        assert 'Missing required section' in error_msg
        assert 'mcp_servers' in error_msg
        assert 'system_prompt' in error_msg
    
    def test_config_validation_invalid_provider(self, temp_config_dir):
        """Test validation with invalid provider configuration."""
        config_file = Path(temp_config_dir) / 'config.yaml'
        invalid_provider_config = {
            'providers': {
                'invalid_provider': {'model_id': 'test'},
                'anthropic': {}  # Missing api_key
            },
            'mcp_servers': {
                'memory': {'command': 'python', 'args': ['test.py'], 'required': True}
            },
            'system_prompt': 'Test prompt'
        }
        
        with open(config_file, 'w') as f:
            yaml.dump(invalid_provider_config, f)
        
        config_manager = ConfigManager(str(config_file))
        
        with pytest.raises(ConfigValidationError) as exc_info:
            config_manager.load_config()
        
        error_msg = str(exc_info.value)
        assert 'Unsupported provider' in error_msg
        assert 'missing required' in error_msg
    
    @patch.dict(os.environ, {'ANTHROPIC_API_KEY': 'sk-ant-test123', 'TEST_VAR': 'test_value'})
    def test_environment_variable_interpolation(self, temp_config_dir):
        """Test environment variable interpolation in config."""
        config_file = Path(temp_config_dir) / 'config.yaml'
        config_with_env = {
            'providers': {
                'anthropic': {
                    'api_key': '${ANTHROPIC_API_KEY}',
                    'model_id': 'claude-3-7-sonnet-20250219'
                }
            },
            'mcp_servers': {
                'memory': {
                    'command': 'python',
                    'args': ['./memory_server/mcp_memory_server.py'],
                    'env': {
                        'TEST_PATH': '${TEST_VAR}/data'
                    },
                    'required': True
                }
            },
            'system_prompt': 'Test prompt'
        }
        
        with open(config_file, 'w') as f:
            yaml.dump(config_with_env, f)
        
        config_manager = ConfigManager(str(config_file))
        loaded_config = config_manager.load_config()
        
        assert loaded_config['providers']['anthropic']['api_key'] == 'sk-ant-test123'
        assert loaded_config['mcp_servers']['memory']['env']['TEST_PATH'] == 'test_value/data'
    
    def test_environment_variable_with_default(self, temp_config_dir):
        """Test environment variable interpolation with default values."""
        config_file = Path(temp_config_dir) / 'config.yaml'
        config_with_defaults = {
            'providers': {
                'ollama': {
                    'base_url': '${OLLAMA_URL:-http://localhost:11434}',
                    'model_id': 'llama3.3'
                }
            },
            'mcp_servers': {
                'memory': {'command': 'python', 'args': ['test.py'], 'required': True}
            },
            'system_prompt': 'Test prompt'
        }
        
        with open(config_file, 'w') as f:
            yaml.dump(config_with_defaults, f)
        
        config_manager = ConfigManager(str(config_file))
        loaded_config = config_manager.load_config()
        
        # Should use default value since OLLAMA_URL is not set
        assert loaded_config['providers']['ollama']['base_url'] == 'http://localhost:11434'
    
    def test_get_provider_config(self, temp_config_dir, valid_config_data):
        """Test getting specific provider configuration."""
        config_file = Path(temp_config_dir) / 'config.yaml'
        with open(config_file, 'w') as f:
            yaml.dump(valid_config_data, f)
        
        config_manager = ConfigManager(str(config_file))
        config_manager.load_config()
        
        anthropic_config = config_manager.get_provider_config('anthropic')
        assert 'api_key' in anthropic_config
        assert anthropic_config['model_id'] == 'claude-3-7-sonnet-20250219'
        
        # Test non-existent provider
        with pytest.raises(ConfigError):
            config_manager.get_provider_config('nonexistent')
    
    def test_get_mcp_server_config(self, temp_config_dir, valid_config_data):
        """Test getting MCP server configuration."""
        config_file = Path(temp_config_dir) / 'config.yaml'
        with open(config_file, 'w') as f:
            yaml.dump(valid_config_data, f)
        
        config_manager = ConfigManager(str(config_file))
        config_manager.load_config()
        
        memory_config = config_manager.get_mcp_server_config('memory')
        assert memory_config['command'] == 'python'
        assert memory_config['required'] is True
        
        # Test non-existent server
        with pytest.raises(ConfigError):
            config_manager.get_mcp_server_config('nonexistent')


class TestProviderDetector:
    """Test cases for ProviderDetector class."""
    
    @pytest.fixture
    def detector(self):
        """Create a ProviderDetector instance."""
        return ProviderDetector()
    
    @patch.dict(os.environ, {}, clear=True)
    def test_detect_provider_fallback_to_ollama(self, detector):
        """Test provider detection falls back to Ollama when no credentials found."""
        detected = detector.detect_provider()
        assert detected == 'ollama'
    
    @patch.dict(os.environ, {'ANTHROPIC_API_KEY': 'sk-ant-test123'})
    def test_detect_provider_anthropic(self, detector):
        """Test provider detection with Anthropic API key."""
        detected = detector.detect_provider()
        assert detected == 'anthropic'
    
    @patch.dict(os.environ, {'OPENAI_API_KEY': 'sk-test123'})
    def test_detect_provider_openai(self, detector):
        """Test provider detection with OpenAI API key."""
        detected = detector.detect_provider()
        assert detected == 'openai'
    
    @patch.dict(os.environ, {'AWS_ACCESS_KEY_ID': 'test', 'AWS_SECRET_ACCESS_KEY': 'test'})
    def test_detect_provider_bedrock(self, detector):
        """Test provider detection with AWS credentials."""
        detected = detector.detect_provider()
        assert detected == 'bedrock'
    
    @patch.dict(os.environ, {
        'ANTHROPIC_API_KEY': 'sk-ant-test123',
        'OPENAI_API_KEY': 'sk-test123'
    })
    def test_detect_provider_priority_anthropic(self, detector):
        """Test provider detection priority - Anthropic should win."""
        detected = detector.detect_provider()
        assert detected == 'anthropic'  # Anthropic has higher priority
    
    def test_validate_anthropic_credentials_valid(self, detector):
        """Test Anthropic credential validation with valid key."""
        with patch.dict(os.environ, {'ANTHROPIC_API_KEY': 'sk-ant-valid123456789'}):
            assert detector._validate_anthropic_credentials() is True
    
    def test_validate_anthropic_credentials_invalid_format(self, detector):
        """Test Anthropic credential validation with invalid format."""
        with patch.dict(os.environ, {'ANTHROPIC_API_KEY': 'invalid-key'}):
            assert detector._validate_anthropic_credentials() is False
    
    def test_validate_anthropic_credentials_missing(self, detector):
        """Test Anthropic credential validation with missing key."""
        with patch.dict(os.environ, {}, clear=True):
            assert detector._validate_anthropic_credentials() is False
    
    def test_validate_openai_credentials_valid(self, detector):
        """Test OpenAI credential validation with valid key."""
        with patch.dict(os.environ, {'OPENAI_API_KEY': 'sk-valid123456789'}):
            assert detector._validate_openai_credentials() is True
    
    def test_validate_openai_credentials_invalid_format(self, detector):
        """Test OpenAI credential validation with invalid format."""
        with patch.dict(os.environ, {'OPENAI_API_KEY': 'invalid-key'}):
            assert detector._validate_openai_credentials() is False
    
    def test_get_available_providers(self, detector):
        """Test getting available providers status."""
        with patch.dict(os.environ, {'ANTHROPIC_API_KEY': 'sk-ant-test123'}):
            availability = detector.get_available_providers()
            
            assert isinstance(availability, dict)
            assert 'anthropic' in availability
            assert 'openai' in availability
            assert 'bedrock' in availability
            assert 'ollama' in availability
            assert availability['anthropic'] is True
            assert availability['ollama'] is True  # Always available
    
    def test_api_key_validation_comprehensive_anthropic(self, detector):
        """Test comprehensive API key validation for Anthropic."""
        # Valid key
        result = detector.validate_api_key_comprehensive('anthropic', 'sk-ant-test123456789')
        assert result['valid'] is True
        assert len(result['errors']) == 0
        
        # Invalid format
        result = detector.validate_api_key_comprehensive('anthropic', 'invalid-key')
        assert result['valid'] is False
        assert 'format is invalid' in result['errors'][0]
        assert any('sk-ant-' in suggestion for suggestion in result['suggestions'])
        
        # Missing key
        result = detector.validate_api_key_comprehensive('anthropic', '')
        assert result['valid'] is False
        assert 'missing' in result['errors'][0]
        assert any('ANTHROPIC_API_KEY' in suggestion for suggestion in result['suggestions'])
    
    def test_api_key_validation_comprehensive_openai(self, detector):
        """Test comprehensive API key validation for OpenAI."""
        # Valid key
        result = detector.validate_api_key_comprehensive('openai', 'sk-test123456789')
        assert result['valid'] is True
        
        # Invalid format
        result = detector.validate_api_key_comprehensive('openai', 'invalid-key')
        assert result['valid'] is False
        assert 'format is invalid' in result['errors'][0]
    
    def test_setup_model_anthropic_valid(self, detector):
        """Test setting up Anthropic model with valid configuration."""
        config = {
            'api_key': 'sk-ant-test123456789',
            'model_id': 'claude-3-7-sonnet-20250219',
            'temperature': 0.7
        }
        
        # Mock the AnthropicModel to avoid actual SDK dependency
        with patch('assistant.providers.AnthropicModel') as mock_model:
            mock_instance = MagicMock()
            mock_model.return_value = mock_instance
            
            result = detector.setup_model('anthropic', config)
            
            mock_model.assert_called_once_with(
                model_id='claude-3-7-sonnet-20250219',
                api_key='sk-ant-test123456789',
                temperature=0.7
            )
            assert result == mock_instance
    
    def test_setup_model_anthropic_invalid_key(self, detector):
        """Test setting up Anthropic model with invalid API key."""
        config = {
            'api_key': 'invalid-key',
            'model_id': 'claude-3-7-sonnet-20250219'
        }
        
        with pytest.raises(APIKeyValidationError) as exc_info:
            detector.setup_model('anthropic', config)
        
        error_msg = str(exc_info.value)
        assert 'API key validation failed' in error_msg
        assert 'format is invalid' in error_msg
        assert 'sk-ant-' in error_msg
    
    def test_setup_model_unsupported_provider(self, detector):
        """Test setting up model with unsupported provider."""
        with pytest.raises(ProviderError) as exc_info:
            detector.setup_model('unsupported', {})
        
        assert 'Unsupported provider' in str(exc_info.value)
    
    def test_get_credential_status_report(self, detector):
        """Test getting comprehensive credential status report."""
        with patch.dict(os.environ, {'ANTHROPIC_API_KEY': 'sk-ant-test123'}):
            report = detector.get_credential_status_report()
            
            assert 'anthropic' in report
            assert 'openai' in report
            assert 'bedrock' in report
            assert 'ollama' in report
            
            # Check Anthropic status
            anthropic_status = report['anthropic']
            assert anthropic_status['available'] is True
            assert anthropic_status['env_var_set'] is True
            assert anthropic_status['validation']['valid'] is True
            
            # Check Ollama status (always available)
            ollama_status = report['ollama']
            assert ollama_status['available'] is True
            assert ollama_status['requires_local_server'] is True


# Integration tests
class TestConfigProviderIntegration:
    """Integration tests between config and provider modules."""
    
    def test_config_provider_integration(self, tmp_path):
        """Test loading config and using it with provider detection."""
        config_file = tmp_path / 'config.yaml'
        test_config = {
            'default_provider': 'anthropic',
            'providers': {
                'anthropic': {
                    'api_key': '${ANTHROPIC_API_KEY}',
                    'model_id': 'claude-3-7-sonnet-20250219'
                }
            },
            'mcp_servers': {
                'memory': {'command': 'python', 'args': ['test.py'], 'required': True}
            },
            'system_prompt': 'Test prompt'
        }
        
        with open(config_file, 'w') as f:
            yaml.dump(test_config, f)
        
        with patch.dict(os.environ, {'ANTHROPIC_API_KEY': 'sk-ant-test123456789'}):
            # Load config
            config_manager = ConfigManager(str(config_file))
            loaded_config = config_manager.load_config()
            
            # Use with provider detector
            detector = ProviderDetector()
            provider_config = config_manager.get_provider_config('anthropic')
            
            # Verify provider validation works
            validation = detector.validate_api_key_comprehensive('anthropic', provider_config['api_key'])
            assert validation['valid'] is True


if __name__ == '__main__':
    pytest.main([__file__]) 