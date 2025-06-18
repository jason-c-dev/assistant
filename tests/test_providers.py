"""
Unit tests for provider detection and model setup.

Tests the assistant.providers module with various provider configurations,
credential validation, and model initialization scenarios.
"""

import os
import boto3
import pytest
from unittest.mock import patch, MagicMock

from assistant.providers import (
    ProviderDetector, ProviderError, APIKeyValidationError
)


class TestProviderDetection:
    """Test provider auto-detection functionality."""
    
    def setup_method(self):
        """Set up test environment before each test."""
        self.detector = ProviderDetector()
    
    @patch.dict(os.environ, {}, clear=True)
    def test_detect_no_credentials_fallback_ollama(self):
        """Test provider detection with no credentials falls back to Ollama."""
        result = self.detector.detect_provider()
        assert result == 'ollama'
    
    @patch.dict(os.environ, {'ANTHROPIC_API_KEY': 'sk-ant-test123456789'})
    def test_detect_anthropic_highest_priority(self):
        """Test Anthropic detection with highest priority."""
        result = self.detector.detect_provider()
        assert result == 'anthropic'
    
    @patch.dict(os.environ, {'OPENAI_API_KEY': 'sk-test123456789'})
    def test_detect_openai_second_priority(self):
        """Test OpenAI detection when Anthropic not available."""
        result = self.detector.detect_provider()
        assert result == 'openai'
    
    @patch.dict(os.environ, {
        'AWS_ACCESS_KEY_ID': 'test_access_key',
        'AWS_SECRET_ACCESS_KEY': 'test_secret_key'
    })
    def test_detect_bedrock_third_priority(self):
        """Test Bedrock detection when others not available."""
        result = self.detector.detect_provider()
        assert result == 'bedrock'
    
    @patch.dict(os.environ, {
        'ANTHROPIC_API_KEY': 'sk-ant-test123',
        'OPENAI_API_KEY': 'sk-test123',
        'AWS_ACCESS_KEY_ID': 'test_key'
    })
    def test_detect_provider_priority_order(self):
        """Test provider priority order when multiple available."""
        result = self.detector.detect_provider()
        # Anthropic should win due to highest priority
        assert result == 'anthropic'
    
    @patch('boto3.Session')
    def test_detect_bedrock_with_boto3_session(self, mock_session):
        """Test Bedrock detection using boto3 session credentials."""
        mock_credentials = MagicMock()
        mock_credentials.access_key = 'test_access'
        mock_credentials.secret_key = 'test_secret'
        
        mock_session.return_value.get_credentials.return_value = mock_credentials
        
        with patch.dict(os.environ, {}, clear=True):
            result = self.detector.detect_provider()
            # Should detect bedrock through boto3 session
            assert result == 'bedrock'


class TestCredentialValidation:
    """Test credential validation for different providers."""
    
    def setup_method(self):
        """Set up test environment before each test."""
        self.detector = ProviderDetector()
    
    def test_validate_anthropic_credentials_valid_formats(self):
        """Test Anthropic API key validation with valid formats."""
        valid_keys = [
            'sk-ant-1234567890',
            'sk-ant-api03-abc123def456'
        ]
        
        for api_key in valid_keys:
            with patch.dict(os.environ, {'ANTHROPIC_API_KEY': api_key}):
                assert self.detector._validate_anthropic_credentials() is True
    
    def test_validate_anthropic_credentials_invalid_formats(self):
        """Test Anthropic API key validation with invalid formats."""
        invalid_keys = [
            'invalid-key',
            'sk-wrong-prefix',
            ''
        ]
        
        for api_key in invalid_keys:
            with patch.dict(os.environ, {'ANTHROPIC_API_KEY': api_key}):
                assert self.detector._validate_anthropic_credentials() is False
    
    def test_validate_anthropic_credentials_missing(self):
        """Test Anthropic credential validation with missing environment variable."""
        with patch.dict(os.environ, {}, clear=True):
            assert self.detector._validate_anthropic_credentials() is False
    
    def test_validate_openai_credentials_valid_formats(self):
        """Test OpenAI API key validation with valid formats."""
        valid_keys = [
            'sk-1234567890abcdef',
            'sk-proj-abcdef1234567890',
            'sk-123456789012345678901234567890123456789012345678'
        ]
        
        for api_key in valid_keys:
            with patch.dict(os.environ, {'OPENAI_API_KEY': api_key}):
                assert self.detector._validate_openai_credentials() is True
    
    def test_validate_openai_credentials_invalid_formats(self):
        """Test OpenAI API key validation with invalid formats."""
        invalid_keys = [
            'invalid-key',
            'openai-key-123',
            'pk-wrong-prefix',
            '',
            'sk-'  # Too short
        ]
        
        for api_key in invalid_keys:
            with patch.dict(os.environ, {'OPENAI_API_KEY': api_key}):
                assert self.detector._validate_openai_credentials() is False
    
    def test_validate_bedrock_credentials_env_vars(self):
        """Test Bedrock credential validation with environment variables."""
        with patch.dict(os.environ, {
            'AWS_ACCESS_KEY_ID': 'test_access_key',
            'AWS_SECRET_ACCESS_KEY': 'test_secret_key'
        }):
            assert self.detector._validate_bedrock_credentials() is True
    
    @patch('boto3.Session')
    def test_validate_bedrock_credentials_boto3_session(self, mock_session):
        """Test Bedrock credential validation with boto3 session."""
        mock_credentials = MagicMock()
        mock_credentials.access_key = 'test_access'
        mock_credentials.secret_key = 'test_secret'
        
        mock_session.return_value.get_credentials.return_value = mock_credentials
        
        with patch.dict(os.environ, {}, clear=True):
            assert self.detector._validate_bedrock_credentials() is True
    
    @patch('boto3.Session')
    def test_validate_bedrock_credentials_no_session(self, mock_session):
        """Test Bedrock credential validation with no credentials."""
        mock_session.return_value.get_credentials.return_value = None
        
        with patch.dict(os.environ, {}, clear=True):
            assert self.detector._validate_bedrock_credentials() is False
    
    def test_validate_ollama_credentials_always_true(self):
        """Test Ollama credential validation (always available locally)."""
        # Ollama should always be available as it's local
        assert self.detector._validate_ollama_credentials() is True


class TestComprehensiveValidation:
    """Test comprehensive API key validation with detailed feedback."""
    
    def setup_method(self):
        """Set up test environment before each test."""
        self.detector = ProviderDetector()
    
    def test_anthropic_comprehensive_validation_valid(self):
        """Test comprehensive Anthropic validation with valid key."""
        result = self.detector.validate_api_key_comprehensive(
            'anthropic', 'sk-ant-test123456789'
        )
        
        assert result['valid'] is True
        assert result['provider'] == 'anthropic'
        assert len(result['errors']) == 0
        assert len(result['suggestions']) == 0
    
    def test_anthropic_comprehensive_validation_invalid_format(self):
        """Test comprehensive Anthropic validation with invalid format."""
        result = self.detector.validate_api_key_comprehensive(
            'anthropic', 'invalid-key-format'
        )
        
        assert result['valid'] is False
        assert result['provider'] == 'anthropic'
        assert len(result['errors']) > 0
        assert 'format is invalid' in result['errors'][0]
        assert len(result['suggestions']) > 0
        assert any('sk-ant-' in suggestion for suggestion in result['suggestions'])
    
    def test_anthropic_comprehensive_validation_missing(self):
        """Test comprehensive Anthropic validation with missing key."""
        result = self.detector.validate_api_key_comprehensive(
            'anthropic', ''
        )
        
        assert result['valid'] is False
        assert 'missing' in result['errors'][0]
        assert any('ANTHROPIC_API_KEY' in suggestion for suggestion in result['suggestions'])
    
    def test_openai_comprehensive_validation_valid(self):
        """Test comprehensive OpenAI validation with valid key."""
        result = self.detector.validate_api_key_comprehensive(
            'openai', 'sk-test123456789abcdef'
        )
        
        assert result['valid'] is True
        assert result['provider'] == 'openai'
        assert len(result['errors']) == 0
    
    def test_openai_comprehensive_validation_invalid_format(self):
        """Test comprehensive OpenAI validation with invalid format."""
        result = self.detector.validate_api_key_comprehensive(
            'openai', 'invalid-key'
        )
        
        assert result['valid'] is False
        assert 'format is invalid' in result['errors'][0]
        assert any('sk-' in suggestion for suggestion in result['suggestions'])
    
    def test_bedrock_comprehensive_validation(self):
        """Test comprehensive Bedrock validation."""
        result = self.detector.validate_api_key_comprehensive(
            'bedrock', ''  # Bedrock doesn't use direct API keys
        )
        
        assert result['provider'] == 'bedrock'
        # Bedrock validation depends on AWS credentials, not direct API key
    
    def test_ollama_comprehensive_validation(self):
        """Test comprehensive Ollama validation."""
        result = self.detector.validate_api_key_comprehensive(
            'ollama', ''  # Ollama doesn't require API keys
        )
        
        assert result['valid'] is True
        assert result['provider'] == 'ollama'
        assert len(result['errors']) == 0
    
    def test_unsupported_provider_validation(self):
        """Test comprehensive validation with unsupported provider."""
        result = self.detector.validate_api_key_comprehensive(
            'unsupported', 'any-key'
        )
        
        assert result['valid'] is False
        assert 'Unsupported provider' in result['errors'][0]


class TestModelSetup:
    """Test model setup functionality for different providers."""
    
    def setup_method(self):
        """Set up test environment before each test."""
        self.detector = ProviderDetector()
    
    @patch('assistant.providers.AnthropicModel')
    def test_setup_anthropic_model_valid(self, mock_anthropic_model):
        """Test setting up Anthropic model with valid configuration."""
        mock_model_instance = MagicMock()
        mock_anthropic_model.return_value = mock_model_instance
        
        config = {
            'api_key': 'sk-ant-test123456789',
            'model_id': 'claude-3-7-sonnet-20250219',
            'temperature': 0.7
        }
        
        result = self.detector.setup_model('anthropic', config)
        
        mock_anthropic_model.assert_called_once_with(
            model_id='claude-3-7-sonnet-20250219',
            api_key='sk-ant-test123456789',
            temperature=0.7
        )
        assert result == mock_model_instance
    
    def test_setup_anthropic_model_invalid_key(self):
        """Test setting up Anthropic model with invalid API key."""
        config = {
            'api_key': 'invalid-key',
            'model_id': 'claude-3-7-sonnet-20250219'
        }
        
        with pytest.raises(APIKeyValidationError) as exc_info:
            self.detector.setup_model('anthropic', config)
        
        error_msg = str(exc_info.value)
        assert 'API key validation failed' in error_msg
    
    @patch('assistant.providers.LiteLLMModel')
    def test_setup_openai_model_valid(self, mock_litellm_model):
        """Test setting up OpenAI model with valid configuration."""
        mock_model_instance = MagicMock()
        mock_litellm_model.return_value = mock_model_instance
        
        config = {
            'api_key': 'sk-test123456789',
            'model_id': 'gpt-4',
            'temperature': 0.8
        }
        
        result = self.detector.setup_model('openai', config)
        
        mock_litellm_model.assert_called_once_with(
            model_id='gpt-4',
            api_key='sk-test123456789',
            temperature=0.8
        )
        assert result == mock_model_instance
    
    @patch('assistant.providers.BedrockModel')
    def test_setup_bedrock_model_valid(self, mock_bedrock_model):
        """Test setting up Bedrock model with valid configuration."""
        mock_model_instance = MagicMock()
        mock_bedrock_model.return_value = mock_model_instance
        
        config = {
            'model_id': 'us.anthropic.claude-3-7-sonnet-20250219-v1:0',
            'region_name': 'us-west-2',
            'temperature': 0.7
        }
        
        result = self.detector.setup_model('bedrock', config)
        
        mock_bedrock_model.assert_called_once_with(
            model_id='us.anthropic.claude-3-7-sonnet-20250219-v1:0',
            region_name='us-west-2',
            temperature=0.7
        )
        assert result == mock_model_instance
    
    @patch('assistant.providers.OllamaModel')
    def test_setup_ollama_model_valid(self, mock_ollama_model):
        """Test setting up Ollama model with valid configuration."""
        mock_model_instance = MagicMock()
        mock_ollama_model.return_value = mock_model_instance
        
        config = {
            'base_url': 'http://localhost:11434',
            'model_id': 'llama3.3',
            'temperature': 0.9
        }
        
        result = self.detector.setup_model('ollama', config)
        
        mock_ollama_model.assert_called_once_with(
            host='http://localhost:11434',
            model_id='llama3.3',
            temperature=0.9
        )
        assert result == mock_model_instance
    
    def test_setup_model_unsupported_provider(self):
        """Test setting up model with unsupported provider."""
        config = {'model_id': 'test-model'}
        
        with pytest.raises(ProviderError) as exc_info:
            self.detector.setup_model('unsupported_provider', config)
        
        assert 'Unsupported provider' in str(exc_info.value)
    
    def test_setup_model_missing_required_config(self):
        """Test setting up model with missing required configuration."""
        # Anthropic model without API key
        config = {'model_id': 'claude-3-7-sonnet-20250219'}
        
        with pytest.raises(APIKeyValidationError):
            self.detector.setup_model('anthropic', config)


class TestProviderAvailability:
    """Test provider availability checking."""
    
    def setup_method(self):
        """Set up test environment before each test."""
        self.detector = ProviderDetector()
    
    @patch.dict(os.environ, {
        'ANTHROPIC_API_KEY': 'sk-ant-test123',
        'OPENAI_API_KEY': 'sk-test123'
    })
    def test_get_available_providers_multiple(self):
        """Test getting availability for multiple providers."""
        availability = self.detector.get_available_providers()
        
        assert isinstance(availability, dict)
        assert 'anthropic' in availability
        assert 'openai' in availability
        assert 'bedrock' in availability
        assert 'ollama' in availability
        
        assert availability['anthropic'] is True
        assert availability['openai'] is True
        assert availability['ollama'] is True  # Always available
    
    @patch.dict(os.environ, {}, clear=True)
    def test_get_available_providers_none(self):
        """Test getting availability when no credentials set."""
        with patch('boto3.Session') as mock_session:
            mock_session.return_value.get_credentials.return_value = None
            
            availability = self.detector.get_available_providers()
            
            assert availability['anthropic'] is False
            assert availability['openai'] is False
            assert availability['bedrock'] is False
            assert availability['ollama'] is True  # Always available
    
    @patch.dict(os.environ, {'ANTHROPIC_API_KEY': 'sk-ant-test123'})
    def test_get_credential_status_report(self):
        """Test getting comprehensive credential status report."""
        report = self.detector.get_credential_status_report()
        
        assert isinstance(report, dict)
        assert 'anthropic' in report
        assert 'openai' in report
        assert 'bedrock' in report
        assert 'ollama' in report
        
        # Check Anthropic status structure
        anthropic_status = report['anthropic']
        assert 'available' in anthropic_status
        assert 'env_var_set' in anthropic_status
        assert 'validation' in anthropic_status
        assert anthropic_status['available'] is True
        assert anthropic_status['env_var_set'] is True
        
        # Check validation details
        validation = anthropic_status['validation']
        assert 'valid' in validation
        assert 'errors' in validation
        assert 'suggestions' in validation
        
        # Check Ollama status
        ollama_status = report['ollama']
        assert ollama_status['available'] is True
        assert ollama_status['requires_local_server'] is True


class TestProviderErrorHandling:
    """Test error handling in provider operations."""
    
    def setup_method(self):
        """Set up test environment before each test."""
        self.detector = ProviderDetector()
    
    def test_api_key_validation_error_details(self):
        """Test APIKeyValidationError contains proper details."""
        try:
            self.detector.setup_model('anthropic', {'api_key': 'invalid'})
            assert False, "Should have raised APIKeyValidationError"
        except APIKeyValidationError as e:
            assert e.provider == 'anthropic'
            assert e.validation_result is not None
            assert not e.validation_result['valid']
            assert len(e.validation_result['errors']) > 0
            assert len(e.validation_result['suggestions']) > 0
    
    def test_provider_error_details(self):
        """Test ProviderError contains proper details."""
        try:
            self.detector.setup_model('invalid_provider', {})
            assert False, "Should have raised ProviderError"
        except ProviderError as e:
            assert 'invalid_provider' in str(e)
            assert 'Unsupported provider' in str(e)
    
    @patch('assistant.providers.AnthropicModel')
    def test_model_initialization_error_handling(self, mock_anthropic_model):
        """Test handling of model initialization errors."""
        mock_anthropic_model.side_effect = Exception("Model initialization failed")
        
        config = {
            'api_key': 'sk-ant-test123456789',
            'model_id': 'claude-3-7-sonnet-20250219'
        }
        
        with pytest.raises(Exception) as exc_info:
            self.detector.setup_model('anthropic', config)
        
        assert "Model initialization failed" in str(exc_info.value)


# Integration tests
class TestProviderIntegration:
    """Integration tests for provider detection and model setup."""
    
    def setup_method(self):
        """Set up test environment before each test."""
        self.detector = ProviderDetector()
    
    @patch.dict(os.environ, {'ANTHROPIC_API_KEY': 'sk-ant-test123456789'})
    @patch('assistant.providers.AnthropicModel')
    def test_full_anthropic_workflow(self, mock_anthropic_model):
        """Test full workflow from detection to model setup for Anthropic."""
        mock_model_instance = MagicMock()
        mock_anthropic_model.return_value = mock_model_instance
        
        # 1. Detect provider
        detected_provider = self.detector.detect_provider()
        assert detected_provider == 'anthropic'
        
        # 2. Validate credentials
        validation = self.detector.validate_api_key_comprehensive(
            'anthropic', 'sk-ant-test123456789'
        )
        assert validation['valid'] is True
        
        # 3. Setup model
        config = {
            'api_key': 'sk-ant-test123456789',
            'model_id': 'claude-3-7-sonnet-20250219',
            'temperature': 0.7
        }
        
        model = self.detector.setup_model('anthropic', config)
        
        assert model == mock_model_instance
        mock_anthropic_model.assert_called_once()
    
    @patch.dict(os.environ, {}, clear=True)
    @patch('assistant.providers.OllamaModel')
    def test_fallback_to_ollama_workflow(self, mock_ollama_model):
        """Test fallback workflow when no credentials available."""
        mock_model_instance = MagicMock()
        mock_ollama_model.return_value = mock_model_instance
        
        # Should fall back to Ollama
        detected_provider = self.detector.detect_provider()
        assert detected_provider == 'ollama'
        
        # Ollama validation should always succeed
        validation = self.detector.validate_api_key_comprehensive('ollama', '')
        assert validation['valid'] is True
        
        # Setup Ollama model
        config = {
            'base_url': 'http://localhost:11434',
            'model_id': 'llama3.3',
            'temperature': 0.7
        }
        
        model = self.detector.setup_model('ollama', config)
        assert model == mock_model_instance


if __name__ == '__main__':
    pytest.main([__file__]) 