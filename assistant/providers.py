"""
Model provider auto-detection and setup for Personal Assistant CLI.

This module handles detection of available AI model providers based on environment
variables and credentials, and provides setup functions using Strands built-in models.
"""

import os
import re
import logging
import requests
from typing import Dict, Any, Optional, Union

try:
    import boto3
    from botocore.exceptions import NoCredentialsError, ClientError
    BOTO3_AVAILABLE = True
except ImportError:
    BOTO3_AVAILABLE = False

# Import Strands model providers
try:
    from strands.models import BedrockModel, AnthropicModel, OllamaModel
    from strands.models.litellm import LiteLLMModel
    STRANDS_AVAILABLE = True
except ImportError:
    STRANDS_AVAILABLE = False
    # Define placeholder classes for development
    class BedrockModel:
        def __init__(self, **kwargs):
            pass
    
    class AnthropicModel:
        def __init__(self, **kwargs):
            pass
    
    class OllamaModel:
        def __init__(self, **kwargs):
            pass
    
    class LiteLLMModel:
        def __init__(self, **kwargs):
            pass

logger = logging.getLogger(__name__)

class ProviderError(Exception):
    """Custom exception for provider-related errors."""
    pass

class APIKeyValidationError(ProviderError):
    """Specific exception for API key validation failures."""
    pass

class ProviderDetector:
    """Handles detection and setup of AI model providers."""
    
    SUPPORTED_PROVIDERS = ['anthropic', 'openai', 'bedrock', 'ollama']
    
    def __init__(self):
        """Initialize the provider detector."""
        if not STRANDS_AVAILABLE:
            logger.warning("Strands SDK not available - provider setup will not work properly")
    
    def detect_provider(self) -> str:
        """
        Auto-detect available provider based on environment variables and credentials.
        
        Detection priority:
        1. Anthropic (ANTHROPIC_API_KEY)
        2. OpenAI (OPENAI_API_KEY)  
        3. AWS Bedrock (AWS credentials)
        4. Ollama (fallback for local development)
        
        Returns:
            String name of the detected provider.
        """
        logger.info("Starting provider auto-detection...")
        
        # Check for Anthropic API key
        if self._validate_anthropic_credentials():
            logger.info("Detected valid Anthropic API key - using Anthropic provider")
            return 'anthropic'
        
        # Check for OpenAI API key
        if self._validate_openai_credentials():
            logger.info("Detected valid OpenAI API key - using OpenAI provider")
            return 'openai'
        
        # Check for AWS credentials
        if self._validate_bedrock_credentials():
            logger.info("Detected valid AWS credentials - using Bedrock provider")
            return 'bedrock'
        
        # Fallback to Ollama for local development
        logger.info("No cloud provider credentials found - falling back to Ollama for local development")
        return 'ollama'
    
    def _validate_anthropic_credentials(self) -> bool:
        """
        Validate Anthropic API key format and presence.
        
        Returns:
            True if Anthropic credentials are valid, False otherwise.
        """
        api_key = os.getenv('ANTHROPIC_API_KEY')
        if not api_key:
            logger.debug("ANTHROPIC_API_KEY environment variable not set")
            return False
        
        # Validate key format (Anthropic keys start with 'sk-ant-')
        if not api_key.startswith('sk-ant-'):
            logger.debug("Anthropic API key does not have expected format (should start with 'sk-ant-')")
            return False
        
        # Basic length check
        if len(api_key) < 20:
            logger.debug("Anthropic API key appears too short")
            return False
        
        logger.debug("Anthropic API key format validation passed")
        return True
    
    def _validate_openai_credentials(self) -> bool:
        """
        Validate OpenAI API key format and presence.
        
        Returns:
            True if OpenAI credentials are valid, False otherwise.
        """
        api_key = os.getenv('OPENAI_API_KEY')
        if not api_key:
            logger.debug("OPENAI_API_KEY environment variable not set")
            return False
        
        # Validate key format (OpenAI keys start with 'sk-')
        if not api_key.startswith('sk-'):
            logger.debug("OpenAI API key does not have expected format (should start with 'sk-')")
            return False
        
        # Basic length check
        if len(api_key) < 20:
            logger.debug("OpenAI API key appears too short")
            return False
        
        logger.debug("OpenAI API key format validation passed")
        return True
    
    def _validate_bedrock_credentials(self) -> bool:
        """
        Validate AWS credentials for Bedrock access.
        
        Returns:
            True if AWS credentials are valid, False otherwise.
        """
        # Check environment variables first
        aws_access_key = os.getenv('AWS_ACCESS_KEY_ID')
        aws_secret_key = os.getenv('AWS_SECRET_ACCESS_KEY')
        
        if aws_access_key and aws_secret_key:
            logger.debug("AWS credentials found in environment variables")
            return True
        
        # Check boto3 credential chain if available
        if BOTO3_AVAILABLE:
            try:
                session = boto3.Session()
                credentials = session.get_credentials()
                if credentials is not None:
                    logger.debug("AWS credentials found via boto3 credential chain")
                    return True
            except (NoCredentialsError, ClientError) as e:
                logger.debug(f"AWS credentials check failed: {e}")
                return False
        
        logger.debug("No valid AWS credentials found")
        return False
    
    def _check_aws_credentials(self) -> bool:
        """
        Check if AWS credentials are available.
        
        Returns:
            True if AWS credentials are available, False otherwise.
        """
        return self._validate_bedrock_credentials()
    
    def validate_api_key_comprehensive(self, provider_name: str, api_key: str) -> Dict[str, Any]:
        """
        Perform comprehensive API key validation with detailed error reporting.
        
        Args:
            provider_name: Name of the provider to validate.
            api_key: API key to validate.
            
        Returns:
            Dictionary with validation results including errors and suggestions.
        """
        result = {
            'valid': False,
            'errors': [],
            'warnings': [],
            'suggestions': []
        }
        
        if provider_name == 'anthropic':
            if not api_key:
                result['errors'].append("Anthropic API key is missing")
                result['suggestions'].append("Set the ANTHROPIC_API_KEY environment variable")
                result['suggestions'].append("Get your API key from https://console.anthropic.com/")
            elif not api_key.startswith('sk-ant-'):
                result['errors'].append("Anthropic API key format is invalid")
                result['suggestions'].append("Anthropic API keys should start with 'sk-ant-'")
                result['suggestions'].append("Verify you copied the complete key from https://console.anthropic.com/")
            elif len(api_key) < 20:
                result['errors'].append("Anthropic API key appears too short")
                result['suggestions'].append("Ensure you copied the complete API key")
            else:
                result['valid'] = True
                
        elif provider_name == 'openai':
            if not api_key:
                result['errors'].append("OpenAI API key is missing")
                result['suggestions'].append("Set the OPENAI_API_KEY environment variable")
                result['suggestions'].append("Get your API key from https://platform.openai.com/api-keys")
            elif not api_key.startswith('sk-'):
                result['errors'].append("OpenAI API key format is invalid")
                result['suggestions'].append("OpenAI API keys should start with 'sk-'")
                result['suggestions'].append("Verify you copied the complete key from https://platform.openai.com/api-keys")
            elif len(api_key) < 20:
                result['errors'].append("OpenAI API key appears too short")
                result['suggestions'].append("Ensure you copied the complete API key")
            else:
                result['valid'] = True
                
        else:
            result['errors'].append(f"API key validation not implemented for provider: {provider_name}")
        
        return result
    
    def get_credential_status_report(self) -> Dict[str, Dict[str, Any]]:
        """
        Generate a comprehensive report of credential status for all providers.
        
        Returns:
            Dictionary with detailed credential status for each provider.
        """
        report = {}
        
        # Anthropic status
        anthropic_key = os.getenv('ANTHROPIC_API_KEY')
        anthropic_validation = self.validate_api_key_comprehensive('anthropic', anthropic_key or '')
        report['anthropic'] = {
            'available': self._validate_anthropic_credentials(),
            'env_var_set': bool(anthropic_key),
            'validation': anthropic_validation
        }
        
        # OpenAI status
        openai_key = os.getenv('OPENAI_API_KEY')
        openai_validation = self.validate_api_key_comprehensive('openai', openai_key or '')
        report['openai'] = {
            'available': self._validate_openai_credentials(),
            'env_var_set': bool(openai_key),
            'validation': openai_validation
        }
        
        # AWS Bedrock status
        report['bedrock'] = {
            'available': self._validate_bedrock_credentials(),
            'env_vars_set': bool(os.getenv('AWS_ACCESS_KEY_ID') and os.getenv('AWS_SECRET_ACCESS_KEY')),
            'boto3_available': BOTO3_AVAILABLE,
            'credential_chain_available': False
        }
        
        if BOTO3_AVAILABLE:
            try:
                session = boto3.Session()
                credentials = session.get_credentials()
                report['bedrock']['credential_chain_available'] = credentials is not None
            except Exception:
                pass
        
        # Ollama status (always available for local development)
        report['ollama'] = {
            'available': True,
            'requires_local_server': True,
            'default_url': 'http://localhost:11434'
        }
        
        return report

    def get_available_providers(self) -> Dict[str, bool]:
        """
        Get a dictionary of all providers and their availability status.
        
        Returns:
            Dictionary mapping provider names to availability status.
        """
        availability = {}
        
        # Check Anthropic
        availability['anthropic'] = self._validate_anthropic_credentials()
        
        # Check OpenAI
        availability['openai'] = self._validate_openai_credentials()
        
        # Check AWS Bedrock
        availability['bedrock'] = self._validate_bedrock_credentials()
        
        # Ollama is always considered available for local development
        availability['ollama'] = True
        
        logger.debug(f"Provider availability: {availability}")
        return availability
    
    def setup_model(self, provider_name: str, provider_config: Dict[str, Any]) -> Union[AnthropicModel, BedrockModel, LiteLLMModel, OllamaModel]:
        """
        Setup a model using the specified provider and configuration.
        
        Args:
            provider_name: Name of the provider to setup.
            provider_config: Configuration dictionary for the provider.
            
        Returns:
            Configured model instance using Strands built-in providers.
            
        Raises:
            ProviderError: If provider setup fails or provider is not supported.
        """
        if provider_name not in self.SUPPORTED_PROVIDERS:
            raise ProviderError(f"Unsupported provider: {provider_name}. Supported providers: {self.SUPPORTED_PROVIDERS}")
        
        logger.info(f"Setting up {provider_name} provider...")
        
        try:
            if provider_name == 'anthropic':
                return self._setup_anthropic(provider_config)
            elif provider_name == 'bedrock':
                return self._setup_bedrock(provider_config)
            elif provider_name == 'openai':
                return self._setup_openai(provider_config)
            elif provider_name == 'ollama':
                return self._setup_ollama(provider_config)
            else:
                raise ProviderError(f"Provider setup not implemented: {provider_name}")
        
        except Exception as e:
            raise ProviderError(f"Failed to setup {provider_name} provider: {e}")
    
    def _setup_anthropic(self, config: Dict[str, Any]) -> AnthropicModel:
        """
        Setup Anthropic provider using AnthropicModel from Strands.
        
        Args:
            config: Anthropic provider configuration.
            
        Returns:
            Configured AnthropicModel instance.
            
        Raises:
            APIKeyValidationError: If API key is invalid or missing.
            ProviderError: If setup fails for other reasons.
        """
        api_key = config.get('api_key')
        
        # Comprehensive API key validation
        validation_result = self.validate_api_key_comprehensive('anthropic', api_key or '')
        if not validation_result['valid']:
            error_msg = "Anthropic API key validation failed:\n"
            for error in validation_result['errors']:
                error_msg += f"  ❌ {error}\n"
            if validation_result['suggestions']:
                error_msg += "Suggestions:\n"
                for suggestion in validation_result['suggestions']:
                    error_msg += f"  💡 {suggestion}\n"
            raise APIKeyValidationError(error_msg.strip())
        
        try:
            model_config = {
                'model_id': config.get('model_id', 'claude-3-7-sonnet-20250219'),
                'api_key': api_key,
                'temperature': config.get('temperature', 0.7),
            }
            
            # Add max_tokens if specified
            if 'max_tokens' in config:
                model_config['max_tokens'] = config['max_tokens']
            
            logger.info(f"Creating AnthropicModel with model_id: {model_config['model_id']}")
            return AnthropicModel(**model_config)
            
        except Exception as e:
            raise ProviderError(f"Failed to create Anthropic model: {e}")
    
    def _setup_bedrock(self, config: Dict[str, Any]) -> BedrockModel:
        """
        Setup AWS Bedrock provider using BedrockModel from Strands.
        
        Args:
            config: Bedrock provider configuration.
            
        Returns:
            Configured BedrockModel instance.
            
        Raises:
            APIKeyValidationError: If AWS credentials are invalid or missing.
            ProviderError: If setup fails for other reasons.
        """
        if not self._validate_bedrock_credentials():
            error_msg = "AWS credentials validation failed:\n"
            error_msg += "  ❌ No valid AWS credentials found\n"
            error_msg += "Suggestions:\n"
            error_msg += "  💡 Set AWS_ACCESS_KEY_ID and AWS_SECRET_ACCESS_KEY environment variables\n"
            error_msg += "  💡 Configure AWS CLI with 'aws configure'\n"
            error_msg += "  💡 Use IAM roles if running on AWS infrastructure\n"
            error_msg += "  💡 Ensure your credentials have Bedrock access permissions\n"
            raise APIKeyValidationError(error_msg.strip())
        
        try:
            model_config = {
                'model_id': config.get('model_id', 'us.anthropic.claude-3-7-sonnet-20250219-v1:0'),
                'region_name': config.get('region_name', 'us-west-2'),
                'temperature': config.get('temperature', 0.7),
            }
            
            logger.info(f"Creating BedrockModel with model_id: {model_config['model_id']} in region: {model_config['region_name']}")
            return BedrockModel(**model_config)
            
        except Exception as e:
            raise ProviderError(f"Failed to create Bedrock model: {e}")
    
    def _setup_openai(self, config: Dict[str, Any]) -> LiteLLMModel:
        """
        Setup OpenAI provider using LiteLLMModel from Strands.
        
        Args:
            config: OpenAI provider configuration.
            
        Returns:
            Configured LiteLLMModel instance.
            
        Raises:
            APIKeyValidationError: If API key is invalid or missing.
            ProviderError: If setup fails for other reasons.
        """
        api_key = config.get('api_key')
        
        # Comprehensive API key validation
        validation_result = self.validate_api_key_comprehensive('openai', api_key or '')
        if not validation_result['valid']:
            error_msg = "OpenAI API key validation failed:\n"
            for error in validation_result['errors']:
                error_msg += f"  ❌ {error}\n"
            if validation_result['suggestions']:
                error_msg += "Suggestions:\n"
                for suggestion in validation_result['suggestions']:
                    error_msg += f"  💡 {suggestion}\n"
            raise APIKeyValidationError(error_msg.strip())
        
        try:
            model_config = {
                'model_id': config.get('model_id', 'gpt-4'),
                'api_key': api_key,
                'temperature': config.get('temperature', 0.7),
            }
            
            # Add max_tokens if specified
            if 'max_tokens' in config:
                model_config['max_tokens'] = config['max_tokens']
            
            logger.info(f"Creating LiteLLMModel for OpenAI with model_id: {model_config['model_id']}")
            return LiteLLMModel(**model_config)
            
        except Exception as e:
            raise ProviderError(f"Failed to create OpenAI model: {e}")
    
    def _setup_ollama(self, config: Dict[str, Any]) -> OllamaModel:
        """
        Setup Ollama provider using OllamaModel from Strands.
        
        Args:
            config: Ollama provider configuration.
            
        Returns:
            Configured OllamaModel instance.
            
        Raises:
            ProviderError: If setup fails.
        """
        model_config = {
            'host': config.get('base_url', 'http://localhost:11434'),
            'model_id': config.get('model_id', 'llama3.3'),
            'temperature': config.get('temperature', 0.7),
        }
        
        logger.info(f"Creating OllamaModel with model_id: {model_config['model_id']} at host: {model_config['host']}")
        return OllamaModel(**model_config)
    
    def validate_provider_config(self, provider_name: str, config: Dict[str, Any]) -> bool:
        """
        Validate provider configuration without setting up the model.
        
        Args:
            provider_name: Name of the provider to validate.
            config: Configuration dictionary for the provider.
            
        Returns:
            True if configuration is valid, False otherwise.
        """
        try:
            if provider_name == 'anthropic':
                api_key = config.get('api_key', '')
                return bool(api_key and not api_key.startswith('${'))
            
            elif provider_name == 'openai':
                api_key = config.get('api_key', '')
                return bool(api_key and not api_key.startswith('${'))
            
            elif provider_name == 'bedrock':
                return self._check_aws_credentials()
            
            elif provider_name == 'ollama':
                # Ollama doesn't require credentials, just check for basic config
                return 'model_id' in config
            
            else:
                return False
                
        except Exception as e:
            logger.debug(f"Provider validation failed for {provider_name}: {e}")
            return False

# Convenience functions for easy access
def detect_provider() -> str:
    """
    Auto-detect available provider using the default ProviderDetector.
    
    Returns:
        String name of the detected provider.
    """
    detector = ProviderDetector()
    return detector.detect_provider()

def setup_model(provider_name: str, provider_config: Dict[str, Any]) -> Union[AnthropicModel, BedrockModel, LiteLLMModel, OllamaModel]:
    """
    Setup a model using the specified provider and configuration.
    
    Args:
        provider_name: Name of the provider to setup.
        provider_config: Configuration dictionary for the provider.
        
    Returns:
        Configured model instance.
    """
    detector = ProviderDetector()
    return detector.setup_model(provider_name, provider_config)

def get_available_providers() -> Dict[str, bool]:
    """
    Get available providers using the default ProviderDetector.
    
    Returns:
        Dictionary mapping provider names to availability status.
    """
    detector = ProviderDetector()
    return detector.get_available_providers() 