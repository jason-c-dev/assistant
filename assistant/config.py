"""
Configuration management for Personal Assistant CLI.

This module handles loading, validating, and managing configuration files
with support for environment variable interpolation and user config directories.
"""

import os
import re
import shutil
import yaml
from pathlib import Path
from typing import Dict, Any, Optional, List, Tuple
import logging
import traceback

logger = logging.getLogger(__name__)

class ConfigError(Exception):
    """Custom exception for configuration-related errors."""
    pass

class YAMLSyntaxError(ConfigError):
    """Specific exception for YAML syntax errors."""
    pass

class ConfigValidationError(ConfigError):
    """Specific exception for configuration validation errors."""
    pass

class ConfigManager:
    """Manages configuration loading, validation, and user setup."""
    
    DEFAULT_CONFIG_FILENAME = "config.yaml"
    USER_CONFIG_DIR = os.path.expanduser("~/.assistant")
    
    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize ConfigManager.
        
        Args:
            config_path: Optional custom path to config file. If None, uses default locations.
        """
        self.config_path = config_path
        self._config_data = None
    
    def load_config(self) -> Dict[str, Any]:
        """
        Load configuration from file with comprehensive validation and error reporting.
        
        Returns:
            Dictionary containing the loaded and processed configuration.
            
        Raises:
            YAMLSyntaxError: If YAML syntax is invalid.
            ConfigValidationError: If configuration validation fails.
            ConfigError: If configuration cannot be loaded for other reasons.
        """
        config_file_path = self._get_config_file_path()
        
        try:
            # Read raw file content for better error reporting
            with open(config_file_path, 'r', encoding='utf-8') as f:
                raw_content = f.read()
            
            if not raw_content.strip():
                raise ConfigError(f"Configuration file {config_file_path} is empty")
            
            # Parse YAML with enhanced error handling
            try:
                raw_config = yaml.safe_load(raw_content)
            except yaml.YAMLError as e:
                yaml_error_details = self._parse_yaml_error(e, raw_content, config_file_path)
                raise YAMLSyntaxError(yaml_error_details)
                
            if raw_config is None:
                raise ConfigError(f"Configuration file {config_file_path} contains only comments or is effectively empty")
                
            # Interpolate environment variables
            try:
                self._config_data = self._interpolate_env_vars(raw_config)
            except Exception as e:
                raise ConfigError(f"Environment variable interpolation failed: {e}")
            
            # Comprehensive validation
            try:
                self._validate_config_comprehensive(self._config_data, config_file_path)
            except ConfigValidationError:
                raise  # Re-raise validation errors as-is
            except Exception as e:
                raise ConfigValidationError(f"Configuration validation failed: {e}")
            
            logger.info(f"Configuration loaded and validated successfully from {config_file_path}")
            return self._config_data
            
        except FileNotFoundError:
            error_msg = f"Configuration file not found: {config_file_path}\n"
            error_msg += "Suggestions:\n"
            error_msg += f"  💡 Run the assistant once to create default config directory\n"
            error_msg += f"  💡 Copy config.yaml to {os.path.dirname(config_file_path)}/\n"
            error_msg += f"  💡 Use --config flag to specify custom config location"
            raise ConfigError(error_msg)
        except (YAMLSyntaxError, ConfigValidationError):
            raise  # Re-raise these specific errors
        except Exception as e:
            raise ConfigError(f"Unexpected error loading configuration: {e}")
    
    def _get_config_file_path(self) -> str:
        """
        Determine the configuration file path to use.
        
        Returns:
            Path to the configuration file.
        """
        if self.config_path:
            # Custom config path provided
            return os.path.expanduser(self.config_path)
        
        # Check user config directory first
        user_config_path = os.path.join(self.USER_CONFIG_DIR, self.DEFAULT_CONFIG_FILENAME)
        if os.path.exists(user_config_path):
            return user_config_path
        
        # Fallback to project root config
        project_root = Path(__file__).parent.parent
        default_config_path = project_root / self.DEFAULT_CONFIG_FILENAME
        
        if os.path.exists(default_config_path):
            return str(default_config_path)
        
        raise ConfigError(f"No configuration file found. Expected locations: {user_config_path} or {default_config_path}")
    
    def _interpolate_env_vars(self, config: Any) -> Any:
        """
        Recursively interpolate environment variables in configuration values.
        
        Supports ${VAR_NAME} and ${VAR_NAME:-default_value} syntax.
        
        Args:
            config: Configuration data (dict, list, or string).
            
        Returns:
            Configuration data with environment variables interpolated.
        """
        if isinstance(config, dict):
            return {key: self._interpolate_env_vars(value) for key, value in config.items()}
        elif isinstance(config, list):
            return [self._interpolate_env_vars(item) for item in config]
        elif isinstance(config, str):
            return self._interpolate_string(config)
        else:
            return config
    
    def _interpolate_string(self, value: str) -> str:
        """
        Interpolate environment variables in a string value.
        
        Args:
            value: String that may contain environment variable references.
            
        Returns:
            String with environment variables interpolated.
        """
        # Pattern matches ${VAR_NAME} or ${VAR_NAME:-default}
        pattern = r'\$\{([^}]+)\}'
        
        def replace_env_var(match):
            var_expr = match.group(1)
            
            # Check for default value syntax
            if ':-' in var_expr:
                var_name, default_value = var_expr.split(':-', 1)
                return os.getenv(var_name.strip(), default_value.strip())
            else:
                var_name = var_expr.strip()
                env_value = os.getenv(var_name)
                if env_value is None:
                    logger.warning(f"Environment variable {var_name} not found")
                    return f"${{{var_name}}}"  # Keep original if not found
                return env_value
        
        return re.sub(pattern, replace_env_var, value)
    
    def _parse_yaml_error(self, yaml_error: yaml.YAMLError, raw_content: str, file_path: str) -> str:
        """
        Parse and format YAML error with detailed information and suggestions.
        
        Args:
            yaml_error: The YAML parsing error.
            raw_content: Raw file content.
            file_path: Path to the configuration file.
            
        Returns:
            Detailed error message with suggestions.
        """
        error_msg = f"YAML syntax error in configuration file: {file_path}\n\n"
        
        # Extract line and column information if available
        if hasattr(yaml_error, 'problem_mark') and yaml_error.problem_mark:
            mark = yaml_error.problem_mark
            line_num = mark.line + 1
            col_num = mark.column + 1
            
            error_msg += f"Error location: Line {line_num}, Column {col_num}\n"
            
            # Show the problematic line and context
            lines = raw_content.split('\n')
            if 0 <= mark.line < len(lines):
                error_msg += f"Problematic line: {lines[mark.line]}\n"
                error_msg += f"Error position: {' ' * (col_num - 1)}^\n\n"
            
            # Show context (lines around the error)
            start_line = max(0, mark.line - 2)
            end_line = min(len(lines), mark.line + 3)
            error_msg += "Context:\n"
            for i in range(start_line, end_line):
                prefix = ">>> " if i == mark.line else "    "
                error_msg += f"{prefix}{i + 1:3}: {lines[i]}\n"
            error_msg += "\n"
        
        # Add the original error message
        if hasattr(yaml_error, 'problem'):
            error_msg += f"Error description: {yaml_error.problem}\n"
        else:
            error_msg += f"Error description: {str(yaml_error)}\n"
        
        # Add common suggestions based on error type
        error_str = str(yaml_error).lower()
        error_msg += "\nCommon YAML syntax issues and fixes:\n"
        
        if 'mapping' in error_str or 'expected' in error_str:
            error_msg += "  💡 Check for missing colons (:) after keys\n"
            error_msg += "  💡 Ensure proper indentation (use spaces, not tabs)\n"
            error_msg += "  💡 Make sure all string values with special characters are quoted\n"
        
        if 'sequence' in error_str or 'list' in error_str:
            error_msg += "  💡 Check list syntax (items should start with '- ')\n"
            error_msg += "  💡 Ensure consistent indentation for list items\n"
        
        if 'found character' in error_str:
            error_msg += "  💡 Check for unescaped special characters\n"
            error_msg += "  💡 Quote strings containing special characters\n"
            error_msg += "  💡 Escape backslashes with double backslashes (\\\\)\n"
        
        if 'indentation' in error_str or 'indent' in error_str:
            error_msg += "  💡 Use consistent indentation (2 or 4 spaces)\n"
            error_msg += "  💡 Do not mix tabs and spaces\n"
            error_msg += "  💡 Ensure child elements are indented more than parent\n"
        
        error_msg += "\nGeneral YAML tips:\n"
        error_msg += "  💡 Use a YAML validator or linter to check syntax\n"
        error_msg += "  💡 Copy from the default config.yaml as a reference\n"
        error_msg += "  💡 Quote all string values that contain special characters\n"
        
        return error_msg.strip()
    
    def _validate_config_comprehensive(self, config: Dict[str, Any], file_path: str) -> None:
        """
        Comprehensive configuration validation with detailed error reporting.
        
        Args:
            config: Configuration dictionary to validate.
            file_path: Path to config file for error reporting.
            
        Raises:
            ConfigValidationError: If validation fails.
        """
        errors = []
        warnings = []
        
        # Check required top-level sections
        required_sections = ['providers', 'mcp_servers', 'system_prompt']
        for section in required_sections:
            if section not in config:
                errors.append(f"Missing required section: '{section}'")
        
        # Validate providers section
        if 'providers' in config:
            provider_errors = self._validate_providers_section(config['providers'])
            errors.extend(provider_errors)
        
        # Validate MCP servers section
        if 'mcp_servers' in config:
            mcp_errors = self._validate_mcp_servers_section(config['mcp_servers'])
            errors.extend(mcp_errors)
        
        # Validate default provider
        if 'default_provider' in config:
            default_provider = config['default_provider']
            if default_provider != 'auto' and 'providers' in config:
                if default_provider not in config['providers']:
                    errors.append(f"Default provider '{default_provider}' not found in providers section")
        
        # Validate system prompt
        if 'system_prompt' in config:
            if not isinstance(config['system_prompt'], str):
                errors.append("System prompt must be a string")
            elif not config['system_prompt'].strip():
                warnings.append("System prompt is empty")
        
        # Validate CLI settings if present
        if 'cli' in config:
            cli_errors = self._validate_cli_section(config['cli'])
            errors.extend(cli_errors)
        
        # Report errors and warnings
        if errors or warnings:
            error_msg = f"Configuration validation failed for: {file_path}\n\n"
            
            if errors:
                error_msg += "❌ ERRORS (must be fixed):\n"
                for i, error in enumerate(errors, 1):
                    error_msg += f"  {i}. {error}\n"
                error_msg += "\n"
            
            if warnings:
                error_msg += "⚠️ WARNINGS (recommended to fix):\n"
                for i, warning in enumerate(warnings, 1):
                    error_msg += f"  {i}. {warning}\n"
                error_msg += "\n"
            
            error_msg += "💡 Suggestions:\n"
            error_msg += "  • Check the default config.yaml for correct structure\n"
            error_msg += "  • Ensure all required environment variables are set\n"
            error_msg += "  • Use proper YAML syntax with consistent indentation\n"
            
            if errors:
                raise ConfigValidationError(error_msg.strip())
            else:
                logger.warning(error_msg.strip())
    
    def _validate_providers_section(self, providers: Dict[str, Any]) -> List[str]:
        """
        Validate the providers section of the configuration.
        
        Args:
            providers: Providers configuration dictionary.
            
        Returns:
            List of validation error messages.
        """
        errors = []
        
        if not providers:
            errors.append("Providers section is empty")
            return errors
        
        supported_providers = ['anthropic', 'openai', 'bedrock', 'ollama']
        
        for provider_name, provider_config in providers.items():
            if provider_name not in supported_providers:
                errors.append(f"Unsupported provider: '{provider_name}' (supported: {supported_providers})")
                continue
            
            if not isinstance(provider_config, dict):
                errors.append(f"Provider '{provider_name}' configuration must be a dictionary")
                continue
            
            # Validate provider-specific requirements
            if provider_name in ['anthropic', 'openai']:
                if 'api_key' not in provider_config:
                    errors.append(f"Provider '{provider_name}' missing required 'api_key' field")
                elif not provider_config['api_key']:
                    errors.append(f"Provider '{provider_name}' has empty 'api_key' field")
            
            if provider_name == 'bedrock':
                if 'model_id' not in provider_config:
                    errors.append(f"Provider '{provider_name}' missing required 'model_id' field")
            
            if provider_name == 'ollama':
                if 'model_id' not in provider_config:
                    errors.append(f"Provider '{provider_name}' missing required 'model_id' field")
                if 'base_url' not in provider_config:
                    errors.append(f"Provider '{provider_name}' missing 'base_url' field (should be Ollama server URL)")
        
        return errors
    
    def _validate_mcp_servers_section(self, mcp_servers: Dict[str, Any]) -> List[str]:
        """
        Validate the MCP servers section of the configuration.
        
        Args:
            mcp_servers: MCP servers configuration dictionary.
            
        Returns:
            List of validation error messages.
        """
        errors = []
        
        if not mcp_servers:
            errors.append("MCP servers section is empty")
            return errors
        
        # Memory server is required
        if 'memory' not in mcp_servers:
            errors.append("Required MCP server 'memory' not found")
        else:
            memory_config = mcp_servers['memory']
            if not isinstance(memory_config, dict):
                errors.append("Memory server configuration must be a dictionary")
            else:
                # Validate memory server configuration
                if 'command' not in memory_config:
                    errors.append("Memory server missing 'command' field")
                if 'args' not in memory_config:
                    errors.append("Memory server missing 'args' field")
                elif not isinstance(memory_config['args'], list):
                    errors.append("Memory server 'args' must be a list")
                
                # Check if memory server is marked as required
                if not memory_config.get('required', False):
                    errors.append("Memory server should be marked as required: true")
        
        # Validate other MCP servers
        for server_name, server_config in mcp_servers.items():
            if server_name == 'memory':
                continue  # Already validated above
            
            if not isinstance(server_config, dict):
                errors.append(f"MCP server '{server_name}' configuration must be a dictionary")
                continue
            
            if 'command' not in server_config:
                errors.append(f"MCP server '{server_name}' missing 'command' field")
            
            if 'args' not in server_config:
                errors.append(f"MCP server '{server_name}' missing 'args' field")
            elif not isinstance(server_config['args'], list):
                errors.append(f"MCP server '{server_name}' 'args' must be a list")
        
        return errors
    
    def _validate_cli_section(self, cli_config: Dict[str, Any]) -> List[str]:
        """
        Validate the CLI section of the configuration.
        
        Args:
            cli_config: CLI configuration dictionary.
            
        Returns:
            List of validation error messages.
        """
        errors = []
        
        if not isinstance(cli_config, dict):
            errors.append("CLI configuration must be a dictionary")
            return errors
        
        # Validate boolean fields
        boolean_fields = ['verbose']
        for field in boolean_fields:
            if field in cli_config and not isinstance(cli_config[field], bool):
                errors.append(f"CLI field '{field}' must be a boolean (true/false)")
        
        # Validate string fields
        string_fields = ['log_level', 'output_format']
        for field in string_fields:
            if field in cli_config and not isinstance(cli_config[field], str):
                errors.append(f"CLI field '{field}' must be a string")
        
        # Validate specific values
        if 'log_level' in cli_config:
            valid_levels = ['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL']
            if cli_config['log_level'] not in valid_levels:
                errors.append(f"CLI log_level must be one of: {valid_levels}")
        
        if 'output_format' in cli_config:
            valid_formats = ['text', 'json', 'markdown']
            if cli_config['output_format'] not in valid_formats:
                errors.append(f"CLI output_format must be one of: {valid_formats}")
        
        return errors
    
    def _validate_config(self, config: Dict[str, Any]) -> None:
        """
        Validate the loaded configuration.
        
        Args:
            config: Configuration dictionary to validate.
            
        Raises:
            ConfigError: If configuration is invalid.
        """
        required_sections = ['providers', 'mcp_servers', 'system_prompt']
        
        for section in required_sections:
            if section not in config:
                raise ConfigError(f"Missing required configuration section: {section}")
        
        # Validate providers section
        providers = config.get('providers', {})
        if not providers:
            raise ConfigError("No providers configured")
        
        # Validate default provider
        default_provider = config.get('default_provider', 'auto')
        if default_provider != 'auto' and default_provider not in providers:
            raise ConfigError(f"Default provider '{default_provider}' not found in providers configuration")
        
        # Validate MCP servers
        mcp_servers = config.get('mcp_servers', {})
        if 'memory' not in mcp_servers:
            raise ConfigError("Memory MCP server configuration is required")
        
        memory_config = mcp_servers['memory']
        if not memory_config.get('required', False):
            logger.warning("Memory server is not marked as required - this may cause issues")
    
    def ensure_user_config_dir(self) -> str:
        """
        Ensure the user configuration directory exists and copy default config if needed.
        
        Returns:
            Path to the user configuration directory.
        """
        os.makedirs(self.USER_CONFIG_DIR, exist_ok=True)
        
        user_config_path = os.path.join(self.USER_CONFIG_DIR, self.DEFAULT_CONFIG_FILENAME)
        
        # Copy default config if user config doesn't exist
        if not os.path.exists(user_config_path):
            project_root = Path(__file__).parent.parent
            default_config_path = project_root / self.DEFAULT_CONFIG_FILENAME
            
            if os.path.exists(default_config_path):
                shutil.copy2(default_config_path, user_config_path)
                logger.info(f"Created user configuration file: {user_config_path}")
            else:
                raise ConfigError(f"Default configuration file not found: {default_config_path}")
        
        return self.USER_CONFIG_DIR
    
    def get_provider_config(self, provider_name: str) -> Dict[str, Any]:
        """
        Get configuration for a specific provider.
        
        Args:
            provider_name: Name of the provider.
            
        Returns:
            Provider configuration dictionary.
            
        Raises:
            ConfigError: If provider not found or config not loaded.
        """
        if self._config_data is None:
            raise ConfigError("Configuration not loaded. Call load_config() first.")
        
        providers = self._config_data.get('providers', {})
        if provider_name not in providers:
            raise ConfigError(f"Provider '{provider_name}' not found in configuration")
        
        return providers[provider_name]
    
    def get_mcp_server_config(self, server_name: str) -> Dict[str, Any]:
        """
        Get configuration for a specific MCP server.
        
        Args:
            server_name: Name of the MCP server.
            
        Returns:
            MCP server configuration dictionary.
            
        Raises:
            ConfigError: If server not found or config not loaded.
        """
        if self._config_data is None:
            raise ConfigError("Configuration not loaded. Call load_config() first.")
        
        mcp_servers = self._config_data.get('mcp_servers', {})
        if server_name not in mcp_servers:
            raise ConfigError(f"MCP server '{server_name}' not found in configuration")
        
        return mcp_servers[server_name]
    
    def get_system_prompt(self) -> str:
        """
        Get the system prompt from configuration.
        
        Returns:
            System prompt string.
            
        Raises:
            ConfigError: If config not loaded or system prompt not found.
        """
        if self._config_data is None:
            raise ConfigError("Configuration not loaded. Call load_config() first.")
        
        return self._config_data.get('system_prompt', '')
    
    def get_cli_settings(self) -> Dict[str, Any]:
        """
        Get CLI settings from configuration.
        
        Returns:
            CLI settings dictionary with defaults.
        """
        if self._config_data is None:
            raise ConfigError("Configuration not loaded. Call load_config() first.")
        
        default_cli_settings = {
            'verbose': False,
            'log_level': 'INFO',
            'output_format': 'text'
        }
        
        cli_settings = self._config_data.get('cli', {})
        return {**default_cli_settings, **cli_settings}

# Convenience functions for easy access
def load_config(config_path: Optional[str] = None) -> Dict[str, Any]:
    """
    Load configuration using the default ConfigManager.
    
    Args:
        config_path: Optional custom path to config file.
        
    Returns:
        Loaded configuration dictionary.
    """
    manager = ConfigManager(config_path)
    manager.ensure_user_config_dir()
    return manager.load_config()

def get_config_manager(config_path: Optional[str] = None) -> ConfigManager:
    """
    Get a ConfigManager instance.
    
    Args:
        config_path: Optional custom path to config file.
        
    Returns:
        ConfigManager instance.
    """
    return ConfigManager(config_path) 