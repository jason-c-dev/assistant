"""
Command-line interface for Personal Assistant CLI.

This module provides the main CLI entry point with support for natural language queries,
configuration management, provider selection, and memory operations.
"""

import os
import sys
import logging
import click
from pathlib import Path
from typing import Optional, Dict, Any

# Import our modules
from assistant import __version__
from assistant.config import ConfigManager, ConfigError
from assistant.providers import ProviderDetector, ProviderError
from assistant.memory_client import MemoryClient, MemoryClientError
from assistant.error_reporting import get_error_reporter, error_context, setup_global_error_handling

# Setup logging
logger = logging.getLogger(__name__)

class CLIError(Exception):
    """Custom exception for CLI-related errors."""
    pass

def setup_logging(verbose: bool = False, log_level: str = "INFO") -> None:
    """
    Setup logging configuration.
    
    Args:
        verbose: Enable verbose logging output.
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR).
    """
    if verbose:
        level = logging.DEBUG
        format_str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    else:
        level = getattr(logging, log_level.upper(), logging.INFO)
        format_str = "%(levelname)s: %(message)s"
    
    logging.basicConfig(
        level=level,
        format=format_str,
        handlers=[logging.StreamHandler()]
    )

def validate_query(query: str) -> str:
    """
    Validate and clean the user query.
    
    Args:
        query: Raw user query string.
        
    Returns:
        Cleaned query string.
        
    Raises:
        CLIError: If query is invalid.
    """
    if not query or not query.strip():
        raise CLIError("Query cannot be empty. Please provide a question or command.")
    
    cleaned_query = query.strip()
    
    # Basic length validation
    if len(cleaned_query) > 10000:
        raise CLIError("Query is too long. Please limit to 10,000 characters.")
    
    return cleaned_query

def format_response(response: str, output_format: str = "text") -> str:
    """
    Format the assistant's response according to the specified format.
    
    Args:
        response: Raw response from the assistant.
        output_format: Output format (text, json, markdown).
        
    Returns:
        Formatted response string.
    """
    if output_format == "json":
        import json
        return json.dumps({"response": response}, indent=2)
    elif output_format == "markdown":
        return f"## Assistant Response\n\n{response}\n"
    else:  # text format (default)
        return response

# Memory management command group
@click.group(invoke_without_command=True)
@click.pass_context
def memory_commands(ctx):
    """Memory management commands."""
    if ctx.invoked_subcommand is None:
        click.echo("Available memory commands: export, import, reset, stats")

@memory_commands.command("export")
@click.option("--output", "-o", help="Output file path for memory export")
@click.option("--config", "-c", help="Path to configuration file")
@click.option("--verbose", "-v", is_flag=True, help="Enable verbose output")
def export_memory(output: Optional[str], config: Optional[str], verbose: bool):
    """Export memory data to JSON file."""
    try:
        setup_logging(verbose)
        
        # Load configuration
        config_manager = ConfigManager(config)
        config_data = config_manager.load_config()
        
        # Get memory server config
        memory_config = config_manager.get_mcp_server_config("memory")
        
        # Create memory client and export
        memory_client = MemoryClient(memory_config)
        export_path = memory_client.export_memory(output)
        
        click.echo(f"✅ Memory exported successfully to: {export_path}")
        
    except (ConfigError, MemoryClientError, CLIError) as e:
        click.echo(f"❌ Error exporting memory: {e}", err=True)
        sys.exit(1)
    except Exception as e:
        click.echo(f"❌ Unexpected error: {e}", err=True)
        sys.exit(1)

@memory_commands.command("import")
@click.argument("import_file", type=click.Path(exists=True))
@click.option("--config", "-c", help="Path to configuration file")
@click.option("--no-backup", is_flag=True, help="Skip backing up existing memory")
@click.option("--verbose", "-v", is_flag=True, help="Enable verbose output")
def import_memory(import_file: str, config: Optional[str], no_backup: bool, verbose: bool):
    """Import memory data from JSON file."""
    try:
        setup_logging(verbose)
        
        # Load configuration
        config_manager = ConfigManager(config)
        config_data = config_manager.load_config()
        
        # Get memory server config
        memory_config = config_manager.get_mcp_server_config("memory")
        
        # Create memory client and import
        memory_client = MemoryClient(memory_config)
        memory_client.import_memory(import_file, backup=not no_backup)
        
        click.echo(f"✅ Memory imported successfully from: {import_file}")
        
    except (ConfigError, MemoryClientError, CLIError) as e:
        click.echo(f"❌ Error importing memory: {e}", err=True)
        sys.exit(1)
    except Exception as e:
        click.echo(f"❌ Unexpected error: {e}", err=True)
        sys.exit(1)

@memory_commands.command("reset")
@click.option("--config", "-c", help="Path to configuration file")
@click.option("--yes", is_flag=True, help="Confirm memory reset without prompting")
@click.option("--verbose", "-v", is_flag=True, help="Enable verbose output")
def reset_memory(config: Optional[str], yes: bool, verbose: bool):
    """Reset (clear) all stored memory."""
    try:
        setup_logging(verbose)
        
        # Confirmation prompt
        if not yes:
            if not click.confirm("⚠️  This will permanently delete all stored memory. Continue?"):
                click.echo("Memory reset cancelled.")
                return
        
        # Load configuration
        config_manager = ConfigManager(config)
        config_data = config_manager.load_config()
        
        # Get memory server config
        memory_config = config_manager.get_mcp_server_config("memory")
        
        # Create memory client and reset
        memory_client = MemoryClient(memory_config)
        memory_client.reset_memory(confirm=True)
        
        click.echo("✅ Memory reset successfully. A backup was created.")
        
    except (ConfigError, MemoryClientError, CLIError) as e:
        click.echo(f"❌ Error resetting memory: {e}", err=True)
        sys.exit(1)
    except Exception as e:
        click.echo(f"❌ Unexpected error: {e}", err=True)
        sys.exit(1)

@memory_commands.command("stats")
@click.option("--config", "-c", help="Path to configuration file")
@click.option("--verbose", "-v", is_flag=True, help="Enable verbose output")
def memory_stats(config: Optional[str], verbose: bool):
    """Show memory statistics and information."""
    try:
        setup_logging(verbose)
        
        # Load configuration
        config_manager = ConfigManager(config)
        config_data = config_manager.load_config()
        
        # Get memory server config
        memory_config = config_manager.get_mcp_server_config("memory")
        
        # Create memory client and get stats
        memory_client = MemoryClient(memory_config)
        stats = memory_client.get_memory_stats()
        validation = memory_client.validate_memory_file()
        
        # Display stats
        click.echo("📊 Memory Statistics:")
        click.echo(f"   File Path: {stats['memory_file_path']}")
        click.echo(f"   Exists: {'✅' if stats['exists'] else '❌'} {stats['exists']}")
        click.echo(f"   Size: {stats['size_mb']} MB ({stats['size_bytes']} bytes)")
        click.echo(f"   Entries: {stats['entry_count']}")
        if stats['last_modified']:
            click.echo(f"   Last Modified: {stats['last_modified']}")
        
        # Display validation
        click.echo(f"\n🔍 File Validation:")
        click.echo(f"   Valid: {'✅' if validation['valid'] else '❌'} {validation['valid']}")
        click.echo(f"   Format: {validation['format']}")
        
        if validation['errors']:
            click.echo("   Errors:")
            for error in validation['errors']:
                click.echo(f"     • {error}")
        
        if validation['warnings']:
            click.echo("   Warnings:")
            for warning in validation['warnings']:
                click.echo(f"     • {warning}")
        
    except (ConfigError, MemoryClientError, CLIError) as e:
        click.echo(f"❌ Error getting memory stats: {e}", err=True)
        sys.exit(1)
    except Exception as e:
        click.echo(f"❌ Unexpected error: {e}", err=True)
        sys.exit(1)

# Main CLI command
@click.command()
@click.argument("query", required=False)
@click.option("--config", "-c", help="Path to configuration file (default: ~/.assistant/config.yaml)")
@click.option("--provider", "-p", 
              type=click.Choice(['anthropic', 'openai', 'bedrock', 'ollama', 'auto'], case_sensitive=False),
              help="Override model provider (auto-detects if not specified)")
@click.option("--verbose", "-v", is_flag=True, help="Enable verbose output showing memory operations")
@click.option("--version", is_flag=True, help="Show version information")
@click.option("--reset-memory", is_flag=True, help="Reset all stored memory (with confirmation)")
@click.option("--export-memory", help="Export memory to specified JSON file")
@click.option("--import-memory", help="Import memory from specified JSON file")
@click.option("--memory-stats", is_flag=True, help="Show memory statistics")
@click.option("--list-providers", is_flag=True, help="List available providers and their status")
@click.option("--output-format", 
              type=click.Choice(['text', 'json', 'markdown'], case_sensitive=False),
              default='text',
              help="Output format for responses")
def main(
    query: Optional[str],
    config: Optional[str],
    provider: Optional[str],
    verbose: bool,
    version: bool,
    reset_memory: bool,
    export_memory: Optional[str],
    import_memory: Optional[str],
    memory_stats: bool,
    list_providers: bool,
    output_format: str
):
    """
    Personal Assistant CLI - A memory-enabled AI assistant.
    
    QUERY is your natural language question or command for the assistant.
    
    Examples:
    
        assistant "What meetings do I have this week?"
        
        assistant --provider anthropic "What did I discuss with John last week?"
        
        assistant --verbose "Update my skills to include Python and FastAPI"
        
        assistant --config /custom/config.yaml "Tell me about my goals"
        
        assistant --export-memory backup.json
        
        assistant --reset-memory
    
    For more help: assistant --help
    """
    # Setup global error handling early
    setup_global_error_handling(verbose)
    error_reporter = get_error_reporter(verbose)
    
    try:
        # Handle version flag
        if version:
            click.echo(f"Personal Assistant CLI v{__version__}")
            click.echo("A memory-enabled AI assistant using AWS Strands Agent SDK and MCP")
            return
        
        # Setup logging early
        setup_logging(verbose)
        
        # Handle memory management commands with error context
        if reset_memory:
            with error_context(error_reporter, "resetting memory", raise_on_error=False):
                ctx = click.Context(reset_memory)
                ctx.invoke(reset_memory, config=config, yes=False, verbose=verbose)
                return
        
        if export_memory:
            with error_context(error_reporter, "exporting memory", raise_on_error=False):
                ctx = click.Context(export_memory)
                ctx.invoke(export_memory, output=export_memory, config=config, verbose=verbose)
                return
        
        if import_memory:
            with error_context(error_reporter, "importing memory", raise_on_error=False):
                if not os.path.exists(import_memory):
                    raise CLIError(f"Import file does not exist: {import_memory}")
                ctx = click.Context(import_memory)
                ctx.invoke(import_memory, import_file=import_memory, config=config, no_backup=False, verbose=verbose)
                return
        
        if memory_stats:
            with error_context(error_reporter, "retrieving memory statistics", raise_on_error=False):
                ctx = click.Context(memory_stats)
                ctx.invoke(memory_stats, config=config, verbose=verbose)
                return
        
        # Load configuration with error context
        with error_context(error_reporter, "loading configuration"):
            config_manager = ConfigManager(config)
            config_manager.ensure_user_config_dir()
            config_data = config_manager.load_config()
        
        # Handle list providers command
        if list_providers:
            with error_context(error_reporter, "checking provider availability", raise_on_error=False):
                detector = ProviderDetector()
                availability = detector.get_available_providers()
                
                click.echo("🔍 Provider Availability:")
                for provider_name, available in availability.items():
                    status = "✅ Available" if available else "❌ Not Available"
                    click.echo(f"   {provider_name}: {status}")
                
                # Show detected provider
                detected = detector.detect_provider()
                click.echo(f"\n🎯 Auto-detected provider: {detected}")
                return
        
        # Validate query is provided for normal operation
        if not query:
            click.echo("❌ No query provided. Use --help for usage information.", err=True)
            click.echo("\nQuick examples:")
            click.echo('  assistant "What can you help me with?"')
            click.echo('  assistant --list-providers')
            click.echo('  assistant --memory-stats')
            sys.exit(1)
        
        # Validate query
        query = validate_query(query)
        
        # Determine provider
        if provider and provider != 'auto':
            selected_provider = provider
        else:
            detector = ProviderDetector()
            selected_provider = detector.detect_provider()
        
        # Initialize and process query with AI agent
        with error_context(error_reporter, "processing your query"):
            from assistant.agent import create_agent, AgentError
            
            if verbose:
                click.echo(f"🚀 Initializing Personal Assistant Agent with {selected_provider} provider...")
            
            # Create and initialize agent
            agent = create_agent(
                config_path=config,
                provider_override=selected_provider,
                verbose=verbose
            )
            
            # Process query with memory integration
            response, metadata = agent.process_query(query)
            
            # Show processing information in verbose mode
            if verbose:
                click.echo(f"\n✅ Query processed in {metadata['processing_time']}s", err=True)
                click.echo(f"🔧 Provider: {metadata['provider']}", err=True)
                if metadata.get('memory_updated'):
                    click.echo("💾 Memory updated with new information", err=True)
                click.echo(f"🔄 Memory operations: {', '.join(metadata['memory_operations'])}", err=True)
            
            # Format and display response
            formatted_response = format_response(response, output_format)
            click.echo(formatted_response)
        
    except KeyboardInterrupt:
        click.echo("\n👋 Goodbye!", err=True)
        sys.exit(0)
    except Exception as e:
        # Use the comprehensive error reporting system
        user_message = error_reporter.report_error(
            e, 
            context={'cli_args': {'query': query, 'provider': provider, 'verbose': verbose}},
            user_action="using the assistant CLI"
        )
        click.echo(user_message, err=True)
        
        # Show error summary in verbose mode
        if verbose:
            error_summary = error_reporter.get_error_summary()
            if error_summary['total_errors'] > 1:
                click.echo(f"\n📊 Session Error Summary: {error_summary['total_errors']} errors occurred", err=True)
        
        sys.exit(1)

# Add memory commands as a subgroup (for future extension)
main.add_command(memory_commands, name="memory")

if __name__ == "__main__":
    main() 