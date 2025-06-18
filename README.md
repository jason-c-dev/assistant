# Personal Assistant CLI

A command-line personal assistant powered by AWS Strands Agent SDK with persistent memory capabilities through MCP (Model Context Protocol) integration. The assistant remembers your conversations, preferences, and context across sessions, making it increasingly useful over time.

## 🚀 30-Second Install

```bash
# 1. Setup virtual environment
python3 -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# 2. Install dependencies
pip install -r requirements.txt
pip install -e .

# 3. Setup memory server
git submodule update --init --recursive
cd memory_server && pip install -r requirements.txt && cd ..

# 4. Create config directory
mkdir -p ~/.assistant
cp config.yaml ~/.assistant/config.yaml

# 5. Set API key and test
export ANTHROPIC_API_KEY="your-key-here"  # or set in config
assistant "Hello, I'm testing the setup"
```

## ✨ Features

- **Persistent Memory**: Remembers your information, preferences, and past interactions
- **Multi-Provider Support**: Works with Anthropic, OpenAI, AWS Bedrock, and Ollama
- **Auto-Detection**: Automatically detects available AI providers based on your credentials
- **MCP Integration**: Extensible architecture supporting additional MCP servers
- **Zero Configuration**: Sensible defaults with optional customization
- **Memory Management**: Export, import, and reset memory functionality
- **Verbose Mode**: Debug memory operations and provider interactions

## 📋 Requirements

- Python 3.10 or higher
- One of the following AI provider credentials:
  - Anthropic API key (recommended)
  - OpenAI API key
  - AWS credentials for Bedrock
  - Ollama running locally (fallback)

## 🔧 Installation

### Option 1: Quick Install (Recommended)

Use the 30-second install guide above for the fastest setup.

### Option 2: Manual Setup

```bash
# Clone the repository
git clone <repository-url>
cd assistant

# Setup virtual environment
python3 -m venv .venv
source .venv/bin/activate

# Install Python dependencies
pip install -r requirements.txt

# Install the package in development mode
pip install -e .

# Initialize memory server submodule
git submodule update --init --recursive

# Install memory server dependencies
cd memory_server
pip install -r requirements.txt
cd ..

# Setup configuration
mkdir -p ~/.assistant
cp config.yaml ~/.assistant/config.yaml

# Configure your API keys (choose one)
export ANTHROPIC_API_KEY="sk-ant-your-key-here"
# OR
export OPENAI_API_KEY="sk-your-key-here"
# OR configure AWS credentials for Bedrock

# Test the installation
assistant "Hello world"
```

## 🎯 Quick Start

### Basic Usage

```bash
# Ask a question
assistant "What's the weather like today?"

# Share information that will be remembered
assistant "I'm working on a Python project called WebStore using React frontend"

# Ask about previously shared information
assistant "What technologies am I using for my WebStore project?"

# Get help
assistant --help
```

### Provider Selection

```bash
# Use specific provider
assistant --provider anthropic "Tell me about my recent projects"
assistant --provider openai "What did I discuss yesterday?"
assistant --provider ollama "Help me with Python coding"

# Use custom config file
assistant --config /path/to/config.yaml "Your question here"
```

### Memory Management

```bash
# Reset memory (with confirmation)
assistant --reset-memory

# Export memory to file
assistant --export-memory ~/my-memory-backup.json

# Import memory from file
assistant --import-memory ~/my-memory-backup.json

# Verbose mode (see memory operations)
assistant --verbose "Tell me about my goals"
```

## ⚙️ Configuration

The assistant uses a YAML configuration file located at `~/.assistant/config.yaml`:

```yaml
# Provider auto-detection priority: anthropic → openai → bedrock → ollama
default_provider: "auto"

# Model provider configurations
providers:
  anthropic:
    api_key: "${ANTHROPIC_API_KEY}"
    model_id: "claude-3-7-sonnet-20250219"
    temperature: 0.7
    max_tokens: 4096
  
  openai:
    api_key: "${OPENAI_API_KEY}"
    model_id: "gpt-4"
    temperature: 0.7
    max_tokens: 4096
  
  bedrock:
    region_name: "us-west-2"
    model_id: "us.anthropic.claude-3-7-sonnet-20250219-v1:0"
    temperature: 0.7
  
  ollama:
    base_url: "http://localhost:11434"
    model_id: "llama3.3"
    temperature: 0.7

# MCP server configurations
mcp_servers:
  memory:
    command: "./memory_server/.venv/bin/python"
    args: ["./memory_server/mcp_memory_server.py"]
    env:
      MEMORY_FILE_PATH: "~/.assistant/memory.json"
    required: true

# System prompt
system_prompt: |
  You are a helpful personal assistant with persistent memory capabilities.
  
  1. **User Identification**: Assume you're interacting with the default user.
  2. **Memory Retrieval**: Always begin by saying "Remembering..." and retrieve relevant information.
  3. **Memory Categories**: Pay attention to: Identity, Behaviors, Goals, Relationships, Events
  4. **Memory Updates**: After each interaction, update memory with new information.
  5. **Conversation Style**: Be conversational and reference past interactions.

# CLI settings
cli:
  verbose: false
  log_level: "INFO"
```

### Environment Variables

You can use environment variables in the config file:

```yaml
providers:
  anthropic:
    api_key: "${ANTHROPIC_API_KEY}"  # Required
  openai:
    api_key: "${OPENAI_API_KEY}"     # Required
```

Set them in your shell:

```bash
export ANTHROPIC_API_KEY="sk-ant-your-key-here"
export OPENAI_API_KEY="sk-your-openai-key"
export AWS_ACCESS_KEY_ID="your-aws-access-key"      # For Bedrock
export AWS_SECRET_ACCESS_KEY="your-aws-secret-key"  # For Bedrock
```

## 🧠 Memory System

The assistant uses a sophisticated memory system that categorizes and stores information:

### Memory Categories

- **Basic Identity**: Age, gender, location, job, education
- **Behaviors**: Interests, habits, preferences, working style
- **Goals**: Targets, aspirations, projects, deadlines
- **Relationships**: Personal and professional contacts
- **Important Events**: Meetings, milestones, deadlines

### Memory Operations

```bash
# The assistant automatically:
# 1. Retrieves relevant memories before responding
# 2. Updates memories after each interaction
# 3. Creates entities for people, organizations, events
# 4. Establishes relationships between entities

# Manual memory management:
assistant --export-memory backup.json    # Backup your memory
assistant --import-memory backup.json    # Restore from backup
assistant --reset-memory                 # Clear all memories
```

## 🔍 Troubleshooting

### Common Issues

**1. Command not found: `assistant`**
```bash
# Make sure you're in the virtual environment
source .venv/bin/activate
pip install -e .
```

**2. No provider available**
```bash
# Set at least one API key
export ANTHROPIC_API_KEY="your-key"
# OR install and run Ollama locally
```

**3. Memory server won't start**
```bash
# Check memory server setup
cd memory_server
pip install -r requirements.txt
python mcp_memory_server.py  # Test manually
```

**4. Permission denied errors**
```bash
# Fix memory server permissions
chmod +x memory_server/mcp_memory_server.py
```

### Verbose Mode

Use `--verbose` to see detailed operation logs:

```bash
assistant --verbose "What are my current projects?"
```

This shows:
- Provider detection and selection
- Memory retrieval operations
- Agent reasoning process
- Memory update operations

### Configuration Validation

```bash
# Test your configuration
assistant --config ~/.assistant/config.yaml "test"
```

## 🚀 Advanced Usage

### Multiple MCP Servers (Future)

The architecture supports additional MCP servers beyond memory:

```yaml
# Example configuration for additional servers
mcp_servers:
  memory:
    command: "./memory_server/.venv/bin/python"
    args: ["./memory_server/mcp_memory_server.py"]
    required: true
  
  filesystem:
    command: "npx"
    args: ["-y", "@modelcontextprotocol/server-filesystem", "/allowed/path"]
    required: false
  
  brave_search:
    command: "npx"
    args: ["-y", "@modelcontextprotocol/server-brave-search"]
    env:
      BRAVE_API_KEY: "${BRAVE_API_KEY}"
    required: false
```

### Custom System Prompts

Customize the assistant's behavior by editing the `system_prompt` in your config:

```yaml
system_prompt: |
  You are a specialized coding assistant with memory.
  Focus on software development topics and remember:
  - Programming languages the user works with
  - Current projects and their status
  - Coding preferences and patterns
  - Tools and frameworks they use
```

### Automation and Scripting

```bash
# Use in scripts
response=$(assistant "What's my next deadline?")
echo "Next deadline: $response"

# Batch processing
echo "What did I work on today?" | assistant
```

## 🧪 Development

### Running Tests

```bash
# Install test dependencies
pip install pytest pytest-asyncio

# Run all tests
pytest

# Run specific test files
pytest tests/test_config.py
pytest tests/test_cli.py
pytest tests/test_providers.py
pytest tests/test_agent.py

# Run with coverage
pytest --cov=assistant
```

### Project Structure

```
assistant/
├── assistant/              # Main package
│   ├── __init__.py        # Package initialization
│   ├── cli.py             # Command-line interface
│   ├── config.py          # Configuration management
│   ├── agent.py           # Strands agent integration
│   ├── providers.py       # AI provider detection/setup
│   ├── memory_client.py   # MCP memory client
│   ├── network_utils.py   # Network error handling
│   └── error_reporting.py # Error reporting system
├── memory_server/         # Git submodule
│   └── mcp_memory_server.py
├── tests/                 # Unit tests
├── config.yaml           # Default configuration
├── requirements.txt      # Dependencies
├── setup.py              # Package setup
└── README.md             # This file
```

## 📚 API Reference

### Command Line Interface

```bash
assistant [OPTIONS] QUERY

Options:
  --config, -c PATH        Configuration file path
  --provider, -p PROVIDER  Provider override (anthropic|openai|bedrock|ollama)
  --verbose, -v           Enable verbose output
  --reset-memory          Reset all stored memory
  --export-memory PATH    Export memory to JSON file
  --import-memory PATH    Import memory from JSON file
  --help                  Show help message
  --version               Show version
```

### Configuration Schema

See the `config.yaml` file for the complete configuration schema with examples.

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- [AWS Strands Agent SDK](https://github.com/awslabs/strands) for the AI agent framework
- [Model Context Protocol](https://modelcontextprotocol.io/) for the extensible server architecture
- [Memory MCP Server](https://github.com/jason-c-dev/memory-mcp-server-py) for persistent memory capabilities

## 📞 Support

- 📧 Open an issue on GitHub for bug reports
- 💬 Join discussions for questions and feature requests
- 📖 Check the troubleshooting section above for common issues

---

**Ready to get started?** Use the 30-second install guide at the top of this README! 🚀 