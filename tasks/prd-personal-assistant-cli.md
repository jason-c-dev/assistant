# Personal Assistant CLI - Product Requirements Document

## Introduction/Overview

The Personal Assistant CLI is a command-line tool that provides users with an intelligent, memory-enabled assistant for both personal and professional use. The assistant integrates AWS Strands Agent SDK with a Python MCP (Model Context Protocol) memory server to deliver persistent memory capabilities across conversations, allowing it to remember user information, preferences, and context over time.

The core problem this solves is the lack of persistent memory in typical AI interactions - users have to re-introduce themselves and re-establish context in every conversation. This assistant learns about users over time and can reference past interactions, making it increasingly useful and personalized.

## Goals

1. **Enable Persistent Memory**: Create an AI assistant that remembers user information, preferences, and past interactions across sessions
2. **Provide Multi-Provider Flexibility**: Support multiple AI model providers (Anthropic, OpenAI, AWS Bedrock, Ollama) with automatic detection
3. **Ensure Easy Setup**: Deliver a zero-configuration experience with sensible defaults and 30-second installation
4. **Maintain Extensibility**: Build an architecture that supports additional MCP servers beyond memory (filesystem, web browsing, etc.)
5. **Optimize Performance**: Achieve fast response times with efficient memory retrieval and model interactions

## User Stories

### Primary Users: Professionals and Personal Users

**As a knowledge worker**, I want to:
- Tell my assistant about my current projects so it can help me track progress and deadlines
- Ask about my goals and have the assistant remind me of what I've shared previously
- Get personalized recommendations based on my stated preferences and past interactions
- Reference past conversations without having to repeat context

**As a project manager**, I want to:
- Store information about team members and their roles so the assistant can help with people-related questions
- Track project milestones and have the assistant remind me of important dates
- Get help organizing my thoughts and plans based on previous discussions

**As a researcher**, I want to:
- Build a knowledge base of my research interests and findings over time
- Ask the assistant to connect new information with previously stored research
- Maintain context about ongoing research projects across multiple sessions

**As a developer enhancing the assistant**, I want to:
- Easily add new MCP servers for additional functionality
- Configure different model providers based on my available API keys
- Debug memory operations when needed

## Functional Requirements

### Core Functionality
1. The system must accept natural language queries via command-line interface
2. The system must maintain persistent memory of user interactions using MCP memory server
3. The system must automatically detect and use available AI model providers (Anthropic, OpenAI, AWS Bedrock, Ollama)
4. The system must retrieve relevant memory before responding to user queries
5. The system must update memory after each interaction with new information
6. The system must support single-command execution (non-interactive mode)

### Memory Management
7. The system must categorize and store information about: Basic Identity, Behaviors/Preferences, Goals, Relationships, Important Events
8. The system must create entities for recurring people, organizations, and significant events
9. The system must establish connections between stored entities with appropriate relations
10. The system must export memory data to JSON format when requested
11. The system must import memory data from JSON format when requested
12. The system must provide memory reset functionality with user confirmation

### Configuration & Setup
13. The system must use a YAML configuration file with provider settings and MCP server configurations
14. The system must auto-detect available providers based on environment variables
15. The system must support configuration file override via command-line arguments
16. The system must create default configuration directory (`~/.assistant/`) automatically
17. The system must expand user home directory paths (`~`) in configuration

### Command-Line Interface
18. The system must accept queries as command arguments (e.g., `assistant "question"`)
19. The system must support help and version information display
20. The system must provide verbose mode showing memory operations
21. The system must support provider override via `--provider` flag
22. The system must support custom configuration file via `--config` flag

### Error Handling
23. The system must gracefully handle missing API keys with clear error messages
24. The system must handle MCP memory server startup failures with appropriate fallbacks
25. The system must manage network connectivity issues with retry logic
26. The system must detect and handle corrupted memory storage
27. The system must validate configuration file format and provide helpful error messages

### Performance
28. The system must start up within 5 seconds in normal conditions
29. The system must respond to queries within 10 seconds for typical interactions
30. The system must handle memory retrieval efficiently for large datasets

## Non-Goals (Out of Scope)

### Version 1.0 Exclusions
- **Interactive/Chat Mode**: No real-time conversation interface - single command execution only
- **Web Interface**: No browser-based UI or web server functionality
- **Advanced MCP Servers**: No implementation of filesystem, web browsing, or other MCP servers (examples only)
- **Multi-User Support**: No user authentication or multi-tenant capabilities
- **Cloud Storage**: No cloud-based memory storage - local file system only
- **Mobile Support**: No mobile app or responsive interface
- **Voice Interface**: No speech-to-text or text-to-speech capabilities
- **Plugin Marketplace**: No dynamic plugin installation or marketplace
- **Advanced Analytics**: No usage analytics or telemetry collection
- **Enterprise Features**: No SSO, audit logging, or enterprise administration
- **Real-time Notifications**: No background processes or notification systems

## Design Considerations

### Configuration Architecture
- YAML-based configuration with environment variable interpolation
- Default configuration template provided in project root
- User configuration stored in `~/.assistant/config.yaml`
- Support for multiple provider configurations with auto-detection

### Memory Integration UX
- Always begin responses with "Remembering..." to indicate memory retrieval
- Proactive user identification if not previously established
- Conversational style that references past interactions naturally
- Clear indication when memory operations occur in verbose mode

### CLI Interaction Pattern
```bash
# Basic usage
assistant "What meetings do I have this week?"

# Configuration examples
assistant --provider anthropic "What did I discuss with John last week?"
assistant --verbose "Update my skills to include Python and FastAPI"
assistant --config /custom/path/config.yaml "Tell me about my goals"
```

## Technical Considerations

### Technology Stack Dependencies
- **AWS Strands Agent SDK**: Primary framework for AI agent functionality
- **Python MCP Memory Server**: From https://github.com/jason-c-dev/memory-mcp-server-py
- **Python 3.10+**: Minimum version requirement
- **Model Context Protocol**: Integration via Strands built-in MCP support

### Architecture Constraints
- Use Strands built-in model providers (no custom provider implementations)
- Follow Strands MCP integration patterns with `MCPClient` and stdio transport
- Leverage context managers for proper MCP server lifecycle management
- Use stdio-based MCP servers running in dedicated threads

### Integration Examples
The system should provide configuration examples for additional MCP servers from the [official MCP servers repository](https://github.com/modelcontextprotocol/servers/tree/main):

```yaml
# Example additional MCP servers (not implemented in v1.0)
mcp_servers:
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

### Security Considerations
- API keys stored in environment variables or configuration files
- Memory data stored locally in user's home directory
- No network exposure of memory server (stdio transport only)
- Configuration validation to prevent injection attacks

## Success Metrics

### Memory Accuracy
- **Memory Recall Rate**: Percentage of previously stored information correctly retrieved when relevant
- **Memory Precision**: Accuracy of stored information compared to original user input
- **Context Relevance**: How well the assistant retrieves appropriate memory for given queries

### Performance Metrics
- **Startup Time**: Time from command execution to first response (target: <5 seconds)
- **Response Time**: Total time for query processing including memory retrieval (target: <10 seconds)
- **Memory Operation Speed**: Time for memory store/retrieve operations (target: <2 seconds)

### User Experience
- **Setup Success Rate**: Percentage of users who complete installation within 2 minutes
- **Error Recovery**: Percentage of common errors that are resolved with clear error messages
- **Feature Discovery**: How easily users discover and use advanced features (verbose mode, provider switching)

## Open Questions

1. **Memory Storage Limits**: Should there be limits on memory storage size, and if so, what cleanup strategies should be implemented?

2. **Offline Mode**: How should the assistant behave when no network connectivity is available (for local models like Ollama)?

3. **Memory Sharing**: Future consideration for sharing memory data between different installations or users?

4. **Model Selection Strategy**: When multiple providers are available, what logic should determine the default selection order?

5. **Configuration Migration**: How should configuration file format changes be handled in future versions?

6. **Logging Strategy**: What level of logging should be implemented for debugging and monitoring?

7. **Update Mechanism**: How should users be notified of and install updates to the assistant or memory server?

8. **Memory Backup**: Should automatic memory backup strategies be implemented to prevent data loss? 