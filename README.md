# Real-Time Chat Application with LangGraph and Deep Research

A sophisticated real-time chat application built with FastAPI, WebSockets, PostgreSQL, and LangGraph that supports multi-tenant chat management, deep research capabilities, OpenAI integration, and comprehensive conversation state management.

## 🎯 Features

### Core Features
- **Real-time Chat**: WebSocket-based bidirectional communication
- **Multi-tenant Support**: User and thread-based conversation management
- **Conversation Types**: Standard chat, web search, deep research, document Q&A
- **State Management**: LangGraph-powered conversation orchestration
- **Summarization**: Automatic conversation summarization for long histories
- **Token Tracking**: Comprehensive OpenAI token usage monitoring

### Advanced Capabilities
- **Deep Research**: Background research execution with OpenAI's response API
- **Web Integration**: Built-in web search functionality
- **Multi-language Support**: English, Spanish, French, German, Chinese
- **Connection Pooling**: Efficient PostgreSQL connection management
- **Error Handling**: Robust error handling and graceful degradation
- **Webhook Processing**: OpenAI deep research result handling

## 🏗️ Architecture

### System Architecture

```mermaid
graph TB
    %% Frontend Layer
    subgraph Frontend
        UI[Web Interface - simple_chat.html]
        WS_FE[WebSocket Client]
    end

    %% API Gateway Layer
    subgraph API_Gateway[FastAPI Application - main.py]
        REST_API[REST Endpoints]
        WS_API[WebSocket Endpoints]
        Chat_Mgr[Chat Manager - chat_manager.py]
        WS_Mgr[WebSocket Manager - websocket_manager.py]
    end

    %% Core Services Layer
    subgraph Core_Services
        Chat_Service[Chat Service - utils.py]
        Graph_Builder[Graph Builder - graph_builder.py]
        Summary_Agent[Summary Agent - summary_agent.py]
    end

    %% Conversation Processing Layer
    subgraph Conversation_Processing
        Standard_Nodes[Nodes - nodes.py]
        Research_Nodes[Deep Research Nodes - deep_research.py]
        Graph[LangGraph State Machine]
    end

    %% Data Layer
    subgraph Data_Layer
        PostgresDB[PostgreSQL Database]
        JSON_Store[Data Store - store.py/data.json]
    end

    %% External Services
    subgraph External_Services
        OpenAI_API[OpenAI API & Webhooks]
    end

    %% Connections
    UI --> WS_FE
    WS_FE --> WS_API
    UI --> REST_API
    
    REST_API --> Chat_Mgr
    WS_API --> WS_Mgr
    
    Chat_Mgr --> PostgresDB
    Chat_Mgr --> Chat_Service
    
    Chat_Service --> Graph_Builder
    Chat_Service --> Summary_Agent
    
    Graph_Builder --> Standard_Nodes
    Graph_Builder --> Research_Nodes
    
    Standard_Nodes --> Graph
    Research_Nodes --> Graph
    
    Graph --> OpenAI_API
    Research_Nodes --> JSON_Store
    JSON_Store --> JSON_Store
    
    OpenAI_API --> WS_API[Webhook Callbacks]
```

### Conversation Flow

```mermaid
flowchart TD
    Start[User Message] --> RouteType{Conversation Type}
    
    RouteType -->|Standard Chat| SummarizeCheck{Check Summarization}
    RouteType -->|Deep Research| Planning[Research Planning Node]
    
    SummarizeCheck -->|Summarize Needed| Summarize[Summarization Node]
    SummarizeCheck -->|No Summarize| Chat[Chat Node]
    
    Summarize --> Chat
    Chat --> End[Response]
    
    Planning --> ClarifyCheck{Clarification Needed?}
    ClarifyCheck -->|Yes| WaitUser[Wait for User Input]
    ClarifyCheck -->|No| Generate[Generate Research Plan]
    
    WaitUser --> Generate
    Generate --> Execute[Execute Research]
    Execute --> Complete[Complete Research]
```

## 📦 Project Structure

```
app/
├── __init__.py                 # Package initialization
├── main.py                     # FastAPI application and endpoints
├── config.py                   # Environment configuration and model settings
├── chat_manager.py             # PostgreSQL-based chat management
├── websocket_manager.py        # WebSocket connection management
├── utils.py                    # Core ChatService orchestrator
├── schema.py                   # Conversation state definitions
├── nodes.py                    # Standard conversation nodes
├── deep_research.py           # Deep research nodes and utilities
├── graph_builder.py           # Conversation graph building
├── helper.py                   # Utility functions and error handling
├── prompt.py                   # AI prompt templates
├── summary_agent.py           # Conversation summarization
├── store.py                    # JSON data store for research tracking
└── data.json                   # Research data storage

ui/
└── simple_chat.html           # Web interface

docker-compose.yml             # PostgreSQL container configuration
requirements.txt               # Python dependencies
README.md                      # This documentation
notes.md                       # Development notes
```

## 🔧 Key Components

### 1. FastAPI Application (`main.py`)
- REST API endpoints for chat management
- WebSocket endpoints for real-time communication
- OpenAI webhook processing for deep research results
- Multi-tenant support with user/thread isolation

### 2. PostgreSQL Chat Manager (`chat_manager.py`)
- Connection pooling with retry logic
- Thread-based conversation storage
- User management and chat history
- Atomic operations with transaction support

### 3. WebSocket Manager (`websocket_manager.py`)
- Real-time message broadcasting
- Connection state management
- Error handling and reconnection logic

### 4. LangGraph Integration (`utils.py`, `graph_builder.py`)
- State machine conversation orchestration
- Multiple conversation type support
- Automatic summarization triggers
- Token usage tracking

### 5. Deep Research Module (`deep_research.py`)
- Research planning and clarification
- Background research execution
- OpenAI response API integration
- Research status tracking and monitoring

### 6. Frontend Interface (`ui/simple_chat.html`)
- Real-time messaging interface
- Research progress monitoring
- Multi-conversation management
- Responsive design with status indicators

## 🚀 Quick Start

### Prerequisites
- Python 3.9+
- PostgreSQL 12+
- OpenAI API key
- Docker (for containerized PostgreSQL)

### Installation
```bash
# Clone and setup
git clone <repository>
cd <project-directory>
pip install -r requirements.txt

# Environment configuration
cp .env.example .env
# Edit .env with your OpenAI API key and database settings

# Start PostgreSQL with Docker
docker-compose up -d

# Run the application
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

### Access the Application
- Web Interface: http://localhost:8000/ui
- API Documentation: http://localhost:8000/docs
- WebSocket Endpoint: ws://localhost:8000/ws/{user_id}/{thread_id}

## ⚙️ Configuration

### Environment Variables
```bash
# OpenAI Configuration
OPENAI_API_KEY=your_openai_api_key_here
OPENAI_WEBHOOK_SECRET=your_webhook_secret
OPENAI_RESPONSE_MODEL=gpt-4

# Database Configuration
DATABASE_URL=postgresql://user:password@localhost:5432/chatdb
DB_MAX_CONNECTIONS=20
DB_POOL_TIMEOUT=30

# Application Settings
HOST=0.0.0.0
PORT=8000
DEBUG=true
```

### Model Configuration
```python
# Default model settings in config.py
DEFAULT_MODELS = {
    "chat": "gpt-4-1106-preview",
    "summarize": "gpt-3.5-turbo",
    "research_plain": "gpt-4-1106-preview"
}
```

## 📋 API Documentation

### REST Endpoints
- `GET /api/chats/{user_id}` - List user chats
- `POST /api/chats/{user_id}` - Create new chat
- `PUT /api/chats/{user_id}/{thread_id}` - Update chat title
- `DELETE /api/chats/{user_id}/{thread_id}` - Delete chat
- `GET /conversation/{user_id}/{thread_id}` - Get conversation messages
- `GET /api/research/{thread_id}` - Get research status

### WebSocket Protocol
- Endpoint: `ws://localhost:8000/ws/{user_id}/{thread_id}`
- Message format: JSON with `type`, `message`, `thread_id`, `language`
- Response types: `start`, `chunk`, `complete`, `error`, `research_initiated`, `webhook_result`

## 🔍 Deep Research Workflow

### Research Process
1. **Planning**: Ask clarifying questions to scope research
2. **Prompt Generation**: Create detailed research instructions
3. **Execution**: Submit to OpenAI's response API
4. **Background Processing**: Monitor research progress
5. **Completion**: Process results via webhook

### Research Status Tracking
- Real-time progress monitoring
- Multi-step progress indication
- Error handling and retry logic
- Results storage in JSON database

## 🛠️ Development

### Adding New Conversation Types
1. Extend `graph_builder.py` with new graph configuration
2. Add nodes in appropriate module (`nodes.py` or new file)
3. Update conversation type routing in `utils.py`
4. Add prompt templates in `prompt.py`

### Customizing Prompts
Edit `app/prompt.py` to modify:
- Chat conversation style
- Summarization format
- Research planning questions
- Research instruction generation

### Monitoring and Debugging
- Built-in token tracking
- Conversation state inspection
- WebSocket connection status
- Research progress monitoring

## 🚨 Troubleshooting

### Common Issues
- **Database connection errors**: Check PostgreSQL is running and DATABASE_URL is correct
- **OpenAI API errors**: Verify OPENAI_API_KEY is set and valid
- **WebSocket connection issues**: Check CORS settings and network connectivity
- **Research not starting**: Verify OPENAI_WEBHOOK_SECRET is configured

### Debug Mode
Enable debug mode for detailed logging:
```bash
export DEBUG=true
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

## 📊 Monitoring

### Built-in Monitoring
- Token usage tracking per conversation
- Database connection health
- WebSocket connection status
- Research task monitoring

### Logging
Application logs include:
- API request/response details
- WebSocket connection events
- OpenAI API interactions
- Database operations
- Research task status updates

## 🤝 Contributing

### Development Setup
1. Fork the repository
2. Create feature branch: `git checkout -b feature/amazing-feature`
3. Commit changes: `git commit -m 'Add amazing feature'`
4. Push to branch: `git push origin feature/amazing-feature`
5. Open a Pull Request

### Code Style
- Follow PEP 8 guidelines
- Use type hints where appropriate
- Include docstrings for public methods
- Write tests for new functionality

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- FastAPI team for the excellent web framework
- LangChain/LangGraph for conversation orchestration
- OpenAI for AI model integration
- PostgreSQL for reliable data storage

---

**Note**: This application is designed for production use with proper security measures. Always secure your API keys and database credentials in production environments.