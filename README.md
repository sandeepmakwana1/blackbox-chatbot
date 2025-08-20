
## Features

- ✅ **Real-time Chat** via WebSocket
- ✅ **Conversation History** with PostgreSQL persistence
- ✅ **Auto-renaming** chats based on first message
- ✅ **Token Tracking** for OpenAI usage
- ✅ **Summarization** of long conversations
- ✅ **Multi-language Support**
- ✅ **Clean Architecture** with proper separation of concerns


## Quick Start

### 1. Environment Setup

Create a `.env` file:

```bash
# Application
APP_NAME="Senior Chat App"
APP_VERSION="1.0.0"
DEBUG=false

# Server
HOST=0.0.0.0
PORT=8000

# Database
POSTGRES_USER=postgres
POSTGRES_PASSWORD=your_password
POSTGRES_HOST=localhost
POSTGRES_PORT=5432
POSTGRES_DB=chat_app

# OpenAI
OPENAI_API_KEY=your_openai_api_key

# Models
CHAT_MODEL=gpt-4o-mini
SUMMARIZE_MODEL=gpt-4o-mini
EMBEDDINGS_MODEL=text-embedding-3-small

# Settings
SUMMARY_TRIGGER_COUNT=5
MAX_TITLE_LENGTH=50

# CORS
CORS_ORIGINS=["http://localhost:3000", "http://localhost:8080"]
```

### 2. Database Setup

Start PostgreSQL with Docker:

```bash
docker-compose up -d
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Run the Application

```bash
# Development mode
uvicorn src.main:app --reload --host 0.0.0.0 --port 8000

# Production mode
uvicorn src.main:app --host 0.0.0.0 --port 8000
```

## API Documentation

### REST Endpoints

- **GET** `/api/chats/{user_id}` - Get user's chats
- **POST** `/api/chats/{user_id}` - Create new chat
- **PUT** `/api/chats/{user_id}/{thread_id}/rename` - Rename chat
- **DELETE** `/api/chats/{user_id}/{thread_id}` - Delete chat
- **GET** `/api/chats/{user_id}/{thread_id}` - Get chat details

### WebSocket Endpoints

- **WebSocket** `/ws/chat` - Real-time chat endpoint
- **WebSocket** `/ws/health` - Health check endpoint

### Health Checks

- **GET** `/api/health` - General health check
- **GET** `/api/health/ready` - Readiness probe
- **GET** `/api/health/live` - Liveness probe

## Development

### Code Quality

```bash
# Format code
black src/

# Sort imports
isort src/

# Lint
flake8 src/

# Type checking
mypy src/

# Run tests
pytest tests/
```

### Testing

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=src tests/

# Run specific test file
pytest tests/unit/test_chat_service.py
```

### Database Migrations

```bash
# Create migration
alembic revision --autogenerate -m "description"

# Apply migrations
alembic upgrade head

# Downgrade migration
alembic downgrade -1
```

## Architecture Decisions

### 1. Layered Architecture
- **API Layer**: REST endpoints and WebSocket handlers
- **Service Layer**: Business logic and orchestration
- **Repository Layer**: Data access and persistence
- **Domain Layer**: Core business models and rules

### 2. Dependency Injection
- Services are injected into routes
- Repositories are injected into services
- Easy testing with mock implementations

### 3. Async/Await
- All I/O operations are async
- PostgreSQL with asyncpg
- WebSocket handling with asyncio

### 4. Error Handling
- Centralized error handling middleware
- Consistent error response format
- Proper logging at all levels

### 5. Configuration Management
- Environment-based configuration
- Type-safe settings with Pydantic
- Easy deployment across environments

## Production Considerations

### Environment Variables

```bash
# Production database
POSTGRES_HOST=your-prod-db-host
POSTGRES_PASSWORD=your-secure-password

# Production OpenAI
OPENAI_API_KEY=your-production-key

# Security
DEBUG=false
CORS_ORIGINS=["https://yourdomain.com"]
```

### Docker Deployment

```dockerfile
FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY src/ ./src/

CMD ["uvicorn", "src.main:app", "--host", "0.0.0.0", "--port", "8000"]
```

### Monitoring

- Health check endpoints for load balancers
- Prometheus metrics (optional)
- Structured logging with correlation IDs

## Contributing

1. Follow the established architecture patterns
2. Write tests for new features
3. Update documentation
4. Use the provided code quality tools
5. Follow semantic versioning

## License

MIT License - see LICENSE file for details.