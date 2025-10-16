# Setup and Installation Guide

This guide provides comprehensive instructions for setting up and running the real-time chat application with LangGraph and deep research capabilities.

## Prerequisites

### System Requirements
- **Python**: 3.9 or higher
- **PostgreSQL**: 12.x or higher
- **OpenAI Account**: API key for AI services
- **Docker**: Optional, for containerized database

### Hardware Requirements
- **RAM**: Minimum 4GB (8GB recommended)
- **Storage**: 2GB free space
- **Network**: Stable internet connection for OpenAI API calls

## Installation Steps

### 1. Clone the Repository
```bash
git clone <your-repository-url>
cd chat-application
```

### 2. Python Environment Setup

#### Using venv (Recommended)
```bash
# Create virtual environment
python -m venv venv

# Activate on Linux/macOS
source venv/bin/activate

# Activate on Windows
venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

#### Using conda
```bash
# Create conda environment
conda create -n chat-app python=3.9
conda activate chat-app

# Install dependencies
pip install -r requirements.txt
```

### 3. Database Setup

#### Option A: Docker PostgreSQL (Recommended)
```bash
# Start PostgreSQL container
docker-compose up -d

# Verify container is running
docker ps

# Check logs
docker-compose logs db
```

#### Option B: Local PostgreSQL Installation
1. Install PostgreSQL from [postgresql.org](https://www.postgresql.org/download/)
2. Create database and user:
```sql
CREATE DATABASE chatdb;
CREATE USER chat_user WITH PASSWORD 'your_password';
GRANT ALL PRIVILEGES ON DATABASE chatdb TO chat_user;
```

### 4. Environment Configuration

Copy the sample environment file:
```bash
cp .env.example .env
```

Edit `.env` with your configuration:
```bash
# OpenAI Configuration
OPENAI_API_KEY=sk-your-openai-api-key-here
OPENAI_WEBHOOK_SECRET=your-random-webhook-secret
OPENAI_RESPONSE_MODEL=gpt-4-1106-preview

# Database Configuration
DATABASE_URL=postgresql://chat_user:your_password@localhost:5432/chatdb
DB_MAX_CONNECTIONS=20
DB_POOL_TIMEOUT=30

# Application Settings
HOST=0.0.0.0
PORT=8000
DEBUG=false
LOG_LEVEL=INFO

# Optional: External Services
SEARCH_API_KEY=your-search-api-key  # For web search functionality

# Optional: Braintrust Telemetry
BRAINTRUST_API_KEY=bt-your-api-key
BRAINTRUST_PROJECT=BlackBox Playground
BRAINTRUST_ENABLED=true
```

### 5. Database Initialization

The application will automatically create required tables on first run. To verify database connection:

```bash
# Test database connection
python -c "
from app.chat_manager import ChatManager
cm = ChatManager()
print('Database connection successful' if cm.test_connection() else 'Connection failed')
"
```

### 6. Run the Application

#### Development Mode
```bash
# Start with auto-reload
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

#### Production Mode
```bash
# Start without reload
uvicorn app.main:app --host 0.0.0.0 --port 8000 --workers 4
```

### 7. Verify Installation

Access the application:
- **Web Interface**: http://localhost:8000/ui
- **API Documentation**: http://localhost:8000/docs
- **Health Check**: http://localhost:8000/health

## Configuration Details

### OpenAI API Key Setup

1. Sign up at [OpenAI Platform](https://platform.openai.com/)
2. Navigate to API Keys section
3. Create a new secret key
4. Copy the key to your `.env` file

### Database Connection Pooling

The application uses connection pooling with these default settings:
- **Max connections**: 20
- **Connection timeout**: 30 seconds
- **Auto-reconnect**: Enabled with exponential backoff

### Webhook Configuration for Deep Research

For deep research functionality, you need to configure OpenAI webhooks:

1. Set `OPENAI_WEBHOOK_SECRET` in your environment
2. The webhook endpoint is automatically exposed at `/webhook/openai`
3. OpenAI will call this endpoint with research results

### Braintrust Telemetry (Optional)

- Braintrust logging automatically activates when `BRAINTRUST_API_KEY` is set. Override the project name with `BRAINTRUST_PROJECT` or disable explicitly with `BRAINTRUST_ENABLED=false`.
- Install the standard dependencies (`braintrust`, `braintrust-langchain`) via `pip install -r requirements.txt`.
- Logged runs appear in the configured Braintrust project with LangChain traces.

## Quick Start Test

### Test the Chat Functionality

1. Open http://localhost:8000/ui in your browser
2. Create a new chat by entering:
   - User ID: `test-user`
   - Thread ID: `test-thread`
3. Send a message to test the system

### Test API Endpoints

```bash
# Test health endpoint
curl http://localhost:8000/health

# Test chat listing
curl http://localhost:8000/api/chats/test-user

# Test WebSocket connection (using wscat)
npm install -g wscat
wscat -c ws://localhost:8000/ws/test-user/test-thread
```

## Production Deployment

### Environment Setup for Production

1. Set appropriate environment variables:
```bash
export DEBUG=false
export LOG_LEVEL=WARNING
export DATABASE_URL=postgresql://prod_user:secure_password@prod-db:5432/chatdb_prod
```

2. Use process manager (recommended):
```bash
# Using PM2
npm install -g pm2
pm2 start "uvicorn app.main:app --host 0.0.0.0 --port 8000 --workers 4" --name chat-app

# Using systemd
sudo cp chat-app.service /etc/systemd/system/
sudo systemctl enable chat-app
sudo systemctl start chat-app
```

### Database Backup

Set up regular backups:
```bash
# Daily backup script
pg_dump -h localhost -U chat_user chatdb > backup-$(date +%Y%m%d).sql

# Automated backup with cron
0 2 * * * pg_dump -h localhost -U chat_user chatdb > /backups/chatdb-$(date +\%Y\%m\%d).sql
```

## Common Setup Issues

### Permission Errors
```bash
# Fix file permissions
chmod +x start.sh
sudo chown -R $USER:$USER .
```

### Port Conflicts
```bash
# Check if port 8000 is in use
lsof -i :8000

# Use different port
uvicorn app.main:app --host 0.0.0.0 --port 8001
```

### Database Connection Issues
```bash
# Test database connectivity
psql -h localhost -p 5432 -U chat_user -d chatdb

# Check PostgreSQL status
sudo systemctl status postgresql
```

### OpenAI API Issues
```bash
# Test OpenAI connection
curl -H "Authorization: Bearer $OPENAI_API_KEY" https://api.openai.com/v1/models
```

## Development Tools

### Useful Commands
```bash
# Run tests
python -m pytest tests/

# Check code style
flake8 app/

# Type checking
mypy app/

# Database migration (if needed)
alembic upgrade head
```

### Debug Mode
Enable debug mode for detailed logging:
```bash
export DEBUG=true
export LOG_LEVEL=DEBUG
uvicorn app.main:app --reload
```

## Monitoring Setup

### Log Monitoring
```bash
# View application logs
tail -f uvicorn.log

# Monitor database queries
sudo tail -f /var/log/postgresql/postgresql-*.log
```

### Health Monitoring
Set up monitoring for:
- Application uptime
- Database connection health
- OpenAI API rate limits
- Memory and CPU usage

## Security Considerations

### Environment Security
- Never commit `.env` files to version control
- Use strong database passwords
- Rotate API keys regularly
- Enable firewall rules for database access

### Network Security
- Use HTTPS in production
- Configure CORS appropriately
- Set up rate limiting
- Enable WebSocket security headers

---

**Next Steps**: After setup, configure your application settings and test the different conversation types as described in the usage documentation.
