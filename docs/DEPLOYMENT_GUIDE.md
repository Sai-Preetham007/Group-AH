# Medical Knowledge RAG Chatbot - Deployment Guide

## Prerequisites
- Python 3.8+
- Virtual environment
- Required dependencies (see requirements.txt)

## Local Development Setup

### 1. Clone and Setup
```bash
git clone <repository-url>
cd Group-AH
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Environment Configuration
Create a `.env` file in the project root:
```bash
# API Configuration
OPENAI_API_KEY=your_openai_api_key_here
API_HOST=0.0.0.0
API_PORT=8000
API_WORKERS=1

# Database Configuration
VECTOR_DB_PATH=./data/vector_db
CHROMA_DB_PATH=./data/chroma_db

# Logging
LOG_LEVEL=INFO
LOG_FILE=./logs/app.log

# Model Configuration
EMBEDDING_MODEL=sentence-transformers/all-MiniLM-L6-v2
LLM_MODEL=gpt-3.5-turbo
```

### 3. Data Setup
```bash
# Setup initial data
python setup_data.py

# Setup Kaggle dataset (if available)
python setup_kaggle_dataset.py
```

### 4. Run the Server
```bash
# Development mode
python src/api/main.py

# Or using the run script
python run_server.py
```

## Production Deployment

### Using Docker (Recommended)
```dockerfile
FROM python:3.9-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .
EXPOSE 8000

CMD ["python", "src/api/main.py"]
```

### Using Gunicorn
```bash
pip install gunicorn
gunicorn src.api.main:app -w 4 -k uvicorn.workers.UvicornWorker --bind 0.0.0.0:8000
```

### Environment Variables for Production
```bash
export OPENAI_API_KEY="your_production_key"
export API_HOST="0.0.0.0"
export API_PORT="8000"
export LOG_LEVEL="WARNING"
```

## Monitoring and Logging
- Logs are stored in `./logs/` directory
- Health checks available at `/health`
- Metrics can be added using Prometheus integration

## Security Considerations
- Use HTTPS in production
- Implement API key authentication
- Set up rate limiting
- Validate all inputs
- Use environment variables for secrets

## Scaling
- Use load balancer for multiple instances
- Consider Redis for session management
- Use external vector database for large datasets
- Implement caching for frequent queries
