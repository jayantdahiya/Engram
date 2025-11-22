#!/bin/bash

# Engram Startup Script

set -e

echo "🚀 Starting Engram..."

# Check if Docker is running
if ! docker info > /dev/null 2>&1; then
    echo "❌ Docker is not running. Please start Docker and try again."
    exit 1
fi

# Check if .env file exists
if [ ! -f .env ]; then
    echo "⚠️  .env file not found. Creating from template..."
    cp env.example .env
    echo "📝 Please edit .env file with your configuration before continuing."
    echo "   Required: OLLAMA_BASE_URL, SECRET_KEY"
    exit 1
fi

# Load environment variables
source .env

# Check required environment variables
if [ -z "$OLLAMA_BASE_URL" ]; then
    echo "❌ OLLAMA_BASE_URL is not set in .env file"
    exit 1
fi

if [ -z "$SECRET_KEY" ] || [ "$SECRET_KEY" = "your-secret-key-change-in-production" ]; then
    echo "❌ SECRET_KEY is not set or is using default value in .env file"
    exit 1
fi

# Check Ollama connectivity
echo "🔗 Checking Ollama connectivity..."
if curl -f "${OLLAMA_BASE_URL}/api/tags" > /dev/null 2>&1; then
    echo "✅ Ollama is accessible at ${OLLAMA_BASE_URL}"
else
    echo "❌ Ollama is not accessible at ${OLLAMA_BASE_URL}"
    echo "   Please ensure Ollama is running and the models are available:"
    echo "   • gemma3:270m"
    echo "   • nomic-embed-text:latest"
    exit 1
fi

echo "✅ Environment configuration validated"

# Create necessary directories
mkdir -p logs
mkdir -p infrastructure/docker/init-scripts

# Start services with Docker Compose
echo "🐳 Starting services with Docker Compose..."
docker-compose -f infrastructure/docker/docker-compose.yml up -d

# Wait for services to be ready
echo "⏳ Waiting for services to be ready..."
sleep 30

# Check service health
echo "🔍 Checking service health..."

# Check API health
if curl -f http://localhost:8000/health/ > /dev/null 2>&1; then
    echo "✅ API service is healthy"
else
    echo "❌ API service is not responding"
    echo "📋 Checking API logs..."
    docker-compose -f infrastructure/docker/docker-compose.yml logs api
    exit 1
fi

# Check PostgreSQL
if docker-compose -f infrastructure/docker/docker-compose.yml exec -T postgres pg_isready -U engram_user -d engram_db > /dev/null 2>&1; then
    echo "✅ PostgreSQL is ready"
else
    echo "❌ PostgreSQL is not ready"
    exit 1
fi

# Check Redis
if docker-compose -f infrastructure/docker/docker-compose.yml exec -T redis redis-cli ping > /dev/null 2>&1; then
    echo "✅ Redis is ready"
else
    echo "❌ Redis is not ready"
    exit 1
fi

# Check Neo4j
if curl -f http://localhost:7474 > /dev/null 2>&1; then
    echo "✅ Neo4j is ready"
else
    echo "❌ Neo4j is not ready"
    exit 1
fi

echo ""
echo "🎉 Engram is now running!"
echo ""
echo "📋 Service URLs:"
echo "   • API Documentation: http://localhost:8000/docs"
echo "   • API Health Check: http://localhost:8000/health/"
echo "   • Flower (Celery): http://localhost:5555"
echo "   • Grafana (Metrics): http://localhost:3000 (admin/admin)"
echo "   • Prometheus: http://localhost:9090"
echo "   • Neo4j Browser: http://localhost:7474 (neo4j/secure_password)"
echo "   • Ollama: ${OLLAMA_BASE_URL}"
echo ""
echo "🤖 AI Models:"
echo "   • LLM Model: ${OLLAMA_LLM_MODEL:-gemma3:270m}"
echo "   • Embedding Model: ${OLLAMA_EMBEDDING_MODEL:-nomic-embed-text:latest}"
echo ""
echo "🔧 Management Commands:"
echo "   • View logs: docker-compose -f infrastructure/docker/docker-compose.yml logs -f"
echo "   • Stop services: docker-compose -f infrastructure/docker/docker-compose.yml down"
echo "   • Restart services: docker-compose -f infrastructure/docker/docker-compose.yml restart"
echo ""
echo "📚 Next Steps:"
echo "   1. Visit http://localhost:8000/docs to explore the API"
echo "   2. Register a user account"
echo "   3. Start processing conversations and memories"
echo "   4. Monitor performance in Grafana"
echo ""
echo "⚠️  Make sure Ollama is running with the required models:"
echo "   ollama pull gemma3:270m"
echo "   ollama pull nomic-embed-text:latest"
echo ""
echo "Happy coding! 🚀"
