# Engram
Persistent long‑term memory for AI agents with dense facts and graph relations.

## Name meaning
Engram is the classical term for a memory "trace": the enduring physical and/or chemical changes in neural circuitry produced by learning, which can later be reactivated to support recall. The concept originates with Richard Semon and is widely used in modern neuroscience to describe the substrate of stored experience. [3][4][5][6][7]

## Why "Engram" for this project?
It reflects durable storage, precise retrieval, and structured consolidation across sessions—exactly what long‑term agent memory systems aim to provide. [4][5]

## 🚀 Features

- **Advanced Memory Management**: Implements Engram architecture with ADD/UPDATE/DELETE/NOOP operations
- **ACAN Retrieval System**: Attention-based Context-Aware Network for intelligent memory retrieval
- **Graph Memory (Engram Graph)**: Neo4j-based entity and relationship storage
- **Production-Ready**: FastAPI, Docker, monitoring, and comprehensive testing
- **Async Processing**: Celery-based background task processing
- **Vector Search**: PostgreSQL with pgvector for efficient similarity search
- **Authentication**: JWT-based user authentication and authorization
- **Monitoring**: Prometheus metrics and Grafana dashboards

## 🏗️ Architecture

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   API Gateway   │────│  FastAPI Core   │────│ Memory Manager  │
│    (Nginx)      │    │   Service       │    │    Service      │
└─────────────────┘    └─────────────────┘    └─────────────────┘
                                │                        │
                                │                        │
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Auth Service  │────│  Celery Workers │────│Vector Database  │
│   (JWT + Redis) │    │ (Memory Tasks)  │    │  (PostgreSQL)   │
└─────────────────┘    └─────────────────┘    └─────────────────┘
                                │                        │
                                │                        │
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│  Graph Database │────│ Embedding API   │────│  Monitoring     │
│    (Neo4j)      │    │   Service       │    │(Prometheus/Graf)│
└─────────────────┘    └─────────────────┘    └─────────────────┘
```

## 🛠️ Tech Stack

### Core Framework
- **Python 3.11+** - Latest stable version with excellent async support
- **FastAPI** - High-performance, auto-documented API framework
- **Pydantic v2** - Data validation and serialization
- **AsyncIO** - For handling concurrent operations

### Database Stack
- **PostgreSQL 16+** with **pgvector 0.7.0** - Vector storage and similarity search
- **Neo4j 5.x** - Graph database for Engram Graph relationships
- **Redis 7.x** - Caching, session storage, and Celery message broker

### AI & ML Components
- **OpenAI API** - GPT-4o-mini for LLM operations
- **sentence-transformers** - Local embedding generation
- **tiktoken** - Token counting for cost management

### Infrastructure
- **Docker & Docker Compose** - Containerization
- **Nginx** - Reverse proxy and load balancing
- **Celery** - Distributed task queue
- **Prometheus + Grafana** - Metrics collection and visualization

## 🚀 Quick Start

### Prerequisites

- Docker and Docker Compose
- OpenAI API key
- At least 8GB RAM and 4 CPU cores

### 1. Clone and Setup

```bash
git clone <repository-url>
cd engram
```

### 2. Environment Configuration

```bash
cp env.example .env
```

Edit `.env` with your configuration:

```env
# Provider Configuration
LLM_PROVIDER=openai
EMBEDDING_PROVIDER=openai

# OpenAI Configuration (required when using openai providers)
OPENAI_API_KEY=your_openai_api_key_here
OPENAI_LLM_MODEL=gpt-5-nano
OPENAI_EMBEDDING_MODEL=text-embedding-3-small

# Ollama Configuration (required when using ollama providers)
OLLAMA_BASE_URL=http://localhost:11434
OLLAMA_LLM_MODEL=gemma3:270m
OLLAMA_EMBEDDING_MODEL=nomic-embed-text:latest

# Security
SECRET_KEY=your_secret_key_here

# Database passwords (change in production!)
POSTGRES_PASSWORD=secure_password
NEO4J_PASSWORD=secure_password
REDIS_PASSWORD=secure_password
```

### 3. Start Services

```bash
# Start all services
docker-compose -f infrastructure/docker/docker-compose.yml up -d

# Check service status
docker-compose -f infrastructure/docker/docker-compose.yml ps
```

### 4. Verify Installation

```bash
# Check API health
curl http://localhost:8000/health/detailed

# Check Flower (Celery monitoring)
open http://localhost:5555

# Check Grafana (metrics)
open http://localhost:3000
# Login: admin/admin
```

## 📚 API Documentation

Once running, visit:
- **API Docs**: http://localhost:8000/docs
- **ReDoc**: http://localhost:8000/redoc

### Key Endpoints

#### Authentication
```bash
# Register user
curl -X POST "http://localhost:8000/auth/register" \
  -H "Content-Type: application/json" \
  -d '{"username": "testuser", "email": "test@example.com", "password": "password123", "full_name": "Test User"}'

# Login
curl -X POST "http://localhost:8000/auth/login" \
  -H "Content-Type: application/x-www-form-urlencoded" \
  -d "username=testuser&password=password123"
```

#### Memory Operations
```bash
# Process conversation turn
curl -X POST "http://localhost:8000/memory/process-turn" \
  -H "Authorization: Bearer YOUR_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{"user_message": "I am vegetarian and avoid dairy", "user_id": "user_id", "conversation_id": "conv_id"}'

# Query memories
curl -X POST "http://localhost:8000/memory/query" \
  -H "Authorization: Bearer YOUR_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{"query": "What are my dietary preferences?", "user_id": "user_id", "top_k": 5}'
```

## 🧪 Testing

### Run Tests

```bash
# Install development dependencies
pip install -r requirements.txt

# Run tests
pytest tests/ -v

# Run with coverage
pytest tests/ --cov=. --cov-report=html
```

### Load Testing

```bash
# Install locust
pip install locust

# Run load tests
locust -f tests/load/locustfile.py --host=http://localhost:8000
```

## 📊 Monitoring

### Metrics

- **Prometheus**: http://localhost:9090
- **Grafana**: http://localhost:3000 (admin/admin)

### Key Metrics

- Request latency and throughput
- Memory operation success rates
- Database connection health
- Celery task queue status
- Vector search performance

### Health Checks

```bash
# Basic health check
curl http://localhost:8000/health/

# Detailed health check
curl http://localhost:8000/health/detailed
```

## 🔧 Configuration

### Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `LLM_PROVIDER` | LLM backend provider (`ollama`, `openai`) | `ollama` |
| `EMBEDDING_PROVIDER` | Embedding backend provider (`ollama`, `openai`, `local`) | `ollama` |
| `OPENAI_API_KEY` | OpenAI API key (required when OpenAI provider is selected) | Empty |
| `OPENAI_LLM_MODEL` | OpenAI chat model | `gpt-5-nano` |
| `OPENAI_EMBEDDING_MODEL` | OpenAI embedding model | `text-embedding-3-small` |
| `OLLAMA_BASE_URL` | Ollama API base URL | `http://localhost:11434` |
| `OLLAMA_LLM_MODEL` | Ollama chat model | `gemma3:270m` |
| `OLLAMA_EMBEDDING_MODEL` | Ollama embedding model | `nomic-embed-text:latest` |
| `SECRET_KEY` | JWT secret key | Required |
| `DATABASE_URL` | PostgreSQL connection string | Auto-generated |
| `REDIS_URL` | Redis connection string | Auto-generated |
| `NEO4J_URI` | Neo4j connection string | Auto-generated |
| `EMBEDDING_DIMENSION` | Target embedding dimension used for storage | `1536` |
| `SIMILARITY_THRESHOLD` | Memory similarity threshold | 0.75 |
| `MAX_MEMORIES_PER_USER` | Max memories per user | 10000 |

### Performance Tuning

#### Database
- Adjust `shared_buffers` and `work_mem` in PostgreSQL
- Tune vector index parameters (`lists` in ivfflat)
- Configure connection pooling

#### Celery
- Adjust worker concurrency based on CPU cores
- Configure task routing and priorities
- Set appropriate timeouts

#### Memory System
- Tune similarity thresholds for your use case
- Adjust ACAN attention dimensions
- Configure memory consolidation frequency

## 🚀 Deployment

### Production Deployment

1. **Security Hardening**:
   - Change all default passwords
   - Use strong JWT secrets
   - Enable HTTPS with proper certificates
   - Configure firewall rules

2. **Scaling**:
   - Use multiple API replicas
   - Scale Celery workers based on load
   - Configure database read replicas
   - Use Redis Cluster for high availability

3. **Monitoring**:
   - Set up alerting rules in Prometheus
   - Configure log aggregation
   - Monitor resource usage
   - Set up backup procedures

### Kubernetes Deployment

```bash
# Apply Kubernetes manifests
kubectl apply -f k8s/

# Check deployment status
kubectl get pods -n engram-production
```

## 🔍 Troubleshooting

### Common Issues

1. **Database Connection Errors**:
   - Check if PostgreSQL is running
   - Verify connection strings
   - Check network connectivity

2. **Memory Operations Failing**:
   - Verify OpenAI API key
   - Check embedding service status
   - Review similarity thresholds

3. **Celery Tasks Not Processing**:
   - Check Redis connection
   - Verify worker status in Flower
   - Review task logs

### Logs

```bash
# View API logs
docker-compose logs -f api

# View worker logs
docker-compose logs -f worker

# View database logs
docker-compose logs -f postgres
```

## 📈 Performance Targets

Based on research benchmarks:

- **Response Latency**: p95 < 1.5 seconds
- **Token Efficiency**: 90%+ reduction vs full-context
- **Memory Footprint**: < 10K tokens per conversation
- **Accuracy**: >65% LLM-as-a-Judge score
- **Throughput**: 1000+ requests/minute per instance

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests
5. Submit a pull request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- Engram research and implementation
- OpenAI for GPT models
- The open-source community for excellent tools and libraries

## 📞 Support

- **Documentation**: Check the `/docs` endpoint when running
- **Issues**: Create GitHub issues for bugs and feature requests
- **Discussions**: Use GitHub discussions for questions

## Glossary

**Engram** — enduring offline physical/chemical changes underlying a memory; "engram cells" are the neuron ensembles that encode and can be reactivated to retrieve the memory. [4][5]

---

**Built with ❤️ for the AI community**
