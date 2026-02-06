# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Engram is a persistent long-term memory system for AI agents. It stores conversation memories as vector embeddings in PostgreSQL (pgvector), entity/relationship graphs in Neo4j, and uses an ACAN (Attention-based Context-Aware Network) retrieval system for intelligent memory recall. All backend code lives in `engram-backend/`.

## Commands

### Development (local, from `engram-backend/`)
```bash
# Activate venv
source .venv/bin/activate

# Run API server
python -m api.main
# or: uvicorn api.main:app --reload --host 0.0.0.0 --port 8000

# Run Celery worker
celery -A tasks.celery_app worker --loglevel=info

# Run Celery beat scheduler
celery -A tasks.celery_app beat --loglevel=info
```

### Docker (full stack)
```bash
# Start all services (API, workers, PostgreSQL, Neo4j, Redis, Nginx, Prometheus, Grafana)
docker-compose -f engram-backend/infrastructure/docker/docker-compose.yml up -d

# Check status
docker-compose -f engram-backend/infrastructure/docker/docker-compose.yml ps
```

### Testing (from `engram-backend/`)
```bash
pytest tests/ -v                          # All tests
pytest tests/unit/ -v                     # Unit tests only
pytest tests/integration/ -v              # Integration tests only
pytest tests/unit/test_memory_manager.py -v  # Single test file
pytest tests/ --cov=. --cov-report=html   # With coverage
```

### Linting (from repo root)
```bash
ruff check engram-backend/                # Lint
ruff check engram-backend/ --fix          # Lint with auto-fix
black engram-backend/                     # Format
isort engram-backend/                     # Sort imports
```

### Benchmarks (from `engram-backend/`)
```bash
python -m benchmarks.run_benchmarks
```

## Architecture

### Request Flow
Nginx -> FastAPI (`api/main.py`) -> Routers (`api/routers/`) -> Services (`services/`) -> Databases

### Core Components

**Memory Manager** (`services/memory_manager.py`): Central orchestrator. On each conversation turn:
1. Generates embedding via `embedding_service`
2. Classifies operation via `llm_service` (ADD/UPDATE/CONSOLIDATE/NOOP)
3. Executes the operation against PostgreSQL
4. Extracts entities via `llm_service` and stores in Neo4j via `graph_service`

**ACAN Retrieval** (in `memory_manager.py`): Retrieves memories using a composite score:
- 40% attention scores (softmax-normalized cross-attention)
- 40% cosine similarity
- 10% recency weight (exponential decay, 72h half-life)
- 10% importance score (log of access count)

**Graph Service** (`services/graph_service.py`): Manages Neo4j entity/relationship storage (Engram Graph). Entities have MERGE semantics; relationships use `:RELATION` type with confidence scores.

**LLM Service** (`services/llm_service.py`): Supports two providers controlled by `LLM_PROVIDER` env var:
- `ollama` (default): Local inference, model `gemma3:270m`, embeddings via `nomic-embed-text`
- `openai`: Cloud inference via GPT-4o-mini, embeddings via `text-embedding-3-small`

**Celery Tasks** (`tasks/`): Two queues:
- `memory` queue: memory extraction, batch processing, consolidation, summaries
- `maintenance` queue: cleanup old memories (hourly), optimize embeddings (daily), generate summaries (6h)

### Database Stack
- **PostgreSQL + pgvector**: Vector storage, memories table with embeddings (default dim: 768 for nomic-embed-text)
- **Neo4j 5.x**: Entity graph with `:Entity` nodes and `:RELATION` edges, scoped by `user_id`
- **Redis 7.x**: Celery broker/backend, caching, session storage

### API Routes
- `/auth/` - JWT authentication (register, login)
- `/memory/` - CRUD, query (ACAN retrieval), process-turn, consolidate, embedding generation, stats
- `/conversation/` - Conversation management
- `/health/` - Health checks (basic and detailed with per-service status)

### Key Patterns
- **Dependency injection** via `api/dependencies.py`: `DatabaseDep`, `AuthUserDep`, `Neo4jDep`, `RedisDep`
- **Async throughout**: SQLAlchemy async sessions, async Neo4j driver, async Redis
- **Global singletons**: `memory_manager`, `graph_service`, `embedding_service`, `llm_service` instantiated at module level
- **Pydantic models** in `models/`: `memory.py` (main), `user.py`, `conversation.py`
- **Config** via `core/config.py`: Pydantic `BaseSettings` loading from `.env`
- Tests use SQLite (`sqlite+aiosqlite`) as a test database substitute

## CI Pipeline

GitHub Actions (`.github/workflows/ci.yml`) runs: ruff + black + isort lint -> pytest with PostgreSQL/Redis services -> Docker build -> Bandit security scan.

## Code Style

- **Ruff**: line-length 100, targeting Python 3.11, rules: E, F, W, I, B, UP, Q
- **Black**: line-length 88, target Python 3.11
- **isort**: black-compatible profile
- First-party packages: `api`, `core`, `models`, `services`, `tasks`, `tests`, `utils`
