"""Shared pytest configuration and fixtures for Engram tests."""

import asyncio
import os
import sys
from collections.abc import AsyncGenerator
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

# Add project paths for imports
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(PROJECT_ROOT, "engram-backend"))
sys.path.insert(0, os.path.join(PROJECT_ROOT, "engram-mcp", "src"))

# Import after sys.path update
from api.main import app
from api.dependencies import (
    get_database_session,
    get_neo4j_dependency,
    get_redis_dependency,
)
from core.database import Base

# Import models to register them with Base
from models import user, memory, conversation


# =============================================================================
# Event Loop Configuration
# =============================================================================


# =============================================================================
# Database Fixtures (Backend Integration)
# =============================================================================

# Test database URL
TEST_DATABASE_URL = "sqlite+aiosqlite:///./test.db"


# Create test engine
@pytest.fixture
async def db_engine():
    """Create test database engine."""
    engine = create_async_engine(
        TEST_DATABASE_URL,
        echo=False,
        pool_pre_ping=True,
    )
    # Create tables manually for SQLite (since no ORM models map to these tables)
    async with engine.begin() as conn:
        await conn.execute(
            text("""
            CREATE TABLE IF NOT EXISTS users (
                id TEXT PRIMARY KEY,
                username TEXT UNIQUE NOT NULL,
                email TEXT UNIQUE NOT NULL,
                full_name TEXT,
                hashed_password TEXT NOT NULL,
                is_active BOOLEAN DEFAULT TRUE,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                last_login TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        )
        await conn.execute(
            text("""
            CREATE TABLE IF NOT EXISTS conversations (
                id TEXT PRIMARY KEY,
                user_id TEXT NOT NULL,
                title TEXT,
                metadata TEXT DEFAULT '{}',
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY(user_id) REFERENCES users(id) ON DELETE CASCADE
            )
        """)
        )
        await conn.execute(
            text("""
            CREATE TABLE IF NOT EXISTS conversation_turns (
                id TEXT PRIMARY KEY,
                conversation_id TEXT NOT NULL,
                user_id TEXT NOT NULL,
                user_message TEXT NOT NULL,
                assistant_response TEXT,
                turn_number INTEGER NOT NULL,
                timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                memory_operations TEXT,
                processing_time_ms FLOAT DEFAULT 0.0,
                FOREIGN KEY(conversation_id) REFERENCES conversations(id) ON DELETE CASCADE,
                FOREIGN KEY(user_id) REFERENCES users(id) ON DELETE CASCADE
            )
        """)
        )
        await conn.execute(
            text("""
            CREATE TABLE IF NOT EXISTS memories (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id TEXT NOT NULL,
                conversation_id TEXT,
                text TEXT NOT NULL,
                embedding TEXT,
                timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                importance_score FLOAT DEFAULT 0.0,
                access_count INTEGER DEFAULT 0,
                metadata TEXT DEFAULT '{}',
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY(user_id) REFERENCES users(id) ON DELETE CASCADE,
                FOREIGN KEY(conversation_id) REFERENCES conversations(id) ON DELETE SET NULL
            )
        """)
        )

    yield engine
    await engine.dispose()


@pytest.fixture
async def db_session(db_engine) -> AsyncGenerator[AsyncSession, None]:
    """Create a test database session for integration tests."""
    TestSessionLocal = async_sessionmaker(
        db_engine,
        class_=AsyncSession,
        expire_on_commit=False,
    )

    async with TestSessionLocal() as session:
        try:
            yield session
            await session.commit()
        except Exception:
            await session.rollback()
            raise
        finally:
            await session.close()


@pytest.fixture
def client(db_session: AsyncSession) -> TestClient:
    """Create a test client with database session override."""

    async def override_get_db():
        yield db_session

    async def override_get_neo4j():
        yield AsyncMock()

    async def override_get_redis():
        yield AsyncMock()

    app.dependency_overrides[get_database_session] = override_get_db
    app.dependency_overrides[get_neo4j_dependency] = override_get_neo4j
    app.dependency_overrides[get_redis_dependency] = override_get_redis

    # Mock database initialization to avoid connecting to real Redis/Neo4j
    with (
        patch("api.main.init_databases", new_callable=AsyncMock),
        patch("api.main.close_databases", new_callable=AsyncMock),
    ):
        with TestClient(app) as test_client:
            yield test_client

    app.dependency_overrides.clear()


@pytest.fixture
async def authenticated_client(client: TestClient, test_user_data: dict):
    """Create an authenticated test client."""
    # Register user
    response = client.post("/auth/register", json=test_user_data)
    # If already registered (e.g. from previous run), login
    if response.status_code != 200:
        pass

    # Login
    login_data = {
        "username": test_user_data["username"],
        "password": test_user_data["password"],
    }
    response = client.post("/auth/login", data=login_data)
    if response.status_code != 200:
        # Try to register if login failed (maybe cleaned up)
        client.post("/auth/register", json=test_user_data)
        response = client.post("/auth/login", data=login_data)

    assert response.status_code == 200
    token = response.json()["access_token"]

    # Set authorization header
    client.headers.update({"Authorization": f"Bearer {token}"})
    return client


# =============================================================================
# Database Fixtures (Backend Unit - Mocks)
# =============================================================================


@pytest.fixture
def mock_db_session():
    """Create mock database session for unit tests."""
    session = AsyncMock()
    session.execute = AsyncMock()
    session.commit = AsyncMock()
    session.rollback = AsyncMock()
    session.close = AsyncMock()
    return session


# =============================================================================
# Test Data Fixtures
# =============================================================================


@pytest.fixture
def test_user_data():
    """Test user data."""
    return {
        "username": "testuser",
        "email": "test@example.com",
        "password": "testpassword123",
        "full_name": "Test User",
    }


@pytest.fixture
def test_memory_data():
    """Test memory data."""
    return {
        "text": "I am vegetarian and avoid dairy products",
        "user_id": "test-user-id",
        "conversation_id": "test-conversation-id",
        "importance_score": 5.0,
    }


@pytest.fixture
def test_conversation_data():
    """Test conversation data."""
    return {
        "title": "Test Conversation",
        "user_id": "test-user-id",
        "metadata": {"test": "data"},
    }


# =============================================================================
# Mock Fixtures
# =============================================================================


@pytest.fixture
def mock_embedding():
    """Create mock embedding vector."""
    import numpy as np

    return np.random.rand(1536).astype(np.float32)


@pytest.fixture
def mock_httpx_client():
    """Create mock httpx async client."""
    client = AsyncMock()
    client.post = AsyncMock()
    client.get = AsyncMock()
    client.delete = AsyncMock()
    client.aclose = AsyncMock()
    return client


# =============================================================================
# MCP Fixtures
# =============================================================================


@pytest.fixture
def mock_mcp_context():
    """Create mock MCP context for tool tests."""
    ctx = MagicMock()
    ctx.request_context = MagicMock()
    ctx.request_context.lifespan_context = {"client": AsyncMock()}
    return ctx


@pytest.fixture
def mcp_client_config():
    """MCP client test configuration."""
    return {
        "api_url": "http://localhost:8000",
        "username": "testuser",
        "password": "testpass123",
    }


@pytest.fixture(autouse=True)
def mock_llm_service():
    """Mock LLM service to avoid external calls."""
    with (
        patch(
            "services.llm_service.LLMService.classify_memory_operation",
            new_callable=AsyncMock,
        ) as mock_classify,
        patch(
            "services.llm_service.LLMService.extract_entities_and_relations",
            new_callable=AsyncMock,
        ) as mock_extract,
        patch(
            "services.llm_service.LLMService.generate_response", new_callable=AsyncMock
        ) as mock_generate,
    ):
        # Mock classify_memory_operation
        # Use side_effect to return different operations for different calls
        mock_classify.side_effect = [
            {
                "operation": "ADD",
                "confidence": 0.9,
                "reasoning": "First memory",
                "related_memory_indices": [],
            },
            {
                "operation": "UPDATE",
                "confidence": 0.95,
                "reasoning": "Updating memory",
                "related_memory_indices": [0],
            },
            {
                "operation": "ADD",
                "confidence": 0.9,
                "reasoning": "New memory",
                "related_memory_indices": [],
            },
            {
                "operation": "ADD",
                "confidence": 0.9,
                "reasoning": "New memory",
                "related_memory_indices": [],
            },
            {
                "operation": "ADD",
                "confidence": 0.9,
                "reasoning": "New memory",
                "related_memory_indices": [],
            },
            {
                "operation": "ADD",
                "confidence": 0.9,
                "reasoning": "New memory",
                "related_memory_indices": [],
            },
            {
                "operation": "ADD",
                "confidence": 0.9,
                "reasoning": "New memory",
                "related_memory_indices": [],
            },
        ]

        # Mock extract_entities_and_relations
        mock_extract.return_value = {
            "entities": [],
            "relationships": [],
        }

        # Mock generate_response
        mock_generate.return_value = "Mocked LLM response"

        yield {
            "classify": mock_classify,
            "extract": mock_extract,
            "generate": mock_generate,
        }
