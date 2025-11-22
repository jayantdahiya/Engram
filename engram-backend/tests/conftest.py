"""Pytest configuration and fixtures"""

import asyncio
from collections.abc import AsyncGenerator

import pytest
from fastapi.testclient import TestClient
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine
from sqlalchemy.orm import DeclarativeBase

from api.main import app
from core.database import get_db_session


class Base(DeclarativeBase):
    """Base class for test database models"""

    pass


# Test database URL
TEST_DATABASE_URL = "sqlite+aiosqlite:///./test.db"

# Create test engine
test_engine = create_async_engine(
    TEST_DATABASE_URL,
    echo=False,
    pool_pre_ping=True,
)

# Test session factory
TestSessionLocal = async_sessionmaker(
    test_engine,
    class_=AsyncSession,
    expire_on_commit=False,
)


@pytest.fixture(scope="session")
def event_loop():
    """Create an instance of the default event loop for the test session."""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()


@pytest.fixture
async def db_session() -> AsyncGenerator[AsyncSession, None]:
    """Create a test database session"""
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
    """Create a test client with database session override"""

    def override_get_db():
        return db_session

    app.dependency_overrides[get_db_session] = override_get_db

    with TestClient(app) as test_client:
        yield test_client

    app.dependency_overrides.clear()


@pytest.fixture
def test_user_data():
    """Test user data"""
    return {
        "username": "testuser",
        "email": "test@example.com",
        "password": "testpassword123",
        "full_name": "Test User",
    }


@pytest.fixture
def test_memory_data():
    """Test memory data"""
    return {
        "text": "I am vegetarian and avoid dairy products",
        "user_id": "test-user-id",
        "conversation_id": "test-conversation-id",
    }


@pytest.fixture
def test_conversation_data():
    """Test conversation data"""
    return {"title": "Test Conversation", "user_id": "test-user-id", "metadata": {"test": "data"}}


@pytest.fixture
async def authenticated_client(client: TestClient, test_user_data: dict):
    """Create an authenticated test client"""

    # Register user
    response = client.post("/auth/register", json=test_user_data)
    assert response.status_code == 200

    # Login
    login_data = {"username": test_user_data["username"], "password": test_user_data["password"]}
    response = client.post("/auth/login", data=login_data)
    assert response.status_code == 200

    token = response.json()["access_token"]

    # Set authorization header
    client.headers.update({"Authorization": f"Bearer {token}"})

    return client
