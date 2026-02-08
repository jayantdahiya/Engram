"""Database connections and session management"""

from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager

import redis.asyncio as redis
from neo4j import AsyncGraphDatabase
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine
from sqlalchemy.orm import DeclarativeBase

from core.config import settings
from core.logging import logger


class Base(DeclarativeBase):
    """Base class for SQLAlchemy models"""

    pass


# PostgreSQL Engine
postgres_engine = create_async_engine(
    settings.database_url,
    echo=settings.debug,
    pool_pre_ping=True,
    pool_recycle=300,
)

# Async Session Factory
AsyncSessionLocal = async_sessionmaker(
    postgres_engine,
    class_=AsyncSession,
    expire_on_commit=False,
)

# Neo4j Driver
neo4j_driver: AsyncGraphDatabase | None = None

# Redis Connection
redis_client: redis.Redis | None = None


async def init_databases() -> None:
    """Initialize all database connections"""
    global neo4j_driver, redis_client

    try:
        # Initialize Neo4j
        neo4j_driver = AsyncGraphDatabase.driver(
            settings.neo4j_uri, auth=(settings.neo4j_user, settings.neo4j_password)
        )
        await neo4j_driver.verify_connectivity()
        logger.info("Neo4j connection established")

        # Initialize Redis
        redis_client = redis.from_url(settings.redis_url)
        await redis_client.ping()
        logger.info("Redis connection established")

        # Create PostgreSQL tables
        async with postgres_engine.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)
        logger.info("PostgreSQL tables created/verified")

    except Exception as e:
        logger.error(f"Database initialization failed: {e}")
        raise


async def close_databases() -> None:
    """Close all database connections"""
    global neo4j_driver, redis_client

    try:
        if neo4j_driver:
            await neo4j_driver.close()
            logger.info("Neo4j connection closed")

        if redis_client:
            await redis_client.close()
            logger.info("Redis connection closed")

        await postgres_engine.dispose()
        logger.info("PostgreSQL connection closed")

    except Exception as e:
        logger.error(f"Error closing databases: {e}")


@asynccontextmanager
async def get_db_session() -> AsyncGenerator[AsyncSession, None]:
    """Get database session with automatic cleanup"""
    async with AsyncSessionLocal() as session:
        try:
            yield session
            await session.commit()
        except Exception:
            await session.rollback()
            raise
        finally:
            await session.close()


@asynccontextmanager
async def get_neo4j_session():
    """Get Neo4j session with automatic cleanup"""
    if not neo4j_driver:
        raise RuntimeError("Neo4j driver not initialized")

    async with neo4j_driver.session() as session:
        yield session


async def get_redis() -> redis.Redis:
    """Get Redis client"""
    if not redis_client:
        raise RuntimeError("Redis client not initialized")
    return redis_client


# Health check functions
async def check_postgres_health() -> bool:
    """Check PostgreSQL connection health"""
    try:
        async with get_db_session() as session:
            await session.execute(text("SELECT 1"))
        return True
    except Exception as e:
        logger.error(f"PostgreSQL health check failed: {e}")
        return False


async def check_neo4j_health() -> bool:
    """Check Neo4j connection health"""
    try:
        async with get_neo4j_session() as session:
            await session.run("RETURN 1")
        return True
    except Exception as e:
        logger.error(f"Neo4j health check failed: {e}")
        return False


async def check_redis_health() -> bool:
    """Check Redis connection health"""
    try:
        client = await get_redis()
        await client.ping()
        return True
    except Exception as e:
        logger.error(f"Redis health check failed: {e}")
        return False
