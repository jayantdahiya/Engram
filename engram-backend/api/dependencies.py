"""FastAPI dependencies for database connections and authentication"""

from collections.abc import AsyncGenerator

from fastapi import Depends, HTTPException, status
from sqlalchemy.ext.asyncio import AsyncSession

from core.database import get_db_session, get_neo4j_session, get_redis
from core.logging import logger
from core.security import get_current_user


async def get_database_session() -> AsyncGenerator[AsyncSession, None]:
    """Dependency to get database session"""
    async with get_db_session() as session:
        try:
            yield session
        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"Database session error: {e}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Database connection error",
            )


async def get_neo4j_dependency():
    """Dependency to get Neo4j session"""
    try:
        async with get_neo4j_session() as session:
            yield session
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Neo4j session error: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Neo4j connection error"
        )


async def get_redis_dependency():
    """Dependency to get Redis client"""
    try:
        redis_client = await get_redis()
        yield redis_client
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Redis connection error: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Redis connection error"
        )


async def get_authenticated_user(current_user: str = Depends(get_current_user)) -> str:
    """Dependency to get authenticated user"""
    return current_user


async def get_optional_user(current_user: str = Depends(get_current_user)) -> str:
    """Dependency to get optional authenticated user (for public endpoints)"""
    return current_user


# Common dependency combinations
DatabaseDep = Depends(get_database_session)
Neo4jDep = Depends(get_neo4j_dependency)
RedisDep = Depends(get_redis_dependency)
AuthUserDep = Depends(get_authenticated_user)
OptionalUserDep = Depends(get_optional_user)
