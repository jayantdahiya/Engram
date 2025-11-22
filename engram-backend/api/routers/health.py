"""Health check endpoints"""

from fastapi import APIRouter, HTTPException, status
from sqlalchemy.ext.asyncio import AsyncSession

from api.dependencies import DatabaseDep
from core.database import check_neo4j_health, check_postgres_health, check_redis_health
from core.logging import logger

router = APIRouter()


@router.get("/")
async def health_check():
    """Basic health check endpoint"""
    return {"status": "healthy", "service": "engram-api", "version": "1.0.0"}


@router.get("/detailed")
async def detailed_health_check(db_session: AsyncSession = DatabaseDep):
    """Detailed health check with database connectivity"""

    health_status = {"status": "healthy", "service": "engram-api", "version": "1.0.0", "checks": {}}

    # Check PostgreSQL
    try:
        postgres_healthy = await check_postgres_health()
        health_status["checks"]["postgresql"] = {
            "status": "healthy" if postgres_healthy else "unhealthy",
            "connected": postgres_healthy,
        }
    except Exception as e:
        logger.error(f"PostgreSQL health check failed: {e}")
        health_status["checks"]["postgresql"] = {"status": "unhealthy", "error": str(e)}

    # Check Neo4j
    try:
        neo4j_healthy = await check_neo4j_health()
        health_status["checks"]["neo4j"] = {
            "status": "healthy" if neo4j_healthy else "unhealthy",
            "connected": neo4j_healthy,
        }
    except Exception as e:
        logger.error(f"Neo4j health check failed: {e}")
        health_status["checks"]["neo4j"] = {"status": "unhealthy", "error": str(e)}

    # Check Redis
    try:
        redis_healthy = await check_redis_health()
        health_status["checks"]["redis"] = {
            "status": "healthy" if redis_healthy else "unhealthy",
            "connected": redis_healthy,
        }
    except Exception as e:
        logger.error(f"Redis health check failed: {e}")
        health_status["checks"]["redis"] = {"status": "unhealthy", "error": str(e)}

    # Determine overall status
    all_healthy = all(
        check.get("status") == "healthy" for check in health_status["checks"].values()
    )

    if not all_healthy:
        health_status["status"] = "unhealthy"
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail=health_status)

    return health_status


@router.get("/ready")
async def readiness_check():
    """Kubernetes readiness probe"""
    return {"status": "ready"}


@router.get("/live")
async def liveness_check():
    """Kubernetes liveness probe"""
    return {"status": "alive"}
