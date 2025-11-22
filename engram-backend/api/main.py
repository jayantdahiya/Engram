"""FastAPI application main entry point"""

from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from api.middleware import PrometheusMiddleware
from api.routers import auth, conversation, health, memory
from core.config import settings
from core.database import close_databases, init_databases
from core.logging import logger, setup_logging


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager"""
    # Startup
    logger.info("Starting Engram API...")
    setup_logging()
    await init_databases()
    logger.info("Application startup completed")

    yield

    # Shutdown
    logger.info("Shutting down Engram API...")
    await close_databases()
    logger.info("Application shutdown completed")


# Create FastAPI application
app = FastAPI(
    title="Engram API",
    description="Persistent long-term memory for AI agents with dense facts and graph relations",
    version="1.0.0",
    lifespan=lifespan,
    docs_url="/docs" if settings.debug else None,
    redoc_url="/redoc" if settings.debug else None,
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"] if settings.debug else ["https://yourdomain.com"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Add Prometheus metrics middleware
app.add_middleware(PrometheusMiddleware)

# Include routers
app.include_router(health.router, prefix="/health", tags=["health"])
app.include_router(auth.router, prefix="/auth", tags=["authentication"])
app.include_router(memory.router, prefix="/memory", tags=["memory"])
app.include_router(conversation.router, prefix="/conversation", tags=["conversation"])


@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "message": "Engram API",
        "version": "1.0.0",
        "status": "running",
        "docs": "/docs" if settings.debug else "disabled",
    }


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "api.main:app",
        host="0.0.0.0",
        port=8000,
        reload=settings.debug,
        log_level=settings.log_level.lower(),
    )
