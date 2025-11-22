"""Structured logging setup with loguru"""

import sys

from loguru import logger

from core.config import settings


def setup_logging() -> None:
    """Configure structured logging"""

    # Remove default handler
    logger.remove()

    # Add console handler with structured format
    logger.add(
        sys.stdout,
        format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | "
        "<level>{level: <8}</level> | "
        "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> | "
        "<level>{message}</level>",
        level=settings.log_level,
        colorize=True,
    )

    # Add file handler for production
    if settings.environment == "production":
        logger.add(
            "logs/engram_{time:YYYY-MM-DD}.log",
            format=(
                "{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{function}:{line} | {message}"
            ),
            level=settings.log_level,
            rotation="1 day",
            retention="30 days",
            compression="zip",
        )

    # Add error file handler
    logger.add(
        "logs/errors_{time:YYYY-MM-DD}.log",
        format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{function}:{line} | {message}",
        level="ERROR",
        rotation="1 day",
        retention="90 days",
        compression="zip",
    )

    logger.info(f"Logging configured for {settings.environment} environment")
