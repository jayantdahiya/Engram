"""Custom middleware for the FastAPI application"""

import time
from collections.abc import Callable

from fastapi import Request, Response
from prometheus_client import Counter, Histogram, generate_latest
from starlette.middleware.base import BaseHTTPMiddleware

from core.logging import logger

# Prometheus metrics
REQUEST_COUNT = Counter(
    "http_requests_total", "Total HTTP requests", ["method", "endpoint", "status_code"]
)

REQUEST_LATENCY = Histogram(
    "http_request_duration_seconds", "HTTP request latency", ["method", "endpoint"]
)

ACTIVE_CONNECTIONS = Counter("http_active_connections", "Number of active HTTP connections")


class PrometheusMiddleware(BaseHTTPMiddleware):
    """Middleware for Prometheus metrics collection"""

    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        """Process request and collect metrics"""

        # Increment active connections
        ACTIVE_CONNECTIONS.inc()

        start_time = time.time()

        try:
            response = await call_next(request)

            # Record metrics
            process_time = time.time() - start_time

            REQUEST_COUNT.labels(
                method=request.method, endpoint=request.url.path, status_code=response.status_code
            ).inc()

            REQUEST_LATENCY.labels(method=request.method, endpoint=request.url.path).observe(
                process_time
            )

            return response

        except Exception as e:
            # Record error metrics
            REQUEST_COUNT.labels(
                method=request.method, endpoint=request.url.path, status_code=500
            ).inc()

            logger.error(f"Request processing error: {e}")
            raise

        finally:
            # Decrement active connections
            ACTIVE_CONNECTIONS.dec()


class LoggingMiddleware(BaseHTTPMiddleware):
    """Middleware for request/response logging"""

    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        """Log request and response details"""

        start_time = time.time()

        # Log request
        logger.info(
            f"Request: {request.method} {request.url.path} "
            f"from {request.client.host if request.client else 'unknown'}"
        )

        try:
            response = await call_next(request)

            # Log response
            process_time = time.time() - start_time
            logger.info(
                f"Response: {response.status_code} "
                f"in {process_time:.3f}s for {request.method} {request.url.path}"
            )

            return response

        except Exception as e:
            process_time = time.time() - start_time
            logger.error(
                f"Error: {str(e)} in {process_time:.3f}s for {request.method} {request.url.path}"
            )
            raise


class SecurityMiddleware(BaseHTTPMiddleware):
    """Security middleware for common security headers"""

    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        """Add security headers to response"""

        response = await call_next(request)

        # Add security headers
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["X-Frame-Options"] = "DENY"
        response.headers["X-XSS-Protection"] = "1; mode=block"
        response.headers["Referrer-Policy"] = "strict-origin-when-cross-origin"

        # Add CORS headers if not already present
        if "Access-Control-Allow-Origin" not in response.headers:
            response.headers["Access-Control-Allow-Origin"] = "*"

        return response


def get_metrics():
    """Get Prometheus metrics"""
    return generate_latest()


def setup_middleware(app):
    """Setup all middleware for the application"""

    # Add middleware in reverse order (last added is first executed)
    app.add_middleware(SecurityMiddleware)
    app.add_middleware(LoggingMiddleware)
    app.add_middleware(PrometheusMiddleware)
