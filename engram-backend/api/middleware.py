"""Custom middleware for the FastAPI application"""

import time
from collections import defaultdict
from collections.abc import Callable

from fastapi import Request, Response, status
from fastapi.responses import JSONResponse
from prometheus_client import Counter, Gauge, Histogram, generate_latest
from starlette.middleware.base import BaseHTTPMiddleware

from core.logging import logger

# Prometheus metrics
REQUEST_COUNT = Counter(
    "http_requests_total", "Total HTTP requests", ["method", "endpoint", "status_code"]
)

REQUEST_LATENCY = Histogram(
    "http_request_duration_seconds", "HTTP request latency", ["method", "endpoint"]
)

ACTIVE_CONNECTIONS = Gauge("http_active_connections", "Number of active HTTP connections")

RATE_LIMIT_HITS = Counter("rate_limit_hits_total", "Total rate limit hits", ["endpoint"])


class RateLimitMiddleware(BaseHTTPMiddleware):
    """Token bucket rate limiting middleware"""

    def __init__(
        self,
        app,
        requests_per_minute: int = 60,
        burst_size: int = 10,
        exclude_paths: list[str] | None = None,
    ):
        super().__init__(app)
        self.requests_per_minute = requests_per_minute
        self.burst_size = burst_size
        self.exclude_paths = exclude_paths or [
            "/health",
            "/metrics",
            "/docs",
            "/redoc",
            "/openapi.json",
        ]

        # Token buckets per client IP
        self.buckets: dict[str, dict] = defaultdict(
            lambda: {"tokens": burst_size, "last_update": time.time()}
        )
        self.tokens_per_second = requests_per_minute / 60.0

    def _get_client_ip(self, request: Request) -> str:
        """Extract client IP from request"""
        # Check for X-Forwarded-For header (behind proxy)
        forwarded = request.headers.get("X-Forwarded-For")
        if forwarded:
            return forwarded.split(",")[0].strip()

        # Check for X-Real-IP header
        real_ip = request.headers.get("X-Real-IP")
        if real_ip:
            return real_ip

        # Fall back to direct client
        return request.client.host if request.client else "unknown"

    def _is_rate_limited(self, client_ip: str) -> tuple[bool, dict]:
        """Check if client is rate limited using token bucket algorithm"""
        current_time = time.time()
        bucket = self.buckets[client_ip]

        # Refill tokens based on elapsed time
        elapsed = current_time - bucket["last_update"]
        bucket["tokens"] = min(self.burst_size, bucket["tokens"] + elapsed * self.tokens_per_second)
        bucket["last_update"] = current_time

        # Check if we have tokens available
        if bucket["tokens"] >= 1:
            bucket["tokens"] -= 1
            return False, {
                "remaining": int(bucket["tokens"]),
                "limit": self.requests_per_minute,
                "reset": int(
                    current_time + (self.burst_size - bucket["tokens"]) / self.tokens_per_second
                ),
            }

        return True, {
            "remaining": 0,
            "limit": self.requests_per_minute,
            "reset": int(current_time + (1 - bucket["tokens"]) / self.tokens_per_second),
        }

    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        """Process request with rate limiting"""

        # Skip rate limiting for excluded paths
        path = request.url.path
        if any(path.startswith(excluded) for excluded in self.exclude_paths):
            return await call_next(request)

        client_ip = self._get_client_ip(request)
        is_limited, rate_info = self._is_rate_limited(client_ip)

        if is_limited:
            RATE_LIMIT_HITS.labels(endpoint=path).inc()
            logger.warning(f"Rate limit exceeded for {client_ip} on {path}")

            return JSONResponse(
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                content={
                    "error": "Rate limit exceeded",
                    "message": f"Too many requests. Limit: {rate_info['limit']}/minute",
                    "retry_after": rate_info["reset"] - int(time.time()),
                },
                headers={
                    "X-RateLimit-Limit": str(rate_info["limit"]),
                    "X-RateLimit-Remaining": "0",
                    "X-RateLimit-Reset": str(rate_info["reset"]),
                    "Retry-After": str(rate_info["reset"] - int(time.time())),
                },
            )

        # Process request and add rate limit headers
        response = await call_next(request)
        response.headers["X-RateLimit-Limit"] = str(rate_info["limit"])
        response.headers["X-RateLimit-Remaining"] = str(rate_info["remaining"])
        response.headers["X-RateLimit-Reset"] = str(rate_info["reset"])

        return response


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
    app.add_middleware(
        RateLimitMiddleware,
        requests_per_minute=120,  # 2 requests per second average
        burst_size=20,  # Allow bursts up to 20 requests
    )
