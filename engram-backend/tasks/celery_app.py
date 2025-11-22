"""Celery application configuration"""

from celery import Celery

from core.config import settings
from core.logging import logger

# Create Celery application
celery_app = Celery(
    "engram_tasks",
    broker=settings.celery_broker_url,
    backend=settings.celery_result_backend,
    include=["tasks.memory_tasks", "tasks.maintenance_tasks"],
)

# Configure Celery
celery_app.conf.update(
    # Serialization
    task_serializer="json",
    accept_content=["json"],
    result_serializer="json",
    # Timezone
    timezone="UTC",
    enable_utc=True,
    # Task routing
    task_routes={
        "tasks.memory_tasks.*": {"queue": "memory"},
        "tasks.maintenance_tasks.*": {"queue": "maintenance"},
    },
    # Task execution
    task_acks_late=True,
    worker_prefetch_multiplier=1,
    # Result backend
    result_expires=3600,  # 1 hour
    # Task time limits
    task_soft_time_limit=300,  # 5 minutes
    task_time_limit=600,  # 10 minutes
    # Retry configuration
    task_default_retry_delay=60,  # 1 minute
    task_max_retries=3,
    # Monitoring
    worker_send_task_events=True,
    task_send_sent_event=True,
    # Beat schedule for periodic tasks
    beat_schedule={
        "cleanup-old-memories": {
            "task": "tasks.maintenance_tasks.cleanup_old_memories",
            "schedule": 3600.0,  # Every hour
        },
        "optimize-embeddings": {
            "task": "tasks.maintenance_tasks.optimize_embeddings",
            "schedule": 86400.0,  # Daily
        },
        "generate-memory-summaries": {
            "task": "tasks.maintenance_tasks.generate_memory_summaries",
            "schedule": 21600.0,  # Every 6 hours
        },
    },
)

# Configure logging
celery_app.conf.update(
    worker_log_format="[%(asctime)s: %(levelname)s/%(processName)s] %(message)s",
    worker_task_log_format="[%(asctime)s: %(levelname)s/%(processName)s][%(task_name)s(%(task_id)s)] %(message)s",
)

logger.info("Celery application configured successfully")


@celery_app.task(bind=True)
def debug_task(self):
    """Debug task to test Celery setup"""
    logger.info(f"Request: {self.request!r}")
    return f"Hello from Celery! Task ID: {self.request.id}"
