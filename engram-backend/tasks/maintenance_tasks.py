"""Maintenance and cleanup tasks"""

import asyncio
from typing import Any

from celery import current_task
from sqlalchemy import text

from core.config import settings
from core.database import get_db_session, get_neo4j_session
from core.logging import logger
from tasks.celery_app import celery_app


@celery_app.task(bind=True)
def cleanup_old_memories(self) -> dict[str, Any]:
    """Clean up old, low-importance memories"""

    try:
        current_task.update_state(
            state="PROGRESS",
            meta={"current": 0, "total": 100, "status": "Starting memory cleanup..."},
        )

        # Run async function
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)

        try:
            result = loop.run_until_complete(_cleanup_old_memories_async(self))
            return result
        finally:
            loop.close()

    except Exception as exc:
        logger.error(f"Memory cleanup failed: {exc}")
        current_task.update_state(
            state="FAILURE", meta={"current": 100, "total": 100, "status": str(exc)}
        )
        raise


async def _cleanup_old_memories_async(celery_task) -> dict[str, Any]:
    """Async implementation of memory cleanup"""

    async with get_db_session() as db_session:
        try:
            # Get users with too many memories
            celery_task.update_state(
                state="PROGRESS",
                meta={
                    "current": 20,
                    "total": 100,
                    "status": "Finding users with excess memories...",
                },
            )

            users_result = await db_session.execute(
                text("""
                SELECT user_id, COUNT(*) as memory_count
                FROM memories
                GROUP BY user_id
                HAVING COUNT(*) > :max_memories
                """),
                {"max_memories": settings.max_memories_per_user},
            )

            users_to_cleanup = users_result.fetchall()

            if not users_to_cleanup:
                return {
                    "current": 100,
                    "total": 100,
                    "status": "No cleanup needed",
                    "users_processed": 0,
                    "memories_deleted": 0,
                }

            # Clean up memories for each user
            celery_task.update_state(
                state="PROGRESS",
                meta={"current": 40, "total": 100, "status": "Cleaning up memories..."},
            )

            total_deleted = 0
            users_processed = 0

            for user_data in users_to_cleanup:
                user_id = user_data.user_id
                memory_count = user_data.memory_count

                # Calculate how many to delete (10% of excess)
                excess = memory_count - settings.max_memories_per_user
                to_delete = max(1, int(excess * 0.1))

                # Delete oldest, lowest importance memories
                delete_result = await db_session.execute(
                    text("""
                    DELETE FROM memories
                    WHERE user_id = :user_id
                    AND id IN (
                        SELECT id FROM memories
                        WHERE user_id = :user_id
                        ORDER BY importance_score ASC, timestamp ASC
                        LIMIT :to_delete
                    )
                    """),
                    {"user_id": user_id, "to_delete": to_delete},
                )

                deleted_count = delete_result.rowcount
                total_deleted += deleted_count
                users_processed += 1

                logger.info(f"Cleaned up {deleted_count} memories for user {user_id}")

            await db_session.commit()

            # Complete
            celery_task.update_state(
                state="SUCCESS",
                meta={
                    "current": 100,
                    "total": 100,
                    "status": "Memory cleanup completed",
                    "users_processed": users_processed,
                    "memories_deleted": total_deleted,
                },
            )

            return {
                "current": 100,
                "total": 100,
                "status": "Memory cleanup completed",
                "users_processed": users_processed,
                "memories_deleted": total_deleted,
            }

        except Exception as e:
            await db_session.rollback()
            logger.error(f"Memory cleanup failed: {e}")
            raise


@celery_app.task(bind=True)
def optimize_embeddings(self) -> dict[str, Any]:
    """Optimize embedding storage and indexing"""

    try:
        current_task.update_state(
            state="PROGRESS",
            meta={"current": 0, "total": 100, "status": "Starting embedding optimization..."},
        )

        # Run async function
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)

        try:
            result = loop.run_until_complete(_optimize_embeddings_async(self))
            return result
        finally:
            loop.close()

    except Exception as exc:
        logger.error(f"Embedding optimization failed: {exc}")
        current_task.update_state(
            state="FAILURE", meta={"current": 100, "total": 100, "status": str(exc)}
        )
        raise


async def _optimize_embeddings_async(celery_task) -> dict[str, Any]:
    """Async implementation of embedding optimization"""

    async with get_db_session() as db_session:
        try:
            # Analyze embedding table
            celery_task.update_state(
                state="PROGRESS",
                meta={"current": 20, "total": 100, "status": "Analyzing embedding storage..."},
            )

            # Get embedding statistics
            stats_result = await db_session.execute(
                text("""
                SELECT
                    COUNT(*) as total_embeddings,
                    AVG(array_length(embedding, 1)) as avg_dimension,
                    MIN(array_length(embedding, 1)) as min_dimension,
                    MAX(array_length(embedding, 1)) as max_dimension
                FROM memories
                WHERE embedding IS NOT NULL
                """)
            )

            stats = stats_result.fetchone()

            # Check for inconsistent dimensions
            celery_task.update_state(
                state="PROGRESS",
                meta={"current": 40, "total": 100, "status": "Checking embedding consistency..."},
            )

            inconsistent_result = await db_session.execute(
                text("""
                SELECT COUNT(*)
                FROM memories
                WHERE array_length(embedding, 1) != :expected_dimension
                """),
                {"expected_dimension": settings.embedding_dimension},
            )

            inconsistent_count = inconsistent_result.scalar()

            # Re-generate embeddings for inconsistent ones
            if inconsistent_count > 0:
                celery_task.update_state(
                    state="PROGRESS",
                    meta={
                        "current": 60,
                        "total": 100,
                        "status": "Regenerating inconsistent embeddings...",
                    },
                )

                from services.embedding_service import embedding_service

                # Get memories with inconsistent embeddings
                inconsistent_memories = await db_session.execute(
                    text("""
                    SELECT id, text
                    FROM memories
                    WHERE array_length(embedding, 1) != :expected_dimension
                    LIMIT 100
                    """),
                    {"expected_dimension": settings.embedding_dimension},
                )

                regenerated_count = 0
                for memory in inconsistent_memories:
                    try:
                        # Generate new embedding
                        new_embedding = await embedding_service.get_embedding(memory.text)

                        # Update memory
                        await db_session.execute(
                            text("UPDATE memories SET embedding = :embedding WHERE id = :id"),
                            {"embedding": str(new_embedding.tolist()), "id": memory.id},
                        )

                        regenerated_count += 1

                    except Exception as e:
                        logger.error(f"Failed to regenerate embedding for memory {memory.id}: {e}")

                await db_session.commit()

            # Update database statistics
            celery_task.update_state(
                state="PROGRESS",
                meta={"current": 80, "total": 100, "status": "Updating database statistics..."},
            )

            await db_session.execute(text("ANALYZE memories"))

            # Complete
            celery_task.update_state(
                state="SUCCESS",
                meta={
                    "current": 100,
                    "total": 100,
                    "status": "Embedding optimization completed",
                    "total_embeddings": stats.total_embeddings,
                    "inconsistent_found": inconsistent_count,
                    "regenerated": regenerated_count,
                },
            )

            return {
                "current": 100,
                "total": 100,
                "status": "Embedding optimization completed",
                "total_embeddings": stats.total_embeddings,
                "inconsistent_found": inconsistent_count,
                "regenerated": regenerated_count,
            }

        except Exception as e:
            await db_session.rollback()
            logger.error(f"Embedding optimization failed: {e}")
            raise


@celery_app.task(bind=True)
def generate_memory_summaries(self) -> dict[str, Any]:
    """Generate memory summaries for active users"""

    try:
        current_task.update_state(
            state="PROGRESS",
            meta={"current": 0, "total": 100, "status": "Starting summary generation..."},
        )

        # Run async function
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)

        try:
            result = loop.run_until_complete(_generate_memory_summaries_async(self))
            return result
        finally:
            loop.close()

    except Exception as exc:
        logger.error(f"Memory summary generation failed: {exc}")
        current_task.update_state(
            state="FAILURE", meta={"current": 100, "total": 100, "status": str(exc)}
        )
        raise


async def _generate_memory_summaries_async(celery_task) -> dict[str, Any]:
    """Async implementation of memory summary generation"""

    async with get_db_session() as db_session:
        try:
            # Get active users (users with recent activity)
            celery_task.update_state(
                state="PROGRESS",
                meta={"current": 20, "total": 100, "status": "Finding active users..."},
            )

            active_users_result = await db_session.execute(
                text("""
                SELECT DISTINCT user_id
                FROM memories
                WHERE timestamp > NOW() - INTERVAL '7 days'
                GROUP BY user_id
                HAVING COUNT(*) >= 10
                LIMIT 50
                """)
            )

            active_users = [row.user_id for row in active_users_result.fetchall()]

            if not active_users:
                return {
                    "current": 100,
                    "total": 100,
                    "status": "No active users found",
                    "summaries_generated": 0,
                }

            # Generate summaries for each user
            celery_task.update_state(
                state="PROGRESS",
                meta={"current": 40, "total": 100, "status": "Generating summaries..."},
            )

            from services.llm_service import llm_service

            summaries_generated = 0

            for i, user_id in enumerate(active_users):
                try:
                    # Get user's recent memories
                    memories_result = await db_session.execute(
                        text("""
                        SELECT text
                        FROM memories
                        WHERE user_id = :user_id
                        ORDER BY timestamp DESC
                        LIMIT 50
                        """),
                        {"user_id": user_id},
                    )

                    memories = [row.text for row in memories_result.fetchall()]

                    if memories:
                        # Generate summary
                        summary = await llm_service.generate_memory_summary(memories, user_id)

                        # Store summary (you might want to create a summaries table)
                        # For now, we'll just log it
                        logger.info(f"Generated summary for user {user_id}: {summary[:100]}...")

                        summaries_generated += 1

                    # Update progress
                    progress = 40 + (i / len(active_users)) * 50
                    celery_task.update_state(
                        state="PROGRESS",
                        meta={
                            "current": int(progress),
                            "total": 100,
                            "status": f"Generated {i + 1}/{len(active_users)} summaries...",
                        },
                    )

                except Exception as e:
                    logger.error(f"Failed to generate summary for user {user_id}: {e}")

            # Complete
            celery_task.update_state(
                state="SUCCESS",
                meta={
                    "current": 100,
                    "total": 100,
                    "status": "Summary generation completed",
                    "active_users": len(active_users),
                    "summaries_generated": summaries_generated,
                },
            )

            return {
                "current": 100,
                "total": 100,
                "status": "Summary generation completed",
                "active_users": len(active_users),
                "summaries_generated": summaries_generated,
            }

        except Exception as e:
            logger.error(f"Memory summary generation failed: {e}")
            raise


@celery_app.task(bind=True)
def cleanup_graph_entities(self) -> dict[str, Any]:
    """Clean up old, unused entities in Neo4j"""

    try:
        current_task.update_state(
            state="PROGRESS",
            meta={"current": 0, "total": 100, "status": "Starting graph cleanup..."},
        )

        # Run async function
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)

        try:
            result = loop.run_until_complete(_cleanup_graph_entities_async(self))
            return result
        finally:
            loop.close()

    except Exception as exc:
        logger.error(f"Graph cleanup failed: {exc}")
        current_task.update_state(
            state="FAILURE", meta={"current": 100, "total": 100, "status": str(exc)}
        )
        raise


async def _cleanup_graph_entities_async(celery_task) -> dict[str, Any]:
    """Async implementation of graph cleanup"""

    async with get_neo4j_session() as neo4j_session:
        try:
            from services.graph_service import graph_service

            # Get all users
            celery_task.update_state(
                state="PROGRESS",
                meta={"current": 20, "total": 100, "status": "Getting users for cleanup..."},
            )

            users_result = await neo4j_session.run(
                "MATCH (e:Entity) RETURN DISTINCT e.user_id as user_id"
            )

            users = [record["user_id"] async for record in users_result]

            total_cleaned = 0

            for i, user_id in enumerate(users):
                try:
                    # Clean up old entities for this user
                    cleaned_count = await graph_service.cleanup_old_entities(
                        user_id, neo4j_session, days_threshold=90
                    )

                    total_cleaned += cleaned_count

                    # Update progress
                    progress = 20 + (i / len(users)) * 70
                    celery_task.update_state(
                        state="PROGRESS",
                        meta={
                            "current": int(progress),
                            "total": 100,
                            "status": f"Cleaned {i + 1}/{len(users)} users...",
                        },
                    )

                except Exception as e:
                    logger.error(f"Failed to cleanup entities for user {user_id}: {e}")

            # Complete
            celery_task.update_state(
                state="SUCCESS",
                meta={
                    "current": 100,
                    "total": 100,
                    "status": "Graph cleanup completed",
                    "users_processed": len(users),
                    "entities_cleaned": total_cleaned,
                },
            )

            return {
                "current": 100,
                "total": 100,
                "status": "Graph cleanup completed",
                "users_processed": len(users),
                "entities_cleaned": total_cleaned,
            }

        except Exception as e:
            logger.error(f"Graph cleanup failed: {e}")
            raise


@celery_app.task(bind=True)
def health_check_all_services(self) -> dict[str, Any]:
    """Comprehensive health check of all services"""

    try:
        current_task.update_state(
            state="PROGRESS",
            meta={"current": 0, "total": 100, "status": "Starting health checks..."},
        )

        # Run async function
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)

        try:
            result = loop.run_until_complete(_health_check_all_services_async(self))
            return result
        finally:
            loop.close()

    except Exception as exc:
        logger.error(f"Health check failed: {exc}")
        current_task.update_state(
            state="FAILURE", meta={"current": 100, "total": 100, "status": str(exc)}
        )
        raise


async def _health_check_all_services_async(celery_task) -> dict[str, Any]:
    """Async implementation of comprehensive health check"""

    health_status = {
        "postgresql": False,
        "neo4j": False,
        "redis": False,
        "openai": False,
        "embedding_service": False,
    }

    try:
        # Check PostgreSQL
        celery_task.update_state(
            state="PROGRESS", meta={"current": 20, "total": 100, "status": "Checking PostgreSQL..."}
        )

        from core.database import check_postgres_health

        health_status["postgresql"] = await check_postgres_health()

        # Check Neo4j
        celery_task.update_state(
            state="PROGRESS", meta={"current": 40, "total": 100, "status": "Checking Neo4j..."}
        )

        from core.database import check_neo4j_health

        health_status["neo4j"] = await check_neo4j_health()

        # Check Redis
        celery_task.update_state(
            state="PROGRESS", meta={"current": 60, "total": 100, "status": "Checking Redis..."}
        )

        from core.database import check_redis_health

        health_status["redis"] = await check_redis_health()

        # Check OpenAI API
        celery_task.update_state(
            state="PROGRESS", meta={"current": 80, "total": 100, "status": "Checking OpenAI API..."}
        )

        try:
            from services.llm_service import llm_service

            # Simple test call
            response = await llm_service.generate_response(
                [{"role": "user", "content": "Hello"}], temperature=0.1, max_tokens=10
            )
            health_status["openai"] = bool(response)
        except Exception as e:
            logger.error(f"OpenAI health check failed: {e}")
            health_status["openai"] = False

        # Check embedding service
        celery_task.update_state(
            state="PROGRESS",
            meta={"current": 90, "total": 100, "status": "Checking embedding service..."},
        )

        try:
            from services.embedding_service import embedding_service

            test_embedding = await embedding_service.get_embedding("test")
            health_status["embedding_service"] = len(test_embedding) > 0
        except Exception as e:
            logger.error(f"Embedding service health check failed: {e}")
            health_status["embedding_service"] = False

        # Complete
        all_healthy = all(health_status.values())

        celery_task.update_state(
            state="SUCCESS" if all_healthy else "FAILURE",
            meta={
                "current": 100,
                "total": 100,
                "status": "Health check completed",
                "health_status": health_status,
                "all_healthy": all_healthy,
            },
        )

        return {
            "current": 100,
            "total": 100,
            "status": "Health check completed",
            "health_status": health_status,
            "all_healthy": all_healthy,
        }

    except Exception as e:
        logger.error(f"Health check failed: {e}")
        raise
