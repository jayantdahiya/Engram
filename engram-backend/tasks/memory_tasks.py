"""Async memory processing tasks"""

import asyncio
from typing import Any

import numpy as np
from celery import current_task
from sqlalchemy import text

from core.database import get_db_session, get_neo4j_session
from core.logging import logger
from models.memory import ConversationTurn
from services.embedding_service import embedding_service
from services.llm_service import llm_service
from services.memory_manager import memory_manager
from tasks.celery_app import celery_app


@celery_app.task(bind=True)
def process_memory_extraction(
    self, user_id: str, message: str, conversation_id: str
) -> dict[str, Any]:
    """Async memory extraction task"""

    try:
        # Update task status
        current_task.update_state(
            state="PROGRESS", meta={"current": 0, "total": 100, "status": "Extracting memories..."}
        )

        # Run async function in event loop
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)

        try:
            result = loop.run_until_complete(
                _process_memory_extraction_async(user_id, message, conversation_id, self)
            )
            return result
        finally:
            loop.close()

    except Exception as exc:
        logger.error(f"Memory extraction task failed: {exc}")
        current_task.update_state(
            state="FAILURE", meta={"current": 100, "total": 100, "status": str(exc)}
        )
        raise


async def _process_memory_extraction_async(
    user_id: str, message: str, conversation_id: str, celery_task
) -> dict[str, Any]:
    """Async implementation of memory extraction"""

    async with get_db_session() as db_session:
        try:
            # Update progress
            celery_task.update_state(
                state="PROGRESS",
                meta={"current": 20, "total": 100, "status": "Generating embedding..."},
            )

            # Generate embedding (not used in this flow)
            await embedding_service.get_embedding(message)

            # Update progress
            celery_task.update_state(
                state="PROGRESS",
                meta={"current": 40, "total": 100, "status": "Classifying operation..."},
            )

            # Create conversation turn
            turn = ConversationTurn(
                user_message=message, user_id=user_id, conversation_id=conversation_id
            )

            # Process turn
            result = await memory_manager.process_conversation_turn(turn, db_session)

            # Update progress
            celery_task.update_state(
                state="PROGRESS",
                meta={"current": 80, "total": 100, "status": "Extracting entities..."},
            )

            # Extract entities for graph memory
            entities_data = await llm_service.extract_entities_and_relations(message, user_id)

            async with get_neo4j_session() as neo4j_session:
                from services.graph_service import graph_service

                await graph_service.store_graph_memory(
                    entities_data["entities"],
                    entities_data["relationships"],
                    user_id,
                    neo4j_session,
                )

            # Complete
            celery_task.update_state(
                state="SUCCESS",
                meta={
                    "current": 100,
                    "total": 100,
                    "status": "Memory processing completed",
                    "result": {
                        "operation": result.operation_performed,
                        "memory_id": result.memory_id,
                        "processing_time_ms": result.processing_time_ms,
                        "entities_extracted": len(entities_data["entities"]),
                        "relationships_extracted": len(entities_data["relationships"]),
                    },
                },
            )

            return {
                "current": 100,
                "total": 100,
                "status": "Memory processing completed",
                "result": {
                    "operation": result.operation_performed,
                    "memory_id": result.memory_id,
                    "processing_time_ms": result.processing_time_ms,
                    "entities_extracted": len(entities_data["entities"]),
                    "relationships_extracted": len(entities_data["relationships"]),
                },
            }

        except Exception as e:
            logger.error(f"Async memory extraction failed: {e}")
            raise


@celery_app.task(bind=True)
def batch_process_memories(
    self, user_id: str, messages: list[str], conversation_id: str
) -> dict[str, Any]:
    """Batch process multiple messages for memory extraction"""

    try:
        current_task.update_state(
            state="PROGRESS",
            meta={"current": 0, "total": len(messages), "status": "Starting batch processing..."},
        )

        # Run async function
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)

        try:
            result = loop.run_until_complete(
                _batch_process_memories_async(user_id, messages, conversation_id, self)
            )
            return result
        finally:
            loop.close()

    except Exception as exc:
        logger.error(f"Batch memory processing failed: {exc}")
        current_task.update_state(
            state="FAILURE",
            meta={"current": len(messages), "total": len(messages), "status": str(exc)},
        )
        raise


async def _batch_process_memories_async(
    user_id: str, messages: list[str], conversation_id: str, celery_task
) -> dict[str, Any]:
    """Async implementation of batch memory processing"""

    results = []
    total_messages = len(messages)

    async with get_db_session() as db_session:
        for i, message in enumerate(messages):
            try:
                # Update progress
                celery_task.update_state(
                    state="PROGRESS",
                    meta={
                        "current": i,
                        "total": total_messages,
                        "status": f"Processing message {i + 1}/{total_messages}...",
                    },
                )

                # Process individual message
                turn = ConversationTurn(
                    user_message=message, user_id=user_id, conversation_id=conversation_id
                )

                result = await memory_manager.process_conversation_turn(turn, db_session)
                results.append(
                    {
                        "message": message,
                        "operation": result.operation_performed,
                        "memory_id": result.memory_id,
                        "processing_time_ms": result.processing_time_ms,
                    }
                )

            except Exception as e:
                logger.error(f"Failed to process message {i + 1}: {e}")
                results.append({"message": message, "error": str(e), "operation": "FAILED"})

        # Complete
        celery_task.update_state(
            state="SUCCESS",
            meta={
                "current": total_messages,
                "total": total_messages,
                "status": "Batch processing completed",
                "results": results,
            },
        )

        return {
            "current": total_messages,
            "total": total_messages,
            "status": "Batch processing completed",
            "results": results,
        }


@celery_app.task(bind=True)
def consolidate_user_memories(
    self, user_id: str, similarity_threshold: float = 0.8
) -> dict[str, Any]:
    """Consolidate similar memories for a user"""

    try:
        current_task.update_state(
            state="PROGRESS",
            meta={"current": 0, "total": 100, "status": "Starting memory consolidation..."},
        )

        # Run async function
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)

        try:
            result = loop.run_until_complete(
                _consolidate_user_memories_async(user_id, similarity_threshold, self)
            )
            return result
        finally:
            loop.close()

    except Exception as exc:
        logger.error(f"Memory consolidation failed: {exc}")
        current_task.update_state(
            state="FAILURE", meta={"current": 100, "total": 100, "status": str(exc)}
        )
        raise


async def _consolidate_user_memories_async(
    user_id: str, similarity_threshold: float, celery_task
) -> dict[str, Any]:
    """Async implementation of memory consolidation"""

    async with get_db_session() as db_session:
        try:
            # Get user memories
            celery_task.update_state(
                state="PROGRESS",
                meta={"current": 20, "total": 100, "status": "Retrieving user memories..."},
            )

            memories = await memory_manager._get_user_memories(user_id, db_session, limit=1000)

            if len(memories) < 2:
                return {
                    "current": 100,
                    "total": 100,
                    "status": "No consolidation needed",
                    "memories_processed": len(memories),
                    "memories_consolidated": 0,
                }

            # Find similar memories
            celery_task.update_state(
                state="PROGRESS",
                meta={"current": 40, "total": 100, "status": "Finding similar memories..."},
            )

            consolidation_pairs = []
            memory_embeddings = [np.array(mem.embedding) for mem in memories]

            for i in range(len(memories)):
                for j in range(i + 1, len(memories)):
                    similarity = await embedding_service.calculate_similarity(
                        memory_embeddings[i], memory_embeddings[j]
                    )

                    if similarity >= similarity_threshold:
                        consolidation_pairs.append((i, j, similarity))

            # Consolidate memories
            celery_task.update_state(
                state="PROGRESS",
                meta={"current": 60, "total": 100, "status": "Consolidating memories..."},
            )

            consolidated_count = 0
            for i, j, _similarity in consolidation_pairs:
                try:
                    memory1 = memories[i]
                    memory2 = memories[j]

                    # Use LLM to consolidate
                    consolidated_text = await llm_service.consolidate_memories(
                        memory1.text,
                        memory2.text,
                        memory1.timestamp.timestamp(),
                        memory2.timestamp.timestamp(),
                    )

                    # Update first memory with consolidated text
                    new_embedding = await embedding_service.get_embedding(consolidated_text)
                    await memory_manager._consolidate_memory(
                        memory1.id, consolidated_text, new_embedding, user_id, db_session
                    )

                    # Delete second memory
                    await db_session.execute(
                        text("DELETE FROM memories WHERE id = :memory_id"), {"memory_id": memory2.id}
                    )

                    consolidated_count += 1

                except Exception as e:
                    logger.error(f"Failed to consolidate memories {i} and {j}: {e}")

            await db_session.commit()

            # Complete
            celery_task.update_state(
                state="SUCCESS",
                meta={
                    "current": 100,
                    "total": 100,
                    "status": "Memory consolidation completed",
                    "memories_processed": len(memories),
                    "memories_consolidated": consolidated_count,
                },
            )

            return {
                "current": 100,
                "total": 100,
                "status": "Memory consolidation completed",
                "memories_processed": len(memories),
                "memories_consolidated": consolidated_count,
            }

        except Exception as e:
            logger.error(f"Memory consolidation failed: {e}")
            raise


@celery_app.task(bind=True)
def generate_memory_summary(self, user_id: str, max_memories: int = 100) -> dict[str, Any]:
    """Generate a summary of user's memories"""

    try:
        current_task.update_state(
            state="PROGRESS",
            meta={"current": 0, "total": 100, "status": "Generating memory summary..."},
        )

        # Run async function
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)

        try:
            result = loop.run_until_complete(
                _generate_memory_summary_async(user_id, max_memories, self)
            )
            return result
        finally:
            loop.close()

    except Exception as exc:
        logger.error(f"Memory summary generation failed: {exc}")
        current_task.update_state(
            state="FAILURE", meta={"current": 100, "total": 100, "status": str(exc)}
        )
        raise


async def _generate_memory_summary_async(
    user_id: str, max_memories: int, celery_task
) -> dict[str, Any]:
    """Async implementation of memory summary generation"""

    async with get_db_session() as db_session:
        try:
            # Get user memories
            celery_task.update_state(
                state="PROGRESS",
                meta={"current": 30, "total": 100, "status": "Retrieving memories..."},
            )

            memories = await memory_manager._get_user_memories(
                user_id, db_session, limit=max_memories
            )

            if not memories:
                return {
                    "current": 100,
                    "total": 100,
                    "status": "No memories to summarize",
                    "summary": "No memories found for this user.",
                }

            # Generate summary
            celery_task.update_state(
                state="PROGRESS",
                meta={"current": 70, "total": 100, "status": "Generating summary..."},
            )

            memory_texts = [mem.text for mem in memories]
            summary = await llm_service.generate_memory_summary(memory_texts, user_id)

            # Complete
            celery_task.update_state(
                state="SUCCESS",
                meta={
                    "current": 100,
                    "total": 100,
                    "status": "Memory summary generated",
                    "summary": summary,
                    "memories_processed": len(memories),
                },
            )

            return {
                "current": 100,
                "total": 100,
                "status": "Memory summary generated",
                "summary": summary,
                "memories_processed": len(memories),
            }

        except Exception as e:
            logger.error(f"Memory summary generation failed: {e}")
            raise
