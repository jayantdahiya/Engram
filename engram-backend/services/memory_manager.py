"""Core memory manager service implementing Engram architecture"""

import json
import time
import uuid
from datetime import datetime
from typing import Any

import numpy as np
from sqlalchemy import text as sa_text
from sqlalchemy.ext.asyncio import AsyncSession

from core.config import settings
from core.database import get_neo4j_session
from core.logging import logger
from models.memory import (
    ConversationTurn,
    ConversationTurnResponse,
    MemoryQuery,
    MemoryQueryResponse,
)
from services.embedding_service import embedding_service
from services.graph_service import GraphService
from services.llm_service import llm_service


class MemoryManager:
    """Core memory manager implementing Engram architecture with ACAN retrieval"""

    def __init__(self):
        self.similarity_threshold = settings.similarity_threshold
        self.max_memories_per_user = settings.max_memories_per_user
        self.graph_service = GraphService()

    async def process_conversation_turn(
        self, turn: ConversationTurn, db_session: AsyncSession
    ) -> ConversationTurnResponse:
        """Process a conversation turn and update memories"""
        start_time = time.time()

        try:
            # Generate embedding for user message
            user_embedding = await embedding_service.get_embedding(turn.user_message)

            # Classify and execute memory operation
            operation_result = await self._classify_and_execute_operation(
                turn.user_message, user_embedding, turn.user_id, turn.conversation_id, db_session
            )

            # Extract entities for graph memory (Mem0g)
            await self._extract_and_store_entities(turn.user_message, turn.user_id, db_session)

            # Determine memory operations
            memory_ops = [operation_result["operation"]]

            # Initialize turn_id for response
            turn_id = str(uuid.uuid4())

            # Persist conversation turn only if conversation_id is provided
            if turn.conversation_id is not None:
                try:
                    # Get next turn number
                    turn_count_result = await db_session.execute(
                        sa_text(
                            "SELECT COUNT(*) FROM conversation_turns WHERE conversation_id = :conversation_id"
                        ),
                        {"conversation_id": turn.conversation_id},
                    )
                    next_turn_number = turn_count_result.scalar() + 1

                    await db_session.execute(
                        sa_text("""
                        INSERT INTO conversation_turns 
                        (id, conversation_id, user_id, user_message, assistant_response, 
                         turn_number, timestamp, memory_operations, processing_time_ms)
                        VALUES (:id, :conversation_id, :user_id, :user_message, :assistant_response,
                                :turn_number, :timestamp, :memory_operations, :processing_time_ms)
                        """),
                        {
                            "id": turn_id,
                            "conversation_id": turn.conversation_id,
                            "user_id": turn.user_id,
                            "user_message": turn.user_message,
                            "assistant_response": turn.assistant_response,
                            "turn_number": next_turn_number,
                            "timestamp": turn.timestamp or datetime.utcnow(),
                            "memory_operations": json.dumps(memory_ops),
                            "processing_time_ms": (time.time() - start_time) * 1000,
                        },
                    )

                    # Update conversation timestamp
                    await db_session.execute(
                        sa_text(
                            "UPDATE conversations SET updated_at = :updated_at WHERE id = :conversation_id"
                        ),
                        {"updated_at": datetime.utcnow(), "conversation_id": turn.conversation_id},
                    )

                    # Commit all changes (memory + turn)
                    await db_session.commit()

                except Exception as e:
                    logger.error(f"Failed to persist conversation turn: {e}")
                    # Don't fail the request if just persistence fails?
                    # Or should we? Probably yes for data integrity.
                    raise
            else:
                # No conversation context - just commit the memory changes
                await db_session.commit()

            processing_time = (time.time() - start_time) * 1000

            return ConversationTurnResponse(
                turn_id=turn_id,
                operation_performed=operation_result["operation"],
                memory_id=operation_result.get("memory_id"),
                processing_time_ms=processing_time,
                memories_affected=operation_result.get("memories_affected", 0),
            )

        except Exception as e:
            logger.error(f"Conversation turn processing failed: {e}")
            raise

    async def _classify_and_execute_operation(
        self,
        text: str,
        embedding: np.ndarray,
        user_id: str,
        conversation_id: str,
        db_session: AsyncSession,
    ) -> dict[str, Any]:
        """Classify memory operation and execute it"""

        # Get existing memories for this user
        existing_memories = await self._get_user_memories(user_id, db_session, limit=50)

        if not existing_memories:
            # No existing memories - ADD operation
            memory_id = await self._add_memory(
                text, embedding, user_id, conversation_id, db_session
            )
            return {"operation": "ADD", "memory_id": memory_id, "memories_affected": 1}

        # Use LLM to classify operation
        existing_texts = [mem.text for mem in existing_memories]
        classification = await llm_service.classify_memory_operation(
            text, existing_texts, f"User: {user_id}"
        )

        operation = classification["operation"]
        related_indices = classification["related_memory_indices"]

        if operation == "ADD":
            memory_id = await self._add_memory(
                text, embedding, user_id, conversation_id, db_session
            )
            return {"operation": "ADD", "memory_id": memory_id, "memories_affected": 1}

        elif operation == "UPDATE" and related_indices:
            # Update existing memory
            memory_to_update = existing_memories[related_indices[0]]
            await self._update_memory(memory_to_update.id, text, embedding, user_id, db_session)
            return {"operation": "UPDATE", "memory_id": memory_to_update.id, "memories_affected": 1}

        elif operation == "CONSOLIDATE" and related_indices:
            # Consolidate conflicting memories
            memory_to_consolidate = existing_memories[related_indices[0]]
            consolidated_text = await llm_service.consolidate_memories(
                memory_to_consolidate.text,
                text,
                memory_to_consolidate.timestamp.timestamp(),
                time.time(),
            )

            await self._consolidate_memory(
                memory_to_consolidate.id, consolidated_text, embedding, user_id, db_session
            )
            return {
                "operation": "CONSOLIDATE",
                "memory_id": memory_to_consolidate.id,
                "memories_affected": 1,
            }

        else:
            # NOOP - no change needed
            return {"operation": "NOOP", "memory_id": None, "memories_affected": 0}

    async def _add_memory(
        self,
        text: str,
        embedding: np.ndarray,
        user_id: str,
        conversation_id: str,
        db_session: AsyncSession,
    ) -> int:
        """Add new memory entry"""
        try:
            # Check memory limit
            memory_count = await self._get_user_memory_count(user_id, db_session)
            if memory_count >= self.max_memories_per_user:
                await self._cleanup_old_memories(user_id, db_session)

            # Create memory entry
            embedding_list = embedding.tolist()
            memory_data = {
                "text": text,
                "embedding": str(embedding_list),
                "user_id": user_id,
                "conversation_id": conversation_id,
                "timestamp": datetime.utcnow(),
                "importance_score": 0.0,
                "access_count": 0,
                "metadata": json.dumps({}),
            }

            # Insert into database (assuming you have a Memory model)
            # This would be implemented with your actual database model
            result = await db_session.execute(
                sa_text("""
                INSERT INTO memories (text, embedding, user_id, conversation_id, timestamp, importance_score, access_count, metadata)
                VALUES (:text, :embedding, :user_id, :conversation_id, :timestamp, :importance_score, :access_count, :metadata)
                RETURNING id
                """),
                memory_data,
            )

            memory_id = result.scalar()
            await db_session.commit()

            logger.info(f"Added memory {memory_id} for user {user_id}")
            return memory_id

        except Exception as e:
            await db_session.rollback()
            logger.error(f"Failed to add memory: {e}")
            raise

    async def _update_memory(
        self,
        memory_id: int,
        new_text: str,
        new_embedding: np.ndarray,
        user_id: str,
        db_session: AsyncSession,
    ):
        """Update existing memory with new information"""
        try:
            # Get existing memory
            result = await db_session.execute(
                sa_text(
                    "SELECT text, embedding FROM memories WHERE id = :memory_id AND user_id = :user_id"
                ),
                {"memory_id": memory_id, "user_id": user_id},
            )
            existing = result.fetchone()

            if not existing:
                raise ValueError(f"Memory {memory_id} not found for user {user_id}")

            # Parse embedding if it's a string (SQLite case)
            existing_embedding = existing.embedding
            if isinstance(existing_embedding, str):
                import json

                try:
                    existing_embedding = json.loads(existing_embedding)
                except json.JSONDecodeError:
                    existing_embedding = []

            # Concatenate new information
            updated_text = f"{existing.text}. {new_text}"
            updated_embedding = (np.array(existing_embedding) + new_embedding) / 2

            # Update memory
            await db_session.execute(
                sa_text("""
                UPDATE memories
                SET text = :text, embedding = :embedding, timestamp = :timestamp
                WHERE id = :memory_id AND user_id = :user_id
                """),
                {
                    "text": updated_text,
                    "embedding": str(updated_embedding.tolist()),
                    "timestamp": datetime.utcnow(),
                    "memory_id": memory_id,
                    "user_id": user_id,
                },
            )

            await db_session.commit()
            logger.info(f"Updated memory {memory_id} for user {user_id}")

        except Exception as e:
            await db_session.rollback()
            logger.error(f"Failed to update memory: {e}")
            raise

    async def _consolidate_memory(
        self,
        memory_id: int,
        consolidated_text: str,
        new_embedding: np.ndarray,
        user_id: str,
        db_session: AsyncSession,
    ):
        """Consolidate memory with temporal awareness"""
        try:
            # Get existing memory
            result = await db_session.execute(
                sa_text(
                    "SELECT embedding FROM memories WHERE id = :memory_id AND user_id = :user_id"
                ),
                {"memory_id": memory_id, "user_id": user_id},
            )
            existing = result.fetchone()

            if not existing:
                raise ValueError(f"Memory {memory_id} not found for user {user_id}")

            # Parse embedding if it's a string (SQLite case)
            existing_embedding = existing.embedding
            if isinstance(existing_embedding, str):
                import json

                try:
                    existing_embedding = json.loads(existing_embedding)
                except json.JSONDecodeError:
                    existing_embedding = []

            # Update with consolidated text
            updated_embedding = (np.array(existing_embedding) + new_embedding) / 2

            await db_session.execute(
                sa_text("""
                UPDATE memories
                SET text = :text, embedding = :embedding, timestamp = :timestamp
                WHERE id = :memory_id AND user_id = :user_id
                """),
                {
                    "text": consolidated_text,
                    "embedding": str(updated_embedding.tolist()),
                    "timestamp": datetime.utcnow(),
                    "memory_id": memory_id,
                    "user_id": user_id,
                },
            )

            await db_session.commit()
            logger.info(f"Consolidated memory {memory_id} for user {user_id}")

        except Exception as e:
            await db_session.rollback()
            logger.error(f"Failed to consolidate memory: {e}")
            raise

    async def retrieve_memories(
        self, query: MemoryQuery, db_session: AsyncSession
    ) -> MemoryQueryResponse:
        """Retrieve relevant memories using ACAN system"""
        start_time = time.time()

        try:
            # Generate query embedding
            query_embedding = await embedding_service.get_embedding(query.query)

            # Get user memories
            user_memories = await self._get_user_memories(query.user_id, db_session, limit=1000)

            if not user_memories:
                return MemoryQueryResponse(
                    query=query.query, memories=[], total_found=0, processing_time_ms=0
                )

            # Apply ACAN retrieval
            relevant_memories = await self._acan_retrieval(
                query_embedding,
                user_memories,
                query.top_k,
                query.similarity_threshold or self.similarity_threshold,
            )

            # Update access counts
            for memory in relevant_memories:
                await self._update_access_count(memory.id, db_session)

            processing_time = (time.time() - start_time) * 1000

            return MemoryQueryResponse(
                query=query.query,
                memories=relevant_memories,
                total_found=len(relevant_memories),
                processing_time_ms=processing_time,
            )

        except Exception as e:
            logger.error(f"Memory retrieval failed: {e}")
            raise

    async def _acan_retrieval(
        self,
        query_embedding: np.ndarray,
        memories: list[Any],
        top_k: int,
        similarity_threshold: float,
    ) -> list[Any]:
        """ACAN (Attention-based Context-Aware Network) retrieval"""

        if not memories:
            return []

        # Calculate composite scores
        attention_scores = await self._compute_attention_scores(
            query_embedding, [mem.embedding for mem in memories]
        )

        cosine_scores = np.array(
            [
                await embedding_service.calculate_similarity(
                    query_embedding, np.array(mem.embedding)
                )
                for mem in memories
            ]
        )

        current_time = time.time()
        recency_weights = np.array(
            [self._recency_weight(mem.timestamp.timestamp(), current_time) for mem in memories]
        )

        importance_scores = np.array([mem.importance_score for mem in memories])

        # Weighted combination
        composite_scores = (
            0.40 * attention_scores
            + 0.40 * cosine_scores
            + 0.10 * recency_weights
            + 0.10 * importance_scores
        )

        # Filter by threshold and get top-k
        valid_indices = np.where(composite_scores >= similarity_threshold)[0]
        if len(valid_indices) == 0:
            # If no memories meet threshold, return top-k anyway
            valid_indices = np.argsort(-composite_scores)[:top_k]

        sorted_indices = valid_indices[np.argsort(-composite_scores[valid_indices])]
        top_indices = sorted_indices[:top_k]

        return [memories[i] for i in top_indices]

    async def _compute_attention_scores(
        self, query_embedding: np.ndarray, memory_embeddings: list[list[float]]
    ) -> np.ndarray:
        """Compute cross-attention scores"""
        # Simplified attention mechanism
        similarities = []
        for mem_emb in memory_embeddings:
            similarity = await embedding_service.calculate_similarity(
                query_embedding, np.array(mem_emb)
            )
            similarities.append(similarity)

        similarities = np.array(similarities)
        # Softmax normalization
        exp_scores = np.exp(similarities - np.max(similarities))
        return exp_scores / np.sum(exp_scores)

    def _recency_weight(
        self, timestamp: float, current_time: float, half_life_hours: float = 72.0
    ) -> float:
        """Calculate recency weight with exponential decay"""
        age_hours = max(0.0, (current_time - timestamp) / 3600.0)
        return np.exp(-np.log(2) * age_hours / half_life_hours)

    async def _get_user_memories(
        self, user_id: str, db_session: AsyncSession, limit: int = 100
    ) -> list[Any]:
        """Get memories for a user"""
        result = await db_session.execute(
            sa_text("""
            SELECT id, text, embedding, timestamp, importance_score, access_count, metadata, conversation_id
            FROM memories
            WHERE user_id = :user_id
            ORDER BY timestamp DESC
            LIMIT :limit
            """),
            {"user_id": user_id, "limit": limit},
        )

        rows = result.fetchall()
        parsed_memories = []

        for row in rows:
            # Create a mutable object from the row
            mem = type("Memory", (), {})()
            mem.id = row.id
            mem.text = row.text
            mem.importance_score = row.importance_score
            mem.access_count = row.access_count
            mem.user_id = user_id  # Set required user_id
            mem.conversation_id = row.conversation_id  # Set conversation_id

            # Parse embedding
            if isinstance(row.embedding, str):
                import json

                try:
                    mem.embedding = json.loads(row.embedding)
                except json.JSONDecodeError:
                    # Handle case where it might be a string rep of list not valid json or just string
                    # For sqlite test it is str([1,2,3...]) which is valid python but maybe not json?
                    # valid json uses [1.0, 2.0]
                    # python str uses [1.0, 2.0].
                    # It should be fine.
                    mem.embedding = []
            else:
                mem.embedding = row.embedding or []

            mem.embedding_dimension = len(mem.embedding)  # Set calculated dimension

            # Parse timestamp
            if isinstance(row.timestamp, str):
                try:
                    mem.timestamp = datetime.fromisoformat(row.timestamp)
                except ValueError:
                    # Fallback or strict?
                    mem.timestamp = datetime.utcnow()
            else:
                mem.timestamp = row.timestamp

            # Parse metadata
            if isinstance(row.metadata, str):
                import json

                try:
                    mem.metadata = json.loads(row.metadata)
                except json.JSONDecodeError:
                    mem.metadata = {}
            else:
                mem.metadata = row.metadata or {}  # Ensure it is a dict if None

            parsed_memories.append(mem)

        return parsed_memories

    async def _get_user_memory_count(self, user_id: str, db_session: AsyncSession) -> int:
        """Get memory count for a user"""
        result = await db_session.execute(
            sa_text("SELECT COUNT(*) FROM memories WHERE user_id = :user_id"), {"user_id": user_id}
        )
        return result.scalar()

    async def _update_access_count(self, memory_id: int, db_session: AsyncSession):
        """Update memory access count and importance score"""
        await db_session.execute(
            sa_text("""
            UPDATE memories
            SET access_count = access_count + 1,
                importance_score = LOG(1 + access_count + 1)
            WHERE id = :memory_id
            """),
            {"memory_id": memory_id},
        )
        await db_session.commit()

    async def _cleanup_old_memories(self, user_id: str, db_session: AsyncSession):
        """Clean up old, low-importance memories"""
        # Remove oldest 10% of memories with lowest importance
        await db_session.execute(
            sa_text("""
            DELETE FROM memories
            WHERE user_id = :user_id
            AND id IN (
                SELECT id FROM memories
                WHERE user_id = :user_id
                ORDER BY importance_score ASC, timestamp ASC
                LIMIT (SELECT COUNT(*) * 0.1 FROM memories WHERE user_id = :user_id)
            )
            """),
            {"user_id": user_id},
        )
        await db_session.commit()

    async def _extract_and_store_entities(self, text: str, user_id: str, db_session: AsyncSession):
        """Extract entities and store in graph database"""
        try:
            entities_data = await llm_service.extract_entities_and_relations(text, user_id)

            async with get_neo4j_session() as neo4j_session:
                await self.graph_service.store_graph_memory(
                    entities_data["entities"],
                    entities_data["relationships"],
                    user_id,
                    neo4j_session,
                )

        except Exception as e:
            logger.error(f"Entity extraction failed: {e}")
            # Don't raise - this is not critical for memory operations


# Global memory manager instance
memory_manager = MemoryManager()
