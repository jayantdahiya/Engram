"""Memory management endpoints"""

import json
from datetime import datetime

from fastapi import APIRouter, HTTPException, Query, status
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession

from api.dependencies import AuthUserDep, DatabaseDep
from core.logging import logger
from models.memory import (
    ConversationTurn,
    ConversationTurnResponse,
    EmbeddingRequest,
    EmbeddingResponse,
    MemoryConsolidationRequest,
    MemoryConsolidationResponse,
    MemoryEntryCreate,
    MemoryEntryResponse,
    MemoryEntryUpdate,
    MemoryQuery,
    MemoryQueryResponse,
    MemoryStats,
)
from services.embedding_service import embedding_service
from services.memory_manager import memory_manager

router = APIRouter()


@router.post("/", response_model=MemoryEntryResponse)
async def create_memory(
    memory_data: MemoryEntryCreate,
    current_user: str = AuthUserDep,
    db_session: AsyncSession = DatabaseDep,
):
    """Create a new memory entry"""

    try:
        # Generate embedding
        embedding = await embedding_service.get_embedding(memory_data.text)

        # Create memory entry
        memory_id = await memory_manager._add_memory(
            memory_data.text,
            embedding,
            current_user,
            memory_data.conversation_id,
            db_session,
            metadata=memory_data.metadata,
        )

        # Get created memory
        result = await db_session.execute(
            text("""
            SELECT id, text, embedding, user_id, conversation_id, timestamp,
                   importance_score, access_count, metadata
            FROM memories
            WHERE id = :memory_id
            """),
            {"memory_id": memory_id},
        )

        memory = result.fetchone()

        # Parse metadata and embedding
        metadata = memory.metadata
        if isinstance(metadata, str):
            try:
                metadata = json.loads(metadata)
            except json.JSONDecodeError:
                metadata = {}

        embedding = memory.embedding
        if isinstance(embedding, str):
            try:
                embedding = json.loads(embedding)
            except json.JSONDecodeError:
                embedding = []

        return MemoryEntryResponse(
            id=memory.id,
            text=memory.text,
            user_id=str(memory.user_id),
            conversation_id=str(memory.conversation_id) if memory.conversation_id else None,
            timestamp=memory.timestamp,
            importance_score=memory.importance_score,
            access_count=memory.access_count,
            metadata=metadata or {},
            embedding_dimension=len(embedding),
        )

    except Exception as e:
        logger.error(f"Memory creation failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Failed to create memory"
        )


@router.get("/{memory_id}", response_model=MemoryEntryResponse)
async def get_memory(
    memory_id: int, current_user: str = AuthUserDep, db_session: AsyncSession = DatabaseDep
):
    """Get a specific memory entry"""

    try:
        result = await db_session.execute(
            text("""
            SELECT id, text, embedding, user_id, conversation_id, timestamp,
                   importance_score, access_count, metadata
            FROM memories
            WHERE id = :memory_id AND user_id = :user_id
            """),
            {"memory_id": memory_id, "user_id": current_user},
        )

        memory = result.fetchone()

        if not memory:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Memory not found")

        # Update access count
        await memory_manager._update_access_count(memory_id, db_session)

        # Parse metadata and embedding
        metadata = memory.metadata
        if isinstance(metadata, str):
            try:
                metadata = json.loads(metadata)
            except json.JSONDecodeError:
                metadata = {}

        embedding = memory.embedding
        if isinstance(embedding, str):
            try:
                embedding = json.loads(embedding)
            except json.JSONDecodeError:
                embedding = []

        return MemoryEntryResponse(
            id=memory.id,
            text=memory.text,
            user_id=str(memory.user_id),
            conversation_id=str(memory.conversation_id) if memory.conversation_id else None,
            timestamp=memory.timestamp,
            importance_score=memory.importance_score,
            access_count=memory.access_count + 1,
            metadata=metadata or {},
            embedding_dimension=len(embedding),
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Get memory failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Failed to get memory"
        )


@router.put("/{memory_id}", response_model=MemoryEntryResponse)
async def update_memory(
    memory_id: int,
    memory_update: MemoryEntryUpdate,
    current_user: str = AuthUserDep,
    db_session: AsyncSession = DatabaseDep,
):
    """Update a memory entry"""

    try:
        # Check if memory exists and belongs to user
        result = await db_session.execute(
            text("SELECT id FROM memories WHERE id = :memory_id AND user_id = :user_id"),
            {"memory_id": memory_id, "user_id": current_user},
        )

        if not result.fetchone():
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Memory not found")

        # Prepare update data
        update_data = {}
        if memory_update.text is not None:
            update_data["text"] = memory_update.text
            # Generate new embedding if text changed
            embedding = await embedding_service.get_embedding(memory_update.text)
            update_data["embedding"] = str(embedding.tolist())

        if memory_update.importance_score is not None:
            update_data["importance_score"] = memory_update.importance_score

        if memory_update.metadata is not None:
            update_data["metadata"] = memory_update.metadata

        if update_data:
            update_data["timestamp"] = datetime.utcnow()

            # Build update query
            set_clauses = [f"{key} = :{key}" for key in update_data.keys()]
            query = f"""
                UPDATE memories 
                SET {', '.join(set_clauses)}
                WHERE id = :memory_id AND user_id = :user_id
            """

            update_data.update({"memory_id": memory_id, "user_id": current_user})

            await db_session.execute(text(query), update_data)
            await db_session.commit()

        # Return updated memory
        return await get_memory(memory_id, current_user, db_session)

    except HTTPException:
        raise
    except Exception as e:
        await db_session.rollback()
        logger.error(f"Memory update failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Failed to update memory"
        )


@router.delete("/{memory_id}")
async def delete_memory(
    memory_id: int, current_user: str = AuthUserDep, db_session: AsyncSession = DatabaseDep
):
    """Delete a memory entry"""

    try:
        result = await db_session.execute(
            text("DELETE FROM memories WHERE id = :memory_id AND user_id = :user_id"),
            {"memory_id": memory_id, "user_id": current_user},
        )

        if result.rowcount == 0:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Memory not found")

        await db_session.commit()

        logger.info(f"Memory {memory_id} deleted for user {current_user}")

        return {"message": "Memory deleted successfully"}

    except HTTPException:
        raise
    except Exception as e:
        await db_session.rollback()
        logger.error(f"Memory deletion failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Failed to delete memory"
        )


@router.post("/query", response_model=MemoryQueryResponse)
async def query_memories(
    query: MemoryQuery, current_user: str = AuthUserDep, db_session: AsyncSession = DatabaseDep
):
    """Query memories using ACAN retrieval system"""

    try:
        # Ensure user can only query their own memories
        if query.user_id != current_user:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN, detail="Cannot query other users' memories"
            )

        # Use memory manager to retrieve memories
        result = await memory_manager.retrieve_memories(query, db_session)

        return result

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Memory query failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Failed to query memories"
        )


@router.post("/process-turn", response_model=ConversationTurnResponse)
async def process_conversation_turn(
    turn: ConversationTurn, current_user: str = AuthUserDep, db_session: AsyncSession = DatabaseDep
):
    """Process a conversation turn and update memories"""

    try:
        # Ensure user can only process their own conversations
        if turn.user_id != current_user:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Cannot process other users' conversations",
            )

        # Process turn using memory manager
        result = await memory_manager.process_conversation_turn(turn, db_session)

        return result

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Conversation turn processing failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to process conversation turn",
        )


@router.get("/", response_model=list[MemoryEntryResponse])
async def list_memories(
    current_user: str = AuthUserDep,
    db_session: AsyncSession = DatabaseDep,
    limit: int = Query(50, ge=1, le=100),
    offset: int = Query(0, ge=0),
    conversation_id: str | None = Query(None),
):
    """List user's memories with pagination"""

    try:
        query = """
            SELECT id, text, embedding, user_id, conversation_id, timestamp, 
                   importance_score, access_count, metadata
            FROM memories 
            WHERE user_id = :user_id
        """
        params = {"user_id": current_user, "limit": limit, "offset": offset}

        if conversation_id:
            query += " AND conversation_id = :conversation_id"
            params["conversation_id"] = conversation_id

        query += " ORDER BY timestamp DESC LIMIT :limit OFFSET :offset"

        result = await db_session.execute(text(query), params)
        memories = result.fetchall()

        results = []
        for mem in memories:
            metadata = mem.metadata
            if isinstance(metadata, str):
                try:
                    metadata = json.loads(metadata)
                except json.JSONDecodeError:
                    metadata = {}

            embedding = mem.embedding
            if isinstance(embedding, str):
                try:
                    embedding = json.loads(embedding)
                except json.JSONDecodeError:
                    embedding = []

            results.append(
                MemoryEntryResponse(
                    id=mem.id,
                    text=mem.text,
                    user_id=str(mem.user_id),
                    conversation_id=str(mem.conversation_id) if mem.conversation_id else None,
                    timestamp=mem.timestamp,
                    importance_score=mem.importance_score,
                    access_count=mem.access_count,
                    metadata=metadata or {},
                    embedding_dimension=len(embedding),
                )
            )

        return results

    except Exception as e:
        logger.error(f"List memories failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Failed to list memories"
        )


@router.get("/stats/overview", response_model=MemoryStats)
async def get_memory_stats(current_user: str = AuthUserDep, db_session: AsyncSession = DatabaseDep):
    """Get memory statistics for the current user"""

    try:
        # Get total memories
        total_result = await db_session.execute(
            text("SELECT COUNT(*) FROM memories WHERE user_id = :user_id"),
            {"user_id": current_user},
        )
        total_memories = total_result.scalar()

        # Get average importance
        avg_result = await db_session.execute(
            text("SELECT AVG(importance_score) FROM memories WHERE user_id = :user_id"),
            {"user_id": current_user},
        )
        avg_importance = avg_result.scalar() or 0.0

        # Get most accessed memories
        most_accessed_result = await db_session.execute(
            text("""
            SELECT id, text, embedding, user_id, conversation_id, timestamp,
                   importance_score, access_count, metadata
            FROM memories
            WHERE user_id = :user_id
            ORDER BY access_count DESC
            LIMIT 5
            """),
            {"user_id": current_user},
        )
        most_accessed = most_accessed_result.fetchall()

        # Get recent memories
        recent_result = await db_session.execute(
            text("""
            SELECT id, text, embedding, user_id, conversation_id, timestamp,
                   importance_score, access_count, metadata
            FROM memories
            WHERE user_id = :user_id
            ORDER BY timestamp DESC
            LIMIT 5
            """),
            {"user_id": current_user},
        )
        recent_memories = recent_result.fetchall()

        parsed_most_accessed = []
        for mem in most_accessed:
            metadata = mem.metadata
            if isinstance(metadata, str):
                try:
                    metadata = json.loads(metadata)
                except json.JSONDecodeError:
                    metadata = {}

            embedding = mem.embedding
            if isinstance(embedding, str):
                try:
                    embedding = json.loads(embedding)
                except json.JSONDecodeError:
                    embedding = []

            parsed_most_accessed.append(
                MemoryEntryResponse(
                    id=mem.id,
                    text=mem.text,
                    user_id=str(mem.user_id),
                    conversation_id=str(mem.conversation_id) if mem.conversation_id else None,
                    timestamp=mem.timestamp,
                    importance_score=mem.importance_score,
                    access_count=mem.access_count,
                    metadata=metadata or {},
                    embedding_dimension=len(embedding),
                )
            )

        parsed_recent = []
        for mem in recent_memories:
            metadata = mem.metadata
            if isinstance(metadata, str):
                try:
                    metadata = json.loads(metadata)
                except json.JSONDecodeError:
                    metadata = {}

            embedding = mem.embedding
            if isinstance(embedding, str):
                try:
                    embedding = json.loads(embedding)
                except json.JSONDecodeError:
                    embedding = []

            parsed_recent.append(
                MemoryEntryResponse(
                    id=mem.id,
                    text=mem.text,
                    user_id=str(mem.user_id),
                    conversation_id=str(mem.conversation_id) if mem.conversation_id else None,
                    timestamp=mem.timestamp,
                    importance_score=mem.importance_score,
                    access_count=mem.access_count,
                    metadata=metadata or {},
                    embedding_dimension=len(embedding),
                )
            )

        return MemoryStats(
            total_memories=total_memories,
            memories_by_user={current_user: total_memories},
            average_importance=float(avg_importance),
            most_accessed_memories=parsed_most_accessed,
            recent_memories=parsed_recent,
        )

    except Exception as e:
        logger.error(f"Get memory stats failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to get memory statistics",
        )


@router.post("/embedding", response_model=EmbeddingResponse)
async def generate_embedding(request: EmbeddingRequest, current_user: str = AuthUserDep):
    """Generate embedding for text"""

    try:
        result = await embedding_service.process_embedding_request(request)
        return result

    except Exception as e:
        logger.error(f"Embedding generation failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Failed to generate embedding"
        )


@router.post("/consolidate", response_model=MemoryConsolidationResponse)
async def consolidate_memories(
    request: MemoryConsolidationRequest,
    current_user: str = AuthUserDep,
    db_session: AsyncSession = DatabaseDep,
):
    """Consolidate similar memories for a user"""

    try:
        # Ensure user can only consolidate their own memories
        if request.user_id != current_user:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Cannot consolidate other users' memories",
            )

        # This would implement memory consolidation logic
        # For now, return a placeholder response
        return MemoryConsolidationResponse(
            user_id=current_user,
            memories_processed=0,
            memories_consolidated=0,
            consolidation_operations=[],
            processing_time_ms=0.0,
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Memory consolidation failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to consolidate memories",
        )
