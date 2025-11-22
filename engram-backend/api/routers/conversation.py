"""Conversation management endpoints"""

import uuid

from fastapi import APIRouter, HTTPException, Query, status
from sqlalchemy.ext.asyncio import AsyncSession

from api.dependencies import AuthUserDep, DatabaseDep
from core.logging import logger
from models.conversation import (
    ConversationCreate,
    ConversationDetailResponse,
    ConversationListResponse,
    ConversationResponse,
    ConversationStats,
    ConversationTurnCreate,
    ConversationTurnResponse,
    ConversationUpdate,
)

router = APIRouter()


@router.post("/", response_model=ConversationResponse)
async def create_conversation(
    conversation_data: ConversationCreate,
    current_user: str = AuthUserDep,
    db_session: AsyncSession = DatabaseDep,
):
    """Create a new conversation"""

    try:
        # Ensure user can only create conversations for themselves
        if conversation_data.user_id != current_user:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Cannot create conversations for other users",
            )

        conversation_id = str(uuid.uuid4())
        now = datetime.utcnow()

        await db_session.execute(
            """
            INSERT INTO conversations (id, user_id, title, metadata, created_at, updated_at)
            VALUES (:id, :user_id, :title, :metadata, :created_at, :updated_at)
            """,
            {
                "id": conversation_id,
                "user_id": conversation_data.user_id,
                "title": conversation_data.title,
                "metadata": conversation_data.metadata,
                "created_at": now,
                "updated_at": now,
            },
        )

        await db_session.commit()

        logger.info(f"Conversation {conversation_id} created for user {current_user}")

        return ConversationResponse(
            id=conversation_id,
            user_id=conversation_data.user_id,
            title=conversation_data.title,
            metadata=conversation_data.metadata,
            created_at=now,
            updated_at=now,
            turn_count=0,
            memory_count=0,
        )

    except HTTPException:
        raise
    except Exception as e:
        await db_session.rollback()
        logger.error(f"Conversation creation failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to create conversation",
        ) from e


@router.get("/{conversation_id}", response_model=ConversationDetailResponse)
async def get_conversation(
    conversation_id: str, current_user: str = AuthUserDep, db_session: AsyncSession = DatabaseDep
):
    """Get a specific conversation with turns and memories"""

    try:
        # Get conversation
        conv_result = await db_session.execute(
            """
            SELECT id, user_id, title, metadata, created_at, updated_at
            FROM conversations
            WHERE id = :conversation_id AND user_id = :user_id
            """,
            {"conversation_id": conversation_id, "user_id": current_user},
        )

        conversation = conv_result.fetchone()

        if not conversation:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND, detail="Conversation not found"
            )

        # Get turns
        turns_result = await db_session.execute(
            """
            SELECT id, conversation_id, user_id, user_message, assistant_response,
                   turn_number, timestamp, memory_operations, processing_time_ms
            FROM conversation_turns
            WHERE conversation_id = :conversation_id
            ORDER BY turn_number ASC
            """,
            {"conversation_id": conversation_id},
        )

        turns = turns_result.fetchall()

        # Get memories from this conversation
        memories_result = await db_session.execute(
            """
            SELECT id, text, timestamp, importance_score, access_count
            FROM memories
            WHERE conversation_id = :conversation_id AND user_id = :user_id
            ORDER BY timestamp DESC
            """,
            {"conversation_id": conversation_id, "user_id": current_user},
        )

        memories = memories_result.fetchall()

        # Get turn and memory counts
        turn_count = len(turns)
        memory_count = len(memories)

        return ConversationDetailResponse(
            id=conversation.id,
            user_id=conversation.user_id,
            title=conversation.title,
            metadata=conversation.metadata,
            created_at=conversation.created_at,
            updated_at=conversation.updated_at,
            turn_count=turn_count,
            memory_count=memory_count,
            turns=[
                ConversationTurnResponse(
                    id=turn.id,
                    conversation_id=turn.conversation_id,
                    user_id=turn.user_id,
                    user_message=turn.user_message,
                    assistant_response=turn.assistant_response,
                    turn_number=turn.turn_number,
                    timestamp=turn.timestamp,
                    memory_operations=turn.memory_operations or [],
                    processing_time_ms=turn.processing_time_ms,
                )
                for turn in turns
            ],
            memories=[
                {
                    "id": mem.id,
                    "text": mem.text,
                    "timestamp": mem.timestamp,
                    "importance_score": mem.importance_score,
                    "access_count": mem.access_count,
                }
                for mem in memories
            ],
            summary=None,  # Could be generated using LLM
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Get conversation failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Failed to get conversation"
        ) from e


@router.put("/{conversation_id}", response_model=ConversationResponse)
async def update_conversation(
    conversation_id: str,
    conversation_update: ConversationUpdate,
    current_user: str = AuthUserDep,
    db_session: AsyncSession = DatabaseDep,
):
    """Update a conversation"""

    try:
        # Check if conversation exists and belongs to user
        result = await db_session.execute(
            "SELECT id FROM conversations WHERE id = :conversation_id AND user_id = :user_id",
            {"conversation_id": conversation_id, "user_id": current_user},
        )

        if not result.fetchone():
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND, detail="Conversation not found"
            )

        # Prepare update data
        update_data = {"updated_at": datetime.utcnow()}

        if conversation_update.title is not None:
            update_data["title"] = conversation_update.title

        if conversation_update.metadata is not None:
            update_data["metadata"] = conversation_update.metadata

        # Build update query
        set_clauses = [f"{key} = :{key}" for key in update_data.keys()]
        query = f"""
            UPDATE conversations 
            SET {', '.join(set_clauses)}
            WHERE id = :conversation_id AND user_id = :user_id
        """

        update_data.update({"conversation_id": conversation_id, "user_id": current_user})

        await db_session.execute(query, update_data)
        await db_session.commit()

        # Return updated conversation
        return await get_conversation(conversation_id, current_user, db_session)

    except HTTPException:
        raise
    except Exception as e:
        await db_session.rollback()
        logger.error(f"Conversation update failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to update conversation",
        )


@router.delete("/{conversation_id}")
async def delete_conversation(
    conversation_id: str, current_user: str = AuthUserDep, db_session: AsyncSession = DatabaseDep
):
    """Delete a conversation and its associated data"""

    try:
        # Check if conversation exists and belongs to user
        result = await db_session.execute(
            "SELECT id FROM conversations WHERE id = :conversation_id AND user_id = :user_id",
            {"conversation_id": conversation_id, "user_id": current_user},
        )

        if not result.fetchone():
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND, detail="Conversation not found"
            )

        # Delete conversation turns
        await db_session.execute(
            "DELETE FROM conversation_turns WHERE conversation_id = :conversation_id",
            {"conversation_id": conversation_id},
        )

        # Delete memories associated with this conversation
        await db_session.execute(
            "DELETE FROM memories WHERE conversation_id = :conversation_id AND user_id = :user_id",
            {"conversation_id": conversation_id, "user_id": current_user},
        )

        # Delete conversation
        await db_session.execute(
            "DELETE FROM conversations WHERE id = :conversation_id AND user_id = :user_id",
            {"conversation_id": conversation_id, "user_id": current_user},
        )

        await db_session.commit()

        logger.info(f"Conversation {conversation_id} deleted for user {current_user}")

        return {"message": "Conversation deleted successfully"}

    except HTTPException:
        raise
    except Exception as e:
        await db_session.rollback()
        logger.error(f"Conversation deletion failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to delete conversation",
        )


@router.get("/", response_model=ConversationListResponse)
async def list_conversations(
    current_user: str = AuthUserDep,
    db_session: AsyncSession = DatabaseDep,
    page: int = Query(1, ge=1),
    page_size: int = Query(20, ge=1, le=100),
    search: str | None = Query(None),
):
    """List user's conversations with pagination and search"""

    try:
        offset = (page - 1) * page_size

        # Build query
        base_query = """
            SELECT c.id, c.user_id, c.title, c.metadata, c.created_at, c.updated_at,
                   COUNT(ct.id) as turn_count,
                   COUNT(m.id) as memory_count
            FROM conversations c
            LEFT JOIN conversation_turns ct ON c.id = ct.conversation_id
            LEFT JOIN memories m ON c.id = m.conversation_id
            WHERE c.user_id = :user_id
        """

        params = {"user_id": current_user, "limit": page_size, "offset": offset}

        if search:
            base_query += " AND (c.title ILIKE :search OR c.metadata::text ILIKE :search)"
            params["search"] = f"%{search}%"

        base_query += """
            GROUP BY c.id
            ORDER BY c.updated_at DESC
            LIMIT :limit OFFSET :offset
        """

        result = await db_session.execute(base_query, params)
        conversations = result.fetchall()

        # Get total count
        count_query = "SELECT COUNT(*) FROM conversations WHERE user_id = :user_id"
        if search:
            count_query += " AND (title ILIKE :search OR metadata::text ILIKE :search)"

        count_result = await db_session.execute(count_query, params)
        total_count = count_result.scalar()

        return ConversationListResponse(
            conversations=[
                ConversationResponse(
                    id=conv.id,
                    user_id=conv.user_id,
                    title=conv.title,
                    metadata=conv.metadata,
                    created_at=conv.created_at,
                    updated_at=conv.updated_at,
                    turn_count=conv.turn_count,
                    memory_count=conv.memory_count,
                )
                for conv in conversations
            ],
            total_count=total_count,
            page=page,
            page_size=page_size,
        )

    except Exception as e:
        logger.error(f"List conversations failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Failed to list conversations"
        )


@router.post("/{conversation_id}/turns", response_model=ConversationTurnResponse)
async def add_conversation_turn(
    conversation_id: str,
    turn_data: ConversationTurnCreate,
    current_user: str = AuthUserDep,
    db_session: AsyncSession = DatabaseDep,
):
    """Add a turn to a conversation"""

    try:
        # Ensure user can only add turns to their own conversations
        if turn_data.user_id != current_user:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Cannot add turns to other users' conversations",
            )

        # Check if conversation exists
        conv_result = await db_session.execute(
            "SELECT id FROM conversations WHERE id = :conversation_id AND user_id = :user_id",
            {"conversation_id": conversation_id, "user_id": current_user},
        )

        if not conv_result.fetchone():
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND, detail="Conversation not found"
            )

        # Get next turn number
        turn_count_result = await db_session.execute(
            "SELECT COUNT(*) FROM conversation_turns WHERE conversation_id = :conversation_id",
            {"conversation_id": conversation_id},
        )
        next_turn_number = turn_count_result.scalar() + 1

        # Create turn
        turn_id = str(uuid.uuid4())
        now = datetime.utcnow()

        await db_session.execute(
            """
            INSERT INTO conversation_turns 
            (id, conversation_id, user_id, user_message, assistant_response, 
             turn_number, timestamp, memory_operations, processing_time_ms)
            VALUES (:id, :conversation_id, :user_id, :user_message, :assistant_response,
                    :turn_number, :timestamp, :memory_operations, :processing_time_ms)
            """,
            {
                "id": turn_id,
                "conversation_id": conversation_id,
                "user_id": turn_data.user_id,
                "user_message": turn_data.user_message,
                "assistant_response": turn_data.assistant_response,
                "turn_number": next_turn_number,
                "timestamp": now,
                "memory_operations": [],
                "processing_time_ms": 0.0,
            },
        )

        # Update conversation timestamp
        await db_session.execute(
            "UPDATE conversations SET updated_at = :updated_at WHERE id = :conversation_id",
            {"updated_at": now, "conversation_id": conversation_id},
        )

        await db_session.commit()

        logger.info(f"Turn {turn_id} added to conversation {conversation_id}")

        return ConversationTurnResponse(
            id=turn_id,
            conversation_id=conversation_id,
            user_id=turn_data.user_id,
            user_message=turn_data.user_message,
            assistant_response=turn_data.assistant_response,
            turn_number=next_turn_number,
            timestamp=now,
            memory_operations=[],
            processing_time_ms=0.0,
        )

    except HTTPException:
        raise
    except Exception as e:
        await db_session.rollback()
        logger.error(f"Add conversation turn failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to add conversation turn",
        )


@router.get("/stats/overview", response_model=ConversationStats)
async def get_conversation_stats(
    current_user: str = AuthUserDep, db_session: AsyncSession = DatabaseDep
):
    """Get conversation statistics for the current user"""

    try:
        # Get total conversations
        conv_count_result = await db_session.execute(
            "SELECT COUNT(*) FROM conversations WHERE user_id = :user_id", {"user_id": current_user}
        )
        total_conversations = conv_count_result.scalar()

        # Get total turns
        turn_count_result = await db_session.execute(
            """
            SELECT COUNT(*) FROM conversation_turns ct
            JOIN conversations c ON ct.conversation_id = c.id
            WHERE c.user_id = :user_id
            """,
            {"user_id": current_user},
        )
        total_turns = turn_count_result.scalar()

        # Calculate average turns per conversation
        avg_turns = total_turns / max(total_conversations, 1)

        # Get recent conversations
        recent_result = await db_session.execute(
            """
            SELECT id, user_id, title, metadata, created_at, updated_at,
                   COUNT(ct.id) as turn_count,
                   COUNT(m.id) as memory_count
            FROM conversations c
            LEFT JOIN conversation_turns ct ON c.id = ct.conversation_id
            LEFT JOIN memories m ON c.id = m.conversation_id
            WHERE c.user_id = :user_id
            GROUP BY c.id
            ORDER BY c.updated_at DESC
            LIMIT 5
            """,
            {"user_id": current_user},
        )
        recent_conversations = recent_result.fetchall()

        return ConversationStats(
            total_conversations=total_conversations,
            total_turns=total_turns,
            average_turns_per_conversation=avg_turns,
            conversations_by_user={current_user: total_conversations},
            recent_conversations=[
                ConversationResponse(
                    id=conv.id,
                    user_id=conv.user_id,
                    title=conv.title,
                    metadata=conv.metadata,
                    created_at=conv.created_at,
                    updated_at=conv.updated_at,
                    turn_count=conv.turn_count,
                    memory_count=conv.memory_count,
                )
                for conv in recent_conversations
            ],
        )

    except Exception as e:
        logger.error(f"Get conversation stats failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to get conversation statistics",
        )


# Import required modules
from datetime import datetime
