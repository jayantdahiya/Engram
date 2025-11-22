"""Pydantic models for conversation management"""

from datetime import datetime
from typing import Any

from pydantic import BaseModel, Field


class ConversationBase(BaseModel):
    """Base conversation model"""

    title: str | None = Field(None, max_length=200, description="Conversation title")
    metadata: dict[str, Any] | None = Field(default_factory=dict, description="Additional metadata")


class ConversationCreate(ConversationBase):
    """Model for creating a new conversation"""

    user_id: str = Field(..., description="User ID who owns this conversation")


class ConversationUpdate(BaseModel):
    """Model for updating conversation"""

    title: str | None = Field(None, max_length=200, description="Updated conversation title")
    metadata: dict[str, Any] | None = Field(None, description="Updated metadata")


class ConversationResponse(ConversationBase):
    """Model for conversation response"""

    id: str = Field(..., description="Conversation ID")
    user_id: str = Field(..., description="User ID who owns this conversation")
    created_at: datetime = Field(..., description="Conversation creation timestamp")
    updated_at: datetime = Field(..., description="Last update timestamp")
    turn_count: int = Field(default=0, description="Number of turns in conversation")
    memory_count: int = Field(default=0, description="Number of memories from this conversation")

    class Config:
        from_attributes = True


class ConversationTurnBase(BaseModel):
    """Base conversation turn model"""

    user_message: str = Field(..., description="User message text")
    assistant_response: str | None = Field(None, description="Assistant response text")
    turn_number: int = Field(..., ge=1, description="Turn number in conversation")


class ConversationTurnCreate(ConversationTurnBase):
    """Model for creating a conversation turn"""

    conversation_id: str = Field(..., description="Conversation ID")
    user_id: str = Field(..., description="User ID")


class ConversationTurnResponse(ConversationTurnBase):
    """Model for conversation turn response"""

    id: str = Field(..., description="Turn ID")
    conversation_id: str = Field(..., description="Conversation ID")
    user_id: str = Field(..., description="User ID")
    timestamp: datetime = Field(..., description="Turn timestamp")
    memory_operations: list[str] = Field(
        default_factory=list, description="Memory operations performed"
    )
    processing_time_ms: float = Field(..., description="Processing time in milliseconds")

    class Config:
        from_attributes = True


class ConversationListResponse(BaseModel):
    """Model for conversation list response"""

    conversations: list[ConversationResponse] = Field(..., description="List of conversations")
    total_count: int = Field(..., description="Total number of conversations")
    page: int = Field(..., description="Current page number")
    page_size: int = Field(..., description="Number of conversations per page")


class ConversationDetailResponse(ConversationResponse):
    """Model for detailed conversation response"""

    turns: list[ConversationTurnResponse] = Field(..., description="Conversation turns")
    memories: list[dict[str, Any]] = Field(..., description="Memories from this conversation")
    summary: str | None = Field(None, description="Conversation summary")


class ConversationQuery(BaseModel):
    """Model for conversation queries"""

    user_id: str = Field(..., description="User ID to search conversations for")
    query: str | None = Field(None, description="Search query")
    limit: int = Field(
        default=20, ge=1, le=100, description="Maximum number of conversations to return"
    )
    offset: int = Field(default=0, ge=0, description="Number of conversations to skip")


class ConversationStats(BaseModel):
    """Model for conversation statistics"""

    total_conversations: int = Field(..., description="Total number of conversations")
    total_turns: int = Field(..., description="Total number of turns")
    average_turns_per_conversation: float = Field(..., description="Average turns per conversation")
    conversations_by_user: dict[str, int] = Field(..., description="Conversation count by user")
    recent_conversations: list[ConversationResponse] = Field(
        ..., description="Recently created conversations"
    )
