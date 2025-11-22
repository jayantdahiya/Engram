"""Pydantic models for memory operations"""

from datetime import datetime
from typing import Any

from pydantic import BaseModel, Field


class MemoryEntryBase(BaseModel):
    """Base memory entry model"""

    text: str = Field(..., description="Memory content text")
    importance_score: float = Field(
        default=0.0, ge=0.0, le=10.0, description="Memory importance score"
    )
    metadata: dict[str, Any] | None = Field(default_factory=dict, description="Additional metadata")


class MemoryEntryCreate(MemoryEntryBase):
    """Model for creating a new memory entry"""

    user_id: str = Field(..., description="User ID who owns this memory")
    conversation_id: str | None = Field(None, description="Conversation ID this memory belongs to")


class MemoryEntryUpdate(BaseModel):
    """Model for updating a memory entry"""

    text: str | None = Field(None, description="Updated memory content")
    importance_score: float | None = Field(
        None, ge=0.0, le=10.0, description="Updated importance score"
    )
    metadata: dict[str, Any] | None = Field(None, description="Updated metadata")


class MemoryEntryResponse(MemoryEntryBase):
    """Model for memory entry response"""

    id: int = Field(..., description="Memory entry ID")
    user_id: str = Field(..., description="User ID who owns this memory")
    conversation_id: str | None = Field(None, description="Conversation ID")
    timestamp: datetime = Field(..., description="Memory creation timestamp")
    access_count: int = Field(default=0, description="Number of times this memory was accessed")
    embedding_dimension: int = Field(..., description="Dimension of the embedding vector")

    class Config:
        from_attributes = True


class MemoryQuery(BaseModel):
    """Model for memory query requests"""

    query: str = Field(..., description="Query text to search memories")
    user_id: str = Field(..., description="User ID to search memories for")
    top_k: int = Field(default=5, ge=1, le=50, description="Number of top memories to retrieve")
    similarity_threshold: float | None = Field(
        None, ge=0.0, le=1.0, description="Minimum similarity threshold"
    )


class MemoryQueryResponse(BaseModel):
    """Model for memory query responses"""

    query: str = Field(..., description="Original query")
    memories: list[MemoryEntryResponse] = Field(..., description="Retrieved memories")
    total_found: int = Field(..., description="Total number of memories found")
    processing_time_ms: float = Field(..., description="Query processing time in milliseconds")


class MemoryOperation(BaseModel):
    """Model for memory operations (ADD/UPDATE/DELETE/NOOP)"""

    operation: str = Field(..., description="Operation type: ADD, UPDATE, DELETE, NOOP")
    memory_id: int | None = Field(None, description="Memory ID (for UPDATE/DELETE operations)")
    text: str = Field(..., description="Memory text content")
    timestamp: datetime = Field(..., description="Operation timestamp")
    user_id: str = Field(..., description="User ID")
    conversation_id: str | None = Field(None, description="Conversation ID")
    confidence: float = Field(default=1.0, ge=0.0, le=1.0, description="Operation confidence score")


class ConversationTurn(BaseModel):
    """Model for processing conversation turns"""

    user_message: str = Field(..., description="User message text")
    assistant_response: str | None = Field(None, description="Assistant response text")
    user_id: str = Field(..., description="User ID")
    conversation_id: str = Field(..., description="Conversation ID")
    timestamp: datetime | None = Field(
        default_factory=datetime.utcnow, description="Turn timestamp"
    )


class ConversationTurnResponse(BaseModel):
    """Model for conversation turn processing response"""

    turn_id: str = Field(..., description="Unique turn ID")
    operation_performed: str = Field(..., description="Memory operation performed")
    memory_id: int | None = Field(None, description="Memory ID if operation created/updated memory")
    processing_time_ms: float = Field(..., description="Processing time in milliseconds")
    memories_affected: int = Field(..., description="Number of memories affected by this turn")


class MemoryStats(BaseModel):
    """Model for memory statistics"""

    total_memories: int = Field(..., description="Total number of memories")
    memories_by_user: dict[str, int] = Field(..., description="Memory count by user")
    average_importance: float = Field(..., description="Average importance score")
    most_accessed_memories: list[MemoryEntryResponse] = Field(
        ..., description="Most frequently accessed memories"
    )
    recent_memories: list[MemoryEntryResponse] = Field(..., description="Recently created memories")


class EmbeddingRequest(BaseModel):
    """Model for embedding generation requests"""

    text: str = Field(..., description="Text to generate embedding for")
    model: str | None = Field(
        default="text-embedding-ada-002", description="Embedding model to use"
    )


class EmbeddingResponse(BaseModel):
    """Model for embedding responses"""

    text: str = Field(..., description="Original text")
    embedding: list[float] = Field(..., description="Generated embedding vector")
    model: str = Field(..., description="Model used for embedding")
    dimension: int = Field(..., description="Embedding dimension")
    tokens_used: int = Field(..., description="Number of tokens used")


class MemoryConsolidationRequest(BaseModel):
    """Model for memory consolidation requests"""

    user_id: str = Field(..., description="User ID to consolidate memories for")
    similarity_threshold: float = Field(
        default=0.8, ge=0.0, le=1.0, description="Similarity threshold for consolidation"
    )
    max_memories_to_consolidate: int = Field(
        default=100, ge=1, le=1000, description="Maximum memories to process"
    )


class MemoryConsolidationResponse(BaseModel):
    """Model for memory consolidation responses"""

    user_id: str = Field(..., description="User ID")
    memories_processed: int = Field(..., description="Number of memories processed")
    memories_consolidated: int = Field(..., description="Number of memories consolidated")
    consolidation_operations: list[MemoryOperation] = Field(
        ..., description="Consolidation operations performed"
    )
    processing_time_ms: float = Field(..., description="Processing time in milliseconds")
