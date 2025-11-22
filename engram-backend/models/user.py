"""Pydantic models for user management"""

from datetime import datetime

from pydantic import BaseModel, EmailStr, Field


class UserBase(BaseModel):
    """Base user model"""

    username: str = Field(..., min_length=3, max_length=50, description="Username")
    email: EmailStr = Field(..., description="User email address")
    full_name: str | None = Field(None, max_length=100, description="Full name")


class UserCreate(UserBase):
    """Model for creating a new user"""

    password: str = Field(..., min_length=8, description="User password")


class UserUpdate(BaseModel):
    """Model for updating user information"""

    username: str | None = Field(None, min_length=3, max_length=50, description="Updated username")
    email: EmailStr | None = Field(None, description="Updated email address")
    full_name: str | None = Field(None, max_length=100, description="Updated full name")
    is_active: bool | None = Field(None, description="User active status")


class UserResponse(UserBase):
    """Model for user response"""

    id: str = Field(..., description="User ID")
    is_active: bool = Field(default=True, description="User active status")
    created_at: datetime = Field(..., description="User creation timestamp")
    last_login: datetime | None = Field(None, description="Last login timestamp")
    memory_count: int = Field(default=0, description="Number of memories owned by user")

    class Config:
        from_attributes = True


class UserLogin(BaseModel):
    """Model for user login"""

    username: str = Field(..., description="Username or email")
    password: str = Field(..., description="User password")


class Token(BaseModel):
    """Model for authentication token"""

    access_token: str = Field(..., description="JWT access token")
    token_type: str = Field(default="bearer", description="Token type")
    expires_in: int = Field(..., description="Token expiration time in seconds")


class TokenData(BaseModel):
    """Model for token data"""

    username: str | None = Field(None, description="Username from token")
    user_id: str | None = Field(None, description="User ID from token")


class UserStats(BaseModel):
    """Model for user statistics"""

    user_id: str = Field(..., description="User ID")
    total_memories: int = Field(..., description="Total number of memories")
    conversations_count: int = Field(..., description="Number of conversations")
    last_activity: datetime | None = Field(None, description="Last activity timestamp")
    average_memory_importance: float = Field(..., description="Average memory importance score")
    most_used_memories: list[str] = Field(..., description="Most frequently accessed memory texts")
