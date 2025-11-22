"""Authentication endpoints"""

import uuid
from datetime import datetime, timedelta

from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.security import OAuth2PasswordRequestForm
from sqlalchemy.ext.asyncio import AsyncSession

from api.dependencies import DatabaseDep
from core.config import settings
from core.logging import logger
from core.security import auth_service, get_current_user, password_service
from models.user import Token, UserCreate, UserResponse

router = APIRouter()


@router.post("/register", response_model=UserResponse)
async def register_user(user_data: UserCreate, db_session: AsyncSession = DatabaseDep):
    """Register a new user"""

    try:
        # Check if user already exists
        result = await db_session.execute(
            "SELECT id FROM users WHERE username = :username OR email = :email",
            {"username": user_data.username, "email": user_data.email},
        )

        if result.fetchone():
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Username or email already registered",
            )

        # Hash password
        hashed_password = password_service.hash_password(user_data.password)

        # Create user
        user_id = str(uuid.uuid4())
        await db_session.execute(
            """
            INSERT INTO users (
                id, username, email, full_name, hashed_password, is_active, created_at
            )
            VALUES (
                :id, :username, :email, :full_name, :hashed_password, :is_active, :created_at
            )
            """,
            {
                "id": user_id,
                "username": user_data.username,
                "email": user_data.email,
                "full_name": user_data.full_name,
                "hashed_password": hashed_password,
                "is_active": True,
                "created_at": datetime.utcnow(),
            },
        )

        await db_session.commit()

        logger.info(f"User registered: {user_data.username}")

        return UserResponse(
            id=user_id,
            username=user_data.username,
            email=user_data.email,
            full_name=user_data.full_name,
            is_active=True,
            created_at=datetime.utcnow(),
            memory_count=0,
        )

    except HTTPException:
        raise
    except Exception as e:
        await db_session.rollback()
        logger.error(f"User registration failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Registration failed"
        ) from e


@router.post("/login", response_model=Token)
async def login_user(
    form_data: OAuth2PasswordRequestForm = Depends(), db_session: AsyncSession = DatabaseDep
):
    """Login user and return access token"""

    try:
        # Get user by username or email
        result = await db_session.execute(
            """
            SELECT id, username, email, hashed_password, is_active
            FROM users
            WHERE (username = :username OR email = :username) AND is_active = true
            """,
            {"username": form_data.username},
        )

        user = result.fetchone()

        if not user or not password_service.verify_password(
            form_data.password, user.hashed_password
        ):
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Incorrect username or password",
                headers={"WWW-Authenticate": "Bearer"},
            )

        # Update last login
        await db_session.execute(
            "UPDATE users SET last_login = :last_login WHERE id = :user_id",
            {"last_login": datetime.utcnow(), "user_id": user.id},
        )
        await db_session.commit()

        # Create access token
        access_token_expires = timedelta(minutes=settings.access_token_expire_minutes)
        access_token = auth_service.create_access_token(
            data={"sub": user.id, "username": user.username}, expires_delta=access_token_expires
        )

        logger.info(f"User logged in: {user.username}")

        return Token(
            access_token=access_token,
            token_type="bearer",
            expires_in=settings.access_token_expire_minutes * 60,
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"User login failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Login failed"
        )


@router.post("/refresh", response_model=Token)
async def refresh_token(
    current_user: str = Depends(get_current_user), db_session: AsyncSession = DatabaseDep
):
    """Refresh access token"""

    try:
        # Get user info
        result = await db_session.execute(
            "SELECT username FROM users WHERE id = :user_id", {"user_id": current_user}
        )

        user = result.fetchone()
        if not user:
            raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="User not found")

        # Create new access token
        access_token_expires = timedelta(minutes=settings.access_token_expire_minutes)
        access_token = auth_service.create_access_token(
            data={"sub": current_user, "username": user.username},
            expires_delta=access_token_expires,
        )

        return Token(
            access_token=access_token,
            token_type="bearer",
            expires_in=settings.access_token_expire_minutes * 60,
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Token refresh failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Token refresh failed"
        )


@router.get("/me", response_model=UserResponse)
async def get_current_user_info(
    current_user: str = Depends(get_current_user), db_session: AsyncSession = DatabaseDep
):
    """Get current user information"""

    try:
        result = await db_session.execute(
            """
            SELECT u.id, u.username, u.email, u.full_name, u.is_active,
                   u.created_at, u.last_login,
                   COUNT(m.id) as memory_count
            FROM users u
            LEFT JOIN memories m ON u.id = m.user_id
            WHERE u.id = :user_id
            GROUP BY u.id
            """,
            {"user_id": current_user},
        )

        user = result.fetchone()

        if not user:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="User not found")

        return UserResponse(
            id=user.id,
            username=user.username,
            email=user.email,
            full_name=user.full_name,
            is_active=user.is_active,
            created_at=user.created_at,
            last_login=user.last_login,
            memory_count=user.memory_count,
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Get user info failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to get user information",
        )


@router.post("/logout")
async def logout_user(current_user: str = Depends(get_current_user)):
    """Logout user (client-side token invalidation)"""

    logger.info(f"User logged out: {current_user}")

    return {"message": "Successfully logged out"}
