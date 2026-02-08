"""Integration tests for authentication endpoints."""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch


class TestAuthRouter:
    """Integration test cases for auth endpoints."""

    @pytest.fixture
    def mock_db_session(self):
        """Create mock database session."""
        session = AsyncMock()
        session.execute = AsyncMock()
        session.commit = AsyncMock()
        session.rollback = AsyncMock()
        return session

    @pytest.fixture
    def mock_password_service(self):
        """Create mock password service."""
        with patch("api.routers.auth.password_service") as mock:
            mock.hash_password.return_value = "hashed_password_123"
            mock.verify_password.return_value = True
            yield mock

    @pytest.fixture
    def mock_auth_service(self):
        """Create mock auth service."""
        with patch("api.routers.auth.auth_service") as mock:
            mock.create_access_token.return_value = "test_access_token_123"
            yield mock

    @pytest.mark.asyncio
    async def test_register_user_success(
        self, mock_db_session, mock_password_service, test_user_data
    ):
        """Test successful user registration."""
        from api.routers.auth import register_user
        from models.user import UserCreate

        # Mock no existing user
        mock_result = MagicMock()
        mock_result.fetchone.return_value = None
        mock_db_session.execute.return_value = mock_result

        user_data = UserCreate(**test_user_data)

        with patch("api.routers.auth.logger"):
            result = await register_user(user_data, mock_db_session)

        assert result.username == test_user_data["username"]
        assert result.email == test_user_data["email"]
        assert result.is_active is True

    @pytest.mark.asyncio
    async def test_register_user_duplicate(
        self, mock_db_session, mock_password_service, test_user_data
    ):
        """Test registration with duplicate username/email."""
        from fastapi import HTTPException
        from api.routers.auth import register_user
        from models.user import UserCreate

        # Mock existing user
        mock_result = MagicMock()
        mock_result.fetchone.return_value = MagicMock(id="existing-user-id")
        mock_db_session.execute.return_value = mock_result

        user_data = UserCreate(**test_user_data)

        with pytest.raises(HTTPException) as exc_info:
            await register_user(user_data, mock_db_session)

        assert exc_info.value.status_code == 400
        assert "already registered" in exc_info.value.detail

    @pytest.mark.asyncio
    async def test_login_success(
        self, mock_db_session, mock_password_service, mock_auth_service, test_user_data
    ):
        """Test successful login."""
        from fastapi.security import OAuth2PasswordRequestForm
        from api.routers.auth import login_user

        # Mock user lookup
        mock_user = MagicMock()
        mock_user.id = "test-user-id"
        mock_user.username = test_user_data["username"]
        mock_user.hashed_password = "hashed_password"
        mock_user.is_active = True

        mock_result = MagicMock()
        mock_result.fetchone.return_value = mock_user
        mock_db_session.execute.return_value = mock_result

        form_data = MagicMock(spec=OAuth2PasswordRequestForm)
        form_data.username = test_user_data["username"]
        form_data.password = test_user_data["password"]

        with (
            patch("api.routers.auth.settings") as mock_settings,
            patch("api.routers.auth.logger"),
        ):
            mock_settings.access_token_expire_minutes = 30
            result = await login_user(form_data, mock_db_session)

        assert result.access_token == "test_access_token_123"
        assert result.token_type == "bearer"

    @pytest.mark.asyncio
    async def test_login_invalid_credentials(
        self, mock_db_session, mock_password_service, test_user_data
    ):
        """Test login with invalid credentials."""
        from fastapi import HTTPException
        from fastapi.security import OAuth2PasswordRequestForm
        from api.routers.auth import login_user

        # Mock no user found
        mock_result = MagicMock()
        mock_result.fetchone.return_value = None
        mock_db_session.execute.return_value = mock_result

        form_data = MagicMock(spec=OAuth2PasswordRequestForm)
        form_data.username = "nonexistent"
        form_data.password = "wrongpassword"

        with patch("api.routers.auth.logger"):
            with pytest.raises(HTTPException) as exc_info:
                await login_user(form_data, mock_db_session)

        assert exc_info.value.status_code == 401

    @pytest.mark.asyncio
    async def test_login_wrong_password(self, mock_db_session, test_user_data):
        """Test login with wrong password."""
        from fastapi import HTTPException
        from fastapi.security import OAuth2PasswordRequestForm
        from api.routers.auth import login_user

        # Mock user found but password doesn't match
        mock_user = MagicMock()
        mock_user.id = "test-user-id"
        mock_user.username = test_user_data["username"]
        mock_user.hashed_password = "hashed_password"

        mock_result = MagicMock()
        mock_result.fetchone.return_value = mock_user
        mock_db_session.execute.return_value = mock_result

        form_data = MagicMock(spec=OAuth2PasswordRequestForm)
        form_data.username = test_user_data["username"]
        form_data.password = "wrongpassword"

        with (
            patch("api.routers.auth.password_service") as mock_pw,
            patch("api.routers.auth.logger"),
        ):
            mock_pw.verify_password.return_value = False

            with pytest.raises(HTTPException) as exc_info:
                await login_user(form_data, mock_db_session)

        assert exc_info.value.status_code == 401

    @pytest.mark.asyncio
    async def test_refresh_token(self, mock_db_session, mock_auth_service):
        """Test token refresh."""
        from api.routers.auth import refresh_token

        # Mock user lookup
        mock_result = MagicMock()
        mock_result.fetchone.return_value = MagicMock(username="testuser")
        mock_db_session.execute.return_value = mock_result

        with patch("api.routers.auth.settings") as mock_settings:
            mock_settings.access_token_expire_minutes = 30
            result = await refresh_token(
                current_user="test-user-id", db_session=mock_db_session
            )

        assert result.access_token == "test_access_token_123"
        assert result.token_type == "bearer"

    @pytest.mark.asyncio
    async def test_refresh_token_user_not_found(self, mock_db_session):
        """Test token refresh when user doesn't exist."""
        from fastapi import HTTPException
        from api.routers.auth import refresh_token

        # Mock user not found
        mock_result = MagicMock()
        mock_result.fetchone.return_value = None
        mock_db_session.execute.return_value = mock_result

        with pytest.raises(HTTPException) as exc_info:
            await refresh_token(
                current_user="nonexistent-user-id", db_session=mock_db_session
            )

        assert exc_info.value.status_code == 401

    @pytest.mark.asyncio
    async def test_get_current_user_info(self, mock_db_session):
        """Test getting current user information."""
        from api.routers.auth import get_current_user_info
        from datetime import datetime

        # Mock user with memory count
        mock_user = MagicMock()
        mock_user.id = "test-user-id"
        mock_user.username = "testuser"
        mock_user.email = "test@example.com"
        mock_user.full_name = "Test User"
        mock_user.is_active = True
        mock_user.created_at = datetime.utcnow()
        mock_user.last_login = datetime.utcnow()
        mock_user.memory_count = 42

        mock_result = MagicMock()
        mock_result.fetchone.return_value = mock_user
        mock_db_session.execute.return_value = mock_result

        with patch("api.routers.auth.logger"):
            result = await get_current_user_info(
                current_user="test-user-id", db_session=mock_db_session
            )

        assert result.username == "testuser"
        assert result.memory_count == 42

    @pytest.mark.asyncio
    async def test_get_current_user_not_found(self, mock_db_session):
        """Test getting user info when user doesn't exist."""
        from fastapi import HTTPException
        from api.routers.auth import get_current_user_info

        # Mock user not found
        mock_result = MagicMock()
        mock_result.fetchone.return_value = None
        mock_db_session.execute.return_value = mock_result

        with patch("api.routers.auth.logger"):
            with pytest.raises(HTTPException) as exc_info:
                await get_current_user_info(
                    current_user="nonexistent-user-id", db_session=mock_db_session
                )

        assert exc_info.value.status_code == 404

    @pytest.mark.asyncio
    async def test_logout_success(self):
        """Test successful logout."""
        from api.routers.auth import logout_user

        with patch("api.routers.auth.logger"):
            result = await logout_user(current_user="test-user-id")

        assert result["message"] == "Successfully logged out"
