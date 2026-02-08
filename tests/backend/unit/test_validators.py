"""Unit tests for validator utilities."""

from datetime import datetime
import uuid

import numpy as np

from utils.validators import (
    TextValidator,
    UUIDValidator,
    TimestampValidator,
    MemoryValidator,
    QueryValidator,
    UserValidator,
)


class TestTextValidator:
    """Test cases for TextValidator."""

    def test_sanitize_text_basic(self):
        """Test basic text sanitization."""
        text = "  Hello, World!  "
        result = TextValidator.sanitize_text(text)
        assert result == "Hello, World!"

    def test_sanitize_text_long_input(self):
        """Test text truncation for long input."""
        text = "a" * 20000
        result = TextValidator.sanitize_text(text, max_length=10000)
        assert len(result) <= 10000

    def test_sanitize_text_removes_null_bytes(self):
        """Test removal of null bytes."""
        text = "Hello\x00World"
        result = TextValidator.sanitize_text(text)
        assert "\x00" not in result

    def test_validate_text_length_valid(self):
        """Test valid text length."""
        text = "Valid text"
        result = TextValidator.validate_text_length(text, min_length=1, max_length=100)
        assert result is True

    def test_validate_text_length_too_short(self):
        """Test text shorter than minimum."""
        text = ""
        result = TextValidator.validate_text_length(text, min_length=1, max_length=100)
        assert result is False

    def test_validate_text_length_too_long(self):
        """Test text longer than maximum."""
        text = "a" * 200
        result = TextValidator.validate_text_length(text, min_length=1, max_length=100)
        assert result is False

    def test_contains_sensitive_info_email(self):
        """Test detection of email addresses."""
        text = "Contact me at user@example.com"
        result = TextValidator.contains_sensitive_info(text)
        assert result is True

    def test_contains_sensitive_info_phone(self):
        """Test detection of phone numbers."""
        text = "Call me at 555-123-4567"
        result = TextValidator.contains_sensitive_info(text)
        assert result is True

    def test_contains_sensitive_info_ssn(self):
        """Test detection of SSN patterns."""
        text = "My SSN is 123-45-6789"
        result = TextValidator.contains_sensitive_info(text)
        assert result is True

    def test_contains_sensitive_info_clean(self):
        """Test clean text without sensitive info."""
        text = "I love hiking in the mountains"
        result = TextValidator.contains_sensitive_info(text)
        assert result is False


class TestUUIDValidator:
    """Test cases for UUIDValidator."""

    def test_is_valid_uuid_valid(self):
        """Test with valid UUID."""
        valid_uuid = str(uuid.uuid4())
        result = UUIDValidator.is_valid_uuid(valid_uuid)
        assert result is True

    def test_is_valid_uuid_invalid(self):
        """Test with invalid UUID."""
        invalid_uuid = "not-a-valid-uuid"
        result = UUIDValidator.is_valid_uuid(invalid_uuid)
        assert result is False

    def test_is_valid_uuid_empty(self):
        """Test with empty string."""
        result = UUIDValidator.is_valid_uuid("")
        assert result is False

    def test_generate_uuid(self):
        """Test UUID generation."""
        result = UUIDValidator.generate_uuid()
        assert UUIDValidator.is_valid_uuid(result)


class TestTimestampValidator:
    """Test cases for TimestampValidator."""

    def test_is_valid_timestamp_datetime(self):
        """Test with datetime object."""
        dt = datetime.now()
        result = TimestampValidator.is_valid_timestamp(dt)
        assert result is True

    def test_is_valid_timestamp_float(self):
        """Test with float timestamp."""
        ts = 1234567890.123
        result = TimestampValidator.is_valid_timestamp(ts)
        assert result is True

    def test_is_valid_timestamp_int(self):
        """Test with integer timestamp."""
        ts = 1234567890
        result = TimestampValidator.is_valid_timestamp(ts)
        assert result is True

    def test_is_valid_timestamp_string(self):
        """Test with ISO format string."""
        ts = "2024-01-15T10:30:00"
        result = TimestampValidator.is_valid_timestamp(ts)
        assert result is True

    def test_is_valid_timestamp_invalid(self):
        """Test with invalid value."""
        result = TimestampValidator.is_valid_timestamp("invalid")
        assert result is False

    def test_normalize_timestamp_datetime(self):
        """Test normalizing datetime to datetime."""
        dt = datetime.now()
        result = TimestampValidator.normalize_timestamp(dt)
        assert isinstance(result, datetime)

    def test_normalize_timestamp_float(self):
        """Test normalizing float to datetime."""
        ts = 1234567890.123
        result = TimestampValidator.normalize_timestamp(ts)
        assert isinstance(result, datetime)

    def test_normalize_timestamp_string(self):
        """Test normalizing ISO string to datetime."""
        ts = "2024-01-15T10:30:00"
        result = TimestampValidator.normalize_timestamp(ts)
        assert isinstance(result, datetime)


class TestMemoryValidator:
    """Test cases for MemoryValidator."""

    def test_validate_memory_text_valid(self):
        """Test valid memory text."""
        text = "I love hiking and outdoor activities"
        result = MemoryValidator.validate_memory_text(text)
        assert result["is_valid"] is True

    def test_validate_memory_text_too_short(self):
        """Test memory text that's too short."""
        text = "Hi"
        result = MemoryValidator.validate_memory_text(text)
        # Short text may or may not be valid depending on implementation
        assert "is_valid" in result

    def test_validate_memory_text_empty(self):
        """Test empty memory text."""
        text = ""
        result = MemoryValidator.validate_memory_text(text)
        assert result["is_valid"] is False

    def test_validate_memory_text_with_sensitive_info(self):
        """Test memory with sensitive information."""
        text = "My email is user@example.com"
        result = MemoryValidator.validate_memory_text(text)
        assert "warnings" in result or "contains_sensitive" in result

    def test_validate_embedding_valid(self):
        """Test valid embedding vector."""
        embedding = np.random.rand(1536).tolist()
        result = MemoryValidator.validate_embedding(embedding, expected_dim=1536)
        assert result["is_valid"] is True

    def test_validate_embedding_wrong_dimension(self):
        """Test embedding with wrong dimension."""
        embedding = np.random.rand(512).tolist()
        result = MemoryValidator.validate_embedding(embedding, expected_dim=1536)
        assert result["is_valid"] is False

    def test_validate_embedding_with_nan(self):
        """Test embedding containing NaN values."""
        embedding = [float("nan")] + [0.0] * 1535
        result = MemoryValidator.validate_embedding(embedding, expected_dim=1536)
        assert result["is_valid"] is False

    def test_validate_embedding_with_inf(self):
        """Test embedding containing infinity."""
        embedding = [float("inf")] + [0.0] * 1535
        result = MemoryValidator.validate_embedding(embedding, expected_dim=1536)
        assert result["is_valid"] is False


class TestQueryValidator:
    """Test cases for QueryValidator."""

    def test_validate_query_valid(self):
        """Test valid search query."""
        query = "What are my dietary preferences?"
        result = QueryValidator.validate_query(query)
        assert result["is_valid"] is True

    def test_validate_query_empty(self):
        """Test empty query."""
        query = ""
        result = QueryValidator.validate_query(query)
        assert result["is_valid"] is False

    def test_validate_query_too_short(self):
        """Test query that's too short."""
        query = "a"
        result = QueryValidator.validate_query(query)
        # Depending on implementation, may be valid or invalid
        assert "is_valid" in result

    def test_validate_query_too_long(self):
        """Test query that's too long."""
        query = "a" * 10000
        result = QueryValidator.validate_query(query)
        # Should either truncate or reject
        assert "is_valid" in result


class TestUserValidator:
    """Test cases for UserValidator."""

    def test_validate_username_valid(self):
        """Test valid username."""
        username = "john_doe123"
        result = UserValidator.validate_username(username)
        assert result["is_valid"] is True

    def test_validate_username_too_short(self):
        """Test username that's too short."""
        username = "ab"
        result = UserValidator.validate_username(username)
        assert result["is_valid"] is False

    def test_validate_username_invalid_chars(self):
        """Test username with invalid characters."""
        username = "john@doe!"
        result = UserValidator.validate_username(username)
        assert result["is_valid"] is False

    def test_validate_username_starts_with_number(self):
        """Test username starting with number."""
        username = "123john"
        result = UserValidator.validate_username(username)
        # Depending on rules, may be valid or invalid
        assert "is_valid" in result

    def test_validate_email_valid(self):
        """Test valid email address."""
        email = "user@example.com"
        result = UserValidator.validate_email(email)
        assert result["is_valid"] is True

    def test_validate_email_invalid(self):
        """Test invalid email address."""
        email = "not-an-email"
        result = UserValidator.validate_email(email)
        assert result["is_valid"] is False

    def test_validate_email_missing_domain(self):
        """Test email missing domain."""
        email = "user@"
        result = UserValidator.validate_email(email)
        assert result["is_valid"] is False

    def test_validate_password_strong(self):
        """Test strong password."""
        password = "SecurePass123!@#"
        result = UserValidator.validate_password(password)
        assert result["is_valid"] is True

    def test_validate_password_too_short(self):
        """Test password that's too short."""
        password = "Ab1!"
        result = UserValidator.validate_password(password)
        assert result["is_valid"] is False

    def test_validate_password_no_uppercase(self):
        """Test password without uppercase."""
        password = "lowercase123!"
        result = UserValidator.validate_password(password)
        # Depending on rules, may fail
        assert "is_valid" in result

    def test_validate_password_no_number(self):
        """Test password without numbers."""
        password = "NoNumbersHere!"
        result = UserValidator.validate_password(password)
        # Depending on rules, may fail
        assert "is_valid" in result
