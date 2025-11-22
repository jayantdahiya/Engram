"""Custom validators for data validation and sanitization"""

import re
import uuid
from datetime import datetime
from typing import Any

import numpy as np

from core.logging import logger


class TextValidator:
    """Text validation utilities"""

    @staticmethod
    def sanitize_text(text: str, max_length: int = 10000) -> str:
        """Sanitize text input"""
        if not isinstance(text, str):
            return ""

        # Remove excessive whitespace
        text = re.sub(r"\s+", " ", text.strip())

        # Truncate if too long
        if len(text) > max_length:
            text = text[:max_length]
            logger.warning(f"Text truncated to {max_length} characters")

        return text

    @staticmethod
    def validate_text_length(text: str, min_length: int = 1, max_length: int = 10000) -> bool:
        """Validate text length"""
        if not isinstance(text, str):
            return False

        return min_length <= len(text) <= max_length

    @staticmethod
    def contains_sensitive_info(text: str) -> bool:
        """Check if text contains potentially sensitive information"""
        sensitive_patterns = [
            r"\b\d{4}[-\s]?\d{4}[-\s]?\d{4}[-\s]?\d{4}\b",  # Credit card
            r"\b\d{3}-\d{2}-\d{4}\b",  # SSN
            r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b",  # Email
            r"\b(?:\d{1,3}\.){3}\d{1,3}\b",  # IP address
            r"\bpassword\s*[:=]\s*\S+",  # Password
            r"\bapi[_-]?key\s*[:=]\s*\S+",  # API key
        ]

        text_lower = text.lower()
        for pattern in sensitive_patterns:
            if re.search(pattern, text_lower):
                return True

        return False


class UUIDValidator:
    """UUID validation utilities"""

    @staticmethod
    def is_valid_uuid(uuid_string: str) -> bool:
        """Check if string is a valid UUID"""
        try:
            uuid.UUID(uuid_string)
            return True
        except ValueError:
            return False

    @staticmethod
    def generate_uuid() -> str:
        """Generate a new UUID string"""
        return str(uuid.uuid4())


class TimestampValidator:
    """Timestamp validation utilities"""

    @staticmethod
    def is_valid_timestamp(timestamp: Any) -> bool:
        """Check if value is a valid timestamp"""
        try:
            if isinstance(timestamp, int | float):
                # Unix timestamp
                datetime.fromtimestamp(timestamp)
                return True
            elif isinstance(timestamp, datetime):
                return True
            elif isinstance(timestamp, str):
                # Try to parse ISO format
                datetime.fromisoformat(timestamp.replace("Z", "+00:00"))
                return True
            return False
        except (ValueError, TypeError, OSError):
            return False

    @staticmethod
    def normalize_timestamp(timestamp: Any) -> datetime | None:
        """Normalize timestamp to datetime object"""
        try:
            if isinstance(timestamp, datetime):
                return timestamp
            elif isinstance(timestamp, int | float):
                return datetime.fromtimestamp(timestamp)
            elif isinstance(timestamp, str):
                return datetime.fromisoformat(timestamp.replace("Z", "+00:00"))
            return None
        except (ValueError, TypeError, OSError):
            return None


class MemoryValidator:
    """Memory-specific validation utilities"""

    @staticmethod
    def validate_memory_text(text: str) -> dict[str, Any]:
        """Validate memory text and return validation results"""
        result = {"is_valid": True, "errors": [], "warnings": [], "sanitized_text": text}

        # Check if text is empty
        if not text or not text.strip():
            result["is_valid"] = False
            result["errors"].append("Memory text cannot be empty")
            return result

        # Sanitize text
        sanitized = TextValidator.sanitize_text(text)
        result["sanitized_text"] = sanitized

        # Check length
        if not TextValidator.validate_text_length(sanitized, min_length=1, max_length=10000):
            result["is_valid"] = False
            result["errors"].append("Memory text length must be between 1 and 10000 characters")

        # Check for sensitive information
        if TextValidator.contains_sensitive_info(sanitized):
            result["warnings"].append("Memory text may contain sensitive information")

        # Check for excessive repetition
        words = sanitized.split()
        if len(words) > 10:
            word_counts = {}
            for word in words:
                word_counts[word] = word_counts.get(word, 0) + 1

            max_repetition = max(word_counts.values())
            if max_repetition > len(words) * 0.3:  # More than 30% repetition
                result["warnings"].append("Memory text contains excessive repetition")

        return result

    @staticmethod
    def validate_embedding(embedding: list[float], expected_dim: int = 1536) -> dict[str, Any]:
        """Validate embedding vector"""
        result = {"is_valid": True, "errors": [], "warnings": []}

        # Check if embedding is a list
        if not isinstance(embedding, list):
            result["is_valid"] = False
            result["errors"].append("Embedding must be a list")
            return result

        # Check dimension
        if len(embedding) != expected_dim:
            result["is_valid"] = False
            result["errors"].append(
                f"Embedding dimension must be {expected_dim}, got {len(embedding)}"
            )
            return result

        # Check for valid float values
        try:
            embedding_array = np.array(embedding, dtype=float)

            # Check for NaN or infinite values
            if np.any(np.isnan(embedding_array)) or np.any(np.isinf(embedding_array)):
                result["is_valid"] = False
                result["errors"].append("Embedding contains NaN or infinite values")
                return result

            # Check for zero vector
            if np.allclose(embedding_array, 0):
                result["warnings"].append("Embedding is a zero vector")

            # Check for very small norm
            norm = np.linalg.norm(embedding_array)
            if norm < 0.01:
                result["warnings"].append("Embedding has very small norm")

        except (ValueError, TypeError):
            result["is_valid"] = False
            result["errors"].append("Embedding contains invalid values")

        return result


class QueryValidator:
    """Query validation utilities"""

    @staticmethod
    def validate_query(query: str) -> dict[str, Any]:
        """Validate search query"""
        result = {"is_valid": True, "errors": [], "warnings": [], "sanitized_query": query}

        # Check if query is empty
        if not query or not query.strip():
            result["is_valid"] = False
            result["errors"].append("Query cannot be empty")
            return result

        # Sanitize query
        sanitized = TextValidator.sanitize_text(query, max_length=1000)
        result["sanitized_query"] = sanitized

        # Check length
        if not TextValidator.validate_text_length(sanitized, min_length=1, max_length=1000):
            result["is_valid"] = False
            result["errors"].append("Query length must be between 1 and 1000 characters")

        # Check for excessive special characters
        special_char_count = len(re.findall(r"[^\w\s]", sanitized))
        if special_char_count > len(sanitized) * 0.5:
            result["warnings"].append("Query contains many special characters")

        return result


class UserValidator:
    """User validation utilities"""

    @staticmethod
    def validate_username(username: str) -> dict[str, Any]:
        """Validate username"""
        result = {"is_valid": True, "errors": [], "warnings": []}

        # Check length
        if not (3 <= len(username) <= 50):
            result["is_valid"] = False
            result["errors"].append("Username must be between 3 and 50 characters")

        # Check format (alphanumeric and underscores only)
        if not re.match(r"^[a-zA-Z0-9_]+$", username):
            result["is_valid"] = False
            result["errors"].append("Username can only contain letters, numbers, and underscores")

        # Check if starts with letter
        if not re.match(r"^[a-zA-Z]", username):
            result["is_valid"] = False
            result["errors"].append("Username must start with a letter")

        return result

    @staticmethod
    def validate_email(email: str) -> dict[str, Any]:
        """Validate email address"""
        result = {"is_valid": True, "errors": [], "warnings": []}

        # Basic email regex
        email_pattern = r"^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$"

        if not re.match(email_pattern, email):
            result["is_valid"] = False
            result["errors"].append("Invalid email format")

        # Check length
        if len(email) > 255:
            result["is_valid"] = False
            result["errors"].append("Email address too long")

        return result

    @staticmethod
    def validate_password(password: str) -> dict[str, Any]:
        """Validate password strength"""
        result = {"is_valid": True, "errors": [], "warnings": [], "strength_score": 0}

        # Check length
        if len(password) < 8:
            result["is_valid"] = False
            result["errors"].append("Password must be at least 8 characters long")

        # Calculate strength score
        score = 0

        # Length bonus
        if len(password) >= 8:
            score += 1
        if len(password) >= 12:
            score += 1

        # Character variety
        if re.search(r"[a-z]", password):
            score += 1
        if re.search(r"[A-Z]", password):
            score += 1
        if re.search(r"[0-9]", password):
            score += 1
        if re.search(r"[^a-zA-Z0-9]", password):
            score += 1

        result["strength_score"] = score

        # Warnings for weak passwords
        if score < 3:
            result["warnings"].append("Password is weak")
        elif score < 4:
            result["warnings"].append("Password could be stronger")

        return result
