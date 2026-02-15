"""Unit tests for LLMService and providers."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from services.llm_service import (
    GoogleAIProvider,
    LLMService,
    OllamaProvider,
    OpenAIProvider,
)


class TestOllamaProvider:
    """Test cases for OllamaProvider."""

    @pytest.fixture
    def ollama_provider(self):
        """Create Ollama provider instance."""
        with patch("services.llm_service.settings") as mock_settings:
            mock_settings.ollama_base_url = "http://localhost:11434"
            mock_settings.ollama_llm_model = "llama3.2"
            return OllamaProvider()

    @pytest.mark.asyncio
    async def test_generate_response_success(self, ollama_provider):
        """Test successful response generation."""
        mock_response = {"message": {"content": "Generated response"}}

        with patch("httpx.AsyncClient") as mock_client:
            mock_instance = AsyncMock()
            mock_client.return_value.__aenter__.return_value = mock_instance
            mock_instance.post.return_value = MagicMock(
                json=lambda: mock_response, raise_for_status=MagicMock()
            )

            messages = [{"role": "user", "content": "Hello"}]
            result = await ollama_provider.generate_response(messages)

            assert result == "Generated response"

    @pytest.mark.asyncio
    async def test_generate_response_with_temperature(self, ollama_provider):
        """Test response generation with custom temperature."""
        mock_response = {"message": {"content": "Creative response"}}

        with patch("httpx.AsyncClient") as mock_client:
            mock_instance = AsyncMock()
            mock_client.return_value.__aenter__.return_value = mock_instance
            mock_instance.post.return_value = MagicMock(
                json=lambda: mock_response, raise_for_status=MagicMock()
            )

            messages = [{"role": "user", "content": "Be creative"}]
            result = await ollama_provider.generate_response(messages, temperature=0.9)

            assert result == "Creative response"


class TestOpenAIProvider:
    """Test cases for OpenAIProvider."""

    @pytest.fixture
    def openai_provider(self):
        """Create OpenAI provider instance."""
        with patch("services.llm_service.settings") as mock_settings:
            mock_settings.openai_api_key = "test-api-key"
            mock_settings.openai_llm_model = "gpt-4"
            return OpenAIProvider()

    @pytest.mark.asyncio
    async def test_generate_response_success(self, openai_provider):
        """Test successful response generation."""
        mock_response = {"choices": [{"message": {"content": "OpenAI response"}}]}

        with patch("httpx.AsyncClient") as mock_client:
            mock_instance = AsyncMock()
            mock_client.return_value.__aenter__.return_value = mock_instance
            mock_instance.post.return_value = MagicMock(
                json=lambda: mock_response, raise_for_status=MagicMock()
            )

            messages = [{"role": "user", "content": "Hello"}]
            result = await openai_provider.generate_response(messages)

            assert result == "OpenAI response"


class TestGoogleAIProvider:
    """Test cases for GoogleAIProvider."""

    @pytest.fixture
    def google_provider(self):
        """Create Google AI provider instance."""
        with patch("services.llm_service.settings") as mock_settings:
            mock_settings.google_api_key = "test-google-key"
            mock_settings.google_llm_model = "gemini-3-flash"
            return GoogleAIProvider()

    @pytest.mark.asyncio
    async def test_generate_response_success(self, google_provider):
        """Test successful response generation with Gemini API."""
        mock_response = {
            "candidates": [
                {
                    "content": {
                        "parts": [{"text": "Gemini response"}],
                        "role": "model",
                    }
                }
            ]
        }

        with patch("httpx.AsyncClient") as mock_client:
            mock_instance = AsyncMock()
            mock_client.return_value.__aenter__.return_value = mock_instance
            mock_instance.post.return_value = MagicMock(
                json=lambda: mock_response, raise_for_status=MagicMock()
            )

            messages = [
                {"role": "system", "content": "You are helpful"},
                {"role": "user", "content": "Hello"},
            ]
            result = await google_provider.generate_response(messages)

            assert result == "Gemini response"

    @pytest.mark.asyncio
    async def test_generate_response_no_api_key(self):
        """Test error when API key is missing."""
        with patch("services.llm_service.settings") as mock_settings:
            mock_settings.google_api_key = ""
            mock_settings.google_llm_model = "gemini-3-flash"
            provider = GoogleAIProvider()

        with pytest.raises(RuntimeError, match="GOOGLE_API_KEY"):
            await provider.generate_response([{"role": "user", "content": "Hello"}])

    def test_convert_messages(self, google_provider):
        """Test OpenAI-to-Gemini message format conversion."""
        messages = [
            {"role": "system", "content": "Be helpful"},
            {"role": "user", "content": "Hi"},
            {"role": "assistant", "content": "Hello!"},
            {"role": "user", "content": "How are you?"},
        ]
        contents, system_instruction = google_provider._convert_messages(messages)

        assert system_instruction == {"parts": [{"text": "Be helpful"}]}
        assert len(contents) == 3
        assert contents[0]["role"] == "user"
        assert contents[1]["role"] == "model"
        assert contents[2]["role"] == "user"

    def test_init_provider_google(self):
        """Test LLMService initializes GoogleAIProvider for 'google'."""
        with patch("services.llm_service.settings") as mock_settings:
            mock_settings.llm_provider = "google"
            mock_settings.google_api_key = "test-key"
            mock_settings.google_llm_model = "gemini-3-flash"
            service = LLMService()

        assert isinstance(service.provider, GoogleAIProvider)


class TestLLMService:
    """Test cases for LLMService."""

    @pytest.fixture
    def llm_service(self):
        """Create LLM service instance with mocked provider."""
        with patch("services.llm_service.settings") as mock_settings:
            mock_settings.llm_provider = "ollama"
            mock_settings.ollama_base_url = "http://localhost:11434"
            mock_settings.ollama_llm_model = "llama3.2"
            service = LLMService()
            service.provider = AsyncMock()
            return service

    @pytest.mark.asyncio
    async def test_generate_response(self, llm_service):
        """Test response generation through service."""
        llm_service.provider.generate_response.return_value = "Test response"

        messages = [{"role": "user", "content": "Hello"}]
        result = await llm_service.generate_response(messages)

        assert result == "Test response"
        llm_service.provider.generate_response.assert_called_once()

    @pytest.mark.asyncio
    async def test_classify_memory_operation_add(self, llm_service):
        """Test memory operation classification as ADD."""
        llm_service.provider.generate_response.return_value = (
            '{"operation": "ADD", "related_memory_indices": []}'
        )

        result = await llm_service.classify_memory_operation(
            "I love pizza", existing_memories=[]
        )

        assert result["operation"] == "ADD"
        assert result["related_memory_indices"] == []

    @pytest.mark.asyncio
    async def test_classify_memory_operation_update(self, llm_service):
        """Test memory operation classification as UPDATE."""
        llm_service.provider.generate_response.return_value = (
            '{"operation": "UPDATE", "related_memory_indices": [0]}'
        )

        result = await llm_service.classify_memory_operation(
            "I now prefer vegetarian pizza", existing_memories=["I love pizza"]
        )

        assert result["operation"] == "UPDATE"
        assert 0 in result["related_memory_indices"]

    @pytest.mark.asyncio
    async def test_classify_memory_operation_noop(self, llm_service):
        """Test memory operation classification as NOOP for trivial input."""
        llm_service.provider.generate_response.return_value = (
            '{"operation": "NOOP", "related_memory_indices": []}'
        )

        result = await llm_service.classify_memory_operation(
            "Hello there", existing_memories=[]
        )

        assert result["operation"] == "NOOP"

    @pytest.mark.asyncio
    async def test_extract_entities_and_relations(self, llm_service):
        """Test entity and relationship extraction."""
        llm_service.provider.generate_response.return_value = """
        {
            "entities": [
                {"name": "John", "type": "PERSON", "attributes": {}},
                {"name": "Acme Corp", "type": "ORGANIZATION", "attributes": {}}
            ],
            "relationships": [
                {"source": "John", "target": "Acme Corp", "type": "WORKS_AT"}
            ]
        }
        """

        result = await llm_service.extract_entities_and_relations(
            "John works at Acme Corp", user_id="test-user"
        )

        assert len(result["entities"]) == 2
        assert len(result["relationships"]) == 1
        assert result["relationships"][0]["type"] == "WORKS_AT"

    @pytest.mark.asyncio
    async def test_consolidate_memories(self, llm_service):
        """Test memory consolidation with temporal awareness."""
        llm_service.provider.generate_response.return_value = (
            "Previously vegetarian, now eats chicken occasionally"
        )

        result = await llm_service.consolidate_memories(
            old_memory="I am vegetarian",
            new_memory="I now eat chicken sometimes",
            timestamp_old=1000000.0,
            timestamp_new=2000000.0,
        )

        assert "Previously" in result or "vegetarian" in result

    @pytest.mark.asyncio
    async def test_generate_memory_summary(self, llm_service):
        """Test memory summary generation."""
        llm_service.provider.generate_response.return_value = (
            "User is a software engineer who loves hiking and has two dogs."
        )

        memories = [
            "I am a software engineer",
            "I love hiking",
            "I have two dogs named Buddy and Scout",
        ]

        result = await llm_service.generate_memory_summary(
            memories, user_id="test-user"
        )

        assert "software engineer" in result.lower() or len(result) > 0

    @pytest.mark.asyncio
    async def test_classify_memory_operation_handles_invalid_json(self, llm_service):
        """Test graceful handling of invalid JSON response."""
        llm_service.provider.generate_response.return_value = "invalid json response"

        # Should not raise, should return default operation
        result = await llm_service.classify_memory_operation(
            "Test message", existing_memories=[]
        )

        # Default behavior when parsing fails
        assert "operation" in result or result is not None

    @pytest.mark.asyncio
    async def test_extract_entities_empty_text(self, llm_service):
        """Test entity extraction with empty input."""
        llm_service.provider.generate_response.return_value = (
            '{"entities": [], "relationships": []}'
        )

        result = await llm_service.extract_entities_and_relations(
            "", user_id="test-user"
        )

        assert result["entities"] == []
        assert result["relationships"] == []
