"""LLM service with support for multiple providers (Ollama and OpenAI)"""

import json
from typing import Any

import httpx

from core.config import settings
from core.logging import logger


class BaseLLMProvider:
    """Base class for LLM providers"""

    async def generate_response(
        self,
        messages: list[dict[str, str]],
        temperature: float = 0.7,
        max_tokens: int | None = None,
    ) -> str:
        raise NotImplementedError


class OllamaProvider(BaseLLMProvider):
    """Ollama LLM provider for local inference"""

    def __init__(self):
        self.base_url = settings.ollama_base_url
        self.model = settings.ollama_llm_model

    async def generate_response(
        self,
        messages: list[dict[str, str]],
        temperature: float = 0.7,
        max_tokens: int | None = None,
    ) -> str:
        """Generate response using Ollama Chat API"""
        try:
            payload = {
                "model": self.model,
                "messages": messages,
                "temperature": temperature,
                "stream": False,
            }

            if max_tokens:
                payload["num_predict"] = max_tokens

            async with httpx.AsyncClient() as client:
                response = await client.post(
                    f"{self.base_url}/api/chat",
                    json=payload,
                    timeout=60.0,
                )
                response.raise_for_status()
                result = response.json()
                return result["message"]["content"]

        except Exception as e:
            logger.error(f"Ollama response generation failed: {e}")
            raise


class OpenAIProvider(BaseLLMProvider):
    """OpenAI LLM provider for cloud inference"""

    def __init__(self):
        self.api_key = settings.openai_api_key
        self.model = settings.openai_llm_model
        self.base_url = "https://api.openai.com/v1"

        if not self.api_key:
            logger.warning("OpenAI API key not configured")

    async def generate_response(
        self,
        messages: list[dict[str, str]],
        temperature: float = 0.7,
        max_tokens: int | None = None,
    ) -> str:
        """Generate response using OpenAI Chat Completions with Responses fallback."""
        if not self.api_key:
            raise RuntimeError("OPENAI_API_KEY is required when LLM_PROVIDER=openai")

        try:
            return await self._generate_with_chat_completions(messages, temperature, max_tokens)
        except httpx.HTTPStatusError as e:
            if e.response.status_code in {400, 404, 422}:
                logger.warning(
                    "OpenAI chat.completions failed for model "
                    f"{self.model}, retrying with responses API: {e.response.text}"
                )
                return await self._generate_with_responses_api(messages, temperature, max_tokens)
            logger.error(f"OpenAI response generation failed: {e}")
            raise
        except Exception as e:
            logger.error(f"OpenAI response generation failed: {e}")
            raise

    async def _generate_with_chat_completions(
        self,
        messages: list[dict[str, str]],
        temperature: float,
        max_tokens: int | None,
    ) -> str:
        """Generate response using OpenAI chat/completions."""
        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature,
        }

        if max_tokens:
            payload["max_completion_tokens"] = max_tokens

        async with httpx.AsyncClient() as client:
            response = await client.post(
                f"{self.base_url}/chat/completions",
                headers={
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json",
                },
                json=payload,
                timeout=60.0,
            )
            response.raise_for_status()

        result = response.json()
        content = result["choices"][0]["message"]["content"]
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            return "".join(
                part.get("text", "") for part in content if isinstance(part, dict)
            ).strip()
        return str(content)

    async def _generate_with_responses_api(
        self,
        messages: list[dict[str, str]],
        temperature: float,
        max_tokens: int | None,
    ) -> str:
        """Generate response using OpenAI responses API."""
        payload: dict[str, Any] = {
            "model": self.model,
            "input": messages,
            "temperature": temperature,
        }

        if max_tokens:
            payload["max_output_tokens"] = max_tokens

        async with httpx.AsyncClient() as client:
            response = await client.post(
                f"{self.base_url}/responses",
                headers={
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json",
                },
                json=payload,
                timeout=60.0,
            )
            response.raise_for_status()

        result = response.json()

        output_text = result.get("output_text")
        if isinstance(output_text, str) and output_text.strip():
            return output_text.strip()

        output_items = result.get("output", [])
        if isinstance(output_items, list):
            chunks: list[str] = []
            for item in output_items:
                if not isinstance(item, dict):
                    continue
                content_items = item.get("content", [])
                if not isinstance(content_items, list):
                    continue
                for content in content_items:
                    if not isinstance(content, dict):
                        continue
                    text = content.get("text")
                    if isinstance(text, str):
                        chunks.append(text)
            if chunks:
                return "".join(chunks).strip()

        raise ValueError("Could not parse text output from OpenAI responses API")


class GoogleAIProvider(BaseLLMProvider):
    """Google AI (Gemini) LLM provider for cloud inference"""

    def __init__(self):
        self.api_key = settings.google_api_key
        self.model = settings.google_llm_model
        self.base_url = "https://generativelanguage.googleapis.com/v1beta"

        if not self.api_key:
            logger.warning("Google AI API key not configured")

    def _convert_messages(self, messages: list[dict[str, str]]) -> tuple[list[dict], dict | None]:
        """Convert OpenAI-style messages to Gemini contents format.

        Returns (contents, system_instruction) tuple.
        """
        contents: list[dict] = []
        system_instruction: dict | None = None

        for msg in messages:
            role = msg.get("role", "user")
            text = msg.get("content", "")

            if role == "system":
                system_instruction = {"parts": [{"text": text}]}
            else:
                gemini_role = "model" if role == "assistant" else "user"
                contents.append({"role": gemini_role, "parts": [{"text": text}]})

        return contents, system_instruction

    async def generate_response(
        self,
        messages: list[dict[str, str]],
        temperature: float = 0.7,
        max_tokens: int | None = None,
    ) -> str:
        """Generate response using Gemini REST API."""
        if not self.api_key:
            raise RuntimeError("GOOGLE_API_KEY is required when LLM_PROVIDER=google")

        contents, system_instruction = self._convert_messages(messages)

        payload: dict[str, Any] = {
            "contents": contents,
            "generationConfig": {
                "temperature": temperature,
            },
        }

        if max_tokens:
            payload["generationConfig"]["maxOutputTokens"] = max_tokens

        if system_instruction:
            payload["systemInstruction"] = system_instruction

        try:
            async with httpx.AsyncClient() as client:
                response = await client.post(
                    f"{self.base_url}/models/{self.model}:generateContent",
                    headers={"Content-Type": "application/json"},
                    params={"key": self.api_key},
                    json=payload,
                    timeout=60.0,
                )
                response.raise_for_status()

            result = response.json()
            candidates = result.get("candidates", [])
            if not candidates:
                raise ValueError("Gemini response missing candidates")

            content = candidates[0].get("content", {})
            parts = content.get("parts", [])
            if not parts:
                raise ValueError("Gemini response missing content parts")

            return "".join(part.get("text", "") for part in parts).strip()

        except Exception as e:
            logger.error(f"Google AI response generation failed: {e}")
            raise


class LLMService:
    """Service for interacting with LLM APIs (supports Ollama and OpenAI)"""

    def __init__(self):
        self.provider = self._init_provider()
        logger.info(f"LLM Service initialized with provider: {settings.llm_provider}")

    def _init_provider(self) -> BaseLLMProvider:
        """Initialize the appropriate LLM provider"""
        provider_name = settings.llm_provider.lower()

        if provider_name == "openai":
            return OpenAIProvider()
        elif provider_name == "google":
            return GoogleAIProvider()
        elif provider_name == "ollama":
            return OllamaProvider()
        else:
            logger.warning(f"Unknown provider '{provider_name}', defaulting to Ollama")
            return OllamaProvider()

    async def generate_response(
        self,
        messages: list[dict[str, str]],
        temperature: float = 0.7,
        max_tokens: int | None = None,
    ) -> str:
        """Generate response using configured LLM provider"""
        return await self.provider.generate_response(messages, temperature, max_tokens)

    async def classify_memory_operation(
        self, new_text: str, existing_memories: list[str], user_context: str | None = None
    ) -> dict[str, Any]:
        """Classify memory operation using LLM"""

        system_prompt = """You are a memory classification system. Analyze the new text and existing memories to determine the appropriate operation.

Operations:
- ADD: New information that doesn't conflict with existing memories
- UPDATE: Information that augments or modifies existing memories
- CONSOLIDATE: Information that contradicts existing memories (requires temporal consolidation)
- NOOP: Information that doesn't add value or is already covered

Return your response as JSON with:
{
    "operation": "ADD|UPDATE|CONSOLIDATE|NOOP",
    "confidence": 0.0-1.0,
    "reasoning": "explanation",
    "related_memory_indices": [0, 1, 2] // indices of related existing memories
}"""

        user_prompt = f"""
New text: "{new_text}"

Existing memories:
{chr(10).join([f"{i}: {mem}" for i, mem in enumerate(existing_memories)])}

{f"User context: {user_context}" if user_context else ""}

Classify the operation needed for this new text.
"""

        try:
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ]

            response = await self.generate_response(messages, temperature=0.1)

            # Parse JSON response
            result = json.loads(response)

            return {
                "operation": result.get("operation", "ADD"),
                "confidence": result.get("confidence", 0.5),
                "reasoning": result.get("reasoning", ""),
                "related_memory_indices": result.get("related_memory_indices", []),
            }

        except Exception as e:
            logger.error(f"Memory operation classification failed: {e}")
            # Fallback to simple classification
            return {
                "operation": "ADD",
                "confidence": 0.5,
                "reasoning": "Classification failed, defaulting to ADD",
                "related_memory_indices": [],
            }

    async def extract_entities_and_relations(self, text: str, user_id: str) -> dict[str, Any]:
        """Extract entities and relationships from text using LLM"""

        system_prompt = """You are an entity and relationship extraction system. Extract entities and their relationships from the given text.

Return your response as JSON with:
{
    "entities": [
        {
            "id": "unique_id",
            "name": "entity_name",
            "type": "PERSON|LOCATION|ORGANIZATION|EVENT|OBJECT|CONCEPT",
            "confidence": 0.0-1.0
        }
    ],
    "relationships": [
        {
            "source": "entity_id_1",
            "target": "entity_id_2",
            "relation": "relationship_type",
            "confidence": 0.0-1.0
        }
    ]
}"""

        user_prompt = f"""
Text: "{text}"

Extract entities and relationships from this text. Focus on:
- People, places, organizations
- Events and activities
- Objects and concepts
- Relationships between entities

User ID: {user_id}
"""

        try:
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ]

            response = await self.generate_response(messages, temperature=0.1)

            # Parse JSON response
            result = json.loads(response)

            return {
                "entities": result.get("entities", []),
                "relationships": result.get("relationships", []),
            }

        except Exception as e:
            logger.error(f"Entity extraction failed: {e}")
            return {"entities": [], "relationships": []}

    async def consolidate_memories(
        self, old_memory: str, new_memory: str, timestamp_old: float, timestamp_new: float
    ) -> str:
        """Consolidate conflicting memories with temporal awareness"""

        system_prompt = """You are a memory consolidation system. When memories conflict, create a temporally-aware consolidated version that acknowledges the change over time.

Guidelines:
- Use temporal language like "previously", "originally", "more recently", "now"
- Acknowledge the contradiction explicitly
- Maintain the timeline of changes
- Be concise but informative

Return only the consolidated memory text."""

        user_prompt = f"""
Old memory (timestamp: {timestamp_old}): "{old_memory}"

New memory (timestamp: {timestamp_new}): "{new_memory}"

These memories appear to contradict each other. Create a consolidated version that acknowledges the temporal change.
"""

        try:
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ]

            consolidated = await self.generate_response(messages, temperature=0.3)
            return consolidated.strip()

        except Exception as e:
            logger.error(f"Memory consolidation failed: {e}")
            # Fallback consolidation
            return f"{old_memory} (Updated: {new_memory})"

    async def generate_memory_summary(self, memories: list[str], user_id: str) -> str:
        """Generate a summary of user's memories"""

        system_prompt = """You are a memory summarization system. Create a concise, informative summary of the user's memories.

Guidelines:
- Highlight key facts about the user
- Organize information logically
- Be respectful and accurate
- Keep it concise but comprehensive"""

        user_prompt = f"""
User ID: {user_id}

Memories:
{chr(10).join([f"- {mem}" for mem in memories])}

Generate a summary of this user's key information.
"""

        try:
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ]

            summary = await self.generate_response(messages, temperature=0.5)
            return summary.strip()

        except Exception as e:
            logger.error(f"Memory summarization failed: {e}")
            return "Unable to generate summary at this time."


# Global LLM service instance
llm_service = LLMService()
