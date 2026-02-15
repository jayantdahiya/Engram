"""Environment configuration for Engram"""

from pydantic import Field
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """Application settings with environment variable support"""

    # Database Configuration
    database_url: str = Field(
        default="postgresql+asyncpg://engram_user:secure_password@localhost:5432/engram_db",
        env="DATABASE_URL",
    )
    redis_url: str = Field(default="redis://localhost:6379/0", env="REDIS_URL")
    neo4j_uri: str = Field(default="bolt://localhost:7687", env="NEO4J_URI")
    neo4j_user: str = Field(default="neo4j", env="NEO4J_USER")
    neo4j_password: str = Field(default="secure_password", env="NEO4J_PASSWORD")

    # LLM Provider Configuration
    # Options: "ollama" (default, local), "openai" (cloud), "google" (cloud)
    llm_provider: str = Field(default="ollama", env="LLM_PROVIDER")
    # Options: "ollama" (default), "openai", "google", "local"
    embedding_provider: str = Field(default="ollama", env="EMBEDDING_PROVIDER")

    # Ollama Configuration (for llm_provider="ollama")
    ollama_base_url: str = Field(default="http://localhost:11434", env="OLLAMA_BASE_URL")
    ollama_llm_model: str = Field(default="gemma3:270m", env="OLLAMA_LLM_MODEL")
    ollama_embedding_model: str = Field(
        default="nomic-embed-text:latest", env="OLLAMA_EMBEDDING_MODEL"
    )

    # OpenAI Configuration (for llm_provider="openai" and/or embedding_provider="openai")
    openai_api_key: str = Field(default="", env="OPENAI_API_KEY")
    openai_llm_model: str = Field(default="gpt-5-nano", env="OPENAI_LLM_MODEL")
    openai_embedding_model: str = Field(
        default="text-embedding-3-small", env="OPENAI_EMBEDDING_MODEL"
    )

    # Google AI Configuration (for llm_provider="google" and/or embedding_provider="google")
    google_api_key: str = Field(default="", env="GOOGLE_API_KEY")
    google_llm_model: str = Field(default="gemini-3-flash", env="GOOGLE_LLM_MODEL")
    google_embedding_model: str = Field(
        default="gemini-embedding-001", env="GOOGLE_EMBEDDING_MODEL"
    )

    # Security Configuration
    secret_key: str = Field(default="your-secret-key-change-in-production", env="SECRET_KEY")
    algorithm: str = Field(default="HS256", env="ALGORITHM")
    access_token_expire_minutes: int = Field(default=15, env="ACCESS_TOKEN_EXPIRE_MINUTES")

    # Application Configuration
    debug: bool = Field(default=False, env="DEBUG")
    log_level: str = Field(default="INFO", env="LOG_LEVEL")
    environment: str = Field(default="development", env="ENVIRONMENT")

    # Celery Configuration
    celery_broker_url: str = Field(default="redis://localhost:6379/0", env="CELERY_BROKER_URL")
    celery_result_backend: str = Field(
        default="redis://localhost:6379/0", env="CELERY_RESULT_BACKEND"
    )

    # Monitoring Configuration
    prometheus_port: int = Field(default=8001, env="PROMETHEUS_PORT")
    grafana_port: int = Field(default=3000, env="GRAFANA_PORT")

    # Memory System Configuration
    similarity_threshold: float = Field(default=0.75, env="SIMILARITY_THRESHOLD")
    max_memories_per_user: int = Field(default=10000, env="MAX_MEMORIES_PER_USER")
    embedding_dimension: int = Field(
        default=1536,
        env="EMBEDDING_DIMENSION",
    )

    class Config:
        env_file = ".env"
        case_sensitive = False
        extra = "allow"


# Global settings instance
settings = Settings()
