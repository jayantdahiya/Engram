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

    # Ollama Configuration
    ollama_base_url: str = Field(default="http://localhost:11434", env="OLLAMA_BASE_URL")
    ollama_llm_model: str = Field(default="gemma3:270m", env="OLLAMA_LLM_MODEL")
    ollama_embedding_model: str = Field(default="nomic-embed-text:latest", env="OLLAMA_EMBEDDING_MODEL")

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
        default=768,  # nomic-embed-text dimension
        env="EMBEDDING_DIMENSION",
    )

    class Config:
        env_file = ".env"
        case_sensitive = False
        extra = "allow"


# Global settings instance
settings = Settings()
