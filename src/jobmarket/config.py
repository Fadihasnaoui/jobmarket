"""Application configuration loaded from environment variables."""

from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Runtime settings for the job market pipeline."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    database_url: str = "postgresql+asyncpg://jobmarket:jobmarket@localhost:5432/jobmarket"
    database_url_sync: str = "postgresql+psycopg://jobmarket:jobmarket@localhost:5432/jobmarket"

    adzuna_app_id: str = ""
    adzuna_app_key: str = ""
    jooble_api_key: str = ""

    llm_provider: str = "groq"
    llm_model: str = "llama-3.3-70b-versatile"
    llm_base_url: str = "https://api.groq.com/openai/v1"
    llm_api_key: str = ""

    embedding_model: str = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    embedding_dimension: int = 384
    embedding_batch_size: int = 64
    embedding_normalize: bool = True
    embedding_model_revision: str = ""


@lru_cache
def get_settings() -> Settings:
    return Settings()
