"""Application settings and environment validation using Pydantic Settings.

Ensures safe defaults, path resolution relative to project root, secret masking,
and non-blocking behavior when running storage-only or offline tests.
"""

from functools import lru_cache
from pathlib import Path
from typing import Optional
from pydantic import Field, SecretStr, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

# Determine project root deterministically from this file's location
# File is at: <project_root>/src/asterline/config/settings.py -> 4 levels up
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent


class Settings(BaseSettings):
    """Application configuration with validation and secret masking."""

    model_config = SettingsConfigDict(
        env_file=str(PROJECT_ROOT / ".env"),
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
    )

    # Model Provider Configuration
    llm_provider: str = Field(
        default="groq",
        description="LLM provider name: groq, openai, gemini, anthropic, or local",
    )
    llm_model: str = Field(
        default="llama-3.3-70b-versatile",
        description="Model name/identifier for the provider",
    )
    llm_api_key: Optional[SecretStr] = Field(
        default=None,
        description="API key for the LLM provider. Not required for storage-only or offline tests.",
    )
    llm_temperature: float = Field(
        default=0.0,
        ge=0.0,
        le=2.0,
        description="Sampling temperature for generation models.",
    )
    llm_max_tokens: int = Field(
        default=2048,
        gt=0,
        description="Maximum generation tokens per request.",
    )

    # Local Embeddings & Vector Storage
    embedding_provider: str = Field(
        default="local_fastembed",
        description="Embedding provider (local_fastembed runs locally on CPU with zero API cost)",
    )
    embedding_model_name: str = Field(
        default="BAAI/bge-small-en-v1.5",
        description="Local embedding model identifier",
    )
    chroma_persist_directory: Path = Field(
        default=Path("data/chroma"),
        description="Path to store local vector database files",
    )
    corpus_path: Path = Field(
        default=Path("data/corpus.jsonl"),
        description="Path to unchanged 75-chunk Asterline documentation corpus",
    )

    # Persistent Storage (SQLite)
    sqlite_db_path: Path = Field(
        default=Path("data/memory.db"),
        description="Path to SQLite database file for conversations and memories",
    )

    # LangSmith Observability
    langchain_tracing_v2: bool = Field(
        default=False,
        description="Whether to enable LangSmith tracing",
    )
    langchain_endpoint: str = Field(
        default="https://api.smith.langchain.com",
        description="LangSmith API endpoint",
    )
    langchain_api_key: Optional[SecretStr] = Field(
        default=None,
        description="LangSmith API key for trace export",
    )
    langchain_project: str = Field(
        default="asterline-research-assistant",
        description="LangSmith project name",
    )

    # Application & Execution Parameters
    app_host: str = Field(default="127.0.0.1", description="Application bind host")
    app_port: int = Field(default=8501, description="Application bind port")
    log_level: str = Field(default="INFO", description="Logging level")
    max_investigation_retries: int = Field(
        default=2,
        ge=0,
        le=5,
        description="Maximum retries permitted for investigation agent upon verification failure",
    )

    def resolve_path(self, relative_or_absolute: Path) -> Path:
        """Resolve a path relative to the project root if it is not absolute."""
        if relative_or_absolute.is_absolute():
            return relative_or_absolute
        return (PROJECT_ROOT / relative_or_absolute).resolve()

    def get_resolved_sqlite_path(self) -> Path:
        """Return the fully resolved SQLite database path."""
        return self.resolve_path(self.sqlite_db_path)

    def get_resolved_corpus_path(self) -> Path:
        """Return the fully resolved corpus.jsonl path."""
        return self.resolve_path(self.corpus_path)

    def get_resolved_chroma_path(self) -> Path:
        """Return the fully resolved Chroma persist directory path."""
        return self.resolve_path(self.chroma_persist_directory)

    def require_llm_api_key(self) -> str:
        """Validate and return LLM API key when model invocation is required.
        
        Raises ValueError with clear guidance if the key is missing.
        """
        if not self.llm_api_key or not self.llm_api_key.get_secret_value().strip():
            raise ValueError(
                f"LLM API key is required to invoke provider '{self.llm_provider}'. "
                "Please configure LLM_API_KEY in your .env file or environment variables."
            )
        return self.llm_api_key.get_secret_value()


@lru_cache()
def get_settings() -> Settings:
    """Return a cached singleton instance of Settings."""
    return Settings()
