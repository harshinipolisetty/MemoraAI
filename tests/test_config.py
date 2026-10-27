"""Tests for application settings, path resolution, and secret masking."""

from pathlib import Path
import pytest
from asterline.config.settings import Settings


def test_settings_default_values_load_without_api_key():
    """Settings can be loaded and imported without requiring an API key for storage tests."""
    settings = Settings(
        llm_api_key=None,
        langchain_api_key=None,
    )
    assert settings.llm_provider == "groq"
    assert settings.llm_model == "llama-3.3-70b-versatile"
    assert settings.sqlite_db_path == Path("data/memory.db")
    assert settings.corpus_path == Path("data/corpus.jsonl")


def test_settings_secret_masking():
    """API keys must not be exposed in plaintext in string representations."""
    settings = Settings(
        llm_api_key="super_secret_test_key_12345",
        langchain_api_key="langsmith_secret_token_abcde",
    )
    repr_str = repr(settings)
    assert "super_secret_test_key_12345" not in repr_str
    assert "langsmith_secret_token_abcde" not in repr_str
    assert "**********" in repr_str


def test_require_llm_api_key_validation():
    """Attempting to access API key when not configured raises clear ValueError."""
    settings = Settings(llm_api_key=None)
    with pytest.raises(ValueError, match="LLM API key is required"):
        settings.require_llm_api_key()

    valid_settings = Settings(llm_api_key="valid-test-key")
    assert valid_settings.require_llm_api_key() == "valid-test-key"


def test_path_resolution():
    """Relative paths resolve predictably from project root."""
    settings = Settings(
        sqlite_db_path=Path("custom/dir/test.db"),
        corpus_path=Path("custom/corpus.jsonl"),
    )
    resolved_sqlite = settings.get_resolved_sqlite_path()
    resolved_corpus = settings.get_resolved_corpus_path()

    assert resolved_sqlite.is_absolute()
    assert resolved_corpus.is_absolute()
    assert str(resolved_sqlite).endswith("custom\\dir\\test.db") or str(resolved_sqlite).endswith("custom/dir/test.db")
