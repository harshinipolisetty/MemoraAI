"""Tests for observability, secret redaction, safe metadata, and LangSmith fallback behavior."""

import os
import uuid
from unittest.mock import patch
import pytest

from asterline.observability.tracing import (
    redact_secrets,
    redact_dict,
    get_safe_trace_metadata,
    check_langsmith_status,
    setup_tracing,
    RedactingTraceCallback,
)


def test_redact_secrets_keys_and_tokens():
    raw_text = (
        "Here is an OpenAI key: sk-abcdef1234567890abcdef12345678901234\n"
        "Here is an Anthropic key: sk-ant-api03-abcdef1234567890abcdef12345678901234\n"
        "Here is an Asterline ingestion key: ast_live_secret1234567890abcdef\n"
        "Here is a generic bearer: Bearer eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.xyz\n"
        "Here is a password: password=SuperSecretPassword123!\n"
    )

    redacted = redact_secrets(raw_text)

    assert "sk-abcdef" not in redacted
    assert "sk-ant" not in redacted
    assert "ast_live" not in redacted
    assert "SuperSecretPassword123!" not in redacted
    assert "[REDACTED_SECRET]" in redacted


def test_redact_dict_nested_and_keys():
    nested_data = {
        "user_id": "alice",
        "api_key": "raw_sensitive_key_999",
        "password": "secret_password",
        "metadata": {
            "token": "token_abc123",
            "message": "Connecting with key sk-12345678901234567890123456789012",
        },
        "list_data": [
            {"client_secret": "my_secret"},
            "Clean string",
        ],
    }

    cleaned = redact_dict(nested_data)

    assert cleaned["user_id"] == "alice"
    assert cleaned["api_key"] == "[REDACTED_CREDENTIAL]"
    assert cleaned["password"] == "[REDACTED_CREDENTIAL]"
    assert cleaned["metadata"]["token"] == "[REDACTED_CREDENTIAL]"
    assert "[REDACTED_SECRET]" in cleaned["metadata"]["message"]
    assert cleaned["list_data"][0]["client_secret"] == "[REDACTED_CREDENTIAL]"
    assert cleaned["list_data"][1] == "Clean string"


def test_get_safe_trace_metadata_excludes_raw_memories():
    meta = get_safe_trace_metadata(
        user_id="alice_researcher",
        conversation_id="conv_99",
        route="factual",
        retry_count=1,
        is_fallback=True,
    )

    assert meta["metadata"]["user_id"] == "alice_researcher"
    assert meta["metadata"]["conversation_id"] == "conv_99"
    assert meta["metadata"]["route"] == "factual"
    assert meta["metadata"]["retry_count"] == 1
    assert meta["metadata"]["is_fallback"] is True
    assert "route:factual" in meta["tags"]
    assert "mode:fallback" in meta["tags"]
    assert "retry:1" in meta["tags"]

    # Verify no raw personal memory field exists in trace metadata
    assert "active_user_memories" not in meta["metadata"]
    assert "content" not in meta["metadata"]


def test_langsmith_status_when_disabled():
    with patch.dict(os.environ, {"LANGSMITH_TRACING": "false", "LANGCHAIN_TRACING_V2": "false"}, clear=True):
        status = check_langsmith_status()
        assert status["tracing_enabled"] is False
        assert "status" in status
        assert "endpoint" in status
        assert "project" in status


def test_setup_tracing_graceful_when_disabled():
    with patch.dict(os.environ, {"LANGSMITH_TRACING": "false", "LANGCHAIN_TRACING_V2": "false"}, clear=True):
        activated = setup_tracing(enable=False)
        assert activated is False
        assert os.environ.get("LANGCHAIN_TRACING_V2") == "false"


def test_redacting_callback_sanitizes_inputs_and_outputs():
    callback = RedactingTraceCallback()

    serialized = {"name": "test_llm"}
    prompts = ["My API key is sk-12345678901234567890123456789012"]

    # Should sanitize prompts in-place without error
    callback.on_llm_start(serialized, prompts)
    assert "[REDACTED_SECRET]" in prompts[0]

    # Tool invocation
    sanitized_input = callback.on_tool_start({"name": "retriever"}, "query for ast_live_secretkey1234567890")
    assert "[REDACTED_SECRET]" in sanitized_input

    sanitized_output = callback.on_tool_end("Found results with password=admin123")
    assert "[REDACTED_SECRET]" in sanitized_output


def test_get_safe_trace_metadata_includes_performance_metrics():
    meta = get_safe_trace_metadata(
        user_id="user_test",
        conversation_id="conv_perf",
        route="factual",
        latency_ms=124.5,
        llm_call_count=1,
        is_cached=True,
    )
    assert meta["metadata"]["latency_ms"] == 124.5
    assert meta["metadata"]["llm_call_count"] == 1
    assert meta["metadata"]["is_cached"] is True
    assert "cache:hit" in meta["tags"]
    assert "llm_calls:1" in meta["tags"]
    assert "latency:124.5ms" in meta["tags"]

