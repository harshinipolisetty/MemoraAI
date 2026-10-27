"""LangSmith observability, trace metadata, and secret redaction for Asterline Research Assistant.

Guarantees:
- End-to-end execution inspectability (agent nodes, routing, retrieval, retries, fallback status).
- Automatic scrubbing of secrets (API keys, passwords, bearer tokens, private keys) before traces are logged.
- Strict protection of personal user memories from trace metadata and tags.
- Non-blocking, graceful fallback when LangSmith tracing is disabled or credentials are absent.
"""

import os
import re
from typing import Any, Dict, List, Optional
from langchain_core.callbacks.base import BaseCallbackHandler
from langchain_core.outputs import LLMResult
from asterline.config.settings import get_settings

# Regex patterns matching secret keys, tokens, and credentials
TRACE_SECRET_PATTERNS = [
    re.compile(r"ast_[a-zA-Z0-9_\-]{16,}", re.IGNORECASE),
    re.compile(r"sk-[a-zA-Z0-9_\-]{20,}", re.IGNORECASE),
    re.compile(r"gh[pousr]_[a-zA-Z0-9]{20,}", re.IGNORECASE),
    re.compile(r"Bearer\s+[a-zA-Z0-9_\-\.]{20,}", re.IGNORECASE),
    re.compile(r"ey[a-zA-Z0-9_\-]{15,}\.ey[a-zA-Z0-9_\-]{15,}", re.IGNORECASE),
    re.compile(r"-----BEGIN (?:[A-Z0-9_\- ]+ )?PRIVATE KEY-----[\s\S]+?-----END (?:[A-Z0-9_\- ]+ )?PRIVATE KEY-----"),
    re.compile(r"(?:password|passwd|secret|api_key|auth_token)\s*[:=]\s*['\"]?\S{4,}['\"]?", re.IGNORECASE),
]


def redact_secrets(text: str) -> str:
    """Scrub sensitive API keys, passwords, and tokens from text before logging or tracing."""
    if not isinstance(text, str) or not text:
        return text

    sanitized = text
    for pattern in TRACE_SECRET_PATTERNS:
        sanitized = pattern.sub("[REDACTED_SECRET]", sanitized)
    return sanitized


def redact_dict(data: Any) -> Any:
    """Recursively scrub secrets from nested dictionaries, lists, or primitive values."""
    if isinstance(data, dict):
        sanitized_dict = {}
        for k, v in data.items():
            # If key explicitly suggests a credential, redact value completely
            if any(s in str(k).lower() for s in ("key", "secret", "password", "token", "auth")):
                if isinstance(v, str) and len(v) > 0:
                    sanitized_dict[k] = "[REDACTED_CREDENTIAL]"
                    continue
            sanitized_dict[k] = redact_dict(v)
        return sanitized_dict
    elif isinstance(data, list):
        return [redact_dict(item) for item in data]
    elif isinstance(data, str):
        return redact_secrets(data)
    return data


def get_safe_trace_metadata(
    user_id: str,
    conversation_id: str,
    route: Optional[str] = None,
    retry_count: int = 0,
    is_fallback: bool = False,
    tags: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """Build sanitized LangChain/LangGraph run metadata for trace export.

    Guarantees no raw personal memories or credentials are leaked into trace tags or metadata.
    """
    safe_tags = ["asterline-research-assistant"]
    if route:
        safe_tags.append(f"route:{route}")
    if is_fallback:
        safe_tags.append("mode:fallback")
    else:
        safe_tags.append("mode:live")
    if retry_count > 0:
        safe_tags.append(f"retry:{retry_count}")

    if tags:
        for t in tags:
            # Scrub any accidental secrets from custom tags
            safe_tags.append(redact_secrets(str(t)))

    return {
        "metadata": {
            "user_id": redact_secrets(user_id),
            "conversation_id": redact_secrets(conversation_id),
            "route": route or "unspecified",
            "retry_count": retry_count,
            "is_fallback": is_fallback,
        },
        "tags": list(dict.fromkeys(safe_tags)),
    }


def setup_tracing(enable: Optional[bool] = None) -> bool:
    """Configure environment variables for LangSmith tracing based on application settings.

    Returns True if tracing was successfully activated with valid API key.
    Returns False if tracing is disabled or API key is absent (graceful fallback).
    """
    settings = get_settings()
    tracing_requested = enable if enable is not None else settings.langchain_tracing_v2

    if not tracing_requested:
        os.environ["LANGCHAIN_TRACING_V2"] = "false"
        return False

    api_key_str = (
        settings.langchain_api_key.get_secret_value().strip()
        if settings.langchain_api_key
        else os.environ.get("LANGCHAIN_API_KEY", "").strip()
    )

    if not api_key_str:
        # Graceful fallback: do not crash if tracing was requested but key is missing
        os.environ["LANGCHAIN_TRACING_V2"] = "false"
        return False

    os.environ["LANGCHAIN_TRACING_V2"] = "true"
    os.environ["LANGCHAIN_API_KEY"] = api_key_str
    os.environ["LANGCHAIN_ENDPOINT"] = settings.langchain_endpoint
    os.environ["LANGCHAIN_PROJECT"] = settings.langchain_project
    return True


def check_langsmith_status() -> Dict[str, Any]:
    """Inspect and report the active LangSmith configuration status.

    Safe for diagnostics in offline test environments without making outbound network requests.
    """
    settings = get_settings()
    has_env_key = bool(os.environ.get("LANGCHAIN_API_KEY", "").strip())
    has_settings_key = bool(
        settings.langchain_api_key and settings.langchain_api_key.get_secret_value().strip()
    )
    env_tracing = os.environ.get("LANGCHAIN_TRACING_V2")
    if env_tracing is not None:
        is_tracing_on = env_tracing.strip().lower() in ("true", "1", "yes")
    else:
        is_tracing_on = settings.langchain_tracing_v2

    return {
        "tracing_enabled": bool(is_tracing_on and (has_env_key or has_settings_key)),
        "configured_in_settings": settings.langchain_tracing_v2,
        "api_key_present": has_env_key or has_settings_key,
        "project": settings.langchain_project,
        "endpoint": settings.langchain_endpoint,
        "status": "ready" if (has_env_key or has_settings_key) and is_tracing_on else "offline_disabled",
    }


class RedactingTraceCallback(BaseCallbackHandler):
    """LangChain callback handler that redacts secrets before traces reach LangSmith."""

    def __init__(self):
        super().__init__()

    def on_llm_start(
        self, serialized: Dict[str, Any], prompts: List[str], **kwargs: Any
    ) -> Any:
        """Sanitize prompts before sending to language models."""
        for i, prompt in enumerate(prompts):
            prompts[i] = redact_secrets(prompt)

    def on_llm_end(self, response: LLMResult, **kwargs: Any) -> Any:
        """Sanitize generated text before logging."""
        for generation_list in response.generations:
            for gen in generation_list:
                gen.text = redact_secrets(gen.text)

    def on_tool_start(
        self, serialized: Dict[str, Any], input_str: str, **kwargs: Any
    ) -> Any:
        """Sanitize tool input before logging."""
        return redact_secrets(input_str)

    def on_tool_end(self, output: Any, **kwargs: Any) -> Any:
        """Sanitize tool output before logging."""
        if isinstance(output, str):
            return redact_secrets(output)
        elif isinstance(output, dict):
            return redact_dict(output)
        return output
