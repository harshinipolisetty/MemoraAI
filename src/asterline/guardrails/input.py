"""Input guardrails for Asterline Research Assistant.

Validates input size, verifies trusted user/conversation identities,
sanitizes prompt injection attempts, and wraps untrusted user text in delimiters.
Preserves benign technical queries mentioning security keywords.
"""

from dataclasses import dataclass, field
import re
from typing import List, Optional

# Patterns explicitly aiming to hijack agent instruction flow or extract prompts/keys
INJECTION_PATTERNS = [
    re.compile(r"(?:ignore|disregard|forget|override)\s+(?:all\s+)?(?:previous|prior|above|system)\s+(?:instructions|prompts|rules)", re.IGNORECASE),
    re.compile(r"\b(?:you are now in|switch to)\s+(?:developer\s+mode|dan\s+mode|jailbreak|unrestricted\s+mode)\b", re.IGNORECASE),
    re.compile(r"\b(?:print|reveal|output|display|show|leak)\s+(?:the\s+)?(?:system\s+prompt|initial\s+prompt|base\s+instructions|secret\s+key|api\s+key)\b", re.IGNORECASE),
    re.compile(r"\b(?:system\s+override|admin\s+override|debug\s+override)\b", re.IGNORECASE),
]

MAX_INPUT_LENGTH = 4000


@dataclass
class InputGuardrailResult:
    """Outcome of input guardrail inspection."""

    is_valid: bool
    sanitized_message: str
    error_message: Optional[str] = None
    flags: List[str] = field(default_factory=list)


def validate_input(
    user_message: str,
    user_id: Optional[str] = None,
    conversation_id: Optional[str] = None,
) -> InputGuardrailResult:
    """Validate and sanitize untrusted user input before workflow processing.

    Rules:
    1. Rejects missing user_id or conversation_id (trusted identity invariant).
    2. Rejects empty or whitespace-only messages.
    3. Rejects inputs exceeding MAX_INPUT_LENGTH (4000 chars).
    4. Detects prompt injection attempts (flags or rejects if hostile).
    5. Safely allows legitimate technical questions mentioning 'token', 'system', 'key'.
    """
    flags: List[str] = []

    # Check trusted application context
    if not user_id or not user_id.strip():
        return InputGuardrailResult(
            is_valid=False,
            sanitized_message="",
            error_message="Missing trusted user_id. Request rejected.",
            flags=["MISSING_USER_ID"],
        )

    if not conversation_id or not conversation_id.strip():
        return InputGuardrailResult(
            is_valid=False,
            sanitized_message="",
            error_message="Missing trusted conversation_id. Request rejected.",
            flags=["MISSING_CONVERSATION_ID"],
        )

    # Check message presence
    if not user_message or not user_message.strip():
        return InputGuardrailResult(
            is_valid=False,
            sanitized_message="",
            error_message="User message cannot be empty.",
            flags=["EMPTY_MESSAGE"],
        )

    trimmed = user_message.strip()

    # Check maximum length
    if len(trimmed) > MAX_INPUT_LENGTH:
        return InputGuardrailResult(
            is_valid=False,
            sanitized_message="",
            error_message=f"Message length ({len(trimmed)} characters) exceeds maximum allowed ({MAX_INPUT_LENGTH}).",
            flags=["MESSAGE_TOO_LONG"],
        )

    # Detect prompt injection
    for pattern in INJECTION_PATTERNS:
        if pattern.search(trimmed):
            flags.append("PROMPT_INJECTION_DETECTED")
            return InputGuardrailResult(
                is_valid=False,
                sanitized_message="",
                error_message="Potential prompt injection or instruction override attempt detected.",
                flags=flags,
            )

    # Sanitized safe message with delimiters for prompt embedding
    return InputGuardrailResult(
        is_valid=True,
        sanitized_message=trimmed,
        error_message=None,
        flags=flags,
    )


def wrap_user_prompt(message: str) -> str:
    """Wrap untrusted user query in unambiguous structural XML tags."""
    return f"<user_query>\n{message}\n</user_query>"
