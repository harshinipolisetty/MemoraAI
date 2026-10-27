"""Selective memory policy for long-term user context.

Enforces privacy, secret rejection, and prevents conversational noise or
unverified company claims from polluting long-term user memory.

IMPORTANT SECURITY LIMITATION:
Heuristic pattern matching and regular expressions provide defense-in-depth against
obvious secrets, but cannot guarantee detection of all steganographic, heavily encoded,
or novel secret formats.
"""

import re
from typing import Optional, Set
from pydantic import BaseModel, Field


class MemoryPolicyDecision(BaseModel):
    """Structured decision returned by memory policy evaluation."""

    allowed: bool = Field(description="True if the candidate meets criteria for long-term storage")
    reason_code: str = Field(description="Machine-readable code explaining the decision")
    message: str = Field(description="Human-readable explanation of the decision")
    sanitized_content: Optional[str] = Field(
        default=None, description="Cleaned content with any trailing or extra whitespace trimmed"
    )


# Regular expression patterns for obvious secrets and credentials
SECRET_PATTERNS = [
    (re.compile(r"ast_[a-zA-Z0-9_\-]{16,}", re.IGNORECASE), "REJECTED_ASTERLINE_KEY", "Candidate contains an Asterline project ingestion key"),
    (re.compile(r"sk-[a-zA-Z0-9_\-]{20,}", re.IGNORECASE), "REJECTED_API_KEY", "Candidate contains an API key pattern (sk-...)"),
    (re.compile(r"gh[pousr]_[a-zA-Z0-9]{20,}", re.IGNORECASE), "REJECTED_GITHUB_TOKEN", "Candidate contains a GitHub access token pattern"),
    (re.compile(r"Bearer\s+[a-zA-Z0-9_\-\.]{20,}", re.IGNORECASE), "REJECTED_BEARER_TOKEN", "Candidate contains an authorization Bearer token"),
    (re.compile(r"ey[a-zA-Z0-9_\-]{15,}\.ey[a-zA-Z0-9_\-]{15,}", re.IGNORECASE), "REJECTED_JWT_TOKEN", "Candidate contains a JWT token pattern"),
    (re.compile(r"(?:password|passwd|secret|api_key|auth_token)\s*[:=]\s*['\"]?\S{4,}['\"]?", re.IGNORECASE), "REJECTED_CREDENTIAL_STRING", "Candidate contains a password or credential assignment pattern"),
    (re.compile(r"-----BEGIN (?:[A-Z0-9_\- ]+ )?PRIVATE KEY-----"), "REJECTED_PRIVATE_KEY", "Candidate contains a cryptographic private key block"),
]

# Transient phrases that must not be stored as durable memory
TRANSIENT_PHRASES: Set[str] = {
    "hi", "hello", "hey", "good morning", "good evening", "howdy",
    "thanks", "thank you", "thx", "appreciate it",
    "ok", "okay", "k", "sure", "cool", "got it", "noted",
    "yes", "no", "yep", "nope", "bye", "goodbye", "see ya",
}

# Regex to detect claims asserting universal product truth rather than user context
PRODUCT_WIDE_ASSERTIONS = re.compile(
    r"\b(?:asterline\s+(?:supports|costs|retains|allows|includes)|(?:launch|team|enterprise)\s+(?:costs|includes|allows))\b",
    re.IGNORECASE,
)


def evaluate_memory_candidate(content: str, category: str = "general") -> MemoryPolicyDecision:
    """Evaluate whether candidate content is eligible for long-term storage.

    Rules:
    1. Rejects empty, whitespace-only, or overly short (< 6 chars) content.
    2. Rejects obvious secrets (API keys, passwords, private keys, tokens).
    3. Rejects transient chatter ("ok", "thanks", "hello").
    4. Distinguishes company-wide claims from personal workspace context.
    """
    if not content or not content.strip():
        return MemoryPolicyDecision(
            allowed=False,
            reason_code="REJECTED_EMPTY",
            message="Memory content cannot be empty.",
        )

    cleaned = content.strip()
    if len(cleaned) < 6:
        return MemoryPolicyDecision(
            allowed=False,
            reason_code="REJECTED_TOO_SHORT",
            message="Memory candidate is too short to represent durable context.",
        )

    # Check for transient conversation filler
    normalized_phrase = cleaned.lower().strip("!.?,")
    if normalized_phrase in TRANSIENT_PHRASES:
        return MemoryPolicyDecision(
            allowed=False,
            reason_code="REJECTED_TRANSIENT",
            message="Transient conversation remarks are not saved as durable memories.",
        )

    # Check for obvious secrets
    for pattern, code, msg in SECRET_PATTERNS:
        if pattern.search(cleaned):
            return MemoryPolicyDecision(
                allowed=False,
                reason_code=code,
                message=f"Memory candidate rejected: {msg}. Never store secrets in persistent memory.",
            )

    # If the candidate attempts to define company policy rather than user preference
    if PRODUCT_WIDE_ASSERTIONS.search(cleaned) and not any(
        kw in cleaned.lower() for kw in ("my", "our", "we use", "i prefer", "workspace", "project")
    ):
        return MemoryPolicyDecision(
            allowed=False,
            reason_code="REJECTED_COMPANY_POLICY_CLAIM",
            message="Company specifications must be verified against documentation, not stored as personal user memory.",
        )

    return MemoryPolicyDecision(
        allowed=True,
        reason_code="ACCEPTED",
        message="Candidate qualifies for durable user memory.",
        sanitized_content=cleaned,
    )
