"""Output guardrails for Asterline Research Assistant.

Validates citation integrity against supplied retrieved chunks,
scrubs accidental secret leakage, verifies cross-user privacy boundaries,
and ensures consistent structured response output.
"""

from dataclasses import dataclass, field
import re
from typing import Dict, List, Optional, Set
from asterline.graph.state import Citation, RetrievedChunk

# Patterns to scrub from final responses to prevent leaking secrets/keys
OUTPUT_SECRET_PATTERNS = [
    re.compile(r"ast_[a-zA-Z0-9_\-]{16,}", re.IGNORECASE),
    re.compile(r"sk-[a-zA-Z0-9_\-]{20,}", re.IGNORECASE),
    re.compile(r"gh[pousr]_[a-zA-Z0-9]{20,}", re.IGNORECASE),
    re.compile(r"Bearer\s+[a-zA-Z0-9_\-\.]{20,}", re.IGNORECASE),
    re.compile(r"(?:password|secret)\s*[:=]\s*['\"]?\S+['\"]?", re.IGNORECASE),
]


@dataclass
class OutputGuardrailResult:
    """Outcome of output guardrail validation."""

    is_valid: bool
    validated_response: str
    validated_citations: List[Citation]
    error_message: Optional[str] = None
    feedback: Optional[str] = None
    flags: List[str] = field(default_factory=list)


def validate_output(
    response_text: str,
    citations: List[Citation],
    retrieved_chunks: List[RetrievedChunk],
    current_user_id: str,
    other_user_ids: Optional[Set[str]] = None,
) -> OutputGuardrailResult:
    """Validate and filter response text and citations before delivery to user.

    Rules:
    1. Every cited chunk_id must exist in the retrieved evidence chunks.
    2. Every cited title must match the title of the retrieved chunk.
    3. Rejects fabricated citations.
    4. Scrubs any accidental secret or API key patterns from response text.
    5. Confirms no identifiers from other users are leaked.
    """
    flags: List[str] = []
    chunk_map: Dict[str, RetrievedChunk] = {c.chunk_id: c for c in retrieved_chunks}

    # 1. Validate Citations
    validated_citations: List[Citation] = []
    invalid_citations: List[str] = []

    for cit in citations:
        if cit.chunk_id not in chunk_map:
            invalid_citations.append(f"Chunk ID '{cit.chunk_id}' was not in retrieved evidence")
            continue

        real_chunk = chunk_map[cit.chunk_id]
        if cit.title != real_chunk.title:
            invalid_citations.append(
                f"Title mismatch for '{cit.chunk_id}': cited '{cit.title}' but real is '{real_chunk.title}'"
            )
            continue

        validated_citations.append(cit)

    if invalid_citations:
        flags.append("INVALID_CITATIONS_DETECTED")
        feedback_msg = "Invalid citations detected: " + "; ".join(invalid_citations)
        return OutputGuardrailResult(
            is_valid=False,
            validated_response="",
            validated_citations=[],
            error_message=feedback_msg,
            feedback=feedback_msg,
            flags=flags,
        )

    # 2. Check for Cross-User Privacy Leaks
    if other_user_ids:
        for other_id in other_user_ids:
            if other_id != current_user_id and other_id in response_text:
                flags.append("CROSS_USER_DATA_LEAK")
                return OutputGuardrailResult(
                    is_valid=False,
                    validated_response="",
                    validated_citations=[],
                    error_message=f"Privacy violation: Output contained reference to unauthorized user '{other_id}'.",
                    feedback="Cross-user data leakage detected. Strip unauthorized context.",
                    flags=flags,
                )

    # 3. Scrub accidental secret patterns
    sanitized_text = response_text
    for pattern in OUTPUT_SECRET_PATTERNS:
        if pattern.search(sanitized_text):
            flags.append("SECRET_REDACTED")
            sanitized_text = pattern.sub("[REDACTED_SECRET]", sanitized_text)

    return OutputGuardrailResult(
        is_valid=True,
        validated_response=sanitized_text,
        validated_citations=validated_citations,
        flags=flags,
    )
