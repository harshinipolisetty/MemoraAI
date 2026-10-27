"""Tests for workflow state schemas, Pydantic models, and type constraints."""

import pytest
from pydantic import ValidationError
from asterline.graph.state import (
    Citation,
    ConversationMessage,
    MemoryItem,
    ResearchAssistantState,
    RetrievedChunk,
    VerificationResult,
)


def test_citation_model_validation():
    """Citation validates required chunk_id, doc_id, and title."""
    cit = Citation(
        chunk_id="pricing:0",
        doc_id="pricing",
        title="Plans and billing",
        source_url="kb://asterline/pricing/pricing",
    )
    assert cit.chunk_id == "pricing:0"
    assert cit.title == "Plans and billing"

    with pytest.raises(ValidationError):
        Citation(chunk_id="only_chunk_id")


def test_retrieved_chunk_model_validation():
    """RetrievedChunk preserves all documentation metadata fields."""
    chunk = RetrievedChunk(
        chunk_id="incident-12:1",
        doc_id="incident-12",
        title="INC-2025-12 duplicate exports",
        category="incident",
        owner="Asterline Documentation",
        source_url="kb://asterline/incident/incident-12",
        published="20251210",
        version="1",
        text="Heron changed query result ordering...",
        score=0.92,
    )
    assert chunk.category == "incident"
    assert chunk.version == "1"
    assert chunk.score == 0.92


def test_verification_result_verdicts():
    """VerificationResult enforces permitted verdict values ('pass', 'revise', 'abstain')."""
    valid_res = VerificationResult(
        verdict="pass",
        is_grounded=True,
        citation_validity=True,
        privacy_compliant=True,
    )
    assert valid_res.verdict == "pass"

    with pytest.raises(ValidationError):
        VerificationResult(
            verdict="invalid_verdict_value",  # type: ignore
            is_grounded=True,
            citation_validity=True,
            privacy_compliant=True,
        )


def test_research_assistant_state_dict_structure():
    """TypedDict ResearchAssistantState accepts valid dictionary matching workflow schema."""
    state: ResearchAssistantState = {
        "user_id": "synthetic_user_alice",
        "conversation_id": "conv_test_123",
        "user_message": "What is the ingestion rate limit for Team plan?",
        "conversation_history": [],
        "active_user_memories": [],
        "route": "factual",
        "routing_reason": "Single plan rate limit lookup",
        "retry_count": 0,
        "search_queries": ["Team plan ingestion rate limit requests per second"],
        "retrieved_chunks": [],
        "draft_response": "",
        "citations": [],
        "verification": None,
        "final_response": "",
        "memory_operations_executed": [],
    }
    assert state["user_id"] == "synthetic_user_alice"
    assert state["route"] == "factual"
    assert state["retry_count"] == 0
