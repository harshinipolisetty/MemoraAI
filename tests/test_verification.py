"""Tests for Verification Agent fact-checking, citation validation, and verdicts."""

import pytest
from asterline.agents.verification import VerificationAgent
from asterline.graph.state import Citation, RetrievedChunk


@pytest.fixture
def sample_chunk() -> RetrievedChunk:
    """Fixture providing an authoritative test evidence chunk."""
    return RetrievedChunk(
        chunk_id="pricing:0",
        doc_id="pricing",
        title="Plans and billing",
        category="pricing",
        owner="Asterline Documentation",
        source_url="kb://asterline/pricing/pricing",
        published="20260310",
        version="3",
        text="Launch costs $59 per month and includes 4 seats and 1 million run records per month.",
    )


def test_verification_valid_pass(sample_chunk: RetrievedChunk):
    """Valid claims citing real chunk IDs and exact titles receive 'pass' verdict."""
    verifier = VerificationAgent()
    citations = [
        Citation(
            chunk_id=sample_chunk.chunk_id,
            doc_id=sample_chunk.doc_id,
            title=sample_chunk.title,
        )
    ]
    res = verifier.verify(
        draft_response="Launch costs $59 per month [pricing:0].",
        citations=citations,
        retrieved_chunks=[sample_chunk],
        user_id="user_alice",
    )
    assert res.verdict == "pass"
    assert res.is_grounded is True
    assert res.citation_validity is True
    assert res.privacy_compliant is True


def test_verification_fabricated_chunk_id_revise(sample_chunk: RetrievedChunk):
    """Fabricated chunk ID not in retrieved evidence triggers 'revise' with feedback."""
    verifier = VerificationAgent()
    fabricated_citation = [
        Citation(
            chunk_id="pricing:999",
            doc_id="pricing",
            title="Plans and billing",
        )
    ]
    res = verifier.verify(
        draft_response="Enterprise costs $10,000 [pricing:999].",
        citations=fabricated_citation,
        retrieved_chunks=[sample_chunk],
        user_id="user_alice",
    )
    assert res.verdict == "revise"
    assert res.citation_validity is False
    assert "Fabricated chunk ID" in (res.feedback or "")


def test_verification_title_mismatch_revise(sample_chunk: RetrievedChunk):
    """Citation with incorrect title triggers 'revise'."""
    verifier = VerificationAgent()
    mismatched_citation = [
        Citation(
            chunk_id="pricing:0",
            doc_id="pricing",
            title="Invented Wrong Title",
        )
    ]
    res = verifier.verify(
        draft_response="Launch costs $59 per month [pricing:0].",
        citations=mismatched_citation,
        retrieved_chunks=[sample_chunk],
        user_id="user_alice",
    )
    assert res.verdict == "revise"
    assert "Citation title mismatch" in (res.feedback or "")


def test_verification_missing_evidence_abstain():
    """When no evidence chunks were retrieved, verifier emits 'abstain'."""
    verifier = VerificationAgent()
    res = verifier.verify(
        draft_response="The available documentation does not contain this information.",
        citations=[],
        retrieved_chunks=[],
        user_id="user_alice",
    )
    assert res.verdict == "abstain"


def test_verification_privacy_violation_revise(sample_chunk: RetrievedChunk):
    """Accidental inclusion of other user data triggers 'revise'."""
    verifier = VerificationAgent()
    citations = [
        Citation(
            chunk_id=sample_chunk.chunk_id,
            doc_id=sample_chunk.doc_id,
            title=sample_chunk.title,
        )
    ]
    res = verifier.verify(
        draft_response="Context for user_bob included here [pricing:0].",
        citations=citations,
        retrieved_chunks=[sample_chunk],
        user_id="user_alice",
        other_user_ids={"user_bob"},
    )
    assert res.verdict == "revise"
    assert res.privacy_compliant is False
    assert "Privacy breach" in (res.feedback or "")


def test_verification_genuine_abstention_passes_without_retries(sample_chunk: RetrievedChunk):
    """When chunks exist but response genuinely abstains without citations or hallucinated facts, it receives 'abstain'."""
    verifier = VerificationAgent()
    res = verifier.verify(
        draft_response="I'm sorry, but the provided documentation does not contain any information about quantum telemetry.",
        citations=[],
        retrieved_chunks=[sample_chunk],
        user_id="user_alice",
    )
    assert res.verdict == "abstain"
    assert res.is_grounded is True
    assert res.citation_validity is True
    assert res.privacy_compliant is True


def test_verification_abstention_with_smuggled_claims_triggers_revise(sample_chunk: RetrievedChunk):
    """Answers smuggling affirmative facts without citation must not be approved merely by saying 'does not contain'."""
    verifier = VerificationAgent()
    res = verifier.verify(
        draft_response="The service costs $500 per month, but the documentation does not contain other details.",
        citations=[],
        retrieved_chunks=[sample_chunk],
        user_id="user_alice",
    )
    assert res.verdict == "revise"
    assert "affirmative factual claims" in (res.feedback or "").lower() or "missing" in (res.feedback or "").lower()


def test_verification_abstention_with_fabricated_citation_triggers_revise(sample_chunk: RetrievedChunk):
    """Abstention text that includes fabricated citations must trigger 'revise'."""
    verifier = VerificationAgent()
    fabricated = [Citation(chunk_id="fake:0", doc_id="fake", title="Fake Doc")]
    res = verifier.verify(
        draft_response="The documentation does not contain information about this [fake:0].",
        citations=fabricated,
        retrieved_chunks=[sample_chunk],
        user_id="user_alice",
    )
    assert res.verdict == "revise"
    assert "Fabricated chunk ID" in (res.feedback or "")


def test_verification_abstention_with_privacy_leak_triggers_revise(sample_chunk: RetrievedChunk):
    """Abstention text that leaks another user's identifier must trigger 'revise'."""
    verifier = VerificationAgent()
    res = verifier.verify(
        draft_response="The documentation does not contain details for user_bob.",
        citations=[],
        retrieved_chunks=[sample_chunk],
        user_id="user_alice",
        other_user_ids={"user_bob"},
    )
    assert res.verdict == "revise"
    assert res.privacy_compliant is False

