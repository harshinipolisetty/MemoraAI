"""Tests for Documentation Agent, Investigation Agent, Conflict Resolution, and Memory Manager."""

from pathlib import Path
import pytest
from asterline.agents.conflict_resolution import ConflictResolutionEngine
from asterline.agents.documentation import DocumentationAgent
from asterline.agents.investigation import InvestigationAgent
from asterline.agents.llm import LLMClient
from asterline.agents.memory_manager import MemoryManagerAgent
from asterline.config.settings import get_settings
from asterline.graph.state import RetrievedChunk
from asterline.memory.service import MemoryService
from asterline.memory.store import SQLiteStore
from asterline.rag.ingestion import load_and_validate_corpus
from asterline.rag.retriever import HybridRetriever


@pytest.fixture(scope="module")
def shared_retriever(tmp_path_factory) -> HybridRetriever:
    """Fixture providing initialized HybridRetriever over corpus."""
    settings = get_settings()
    corpus_path = settings.get_resolved_corpus_path()
    chunks, _ = load_and_validate_corpus(corpus_path)
    chroma_dir = tmp_path_factory.mktemp("chroma_agents")
    return HybridRetriever(corpus_chunks=chunks, chroma_persist_dir=chroma_dir)


@pytest.fixture
def memory_service(tmp_path: Path) -> MemoryService:
    """Fixture providing isolated MemoryService."""
    store = SQLiteStore(tmp_path / "agents_test_memory.db")
    return MemoryService(store)


# -----------------------------------------------------------------------------
# 1. Documentation Research Agent
# -----------------------------------------------------------------------------

def test_doc_agent_factual_query(shared_retriever: HybridRetriever):
    """Documentation Agent answers direct factual questions using retrieved evidence."""
    agent = DocumentationAgent(retriever=shared_retriever)
    result = agent.research("What is the cost of the Launch plan?")

    assert len(result["retrieved_chunks"]) > 0
    assert len(result["citations"]) > 0
    assert any(c.chunk_id == "pricing:0" for c in result["citations"])
    assert "$59" in result["draft_response"] or "pricing:0" in result["draft_response"]


def test_doc_agent_missing_evidence_abstains(shared_retriever: HybridRetriever):
    """Documentation Agent states when evidence is missing and does not fabricate citations."""
    agent = DocumentationAgent(retriever=shared_retriever)
    # Query about completely unrelated topic outside Asterline documentation
    result = agent.research("What is the quantum teleportation protocol for Mars rovers?")

    # Should not fabricate any chunk IDs
    valid_cids = {c.chunk_id for c in shared_retriever.corpus_chunks}
    for c in result["citations"]:
        assert c.chunk_id in valid_cids


def test_doc_agent_preserves_metadata(shared_retriever: HybridRetriever):
    """Documentation Agent preserves chunk title, version, and doc_id in citations."""
    agent = DocumentationAgent(retriever=shared_retriever)
    result = agent.research("What are the ingestion rate limits across plans?")

    assert len(result["citations"]) > 0
    top_cit = result["citations"][0]
    assert top_cit.title != ""
    assert top_cit.doc_id != ""


# -----------------------------------------------------------------------------
# 2. Temporal & Conflict Resolution (The 5 Key Scenarios + Unresolved)
# -----------------------------------------------------------------------------

def test_conflict_resolution_retention_supersession():
    """Case 1: retention:0 explicitly supersedes security:1 cold recovery period."""
    chunks = [
        RetrievedChunk(
            chunk_id="security:1",
            doc_id="security",
            title="Security overview",
            category="policy",
            owner="Asterline Documentation",
            source_url="",
            published="20250315",
            version="1",
            text="raw run records remain in cold recovery storage for 60 days",
        ),
        RetrievedChunk(
            chunk_id="retention:0",
            doc_id="retention",
            title="Data retention policy",
            category="policy",
            owner="Asterline Documentation",
            source_url="",
            published="20260115",
            version="2",
            text="Effective 2026-01-15, raw run records are held in cold recovery storage for 20 days... This policy supersedes the earlier 60-day cold-storage period.",
        ),
    ]
    decision = ConflictResolutionEngine.resolve(chunks, "cold recovery storage retention")
    assert decision.tier == 1
    assert "retention:0" in decision.controlling_chunk_ids
    assert "security:1" in decision.superseded_chunk_ids
    assert "supersedes" in decision.reasoning.lower()


def test_conflict_resolution_oncall_stipend_supersession():
    """Case 2: handbook:1 explicitly supersedes oncall:0 on-call compensation."""
    chunks = [
        RetrievedChunk(
            chunk_id="oncall:0",
            doc_id="oncall",
            title="On-call runbook",
            category="engineering",
            owner="Asterline Documentation",
            source_url="",
            published="20250410",
            version="1",
            text="lists an on-call stipend of $140 per week.",
        ),
        RetrievedChunk(
            chunk_id="handbook:1",
            doc_id="handbook",
            title="Employee handbook",
            category="policy",
            owner="Asterline Documentation",
            source_url="",
            published="20260105",
            version="3",
            text="Effective 2026-01-01 the weekly on-call stipend is $175, replacing $140. This compensation update supersedes stipend figures in older operational runbooks.",
        ),
    ]
    decision = ConflictResolutionEngine.resolve(chunks, "weekly on-call stipend")
    assert decision.tier == 1
    assert "handbook:1" in decision.controlling_chunk_ids
    assert "oncall:0" in decision.superseded_chunk_ids
    assert "$175" in decision.reasoning


def test_conflict_resolution_aql_timeout():
    """Case 3: query:0 and release-23:0 (v2.3, 50s) supersede historical 25s limit."""
    chunks = [
        RetrievedChunk(
            chunk_id="release-20:2",
            doc_id="release-20",
            title="Version 2.0 release notes",
            category="release-notes",
            owner="Asterline Documentation",
            source_url="",
            published="20250812",
            version="2.0",
            text="Version 2.0 retained a 25-second AQL query timeout.",
        ),
        RetrievedChunk(
            chunk_id="query:0",
            doc_id="query",
            title="AQL query reference",
            category="product",
            owner="Asterline Documentation",
            source_url="",
            published="20260310",
            version="2.3",
            text="Queries return at most 8000 rows and time out after 50 seconds in version 2.3. Earlier versions used a 25-second timeout.",
        ),
    ]
    decision = ConflictResolutionEngine.resolve(chunks, "AQL query timeout")
    assert decision.tier == 3
    assert "query:0" in decision.controlling_chunk_ids
    assert "release-20:2" in decision.superseded_chunk_ids


def test_conflict_resolution_export_incident():
    """Case 4: incident-12:1 and release-21:2 resolve duplicate export cause & fix."""
    chunks = [
        RetrievedChunk(
            chunk_id="incident-12:1",
            doc_id="incident-12",
            title="INC-2025-12 duplicate exports",
            category="incident",
            owner="Asterline Documentation",
            source_url="",
            published="20251210",
            version="1",
            text="Heron changed query result ordering, which broke position-based export deduplication. The fix replaced row position with the stable pair (export_job_id, record_hash).",
        ),
        RetrievedChunk(
            chunk_id="release-21:2",
            doc_id="release-21",
            title="Version 2.1 release notes",
            category="release-notes",
            owner="Asterline Documentation",
            source_url="",
            published="20251208",
            version="2.1.2",
            text="Patch 2.1.2 shipped on 2025-12-08. It changed Warehouse Export deduplication to (export_job_id, record_hash), resolving INC-2025-12.",
        ),
    ]
    decision = ConflictResolutionEngine.resolve(chunks, "duplicate exports INC-2025-12")
    assert "incident-12:1" in decision.controlling_chunk_ids
    assert "export_job_id" in decision.reasoning


def test_conflict_resolution_apac_launch_status():
    """Case 5: regions:2 and faq:1 acknowledge lack of announced timeline for APAC."""
    chunks = [
        RetrievedChunk(
            chunk_id="regions:2",
            doc_id="regions",
            title="Regions and workspace setup",
            category="product",
            owner="Asterline Documentation",
            source_url="",
            published="20260115",
            version="2.1",
            text="No launch date has been announced for an Asia-Pacific region.",
        ),
        RetrievedChunk(
            chunk_id="faq:1",
            doc_id="faq",
            title="Product FAQ",
            category="onboarding",
            owner="Asterline Documentation",
            source_url="",
            published="20260311",
            version="3",
            text="There is no announced timeline for an Asia-Pacific region.",
        ),
    ]
    decision = ConflictResolutionEngine.resolve(chunks, "Has Asterline launched in APAC?")
    assert decision.is_unresolved is False
    assert "announced" in decision.reasoning.lower()


def test_investigation_agent_synthesizes_multidoc(shared_retriever: HybridRetriever):
    """Investigation Agent investigates multi-document inquiries and applies conflict resolution."""
    agent = InvestigationAgent(retriever=shared_retriever)
    result = agent.investigate("Why did duplicate warehouse export rows occur in INC-2025-12?")

    assert len(result["retrieved_chunks"]) >= 2
    assert len(result["citations"]) > 0
    assert "incident-12:1" in [c.chunk_id for c in result["citations"]] or "release-21:2" in [c.chunk_id for c in result["citations"]]
    assert "Heron" in result["draft_response"] or "export_job_id" in result["draft_response"]


# -----------------------------------------------------------------------------
# 3. Memory Manager Agent
# -----------------------------------------------------------------------------

def test_memory_manager_remember_and_view(memory_service: MemoryService):
    """Memory Manager stores valid user preference and retrieves it on demand."""
    agent = MemoryManagerAgent(memory_service)

    # Store memory
    res_store = agent.handle_request(
        user_id="user_alice",
        user_message="Please remember that our team uses Snowflake for warehouse export in the EU region.",
    )
    assert res_store["status"] == "SUCCESS"
    assert "remembered" in res_store["response"].lower()

    # List memories
    res_view = agent.handle_request(
        user_id="user_alice",
        user_message="What do you remember about me?",
    )
    assert res_view["status"] == "COMPLETED"
    assert "Snowflake" in res_view["response"]


def test_memory_manager_rejects_secrets(memory_service: MemoryService):
    """Memory Manager rejects secrets and does not persist them."""
    agent = MemoryManagerAgent(memory_service)
    res = agent.handle_request(
        user_id="user_alice",
        user_message="Remember that my ingestion key is ast_secret_key_1234567890abcdef",
    )
    assert res["status"] == "POLICY_REJECTED"
    assert "cannot remember" in res["response"].lower()

    # Verify storage remains empty
    assert len(memory_service.list_user_memories("user_alice")) == 0


def test_memory_manager_forget_and_verifies_deletion(memory_service: MemoryService):
    """Memory Manager forgets context, verifies deletion at persistence layer, and audits."""
    agent = MemoryManagerAgent(memory_service)

    # First store memory
    agent.handle_request("user_alice", "Remember that my preferred tier is Team.")
    assert len(memory_service.list_user_memories("user_alice")) == 1

    # Now forget
    res_forget = agent.handle_request("user_alice", "Please forget my preferred tier.")
    assert res_forget["status"] == "SUCCESS"
    assert "forgotten" in res_forget["response"].lower()

    # Verified at persistence layer
    assert len(memory_service.list_user_memories("user_alice")) == 0


def test_memory_manager_user_isolation(memory_service: MemoryService):
    """Memory Manager enforces isolation between user_alice and user_bob."""
    agent = MemoryManagerAgent(memory_service)

    agent.handle_request("user_alice", "Remember that Alice leads the telemetry team.")
    agent.handle_request("user_bob", "Remember that Bob leads the billing team.")

    alice_view = agent.handle_request("user_alice", "What do you know about me?")
    bob_view = agent.handle_request("user_bob", "What do you know about me?")

    assert "Alice" in alice_view["response"]
    assert "Bob" not in alice_view["response"]

    assert "Bob" in bob_view["response"]
    assert "Alice" not in bob_view["response"]
