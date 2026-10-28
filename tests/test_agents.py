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


# -----------------------------------------------------------------------------
# 5. Provider Failures, Timeouts, and Explicit Fallback Metadata
# -----------------------------------------------------------------------------

def test_llm_client_offline_fallback_metadata():
    """LLMClient in offline mode returns is_fallback=True and does not claim live model success."""
    from asterline.config.settings import Settings
    client = LLMClient()
    client.settings = Settings(llm_api_key=None, langchain_api_key=None, _env_file=None)
    res = client.generate_with_metadata("Test prompt")
    assert res.is_fallback is True
    assert res.fallback_reason is not None
    assert "offline" in res.fallback_reason.lower()
    assert res.content != ""


def test_llm_client_handles_provider_exception_with_explicit_fallback(monkeypatch):
    """When a live LLM provider fails (e.g. timeout or connection failure), LLMClient
    catches the exception, marks is_fallback=True, captures error metadata, and never
    reports it as a successful live-model response.
    """
    settings = get_settings()
    # Simulate a configured provider and API key
    monkeypatch.setattr(settings, "llm_api_key", type("Secret", (), {"get_secret_value": lambda self: "fake-key"})())
    monkeypatch.setattr(settings, "llm_provider", "groq")

    client = LLMClient()

    # Simulate provider raising a connection timeout
    def mock_invoke_failing(*args, **kwargs):
        raise TimeoutError("Connection to Groq API timed out after 30.0s")

    # Monkeypatch the provider instantiation or invocation
    import langchain_groq
    class MockChatGroq:
        def __init__(self, *args, **kwargs):
            pass
        def invoke(self, *args, **kwargs):
            mock_invoke_failing()

    monkeypatch.setattr(langchain_groq, "ChatGroq", MockChatGroq)

    res = client.generate_with_metadata("Test prompt under provider failure")
    assert res.is_fallback is True
    assert res.error is not None
    assert "timed out" in res.error
    assert res.fallback_reason == "provider_exception"
    assert "Fallback Mode" in res.content


def test_documentation_agent_preserves_existing_chunks(shared_retriever: HybridRetriever):
    """Documentation Agent reuses existing_chunks when provided instead of re-searching."""
    agent = DocumentationAgent(retriever=shared_retriever)
    custom_chunk = RetrievedChunk(
        chunk_id="pricing:0",
        doc_id="pricing",
        title="Plans and billing",
        category="pricing",
        owner="Asterline Documentation",
        source_url="",
        published="20260310",
        version="3",
        text="Launch costs $59 per month custom preserved test evidence.",
    )
    result = agent.research("Cost of Launch plan?", existing_chunks=[custom_chunk])
    assert len(result["retrieved_chunks"]) == 1
    assert result["retrieved_chunks"][0].text == "Launch costs $59 per month custom preserved test evidence."
    assert "is_fallback" in result
    assert "llm_metadata" in result


def test_investigation_agent_preserves_existing_chunks(shared_retriever: HybridRetriever):
    """Investigation Agent reuses existing_chunks when provided instead of re-searching."""
    agent = InvestigationAgent(retriever=shared_retriever)
    custom_chunk = RetrievedChunk(
        chunk_id="retention:0",
        doc_id="retention",
        title="Data retention policy",
        category="policy",
        owner="Asterline Documentation",
        source_url="",
        published="20260115",
        version="2",
        text="Effective 2026-01-15, raw run records are held in cold recovery storage for 20 days.",
    )
    result = agent.investigate("Retention conflict?", existing_chunks=[custom_chunk])
    assert len(result["retrieved_chunks"]) == 1
    assert result["retrieved_chunks"][0].chunk_id == "retention:0"
    assert "is_fallback" in result
    assert "llm_metadata" in result


# -----------------------------------------------------------------------------
# 6. ChatGroq Client Reuse & Concurrency Safety
# -----------------------------------------------------------------------------

def test_chatgroq_client_reuse_returns_same_instance():
    """Verify that ChatGroq instances with identical configuration are reused across calls."""
    LLMClient.clear_client_cache()
    client1 = LLMClient._get_groq_client(
        model_name="llama-3.3-70b-versatile",
        api_key="test-api-key-reuse",
        temperature=0.0,
        max_tokens=1024,
        request_timeout=30.0,
    )
    client2 = LLMClient._get_groq_client(
        model_name="llama-3.3-70b-versatile",
        api_key="test-api-key-reuse",
        temperature=0.0,
        max_tokens=1024,
        request_timeout=30.0,
    )
    assert client1 is client2, "Expected identical ChatGroq instance to be returned from cache"


def test_chatgroq_client_distinct_configs_create_distinct_instances():
    """Verify that different parameters produce distinct cached clients."""
    LLMClient.clear_client_cache()
    client1 = LLMClient._get_groq_client(
        model_name="llama-3.3-70b-versatile",
        api_key="test-api-key",
        temperature=0.0,
        max_tokens=512,
        request_timeout=30.0,
    )
    client2 = LLMClient._get_groq_client(
        model_name="llama-3.3-70b-versatile",
        api_key="test-api-key",
        temperature=0.0,
        max_tokens=1500,
        request_timeout=30.0,
    )
    assert client1 is not client2, "Expected distinct instances for distinct max_tokens settings"


def test_chatgroq_client_thread_safety_concurrent_access():
    """Verify concurrent requests for the client do not race or corrupt the cache."""
    import concurrent.futures

    LLMClient.clear_client_cache()
    instances = []

    def get_client():
        return LLMClient._get_groq_client(
            model_name="llama-3.3-70b-versatile",
            api_key="concurrent-key-123",
            temperature=0.0,
            max_tokens=1024,
            request_timeout=30.0,
        )

    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as executor:
        futures = [executor.submit(get_client) for _ in range(20)]
        for fut in concurrent.futures.as_completed(futures):
            instances.append(fut.result())

    assert len(instances) == 20
    first_instance = instances[0]
    for inst in instances:
        assert inst is first_instance, "All concurrent calls must return the identical cached instance"


def test_chatgroq_client_clear_cache():
    """Verify clear_client_cache removes previously cached instances."""
    LLMClient.clear_client_cache()
    c1 = LLMClient._get_groq_client("llama-3.3-70b-versatile", "key-clear", 0.0, 1024, 30.0)
    LLMClient.clear_client_cache()
    c2 = LLMClient._get_groq_client("llama-3.3-70b-versatile", "key-clear", 0.0, 1024, 30.0)
    assert c1 is not c2, "Clearing cache must result in a new instance on next call"


# -----------------------------------------------------------------------------
# 7. Priority 3: Route-Specific Output Limits & Prompt Optimization
# -----------------------------------------------------------------------------

def test_route_specific_output_limits_configured(shared_retriever: HybridRetriever):
    """Verify DocumentationAgent defaults to 512 tokens and InvestigationAgent defaults to 1500 tokens."""
    captured_kwargs = {}

    class MockTrackingLLM:
        def __init__(self):
            self.settings = type("MockSettings", (), {
                "llm_api_key": type("Secret", (), {"get_secret_value": lambda self: "fake-key"})(),
                "llm_provider": "groq",
                "llm_model": "llama-3.3-70b-versatile",
                "llm_max_tokens": 2048,
            })()
            self.mock_synthesizer = None

        def generate_with_metadata(self, prompt: str, system_prompt: str, max_tokens: Optional[int] = None):
            captured_kwargs["max_tokens"] = max_tokens
            from asterline.agents.llm import LLMResponse
            return LLMResponse(
                content="Sample response [pricing:0]",
                is_fallback=False,
                provider="mock",
                model="mock",
            )

    tracking_llm = MockTrackingLLM()

    # Documentation Agent
    doc_agent = DocumentationAgent(retriever=shared_retriever, llm_client=tracking_llm)
    assert doc_agent.max_tokens == 512
    doc_agent.research("What is the cost of Team plan?")
    assert captured_kwargs.get("max_tokens") == 512, f"Expected 512 max_tokens for doc agent, got {captured_kwargs.get('max_tokens')}"

    # Investigation Agent
    invest_agent = InvestigationAgent(retriever=shared_retriever, llm_client=tracking_llm)
    assert invest_agent.max_tokens == 1500
    invest_agent.investigate("Investigate incident INC-2025-12 duplicate exports")
    assert captured_kwargs.get("max_tokens") == 1500, f"Expected 1500 max_tokens for investigation agent, got {captured_kwargs.get('max_tokens')}"


# -----------------------------------------------------------------------------
# 8. Priority 4: Safe Public Documentation Cache & Memory Isolation
# -----------------------------------------------------------------------------

def test_cache_hit_for_public_documentation_query(shared_retriever: HybridRetriever):
    """Public query without personal memories is served from cache on subsequent call."""
    from asterline.rag.cache import DocumentationAnswerCache
    cache = DocumentationAnswerCache(corpus_hash="CORPUS_V1")

    call_count = 0
    class CountingMockLLM:
        def __init__(self):
            self.settings = type("MockSettings", (), {
                "llm_api_key": type("Secret", (), {"get_secret_value": lambda self: "fake-key"})(),
                "llm_provider": "groq",
                "llm_model": "llama-3.3-70b-versatile",
                "llm_max_tokens": 512,
            })()
            self.mock_synthesizer = None

        def generate_with_metadata(self, prompt: str, system_prompt: str, max_tokens: Optional[int] = None):
            nonlocal call_count
            call_count += 1
            from asterline.agents.llm import LLMResponse
            return LLMResponse(
                content="The Team plan costs $449/mo [pricing:0].",
                is_fallback=False,
                provider="mock",
                model="mock",
            )

    agent = DocumentationAgent(retriever=shared_retriever, llm_client=CountingMockLLM(), cache=cache)

    # First call: executes LLM and populates cache
    res1 = agent.research("What is the cost of Team plan?")
    assert call_count == 1
    assert res1.get("is_cached") is not True

    # Second identical public call: served directly from cache without LLM invocation
    res2 = agent.research("What is the cost of Team plan?")
    assert call_count == 1, "LLM must NOT be called on cache hit"
    assert res2.get("is_cached") is True
    assert res2["draft_response"] == res1["draft_response"]
    assert len(res2["citations"]) == len(res1["citations"])


def test_cache_never_serves_or_caches_with_user_memories(shared_retriever: HybridRetriever):
    """Personalized queries with user memories bypass cache and never store private context."""
    from asterline.rag.cache import DocumentationAnswerCache
    cache = DocumentationAnswerCache(corpus_hash="CORPUS_V1")

    call_count = 0
    class CountingMockLLM:
        def __init__(self):
            self.settings = type("MockSettings", (), {
                "llm_api_key": type("Secret", (), {"get_secret_value": lambda self: "fake-key"})(),
                "llm_provider": "groq",
                "llm_model": "llama-3.3-70b-versatile",
                "llm_max_tokens": 512,
            })()
            self.mock_synthesizer = None

        def generate_with_metadata(self, prompt: str, system_prompt: str, max_tokens: Optional[int] = None):
            nonlocal call_count
            call_count += 1
            from asterline.agents.llm import LLMResponse
            return LLMResponse(
                content="The Team plan costs $449/mo [pricing:0].",
                is_fallback=False,
                provider="mock",
                model="mock",
            )

    agent = DocumentationAgent(retriever=shared_retriever, llm_client=CountingMockLLM(), cache=cache)
    personal_memories = [{"id": "m1", "content": "User prefers Snowflake warehouse", "is_active": True}]

    # Query with memories
    res = agent.research("What is the cost of Team plan?", user_memories=personal_memories)
    assert call_count == 1
    assert res.get("is_cached") is not True

    # Check cache is completely empty
    assert len(cache) == 0, "Personalized query must NEVER be written to public documentation cache"


def test_cache_invalidated_by_corpus_hash_change():
    """Cache misses when corpus hash changes."""
    from asterline.graph.state import Citation
    from asterline.rag.cache import DocumentationAnswerCache
    cache1 = DocumentationAnswerCache(corpus_hash="HASH_OLD")
    cache1.set("query test", {"draft_response": "answer [doc:0]", "citations": [Citation(chunk_id="doc:0", doc_id="doc", title="T", source_url="")]})

    cache2 = DocumentationAnswerCache(corpus_hash="HASH_NEW")
    assert cache2.get("query test") is None, "Different corpus hash must result in cache miss"


def test_cache_never_stores_transient_fallback():
    """Cache rejects transient fallback results."""
    from asterline.rag.cache import DocumentationAnswerCache
    cache = DocumentationAnswerCache(corpus_hash="HASH_A")
    cache.set("query fallback", {"draft_response": "fallback", "is_fallback": True, "citations": []})
    assert cache.get("query fallback") is None, "Fallback answers must never be cached"


def test_memory_manager_personal_context_recall(memory_service: MemoryService):
    """Memory Manager answers personal context inquiries directly from user memories with isolation."""
    agent = MemoryManagerAgent(memory_service)

    # Alice remembers her warehouse
    agent.handle_request("user_alice", "Remember that our team uses Snowflake for data warehousing in eu-west-1.")

    # Alice asks what data warehouse our team uses
    res_alice = agent.handle_request("user_alice", "What data warehouse does our team use?")
    assert res_alice["status"] == "COMPLETED"
    assert "Snowflake" in res_alice["response"]

    # Bob (who has no memories) asks the same question
    res_bob = agent.handle_request("user_bob", "What data warehouse does our team use?")
    assert res_bob["status"] == "COMPLETED"
    assert "Snowflake" not in res_bob["response"]
    assert "do not have any stored personal memories" in res_bob["response"].lower()





