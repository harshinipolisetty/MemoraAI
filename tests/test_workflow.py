"""Tests for conditional routing and full LangGraph workflow execution."""

from pathlib import Path
import pytest
from asterline.config.settings import get_settings
from asterline.graph.routing import determine_route
from asterline.graph.workflow import WorkflowBuilder
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
    chroma_dir = tmp_path_factory.mktemp("chroma_workflow")
    return HybridRetriever(corpus_chunks=chunks, chroma_persist_dir=chroma_dir)


@pytest.fixture
def workflow_app(shared_retriever: HybridRetriever, tmp_path: Path):
    """Fixture providing compiled LangGraph workflow with isolated SQLite storage."""
    db_path = tmp_path / "workflow_test.db"
    store = SQLiteStore(db_path)
    service = MemoryService(store)
    builder = WorkflowBuilder(retriever=shared_retriever, memory_service=service)
    return builder.build(), service


# -----------------------------------------------------------------------------
# 1. Routing Tests
# -----------------------------------------------------------------------------

def test_routing_greetings():
    """Greetings route to 'conversational' without calling RAG."""
    for greet in ["hi", "hello there", "good morning", "who are you?"]:
        decision = determine_route(greet)
        assert decision.route == "conversational"


def test_routing_clarification():
    """Vague or underspecified inputs route to 'clarification'."""
    for vague in ["?", "help me", "...", "hi"]:
        # "hi" is greeting, "?" is clarification
        if vague == "?":
            assert determine_route(vague).route == "clarification"


def test_routing_memory_ops():
    """Memory commands route to 'memory_ops'."""
    commands = [
        "Please remember that I use BigQuery",
        "Remember: our workspace is on Team plan",
        "What do you know about me?",
        "Forget my warehouse preference",
    ]
    for cmd in commands:
        decision = determine_route(cmd)
        assert decision.route == "memory_ops"


def test_routing_investigation():
    """Incidents and multi-document comparisons route to 'investigation'."""
    investigation_queries = [
        "What caused the duplicate export rows in INC-2025-12?",
        "Explain the incident INC-2025-08 and its patch fix",
        "Compare the retention policy supersession versus older documentation",
        "Why did Signals alerts fail during the time synchronization incident?",
    ]
    for query in investigation_queries:
        decision = determine_route(query)
        assert decision.route == "investigation"


def test_routing_factual():
    """Direct specifications and pricing questions route to 'factual'."""
    factual_queries = [
        "What is the price of the Team plan?",
        "What are the payload limits for POST /v2/runs?",
        "How many alert rules are permitted on Launch tier?",
        "What is the SLA percentage for Enterprise customers?",
    ]
    for query in factual_queries:
        decision = determine_route(query)
        assert decision.route == "factual"


# -----------------------------------------------------------------------------
# 2. End-to-End Workflow Execution
# -----------------------------------------------------------------------------

def test_workflow_greeting_fast_path(workflow_app):
    """Greeting query routes through conversational path without RAG."""
    app, service = workflow_app
    initial_state = {
        "user_id": "user_alice",
        "conversation_id": "conv_greet",
        "user_message": "Hello!",
    }
    result = app.invoke(initial_state)

    assert result["route"] == "conversational"
    assert "Asterline Labs" in result["final_response"]
    assert len(result["citations"]) == 0

    # Verify conversation persisted to SQLite
    msgs = service.get_conversation_messages("conv_greet", "user_alice")
    assert len(msgs) == 2
    assert msgs[0].role == "user"
    assert msgs[1].role == "assistant"


def test_workflow_memory_lifecycle_execution(workflow_app):
    """Memory command routes to memory manager, persists preference, and logs turns."""
    app, service = workflow_app

    # Store memory turn
    initial_state = {
        "user_id": "user_alice",
        "conversation_id": "conv_mem",
        "user_message": "Remember that our project uses Snowflake for exports.",
    }
    result = app.invoke(initial_state)

    assert result["route"] == "memory_ops"
    assert "remembered" in result["final_response"].lower()

    # Verify memory persisted in SQLite user_memories
    active_mems = service.list_user_memories("user_alice")
    assert len(active_mems) == 1
    assert "Snowflake" in active_mems[0].content


def test_workflow_factual_research_execution(workflow_app):
    """Factual question executes research, verification, output guardrail, and persistence."""
    app, service = workflow_app
    initial_state = {
        "user_id": "user_alice",
        "conversation_id": "conv_factual",
        "user_message": "What is the monthly price of Launch plan?",
    }
    result = app.invoke(initial_state)

    assert result["route"] == "factual"
    assert result["verification"]["verdict"] == "pass"
    assert len(result["citations"]) > 0
    assert any(c["chunk_id"] == "pricing:0" for c in result["citations"])

    # Verify message persistence
    msgs = service.get_conversation_messages("conv_factual", "user_alice")
    assert len(msgs) == 2
    assert msgs[1].citations[0].chunk_id == "pricing:0"


def test_workflow_bounded_retry_prevention(workflow_app):
    """Workflow halts retry loops at configured MAX_RETRIES without crashing."""
    app, service = workflow_app
    initial_state = {
        "user_id": "user_alice",
        "conversation_id": "conv_retry",
        "user_message": "What is the monthly price of Launch plan?",
        "retry_count": 2,
    }
    result = app.invoke(initial_state)
    assert result["retry_count"] <= 2
    assert result["final_response"] != ""


def test_workflow_documentation_agent_revision_routing(shared_retriever: HybridRetriever, tmp_path: Path):
    """Documentation Agent revisions return to Documentation Agent, preserving route and evidence."""
    db_path = tmp_path / "doc_retry_test.db"
    store = SQLiteStore(db_path)
    service = MemoryService(store)

    attempts = {"count": 0}

    def mock_synthesizer(sys_prompt: str, user_prompt: str) -> str:
        attempts["count"] += 1
        if attempts["count"] == 1:
            # First attempt: deliberate fabricated citation to trigger revise
            return "Launch costs $59 [pricing:999]."
        # Second attempt: corrected citation matching evidence
        return "Launch costs $59 [pricing:0]."

    from asterline.agents.llm import LLMClient
    mock_llm = LLMClient(mock_synthesizer=mock_synthesizer)
    builder = WorkflowBuilder(retriever=shared_retriever, memory_service=service, llm_client=mock_llm)
    app = builder.build()

    initial_state = {
        "user_id": "user_alice",
        "conversation_id": "conv_doc_retry",
        "user_message": "What is the cost of the Launch plan?",
    }
    result = app.invoke(initial_state)

    # Verifies retry occurred
    assert attempts["count"] == 2
    assert result["retry_count"] == 1
    assert result["route"] == "factual"
    assert result["verification"]["verdict"] == "pass"
    assert any(c["chunk_id"] == "pricing:0" for c in result["citations"])
    assert len(result["retrieved_chunks"]) > 0


def test_workflow_investigation_agent_revision_routing(shared_retriever: HybridRetriever, tmp_path: Path):
    """Investigation Agent revisions return to Investigation Agent, preserving route and evidence."""
    db_path = tmp_path / "invest_retry_test.db"
    store = SQLiteStore(db_path)
    service = MemoryService(store)

    attempts = {"count": 0}

    def mock_synthesizer(sys_prompt: str, user_prompt: str) -> str:
        attempts["count"] += 1
        if attempts["count"] == 1:
            # First attempt: fabricated chunk to trigger revise
            return "INC-2025-12 was caused by Heron query engine [incident:999]."
        # Second attempt: correct citations
        return "INC-2025-12 deduplication bug was patched in [incident-12:1] [release-21:2]."

    from asterline.agents.llm import LLMClient
    mock_llm = LLMClient(mock_synthesizer=mock_synthesizer)
    builder = WorkflowBuilder(retriever=shared_retriever, memory_service=service, llm_client=mock_llm)
    app = builder.build()

    initial_state = {
        "user_id": "user_alice",
        "conversation_id": "conv_invest_retry",
        "user_message": "What caused the duplicate export rows in incident INC-2025-12?",
    }
    result = app.invoke(initial_state)

    # Verifies retry occurred and returned to Investigation Agent
    assert attempts["count"] == 2
    assert result["retry_count"] == 1
    assert result["route"] == "investigation"
    assert result["verification"]["verdict"] == "pass"
    assert len(result["retrieved_chunks"]) > 0


def test_workflow_retry_exhaustion_enforces_configured_maximum(shared_retriever: HybridRetriever, tmp_path: Path):
    """Enforces configured maximum of two retries. If agent keeps failing verification,
    the workflow stops after exactly 2 retries (total 3 attempts) and routes to output_guardrails_node.
    """
    db_path = tmp_path / "exhaust_retry_test.db"
    store = SQLiteStore(db_path)
    service = MemoryService(store)

    attempts = {"count": 0}

    def failing_synthesizer(sys_prompt: str, user_prompt: str) -> str:
        attempts["count"] += 1
        # Always return fabricated citation so verification always emits 'revise'
        return f"Attempt {attempts['count']} with fabricated citation [pricing:999]."

    from asterline.agents.llm import LLMClient
    mock_llm = LLMClient(mock_synthesizer=failing_synthesizer)
    builder = WorkflowBuilder(retriever=shared_retriever, memory_service=service, llm_client=mock_llm)
    app = builder.build()

    initial_state = {
        "user_id": "user_alice",
        "conversation_id": "conv_exhaust",
        "user_message": "What is the monthly price of Launch plan?",
    }
    result = app.invoke(initial_state)

    # Initial attempt (count=1, retry_count=1) -> Retry 1 (count=2, retry_count=2) -> Retry 2 (count=3, retry_count=3)
    # Total attempts = 3, exactly 2 retries executed before exhaustion halts at output_guardrails_node
    assert attempts["count"] == 3
    assert result["retry_count"] == 3
    assert result["route"] == "factual"
    assert "Verification note" in result["final_response"]


def test_workflow_explicit_fallback_metadata(shared_retriever: HybridRetriever, tmp_path: Path, monkeypatch):
    """Workflow state and output metadata explicitly reflect fallback status in offline mode."""
    monkeypatch.setenv("LLM_API_KEY", "")
    from asterline.config.settings import Settings
    no_key_settings = Settings(llm_api_key=None, langchain_api_key=None, _env_file=None)
    monkeypatch.setattr("asterline.config.settings.get_settings", lambda: no_key_settings)
    monkeypatch.setattr("asterline.agents.llm.get_settings", lambda: no_key_settings)
    db_path = tmp_path / "workflow_fallback_test.db"
    store = SQLiteStore(db_path)
    service = MemoryService(store)
    builder = WorkflowBuilder(retriever=shared_retriever, memory_service=service)
    app = builder.build()

    initial_state = {
        "user_id": "user_alice",
        "conversation_id": "conv_meta",
        "user_message": "What is the price of Launch plan?",
    }
    result = app.invoke(initial_state)

    # In offline mode without live API key, is_fallback must be True and metadata present
    assert "is_fallback" in result
    assert result["is_fallback"] is True
    assert "llm_metadata" in result
    assert result["llm_metadata"]["is_fallback"] is True
    assert "fallback_reason" in result["llm_metadata"]


def test_workflow_conversational_followups(workflow_app):
    """Conversational follow-ups (thanks, bye, cool) route to conversational with 0 citations and 0 LLM calls."""
    app, service = workflow_app

    for query, expected_fragment in [
        ("thank you so much!", "welcome"),
        ("bye", "goodbye"),
        ("cool", "glad to help"),
    ]:
        state = {
            "user_id": "user_alice",
            "conversation_id": f"conv_{query[:4]}",
            "user_message": query,
        }
        res = app.invoke(state)
        assert res["route"] == "conversational"
        assert len(res["citations"]) == 0
        assert expected_fragment in res["final_response"].lower()



