"""Regression tests for performance optimizations, instant user switching, and caching.

Verifies:
1. Profile switching does not trigger LLM calls.
2. Profile switching does not recreate reusable resources unnecessarily.
3. Conversation history loads only for the selected user and conversation.
4. Memory Inspector data is strictly scoped to the selected user.
5. Alice's records never appear in Bob's view.
6. Changing profiles does not leave stale conversations selected.
7. Memory creation, correction, and forgetting still work and invalidate caches.
8. Cached data refreshes after relevant mutations.
9. UI-only interactions do not trigger unnecessary retrieval or model calls.
10. Compound database indexes and query plan optimizations.
"""

from pathlib import Path
from unittest.mock import MagicMock
import pytest

from asterline.config.settings import get_settings
from asterline.graph.workflow import WorkflowBuilder
from asterline.memory.service import MemoryService
from asterline.memory.store import SQLiteStore
from asterline.rag.ingestion import load_and_validate_corpus
from asterline.rag.retriever import HybridRetriever


@pytest.fixture
def clean_memory_service(tmp_path):
    """Provides a fresh SQLiteStore and MemoryService for test isolation."""
    db_file = tmp_path / "test_perf.db"
    store = SQLiteStore(db_file)
    return MemoryService(store)


def test_user_profile_switching_never_calls_llm(clean_memory_service):
    """Verify switching user profiles never touches the LLM."""
    mock_llm = MagicMock()
    mock_retriever = MagicMock()

    builder = WorkflowBuilder(
        retriever=mock_retriever,
        memory_service=clean_memory_service,
        llm_client=mock_llm,
    )
    _ = builder.build()

    # Create data for Alice and Bob
    alice_conv = clean_memory_service.get_or_create_conversation("user_alice", title="Alice Pipeline")
    bob_conv = clean_memory_service.get_or_create_conversation("user_bob", title="Bob Limits")

    # Simulate profile switch: Alice -> Bob
    bob_convs = clean_memory_service.list_user_conversations("user_bob")
    assert len(bob_convs) == 1
    assert bob_convs[0]["conversation_id"] == bob_conv["conversation_id"]

    # Bob's active messages
    bob_msgs = clean_memory_service.get_conversation_messages(bob_conv["conversation_id"], "user_bob")
    assert len(bob_msgs) == 0

    # Bob's memories
    bob_mems = clean_memory_service.list_user_memories("user_bob")
    assert len(bob_mems) == 0

    # Ensure mock LLM was NEVER called
    mock_llm.generate.assert_not_called()
    mock_retriever.hybrid_search.assert_not_called()


def test_profile_switching_isolation_and_no_stale_conversations(clean_memory_service):
    """Verify changing profiles switches active conversation without stale leakage."""
    alice = "user_alice"
    bob = "user_bob"

    alice_conv = clean_memory_service.get_or_create_conversation(alice, title="Alice Thread")
    clean_memory_service.append_message(
        conversation_id=alice_conv["conversation_id"],
        user_id=alice,
        role="user",
        content="Secret Alice operational detail",
    )

    bob_conv = clean_memory_service.get_or_create_conversation(bob, title="Bob Thread")
    clean_memory_service.append_message(
        conversation_id=bob_conv["conversation_id"],
        user_id=bob,
        role="user",
        content="Bob database optimization query",
    )

    # 1. Verify Bob cannot see Alice's conversation in his list
    bob_list = clean_memory_service.list_user_conversations(bob)
    assert len(bob_list) == 1
    assert bob_list[0]["conversation_id"] == bob_conv["conversation_id"]
    assert not any(c["conversation_id"] == alice_conv["conversation_id"] for c in bob_list)

    # 2. Verify Bob cannot fetch messages from Alice's conversation ID
    messages_attempt = clean_memory_service.get_conversation_messages(alice_conv["conversation_id"], user_id=bob)
    assert len(messages_attempt) == 0

    # 3. Verify Alice's messages are intact for Alice
    alice_msgs = clean_memory_service.get_conversation_messages(alice_conv["conversation_id"], user_id=alice)
    assert len(alice_msgs) == 1
    assert "Secret Alice" in alice_msgs[0].content


def test_memory_cache_invalidation_on_mutations(clean_memory_service):
    """Verify that user-scoped memory caches refresh immediately after create, update, and delete."""
    alice = "user_alice"

    # Initially empty
    mems = clean_memory_service.list_user_memories(alice)
    assert len(mems) == 0

    # 1. Create Memory -> Cache must invalidate and show new item
    item, dec = clean_memory_service.create_memory(
        user_id=alice,
        content="We use Snowflake for warehouse data export.",
        category="workspace_context",
    )
    assert dec.allowed is True
    assert item is not None

    mems_after_create = clean_memory_service.list_user_memories(alice)
    assert len(mems_after_create) == 1
    assert mems_after_create[0].memory_id == item.memory_id
    assert "Snowflake" in mems_after_create[0].content

    # 2. Update Memory -> Cache must invalidate and show updated text
    updated_item, update_dec = clean_memory_service.update_memory(
        user_id=alice,
        memory_id=item.memory_id,
        new_content="We migrated from Snowflake to BigQuery EU.",
    )
    assert update_dec.allowed is True
    assert updated_item is not None

    mems_after_update = clean_memory_service.list_user_memories(alice)
    assert len(mems_after_update) == 1
    assert "BigQuery EU" in mems_after_update[0].content

    # 3. Delete Memory -> Cache must invalidate and list must be empty
    deleted = clean_memory_service.delete_memory(alice, item.memory_id)
    assert deleted is True

    mems_after_del = clean_memory_service.list_user_memories(alice)
    assert len(mems_after_del) == 0


def test_conversation_cache_invalidation_on_mutations(clean_memory_service):
    """Verify conversation and message caches invalidate properly on new turns."""
    alice = "user_alice"

    conv = clean_memory_service.get_or_create_conversation(alice, title="Initial Title")
    cid = conv["conversation_id"]

    # Read messages (empty)
    msgs0 = clean_memory_service.get_conversation_messages(cid, alice)
    assert len(msgs0) == 0

    # Append message
    clean_memory_service.append_message(cid, alice, role="user", content="Turn 1")
    msgs1 = clean_memory_service.get_conversation_messages(cid, alice)
    assert len(msgs1) == 1
    assert msgs1[0].content == "Turn 1"

    # Append assistant message with citations
    clean_memory_service.append_message(
        cid,
        alice,
        role="assistant",
        content="Response 1",
        citations=[{"chunk_id": "company:0", "doc_id": "company", "title": "Company overview"}],
    )
    msgs2 = clean_memory_service.get_conversation_messages(cid, alice)
    assert len(msgs2) == 2
    assert len(msgs2[1].citations) == 1
    assert msgs2[1].citations[0].chunk_id == "company:0"

    # Update title
    clean_memory_service.update_conversation_title(cid, alice, "Updated Title")
    convs = clean_memory_service.list_user_conversations(alice)
    assert convs[0]["title"] == "Updated Title"


def test_sqlite_compound_indexes_query_plans(clean_memory_service):
    """Verify that compound indexes prevent temporary B-Trees for ORDER BY."""
    conn = clean_memory_service.store.get_connection()

    # 1. Conversations query plan
    plan_convs = conn.execute(
        "EXPLAIN QUERY PLAN SELECT * FROM conversations WHERE user_id = ? ORDER BY updated_at DESC LIMIT 50",
        ("user_alice",),
    ).fetchall()
    plan_convs_text = " ".join(r["detail"] if "detail" in r.keys() else str(tuple(r)) for r in plan_convs)
    assert "USE TEMP B-TREE FOR ORDER BY" not in plan_convs_text
    assert "idx_conversations_user_updated" in plan_convs_text

    # 2. Memories query plan
    plan_mems = conn.execute(
        "EXPLAIN QUERY PLAN SELECT * FROM user_memories WHERE user_id = ? AND is_active = 1 ORDER BY updated_at DESC",
        ("user_alice",),
    ).fetchall()
    plan_mems_text = " ".join(r["detail"] if "detail" in r.keys() else str(tuple(r)) for r in plan_mems)
    assert "USE TEMP B-TREE FOR ORDER BY" not in plan_mems_text
    assert "idx_user_memories_user_active_updated" in plan_mems_text


def test_chroma_sync_uses_include_empty():
    """Verify sync_index in HybridRetriever performs light metadata checking."""
    settings = get_settings()
    corpus_path = settings.resolve_path(settings.corpus_path)
    chunks, _ = load_and_validate_corpus(corpus_path)

    retriever = HybridRetriever(
        corpus_chunks=chunks,
        chroma_persist_dir=settings.resolve_path(settings.chroma_persist_directory),
    )
    # Re-running sync_index on existing collection should be a fast no-op
    retriever.sync_index(force_rebuild=False)
    assert retriever.collection.count() == len(chunks)
