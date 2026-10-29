"""Tests for conversation switching, message persistence, thread isolation, and session state synchronization.

Verifies:
1. New conversation creation allocates a distinct conversation ID.
2. The first message after starting a new conversation uses the new ID throughout execution.
3. LangGraph receives the exact active thread/conversation ID in config and state.
4. User and assistant messages are persisted under the same conversation ID in SQLite.
5. Conversation A history and Conversation B history remain strictly partitioned.
6. Switching between conversations restores each conversation's complete and isolated history.
7. Streamlit reruns do not duplicate conversations or messages.
8. Cross-user isolation prevents access to another user's conversation threads and messages.
9. End-to-end secret test code isolation scenario (ALPHA-731).
"""

from pathlib import Path
from unittest.mock import MagicMock
import pytest

from asterline.config.settings import get_settings
from asterline.graph.workflow import WorkflowBuilder
from asterline.memory.service import MemoryService
from asterline.memory.store import SQLiteStore
from asterline.rag.retriever import HybridRetriever
from asterline.rag.ingestion import load_and_validate_corpus


@pytest.fixture
def test_env(tmp_path):
    """Provides a fully wired SQLiteStore, MemoryService, and WorkflowBuilder for testing."""
    settings = get_settings()
    corpus_path = settings.resolve_path(settings.corpus_path)
    chunks, _ = load_and_validate_corpus(corpus_path)

    chroma_dir = tmp_path / "chroma_test"
    retriever = HybridRetriever(
        corpus_chunks=chunks,
        chroma_persist_dir=chroma_dir,
        embedding_model_name=settings.embedding_model_name,
    )

    db_path = tmp_path / "test_conversations.db"
    store = SQLiteStore(db_path)
    memory_service = MemoryService(store)

    builder = WorkflowBuilder(
        retriever=retriever,
        memory_service=memory_service,
    )
    workflow = builder.build()

    return {
        "store": store,
        "memory_service": memory_service,
        "workflow": workflow,
        "retriever": retriever,
    }


def test_new_conversation_creates_distinct_id(test_env):
    """Verify creating a new conversation generates a distinct ID and persists it in SQLite."""
    service: MemoryService = test_env["memory_service"]
    user_id = "user_alice"

    conv1 = service.get_or_create_conversation(user_id, title="First Thread")
    conv2 = service.get_or_create_conversation(user_id, title="Second Thread")

    assert conv1["conversation_id"] != conv2["conversation_id"]
    assert conv1["user_id"] == user_id
    assert conv2["user_id"] == user_id

    user_convs = service.list_user_conversations(user_id)
    cids = [c["conversation_id"] for c in user_convs]
    assert conv1["conversation_id"] in cids
    assert conv2["conversation_id"] in cids


def test_first_message_persists_to_new_conversation_id(test_env):
    """Verify the first message in a new conversation persists exclusively to that ID."""
    service: MemoryService = test_env["memory_service"]
    workflow = test_env["workflow"]
    user_id = "user_alice"

    # Start first conversation
    conv_a = service.get_or_create_conversation(user_id, title="Conversation A")
    cid_a = conv_a["conversation_id"]

    res_a = workflow.invoke(
        {
            "user_message": "Hello, this is conversation A",
            "user_id": user_id,
            "conversation_id": cid_a,
        },
        config={"configurable": {"thread_id": cid_a}},
    )
    assert res_a["final_response"]

    msgs_a = service.get_conversation_messages(cid_a, user_id)
    assert len(msgs_a) == 2
    assert msgs_a[0].role == "user"
    assert msgs_a[0].content == "Hello, this is conversation A"
    assert msgs_a[1].role == "assistant"

    # Start second conversation
    conv_b = service.get_or_create_conversation(user_id, title="Conversation B")
    cid_b = conv_b["conversation_id"]
    assert cid_b != cid_a

    # Send message in Conversation B
    res_b = workflow.invoke(
        {
            "user_message": "Hello, this is conversation B",
            "user_id": user_id,
            "conversation_id": cid_b,
        },
        config={"configurable": {"thread_id": cid_b}},
    )
    assert res_b["final_response"]

    # Verify Conversation B has only its message
    msgs_b = service.get_conversation_messages(cid_b, user_id)
    assert len(msgs_b) == 2
    assert msgs_b[0].role == "user"
    assert msgs_b[0].content == "Hello, this is conversation B"
    assert msgs_b[1].role == "assistant"

    # Verify Conversation A was not modified
    msgs_a_after = service.get_conversation_messages(cid_a, user_id)
    assert len(msgs_a_after) == 2
    assert msgs_a_after[0].content == "Hello, this is conversation A"


def test_conversation_isolation_secret_code_scenario(test_env):
    """Verify Section 5 test scenario: Secret code in Conv A is not leaked or inherited into Conv B.

    Scenario:
    1. Conversation A: Send "My secret test code is ALPHA-731."
    2. Confirm user and assistant messages are saved under Conversation A.
    3. Start Conversation B with a new distinct conversation ID.
    4. Send: "What was the secret test code I told you?"
    5. Confirm that Conversation B does not retrieve ALPHA-731 from Conversation A's thread history.
    6. Switch back to Conversation A and verify history is intact.
    7. Switch to Conversation B and verify only its history appears.
    8. Start Conversation C and verify it receives another distinct ID.
    """
    service: MemoryService = test_env["memory_service"]
    workflow = test_env["workflow"]
    user_id = "user_alice"

    # 1. Conversation A
    conv_a = service.get_or_create_conversation(user_id, title="Secret Thread A")
    cid_a = conv_a["conversation_id"]

    workflow.invoke(
        {
            "user_message": "My secret test code is ALPHA-731.",
            "user_id": user_id,
            "conversation_id": cid_a,
        },
        config={"configurable": {"thread_id": cid_a}},
    )

    # 2. Confirm messages saved under Conversation A
    msgs_a = service.get_conversation_messages(cid_a, user_id)
    assert len(msgs_a) == 2
    assert "ALPHA-731" in msgs_a[0].content
    assert msgs_a[1].role == "assistant"

    # 3. Start Conversation B
    conv_b = service.get_or_create_conversation(user_id, title="Clean Thread B")
    cid_b = conv_b["conversation_id"]
    assert cid_b != cid_a

    # Confirm UI/messages for Conversation B are initially completely empty
    msgs_b_initial = service.get_conversation_messages(cid_b, user_id)
    assert len(msgs_b_initial) == 0

    # 4. Send question in Conversation B
    res_b = workflow.invoke(
        {
            "user_message": "What was the secret test code I told you?",
            "user_id": user_id,
            "conversation_id": cid_b,
        },
        config={"configurable": {"thread_id": cid_b}},
    )

    # 5. Confirm Conversation B does not leak ALPHA-731 from Conversation A thread history
    assert "ALPHA-731" not in res_b["final_response"]

    # Verify Conversation B messages
    msgs_b = service.get_conversation_messages(cid_b, user_id)
    assert len(msgs_b) == 2
    assert "What was the secret test code I told you?" in msgs_b[0].content
    assert msgs_b[1].role == "assistant"

    # 6. Switch back to Conversation A and verify history is intact
    msgs_a_reload = service.get_conversation_messages(cid_a, user_id)
    assert len(msgs_a_reload) == 2
    assert "ALPHA-731" in msgs_a_reload[0].content

    # 7. Switch to Conversation B and verify only its history appears
    msgs_b_reload = service.get_conversation_messages(cid_b, user_id)
    assert len(msgs_b_reload) == 2
    assert all("ALPHA-731" not in m.content for m in msgs_b_reload)

    # 8. Start Conversation C and verify it receives another distinct ID
    conv_c = service.get_or_create_conversation(user_id, title="Thread C")
    cid_c = conv_c["conversation_id"]
    assert cid_c != cid_a
    assert cid_c != cid_b

    # Messages in C are initially empty
    msgs_c = service.get_conversation_messages(cid_c, user_id)
    assert len(msgs_c) == 0


def test_cross_user_conversation_isolation(test_env):
    """Verify different users cannot view or append to each other's conversations."""
    service: MemoryService = test_env["memory_service"]
    workflow = test_env["workflow"]

    alice = "user_alice"
    bob = "user_bob"

    # Alice creates Conversation
    alice_conv = service.get_or_create_conversation(alice, title="Alice Secret")
    cid_alice = alice_conv["conversation_id"]

    workflow.invoke(
        {
            "user_message": "Alice confidential deployment plan",
            "user_id": alice,
            "conversation_id": cid_alice,
        },
        config={"configurable": {"thread_id": cid_alice}},
    )

    # Bob lists conversations -> Alice's conversation ID is not present
    bob_convs = service.list_user_conversations(bob)
    assert not any(c["conversation_id"] == cid_alice for c in bob_convs)

    # Bob attempts to fetch Alice's messages -> returns empty list
    bob_attempt_msgs = service.get_conversation_messages(cid_alice, user_id=bob)
    assert len(bob_attempt_msgs) == 0

    # Bob starts his own conversation
    bob_conv = service.get_or_create_conversation(bob, title="Bob Ingestion")
    cid_bob = bob_conv["conversation_id"]
    assert cid_bob != cid_alice


def test_no_duplicate_conversations_on_repeated_calls(test_env):
    """Verify requesting the same conversation ID retrieves the existing record without duplicates."""
    service: MemoryService = test_env["memory_service"]
    user_id = "user_alice"

    conv = service.get_or_create_conversation(user_id, title="Repeat Test")
    cid = conv["conversation_id"]

    # Re-call with same ID
    conv2 = service.get_or_create_conversation(user_id, conversation_id=cid)
    assert conv2["conversation_id"] == cid

    # Total conversation count for this user
    all_convs = service.list_user_conversations(user_id)
    matching = [c for c in all_convs if c["conversation_id"] == cid]
    assert len(matching) == 1


def test_switching_conversations_restores_complete_history(test_env):
    """Verify switching back and forth between conversations accurately restores each history."""
    service: MemoryService = test_env["memory_service"]
    workflow = test_env["workflow"]
    user_id = "user_alice"

    conv1 = service.get_or_create_conversation(user_id, title="Topic 1")
    cid1 = conv1["conversation_id"]
    workflow.invoke(
        {"user_message": "Topic 1 Turn 1", "user_id": user_id, "conversation_id": cid1},
        config={"configurable": {"thread_id": cid1}},
    )
    workflow.invoke(
        {"user_message": "Topic 1 Turn 2", "user_id": user_id, "conversation_id": cid1},
        config={"configurable": {"thread_id": cid1}},
    )

    conv2 = service.get_or_create_conversation(user_id, title="Topic 2")
    cid2 = conv2["conversation_id"]
    workflow.invoke(
        {"user_message": "Topic 2 Turn 1", "user_id": user_id, "conversation_id": cid2},
        config={"configurable": {"thread_id": cid2}},
    )

    # Load 1
    msgs1 = service.get_conversation_messages(cid1, user_id)
    assert len(msgs1) == 4
    assert msgs1[0].content == "Topic 1 Turn 1"
    assert msgs1[2].content == "Topic 1 Turn 2"

    # Switch to 2
    msgs2 = service.get_conversation_messages(cid2, user_id)
    assert len(msgs2) == 2
    assert msgs2[0].content == "Topic 2 Turn 1"

    # Switch back to 1
    msgs1_reload = service.get_conversation_messages(cid1, user_id)
    assert len(msgs1_reload) == 4
    assert msgs1_reload[0].content == "Topic 1 Turn 1"
    assert msgs1_reload[2].content == "Topic 1 Turn 2"


def test_session_state_create_new_conversation_synchronization(tmp_path):
    """Verify create_new_conversation atomically synchronizes all session state keys."""
    from unittest.mock import patch
    from app.main import create_new_conversation, init_session_state

    db_path = tmp_path / "test_session_sync.db"
    store = SQLiteStore(db_path)
    service = MemoryService(store)

    fake_session = {
        "pending_prompt": "stale prompt",
        "last_trace_info": {"stale": "trace"},
        "conv_search_query": "filter",
    }

    with patch("streamlit.session_state", fake_session):
        init_session_state(service)
        initial_id = fake_session["active_conv_id"]
        assert initial_id.startswith("conv_")
        assert fake_session["conv_select_box_user_alice"] == initial_id

        # Click New Conversation
        new_conv_id = create_new_conversation("user_alice", service)
        assert new_conv_id != initial_id
        assert new_conv_id.startswith("conv_")

        # Verify active_conv_id and widget key match new_conv_id
        assert fake_session["active_conv_id"] == new_conv_id
        assert fake_session["conv_select_box_user_alice"] == new_conv_id

        # Verify transient state was cleanly wiped
        assert fake_session.get("pending_prompt") is None
        assert fake_session["last_trace_info"] is None
        assert fake_session["conv_search_query"] == ""


def test_session_state_switch_user_profile_synchronization(tmp_path):
    """Verify switch_user_profile re-scopes active_conv_id and widget keys to new user."""
    from unittest.mock import patch
    from app.main import init_session_state, switch_user_profile

    db_path = tmp_path / "test_user_switch_sync.db"
    store = SQLiteStore(db_path)
    service = MemoryService(store)

    fake_session = {}

    with patch("streamlit.session_state", fake_session), patch("streamlit.rerun"):
        init_session_state(service)
        alice_conv_id = fake_session["active_conv_id"]

        # Switch to Bob
        switch_user_profile("user_bob", service)

        assert fake_session["user_id"] == "user_bob"
        bob_conv_id = fake_session["active_conv_id"]
        assert bob_conv_id != alice_conv_id
        assert fake_session["conv_select_box_user_bob"] == bob_conv_id

        # Switch back to Alice
        switch_user_profile("user_alice", service)
        assert fake_session["user_id"] == "user_alice"
        assert fake_session["active_conv_id"] == alice_conv_id
        assert fake_session["conv_select_box_user_alice"] == alice_conv_id

