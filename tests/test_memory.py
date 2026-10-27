"""Tests for SQLite persistence, user isolation, memory policy, and audit logging.

Uses temporary SQLite databases to ensure total test isolation and no pollution of user state.
Tests with at least two synthetic users: 'user_alice' and 'user_bob'.
"""

from pathlib import Path
import pytest
from asterline.memory.policy import evaluate_memory_candidate
from asterline.memory.service import MemoryService
from asterline.memory.store import SQLiteStore


@pytest.fixture
def temp_db(tmp_path: Path) -> Path:
    """Fixture providing a temporary SQLite database path."""
    return tmp_path / "test_memory.db"


@pytest.fixture
def store(temp_db: Path) -> SQLiteStore:
    """Fixture providing an initialized SQLiteStore using a temporary database."""
    return SQLiteStore(temp_db)


@pytest.fixture
def service(store: SQLiteStore) -> MemoryService:
    """Fixture providing a MemoryService instance."""
    return MemoryService(store)


# -----------------------------------------------------------------------------
# 1. Memory Creation & Retrieval (Positive Flow)
# -----------------------------------------------------------------------------

def test_memory_can_be_created_and_retrieved(service: MemoryService):
    """Test 1 & 12: Valid durable preferences pass policy, are stored, and can be retrieved."""
    content = "User prefers BigQuery for Warehouse Export destinations"
    item, decision = service.create_memory(
        user_id="user_alice",
        content=content,
        category="preference",
    )
    assert decision.allowed is True
    assert item is not None
    assert item.user_id == "user_alice"
    assert item.content == content

    # Retrieve by ID
    retrieved = service.get_user_memory(user_id="user_alice", memory_id=item.memory_id)
    assert retrieved is not None
    assert retrieved.memory_id == item.memory_id
    assert retrieved.content == content
    assert retrieved.is_active is True


# -----------------------------------------------------------------------------
# 2. User Isolation & Multi-User Partitioning
# -----------------------------------------------------------------------------

def test_memory_lists_contain_only_requested_user_records(service: MemoryService):
    """Test 2: Memory lists are strictly partitioned per user."""
    service.create_memory("user_alice", "Alice operates on Team plan EU workspace", "workspace_context")
    service.create_memory("user_bob", "Bob uses Python async SDK for ingest testing", "workspace_context")

    alice_memories = service.list_user_memories("user_alice")
    bob_memories = service.list_user_memories("user_bob")

    assert len(alice_memories) == 1
    assert "Alice" in alice_memories[0].content
    assert len(bob_memories) == 1
    assert "Bob" in bob_memories[0].content


def test_user_cannot_read_another_user_memory_by_id(service: MemoryService):
    """Test 3: A user cannot read another user's memory even if guessing the memory_id."""
    alice_mem, _ = service.create_memory("user_alice", "Alice confidential team note", "preference")
    assert alice_mem is not None

    # Bob attempts to read Alice's memory ID
    bob_attempt = service.get_user_memory("user_bob", alice_mem.memory_id)
    assert bob_attempt is None


def test_user_cannot_update_another_user_memory(service: MemoryService):
    """Test 4: A user cannot modify another user's memory."""
    alice_mem, _ = service.create_memory("user_alice", "Original Alice Preference", "preference")
    assert alice_mem is not None

    # Bob attempts to overwrite Alice's memory
    updated, decision = service.update_memory(
        user_id="user_bob",
        memory_id=alice_mem.memory_id,
        new_content="Bob malicious overwrite",
    )
    assert updated is None
    assert decision.allowed is False
    assert decision.reason_code == "NOT_FOUND_OR_UNAUTHORIZED"

    # Verify Alice's memory remained unchanged
    alice_check = service.get_user_memory("user_alice", alice_mem.memory_id)
    assert alice_check is not None
    assert alice_check.content == "Original Alice Preference"


def test_user_cannot_delete_another_user_memory(service: MemoryService):
    """Test 5: A user cannot delete another user's memory."""
    alice_mem, _ = service.create_memory("user_alice", "Alice critical preference", "preference")
    assert alice_mem is not None

    # Bob attempts to delete Alice's memory
    success = service.delete_memory("user_bob", alice_mem.memory_id)
    assert success is False

    # Alice's memory is still active
    assert service.is_memory_active("user_alice", alice_mem.memory_id) is True


# -----------------------------------------------------------------------------
# 3. Deletion ("Forget") & Audit Logging
# -----------------------------------------------------------------------------

def test_deleting_owned_memory_removes_from_active_and_audits(service: MemoryService):
    """Test 6 & 7: Deleting an owned memory removes it from active list and logs an audit entry."""
    item, _ = service.create_memory("user_alice", "User prefers dark mode UI theme", "preference")
    assert item is not None

    deleted = service.delete_memory("user_alice", item.memory_id, reason="Right to be forgotten")
    assert deleted is True

    # Memory is no longer retrievable or active
    assert service.get_user_memory("user_alice", item.memory_id) is None
    assert service.is_memory_active("user_alice", item.memory_id) is False
    assert len(service.list_user_memories("user_alice")) == 0

    # Audit log contains DELETE action
    audit_events = service.get_audit_log("user_alice")
    actions = [e["action"] for e in audit_events]
    assert "DELETE" in actions
    assert "CREATE" in actions

    # Verify non-sensitive audit logging (no raw content leak)
    delete_event = next(e for e in audit_events if e["action"] == "DELETE")
    assert "dark mode" not in (delete_event["previous_summary"] or "")
    assert "Length:" in (delete_event["previous_summary"] or "")


def test_failed_mutation_does_not_create_misleading_audit_event(service: MemoryService):
    """Test 8: If a mutation fails (e.g. non-existent ID), no audit event is logged."""
    audit_count_before = len(service.get_audit_log("user_alice"))
    success = service.delete_memory("user_alice", "non_existent_memory_id")
    assert success is False

    audit_count_after = len(service.get_audit_log("user_alice"))
    assert audit_count_after == audit_count_before


def test_memory_creation_and_audit_logging_are_transactional(store: SQLiteStore):
    """Test 9: Memory creation and its audit log record commit together atomically."""
    # Insert via store method
    item = store.insert_memory_with_audit(
        memory_id="mem_trans_test",
        user_id="user_alice",
        category="preference",
        content="Transactional test memory content",
        audit_id="aud_trans_test",
    )
    assert item["memory_id"] == "mem_trans_test"

    # Both memory and audit log exist
    mem = store.get_memory("mem_trans_test", "user_alice")
    assert mem is not None
    audits = store.list_audit_events("user_alice")
    assert any(a["audit_id"] == "aud_trans_test" for a in audits)


# -----------------------------------------------------------------------------
# 4. Memory Policy & Secret Rejection
# -----------------------------------------------------------------------------

@pytest.mark.parametrize(
    "secret_candidate, expected_code",
    [
        ("My ingestion key is ast_abc12345678901234567890", "REJECTED_ASTERLINE_KEY"),
        ("Here is my OpenAI key sk-1234567890abcdef1234567890", "REJECTED_API_KEY"),
        ("My GitHub token is ghp_1234567890abcdefghijklmnopqrst", "REJECTED_GITHUB_TOKEN"),
        ("Bearer eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJzdWIiOiIxMjM0NTY3ODkwIn0", "REJECTED_BEARER_TOKEN"),
        ("Please remember password: super_secret_pass_123", "REJECTED_CREDENTIAL_STRING"),
        ("secret = 'production_vault_secret_val'", "REJECTED_CREDENTIAL_STRING"),
    ],
)
def test_secrets_and_credentials_are_rejected_by_policy(
    service: MemoryService, secret_candidate: str, expected_code: str
):
    """Test 10 & 11: Candidate memories with obvious secrets are rejected and NEVER persisted."""
    decision = evaluate_memory_candidate(secret_candidate)
    assert decision.allowed is False
    assert decision.reason_code == expected_code

    # Attempt to create via service
    item, service_decision = service.create_memory("user_alice", secret_candidate)
    assert item is None
    assert service_decision.allowed is False
    assert service_decision.reason_code == expected_code

    # Verify nothing was saved to SQLite
    active = service.list_user_memories("user_alice")
    assert len(active) == 0


@pytest.mark.parametrize(
    "transient_phrase",
    ["hi", "hello", "thanks!", "okay", "yes", "bye", "cool", "got it."],
)
def test_transient_conversation_chatter_is_rejected(
    service: MemoryService, transient_phrase: str
):
    """Test 10 & 11: Pure conversational noise is rejected and not persisted."""
    item, decision = service.create_memory("user_alice", transient_phrase)
    assert item is None
    assert decision.allowed is False
    assert decision.reason_code in ("REJECTED_TRANSIENT", "REJECTED_TOO_SHORT")


def test_empty_or_whitespace_candidate_rejected(service: MemoryService):
    """Empty or whitespace candidates are rejected."""
    item, decision = service.create_memory("user_alice", "   ")
    assert item is None
    assert decision.allowed is False
    assert decision.reason_code == "REJECTED_EMPTY"


# -----------------------------------------------------------------------------
# 5. Conversation History Continuity & Ownership
# -----------------------------------------------------------------------------

def test_conversation_messages_persist_and_load_in_order(temp_db: Path):
    """Test 13: Conversation messages survive store re-opening and load in chronological order."""
    # First session: create conversation and append turns
    store1 = SQLiteStore(temp_db)
    service1 = MemoryService(store1)

    conv = service1.get_or_create_conversation("user_alice", title="Incident Investigation")
    conv_id = conv["conversation_id"]

    service1.append_message(
        conversation_id=conv_id,
        user_id="user_alice",
        role="user",
        content="What caused the duplicate export rows in INC-2025-12?",
    )
    service1.append_message(
        conversation_id=conv_id,
        user_id="user_alice",
        role="assistant",
        content="Heron query engine ordering broke position-based deduplication.",
        citations=[{"chunk_id": "incident-12:1", "doc_id": "incident-12", "title": "INC-2025-12 duplicate exports", "source_url": ""}],
        route_taken="investigation",
    )

    # Reopen database simulating application restart
    store2 = SQLiteStore(temp_db)
    service2 = MemoryService(store2)

    messages = service2.get_conversation_messages(conv_id, "user_alice")
    assert len(messages) == 2
    assert messages[0].role == "user"
    assert "INC-2025-12" in messages[0].content
    assert messages[1].role == "assistant"
    assert len(messages[1].citations) == 1
    assert messages[1].citations[0].chunk_id == "incident-12:1"


def test_conversations_cannot_be_accessed_by_different_user(temp_db: Path):
    """Test 14: Conversations belonging to Alice cannot be loaded or appended by Bob."""
    store = SQLiteStore(temp_db)
    service = MemoryService(store)

    alice_conv = service.get_or_create_conversation("user_alice", title="Alice Private Workspace Thread")
    conv_id = alice_conv["conversation_id"]

    # Bob attempts to read Alice's messages
    bob_messages = service.get_conversation_messages(conv_id, "user_bob")
    assert len(bob_messages) == 0

    # Bob attempts to append a message into Alice's conversation
    with pytest.raises(PermissionError, match="access denied"):
        service.append_message(
            conversation_id=conv_id,
            user_id="user_bob",
            role="user",
            content="Bob trying to inject into Alice thread",
        )


# -----------------------------------------------------------------------------
# 6. Database Initialization Idempotence
# -----------------------------------------------------------------------------

def test_repeated_init_schema_does_not_destroy_records(temp_db: Path):
    """Test 15: Running init_schema repeatedly preserves all existing rows and tables."""
    store = SQLiteStore(temp_db)
    service = MemoryService(store)

    # Insert a conversation and memory
    conv = service.get_or_create_conversation("user_alice", title="Persisted Conversation")
    mem, _ = service.create_memory("user_alice", "Persistent user preference across re-inits", "preference")
    assert mem is not None

    # Call init_schema multiple times
    store.init_schema()
    store.init_schema()

    # Verify both records still exist
    conv_check = store.get_conversation(conv["conversation_id"], "user_alice")
    mem_check = service.get_user_memory("user_alice", mem.memory_id)

    assert conv_check is not None
    assert conv_check["title"] == "Persisted Conversation"
    assert mem_check is not None
    assert mem_check.content == "Persistent user preference across re-inits"


def test_forgotten_memory_purged_completely_from_raw_sqlite(temp_db: Path):
    """Regression Test: Forgetting a memory permanently deletes the row from user_memories table.
    
    Proves content cannot be retrieved even via raw SQL inspection, and audit log contains
    only non-sensitive metadata (no raw content leakage).
    """
    store = SQLiteStore(temp_db)
    service = MemoryService(store)

    target_content = "Alice private confidential project codename Nebula"
    mem, decision = service.create_memory("user_alice", target_content, "workspace_context")
    assert mem is not None
    assert decision.allowed is True

    # Verify present in raw SQLite
    with store.get_connection() as conn:
        row = conn.execute("SELECT * FROM user_memories WHERE memory_id = ?", (mem.memory_id,)).fetchone()
        assert row is not None
        assert row["content"] == target_content

    # Execute forget
    deleted = service.delete_memory("user_alice", mem.memory_id, reason="User requested forget")
    assert deleted is True

    # Verify completely purged from raw SQLite table
    with store.get_connection() as conn:
        # Row must not exist at all (no soft delete row left behind)
        row = conn.execute("SELECT * FROM user_memories WHERE memory_id = ?", (mem.memory_id,)).fetchone()
        assert row is None

        # Content must not exist anywhere in user_memories table
        all_rows = conn.execute("SELECT * FROM user_memories WHERE content LIKE '%Nebula%'").fetchall()
        assert len(all_rows) == 0

        # Audit log must contain action DELETE with only non-sensitive summary
        audit_rows = conn.execute(
            "SELECT * FROM memory_audit_log WHERE memory_id = ? AND action = 'DELETE'",
            (mem.memory_id,),
        ).fetchall()
        assert len(audit_rows) == 1
        audit_entry = dict(audit_rows[0])
        assert "Nebula" not in audit_entry["previous_summary"]
        assert "Category: workspace_context" in audit_entry["previous_summary"]
        assert f"Length: {len(target_content)}" in audit_entry["previous_summary"]


def test_cross_user_cannot_access_or_purge_another_user_memory(temp_db: Path):
    """Regression Test: User Bob cannot access, update, or purge User Alice's memory."""
    store = SQLiteStore(temp_db)
    service = MemoryService(store)

    alice_mem, _ = service.create_memory("user_alice", "Alice critical production workspace context", "workspace_context")
    assert alice_mem is not None

    # Bob tries to read Alice's memory
    assert service.get_user_memory("user_bob", alice_mem.memory_id) is None

    # Bob tries to delete Alice's memory
    bob_delete_success = service.delete_memory("user_bob", alice_mem.memory_id)
    assert bob_delete_success is False

    # Alice's memory must remain untouched in store
    alice_check = service.get_user_memory("user_alice", alice_mem.memory_id)
    assert alice_check is not None
    assert alice_check.content == "Alice critical production workspace context"

