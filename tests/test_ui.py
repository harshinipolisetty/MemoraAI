"""Tests for Streamlit UI helpers, profile switching, memory inspector operations, and observability displays."""

from pathlib import Path
import pytest

from asterline.config.settings import Settings
from asterline.memory.store import SQLiteStore
from asterline.memory.service import MemoryService
from app.components import (
    DEMO_PROFILES,
    ROUTE_DISPLAY_META,
    format_timestamp,
    render_route_pill,
    render_fallback_pill,
)


def test_ui_profile_definitions():
    """Verify demo profile selector metadata and configuration."""
    assert "user_alice" in DEMO_PROFILES
    assert "user_bob" in DEMO_PROFILES
    assert "user_charlie" in DEMO_PROFILES

    for uid, prof in DEMO_PROFILES.items():
        assert prof["id"] == uid
        assert "name" in prof
        assert "role" in prof
        assert "focus" in prof
        assert "avatar" in prof


def test_ui_timestamp_formatting():
    """Verify robust timestamp formatting for ISO, SQLite, and empty strings."""
    assert format_timestamp(None) == "Unknown"
    assert format_timestamp("") == "Unknown"
    iso_ts = "2026-10-27T14:30:00Z"
    formatted = format_timestamp(iso_ts)
    assert "Oct 27, 2026" in formatted
    assert "14:30" in formatted


def test_ui_route_and_fallback_pills():
    """Verify route pills and fallback status indicators."""
    for route in ("conversational", "clarification", "memory_ops", "factual", "investigation"):
        pill = render_route_pill(route)
        assert ROUTE_DISPLAY_META[route]["label"] in pill
        assert "pill-badge" in pill

    # Fallback pill
    fallback_pill = render_fallback_pill(True)
    assert "Deterministic Fallback" in fallback_pill

    # Live pill
    live_pill = render_fallback_pill(False, {"provider": "groq", "model": "llama-3.3-70b"})
    assert "Live: groq/llama-3.3-70b" in live_pill


def test_user_profile_switching_and_conversation_isolation(tmp_path):
    """Verify profile switching strictly partitions conversation history in SQLite."""
    store = SQLiteStore(tmp_path / "test_ui_conversations.db")
    service = MemoryService(store)

    alice = "user_alice"
    bob = "user_bob"

    # Alice creates a thread and message
    alice_conv = service.get_or_create_conversation(alice, title="Alice Pipeline Debug")
    service.append_message(
        conversation_id=alice_conv["conversation_id"],
        user_id=alice,
        role="user",
        content="Checking BigQuery export status",
    )

    # Bob lists his conversations -> Alice's conversation MUST NOT appear
    bob_convs = service.list_user_conversations(bob)
    assert len(bob_convs) == 0
    assert not any(c["conversation_id"] == alice_conv["conversation_id"] for c in bob_convs)

    # Bob creates his own conversation
    bob_conv = service.get_or_create_conversation(bob, title="Bob Ingestion Limit Check")
    service.append_message(
        conversation_id=bob_conv["conversation_id"],
        user_id=bob,
        role="user",
        content="What is POST /v2/runs limit?",
    )

    # Verify Alice cannot see Bob's messages
    alice_messages = service.get_conversation_messages(bob_conv["conversation_id"], user_id=alice)
    assert len(alice_messages) == 0

    # Verify Bob can see his messages
    bob_messages = service.get_conversation_messages(bob_conv["conversation_id"], user_id=bob)
    assert len(bob_messages) == 1
    assert "POST /v2/runs" in bob_messages[0].content


def test_memory_inspector_crud_and_policy_enforcement(tmp_path):
    """Verify Memory Inspector add, edit, forget, and policy checks via MemoryService."""
    store = SQLiteStore(tmp_path / "test_ui_memory.db")
    service = MemoryService(store)
    alice = "user_alice"

    # 1. Add valid memory
    item, decision = service.create_memory(
        user_id=alice,
        content="We prefer dark mode and weekly summaries.",
        category="preference",
    )
    assert decision.allowed is True
    assert item is not None
    assert item.content == "We prefer dark mode and weekly summaries."

    # 2. Reject secret key input in memory inspector
    _, bad_decision = service.create_memory(
        user_id=alice,
        content="My secret key is ast_live_secret1234567890abcdef",
        category="preference",
    )
    assert bad_decision.allowed is False
    assert bad_decision.reason_code == "REJECTED_ASTERLINE_KEY"

    # 3. Update memory
    updated_item, update_decision = service.update_memory(
        user_id=alice,
        memory_id=item.memory_id,
        new_content="We prefer high-contrast dark mode.",
    )
    assert update_decision.allowed is True
    assert updated_item is not None
    assert updated_item.content == "We prefer high-contrast dark mode."

    # 4. Forget memory (permanent purge)
    deleted = service.delete_memory(alice, item.memory_id)
    assert deleted is True
    assert service.is_memory_active(alice, item.memory_id) is False

    # Verify raw SQLite table has zero records
    memories = service.list_user_memories(alice)
    assert len(memories) == 0


def test_cross_user_memory_isolation(tmp_path):
    """Verify Charlie cannot view or modify Alice's memory."""
    store = SQLiteStore(tmp_path / "test_ui_isolation.db")
    service = MemoryService(store)

    alice = "user_alice"
    charlie = "user_charlie"

    item, _ = service.create_memory(alice, "Alice private deployment secret notes", "workspace_context")
    assert item is not None

    # Charlie lists memories -> empty
    charlie_mems = service.list_user_memories(charlie)
    assert len(charlie_mems) == 0

    # Charlie cannot fetch Alice memory by ID
    assert service.get_user_memory(charlie, item.memory_id) is None

    # Charlie cannot delete Alice memory
    charlie_del = service.delete_memory(charlie, item.memory_id)
    assert charlie_del is False
    # Alice memory remains active
    assert service.is_memory_active(alice, item.memory_id) is True
