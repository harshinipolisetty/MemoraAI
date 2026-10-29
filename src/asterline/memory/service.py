"""High-level application service for conversations and long-term memory.

Coordinates between SQLiteStore and MemoryPolicy to guarantee:
- Strict user-level isolation and ownership verification.
- Policy enforcement: rejecting secrets and transient chatter before persistence.
- Transactional mutations with non-sensitive audit logging.
- Conversation continuity across sessions and restarts.
"""

import json
from typing import Any, Dict, List, Optional
import uuid
from asterline.graph.state import Citation, ConversationMessage, MemoryItem
from asterline.memory.policy import MemoryPolicyDecision, evaluate_memory_candidate
from asterline.memory.store import SQLiteStore


class MemoryService:
    """Coordinates persistent memory operations, policy checks, and conversation history."""

    def __init__(self, store: SQLiteStore):
        self.store = store
        self._conv_cache: Dict[str, List[Dict[str, Any]]] = {}
        self._msg_cache: Dict[tuple[str, str], List[ConversationMessage]] = {}
        self._mem_cache: Dict[str, List[MemoryItem]] = {}

    def clear_cache(self, user_id: Optional[str] = None) -> None:
        """Clear user-scoped caches, either for a specific user or globally."""
        if user_id:
            self._conv_cache.pop(user_id, None)
            self._mem_cache.pop(user_id, None)
            keys_to_del = [k for k in self._msg_cache if k[1] == user_id]
            for k in keys_to_del:
                self._msg_cache.pop(k, None)
        else:
            self._conv_cache.clear()
            self._msg_cache.clear()
            self._mem_cache.clear()

    # -------------------------------------------------------------------------
    # Conversation Methods
    # -------------------------------------------------------------------------

    def get_or_create_conversation(
        self,
        user_id: str,
        conversation_id: Optional[str] = None,
        title: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Retrieve an existing conversation or create a new one owned by user_id."""
        if conversation_id:
            existing = self.store.get_conversation(conversation_id, user_id)
            if existing:
                return existing
            # If conversation_id was requested but does not exist under this user, create it with that ID
            conv_id = conversation_id
        else:
            conv_id = f"conv_{uuid.uuid4().hex[:12]}"

        conv_title = title or "New Investigation"
        result = self.store.create_conversation(conv_id, user_id, conv_title)
        self._conv_cache.pop(user_id, None)
        return result

    def list_user_conversations(self, user_id: str, limit: int = 50) -> List[Dict[str, Any]]:
        """List conversation threads owned by user_id with user-scoped caching."""
        if limit == 50 and user_id in self._conv_cache:
            return list(self._conv_cache[user_id])

        convs = self.store.list_conversations(user_id, limit=limit)
        if limit == 50:
            self._conv_cache[user_id] = convs
        return convs

    def update_conversation_title(self, conversation_id: str, user_id: str, title: str) -> bool:
        """Update the title of a conversation thread owned by user_id."""
        updated = self.store.update_conversation_title(conversation_id, user_id, title)
        if updated:
            self._conv_cache.pop(user_id, None)
        return updated

    def append_message(
        self,
        conversation_id: str,
        user_id: str,
        role: str,
        content: str,
        citations: Optional[List[Dict[str, Any]]] = None,
        route_taken: Optional[str] = None,
    ) -> ConversationMessage:
        """Append a message turn to a conversation owned by user_id."""
        message_id = f"msg_{uuid.uuid4().hex[:12]}"
        citations_json = json.dumps(citations) if citations else None

        row = self.store.add_message(
            message_id=message_id,
            conversation_id=conversation_id,
            user_id=user_id,
            role=role,
            content=content,
            citations_json=citations_json,
            route_taken=route_taken,
        )

        parsed_citations: List[Citation] = []
        if row["citations_json"]:
            try:
                raw_cits = json.loads(row["citations_json"])
                parsed_citations = [Citation(**c) for c in raw_cits]
            except Exception:
                parsed_citations = []

        msg = ConversationMessage(
            message_id=row["message_id"],
            conversation_id=row["conversation_id"],
            user_id=row["user_id"],
            role=role,
            content=row["content"],
            citations=parsed_citations,
            route_taken=row["route_taken"],
            created_at=str(row["created_at"]),
        )

        # Invalidate affected user and conversation caches
        self._conv_cache.pop(user_id, None)
        self._msg_cache.pop((conversation_id, user_id), None)
        return msg

    def get_conversation_messages(
        self, conversation_id: str, user_id: str
    ) -> List[ConversationMessage]:
        """Retrieve chronological messages for a conversation, enforcing ownership with caching."""
        cache_key = (conversation_id, user_id)
        if cache_key in self._msg_cache:
            return list(self._msg_cache[cache_key])

        rows = self.store.get_messages(conversation_id, user_id)
        messages: List[ConversationMessage] = []
        for r in rows:
            parsed_cits = []
            if r["citations_json"]:
                try:
                    parsed_cits = [Citation(**c) for c in json.loads(r["citations_json"])]
                except Exception:
                    pass
            messages.append(
                ConversationMessage(
                    message_id=r["message_id"],
                    conversation_id=r["conversation_id"],
                    user_id=r["user_id"],
                    role=r["role"],
                    content=r["content"],
                    citations=parsed_cits,
                    route_taken=r["route_taken"],
                    created_at=str(r["created_at"]),
                )
            )

        self._msg_cache[cache_key] = messages
        return messages

    # -------------------------------------------------------------------------
    # Long-Term User Memory Methods
    # -------------------------------------------------------------------------

    def create_memory(
        self,
        user_id: str,
        content: str,
        category: str = "general",
        source_message_id: Optional[str] = None,
        reason: Optional[str] = None,
    ) -> tuple[Optional[MemoryItem], MemoryPolicyDecision]:
        """Evaluate content against memory policy; persist if accepted."""
        decision = evaluate_memory_candidate(content, category)
        if not decision.allowed:
            return None, decision

        memory_id = f"mem_{uuid.uuid4().hex[:12]}"
        audit_id = f"aud_{uuid.uuid4().hex[:12]}"
        sanitized_content = decision.sanitized_content or content.strip()

        row = self.store.insert_memory_with_audit(
            memory_id=memory_id,
            user_id=user_id,
            category=category,
            content=sanitized_content,
            audit_id=audit_id,
            confidence=1.0,
            source_message_id=source_message_id,
            reason=reason or "Saved user memory",
        )

        item = MemoryItem(
            memory_id=row["memory_id"],
            user_id=row["user_id"],
            category=row["category"],
            content=row["content"],
            confidence=row["confidence"],
            source_message_id=row["source_message_id"],
            is_active=bool(row["is_active"]),
            created_at=str(row["created_at"]),
            updated_at=str(row["updated_at"]),
        )
        self._mem_cache.pop(user_id, None)
        return item, decision

    def list_user_memories(self, user_id: str) -> List[MemoryItem]:
        """List all active memories for the specified user with user-scoped caching."""
        if user_id in self._mem_cache:
            return list(self._mem_cache[user_id])

        rows = self.store.list_active_memories(user_id)
        items = [
            MemoryItem(
                memory_id=r["memory_id"],
                user_id=r["user_id"],
                category=r["category"],
                content=r["content"],
                confidence=r["confidence"],
                source_message_id=r["source_message_id"],
                is_active=bool(r["is_active"]),
                created_at=str(r["created_at"]),
                updated_at=str(r["updated_at"]),
            )
            for r in rows
        ]
        self._mem_cache[user_id] = items
        return items

    def get_user_memory(self, user_id: str, memory_id: str) -> Optional[MemoryItem]:
        """Retrieve a specific active memory, strictly validating ownership."""
        row = self.store.get_memory(memory_id, user_id)
        if not row:
            return None
        return MemoryItem(
            memory_id=row["memory_id"],
            user_id=row["user_id"],
            category=row["category"],
            content=row["content"],
            confidence=row["confidence"],
            source_message_id=row["source_message_id"],
            is_active=bool(row["is_active"]),
            created_at=str(row["created_at"]),
            updated_at=str(row["updated_at"]),
        )

    def update_memory(
        self,
        user_id: str,
        memory_id: str,
        new_content: str,
        reason: Optional[str] = None,
    ) -> tuple[Optional[MemoryItem], MemoryPolicyDecision]:
        """Update an existing memory if the user owns it and the new content passes policy."""
        # Check ownership first
        existing = self.store.get_memory(memory_id, user_id)
        if not existing:
            return None, MemoryPolicyDecision(
                allowed=False,
                reason_code="NOT_FOUND_OR_UNAUTHORIZED",
                message="Target memory not found or not owned by the specified user.",
            )

        decision = evaluate_memory_candidate(new_content, existing["category"])
        if not decision.allowed:
            return None, decision

        audit_id = f"aud_{uuid.uuid4().hex[:12]}"
        sanitized = decision.sanitized_content or new_content.strip()

        row = self.store.update_memory_with_audit(
            memory_id=memory_id,
            user_id=user_id,
            new_content=sanitized,
            audit_id=audit_id,
            reason=reason or "User updated memory content",
        )
        if not row:
            return None, MemoryPolicyDecision(
                allowed=False,
                reason_code="UPDATE_FAILED",
                message="Memory update failed.",
            )

        item = MemoryItem(
            memory_id=row["memory_id"],
            user_id=row["user_id"],
            category=row["category"],
            content=row["content"],
            confidence=row["confidence"],
            source_message_id=row["source_message_id"],
            is_active=bool(row["is_active"]),
            created_at=str(row["created_at"]),
            updated_at=str(row["updated_at"]),
        )
        self._mem_cache.pop(user_id, None)
        return item, decision

    def delete_memory(
        self,
        user_id: str,
        memory_id: str,
        reason: Optional[str] = None,
    ) -> bool:
        """Deactivate (forget) an owned memory and record an audit log event."""
        audit_id = f"aud_{uuid.uuid4().hex[:12]}"
        deleted = self.store.delete_memory_with_audit(
            memory_id=memory_id,
            user_id=user_id,
            audit_id=audit_id,
            reason=reason or "User requested deletion (forget)",
        )
        if deleted:
            self._mem_cache.pop(user_id, None)
        return deleted

    def is_memory_active(self, user_id: str, memory_id: str) -> bool:
        """Check whether a specific memory remains active for this user."""
        return self.store.get_memory(memory_id, user_id) is not None

    def get_audit_log(self, user_id: str) -> List[Dict[str, Any]]:
        """Retrieve audit log events for the specified user."""
        return self.store.list_audit_events(user_id)
