"""Memory Manager Agent for Asterline Labs.

Interprets requests to remember, view, update, or forget durable user context.
Enforces user isolation, applies memory policy to reject secrets, coordinates
transactional persistence with SQLite, and verifies deletion outcomes.
"""

import re
from typing import Any, Dict, List, Optional
from asterline.graph.state import MemoryItem
from asterline.memory.policy import evaluate_memory_candidate
from asterline.memory.service import MemoryService


class MemoryManagerAgent:
    """Manages user-specific long-term memory operations with strict authorization."""

    def __init__(self, memory_service: MemoryService):
        self.service = memory_service

    def handle_request(
        self,
        user_id: str,
        user_message: str,
    ) -> Dict[str, Any]:
        """Parse user intent regarding long-term memory and execute authorized operation."""
        msg_lower = user_message.lower().strip()

        # ---------------------------------------------------------------------
        # 1. Forget / Delete Memory Request
        # ---------------------------------------------------------------------
        if any(kw in msg_lower for kw in ("forget", "delete memory", "remove memory", "erase")):
            active_memories = self.service.list_user_memories(user_id)
            if not active_memories:
                return {
                    "response": "You currently have no active remembered context to forget.",
                    "operation": "FORGET_NONE",
                    "status": "COMPLETED",
                }

            # Identify target memory based on keywords in user message
            target_mem: Optional[MemoryItem] = None
            for mem in active_memories:
                mem_tokens = set(mem.content.lower().split())
                # Check for token overlap with forget command
                if any(t in msg_lower for t in mem_tokens if len(t) > 3):
                    target_mem = mem
                    break

            if not target_mem:
                # If specific target not identified, target the most recent memory or request clarification
                target_mem = active_memories[0]

            # Execute deletion
            success = self.service.delete_memory(
                user_id=user_id,
                memory_id=target_mem.memory_id,
                reason="User requested deletion via chat command",
            )

            # Verification of deletion at persistence layer
            is_still_active = self.service.is_memory_active(user_id, target_mem.memory_id)

            if success and not is_still_active:
                return {
                    "response": f"I have forgotten your remembered context: '{target_mem.content}'. This record has been deactivated and will no longer be used.",
                    "operation": f"FORGET {target_mem.memory_id}",
                    "status": "SUCCESS",
                }
            else:
                return {
                    "response": "Failed to complete forget request at the persistence layer. Please try again.",
                    "operation": f"FORGET {target_mem.memory_id}",
                    "status": "FAILED",
                }

        # ---------------------------------------------------------------------
        # 2. View / List / Recall Memories Request
        # ---------------------------------------------------------------------
        if any(kw in msg_lower for kw in ("what do you remember", "what do you know about me", "list my memories", "my preferences", "show memories")):
            active_memories = self.service.list_user_memories(user_id)
            if not active_memories:
                return {
                    "response": "I do not have any stored personal memories or workspace context for your profile.",
                    "operation": "LIST_MEMORIES",
                    "status": "COMPLETED",
                }
            memory_list_str = "\n".join(f"- [{m.category}] {m.content}" for m in active_memories)
            return {
                "response": f"Here is what I currently remember about your profile:\n{memory_list_str}",
                "operation": "LIST_MEMORIES",
                "status": "COMPLETED",
            }

        # Check for specific personal context recall question
        is_explicit_remember = any(
            candidate_prefix in msg_lower
            for candidate_prefix in ("remember that", "please remember", "remember:", "take note", "note that")
        ) or (msg_lower.startswith("remember") and not msg_lower.endswith("?"))

        if not is_explicit_remember and (
            any(msg_lower.startswith(w) for w in ("what", "which", "where", "who", "does", "do", "is", "are", "how"))
            or "?" in msg_lower
            or any(kw in msg_lower for kw in ("warehouse", "database", "tier", "region", "role", "lead", "team"))
        ):
            active_memories = self.service.list_user_memories(user_id)
            if not active_memories:
                return {
                    "response": "I do not have any stored personal memories or workspace context for your profile.",
                    "operation": "RECALL_NONE",
                    "status": "COMPLETED",
                }

            stop_words = {"what", "which", "where", "who", "does", "do", "our", "team", "we", "use", "is", "are", "the", "a", "an", "in", "for", "to", "of", "my", "me", "about"}
            query_tokens = {t for t in re.findall(r"\b[a-zA-Z0-9_\-]+\b", msg_lower) if t not in stop_words and len(t) > 2}

            best_match: Optional[MemoryItem] = None
            best_overlap = 0
            for mem in active_memories:
                mem_tokens = {t.lower() for t in re.findall(r"\b[a-zA-Z0-9_\-]+\b", mem.content)}
                overlap = len(query_tokens & mem_tokens)
                if overlap > best_overlap:
                    best_overlap = overlap
                    best_match = mem

            if best_match and best_overlap > 0:
                return {
                    "response": f"Based on your remembered context: {best_match.content}",
                    "operation": f"RECALL {best_match.memory_id}",
                    "status": "COMPLETED",
                }
            else:
                memory_list_str = "\n".join(f"- [{m.category}] {m.content}" for m in active_memories)
                return {
                    "response": f"Here is your remembered context:\n{memory_list_str}",
                    "operation": "RECALL_ALL",
                    "status": "COMPLETED",
                }

        # ---------------------------------------------------------------------
        # 3. Remember / Store New Context
        # ---------------------------------------------------------------------
        # Extract candidate text by stripping prefixes like "remember that", "please remember", etc.
        candidate_text = user_message
        for prefix in (
            "please remember that",
            "remember that",
            "remember:",
            "remember",
            "take note that",
            "note that",
        ):
            if candidate_text.lower().startswith(prefix):
                candidate_text = candidate_text[len(prefix):].strip(" :-,")
                break

        # Category inference
        category = "preference"
        if any(w in candidate_text.lower() for w in ("workspace", "project", "tier", "plan", "region")):
            category = "workspace_context"
        elif any(w in candidate_text.lower() for w in ("role", "lead", "engineer", "devops", "team")):
            category = "project_role"

        # Apply memory policy and create memory
        item, decision = self.service.create_memory(
            user_id=user_id,
            content=candidate_text,
            category=category,
            reason="User requested remember via chat",
        )

        if not decision.allowed:
            return {
                "response": f"I cannot remember this information: {decision.message}",
                "operation": f"REJECTED ({decision.reason_code})",
                "status": "POLICY_REJECTED",
            }

        assert item is not None
        return {
            "response": f"I have remembered: '{item.content}' (Category: {item.category}).",
            "operation": f"CREATED {item.memory_id}",
            "status": "SUCCESS",
        }
