"""Typed state definitions for the Asterline Research Assistant LangGraph workflow.

Distinguishes trusted application inputs from untrusted user and document data.
Provides validated Pydantic models for structured components alongside
TypedDict definitions for LangGraph state passing.
"""

from typing import Any, Dict, List, Literal, Optional, TypedDict
from pydantic import BaseModel, Field


class Citation(BaseModel):
    """Source citation identifying evidence chunk and document title."""

    chunk_id: str = Field(description="Unique identifier of chunk, e.g. pricing:0 or incident-12:1")
    doc_id: str = Field(description="Document identifier, e.g. pricing or incident-12")
    title: str = Field(description="Title of document, e.g. Plans and billing")
    source_url: str = Field(default="", description="Internal documentation source URL")


class RetrievedChunk(BaseModel):
    """Corpus evidence chunk retrieved from vector store."""

    chunk_id: str
    doc_id: str
    title: str
    category: str
    owner: str
    source_url: str
    published: str
    version: str
    text: str
    score: float = Field(default=0.0, description="Relevance similarity score")


class VerificationResult(BaseModel):
    """Structured fact-checking verdict from Verification Agent."""

    verdict: Literal["pass", "revise", "abstain"] = Field(
        description="Verification outcome. 'pass' allows final delivery, 'revise' triggers bounded retry, 'abstain' flags missing evidence."
    )
    is_grounded: bool = Field(description="True if all claims are supported by retrieved chunks")
    citation_validity: bool = Field(description="True if cited chunks exist in retrieved set and contain claimed facts")
    privacy_compliant: bool = Field(description="True if no unauthorized user data or cross-user leakage was detected")
    feedback: Optional[str] = Field(default=None, description="Actionable revision feedback for research agents")
    unsupported_claims: List[str] = Field(default_factory=list, description="List of unverified claims detected in draft")


class MemoryItem(BaseModel):
    """Durable long-term memory record for a specific user."""

    memory_id: str
    user_id: str
    category: Literal["preference", "workspace_context", "project_role", "investigation_note", "general"]
    content: str
    confidence: float = 1.0
    source_message_id: Optional[str] = None
    is_active: bool = True
    created_at: str
    updated_at: str


class ConversationMessage(BaseModel):
    """Single turn in a persistent conversation."""

    message_id: str
    conversation_id: str
    user_id: str
    role: Literal["user", "assistant", "system"]
    content: str
    citations: List[Citation] = Field(default_factory=list)
    route_taken: Optional[str] = None
    created_at: str


class ResearchAssistantState(TypedDict, total=False):
    """Shared execution state for LangGraph workflow.

    TRUST BOUNDARIES:
    - Trusted Application Inputs: user_id, conversation_id, active_user_memories, conversation_history
    - Untrusted User Inputs: user_message
    - Untrusted Retrieved Data: retrieved_chunks
    - Generated Intermediate Data: route, routing_reason, search_queries, draft_response, citations, verification
    - Final Output Data: final_response, memory_operations_executed, error
    """

    # --- 1. Trusted Application Session Context ---
    user_id: str
    """Trusted authenticated/selected user ID. Set by application layer, never inferred from chat."""

    conversation_id: str
    """Active conversation/thread identifier. Set by application layer."""

    conversation_history: List[Dict[str, Any]]
    """Chronological conversation turns loaded from SQLite storage."""

    active_user_memories: List[Dict[str, Any]]
    """Curated active long-term memories retrieved from SQLite storage for this user."""

    # --- 2. Untrusted User Input ---
    user_message: str
    """Current raw message provided by user. Subject to input guardrails."""

    # --- 3. Routing & Orchestration Decisions ---
    route: Literal["factual", "investigation", "memory_ops", "conversational", "clarification"]
    """Workflow branch selected by Conditional Router."""

    routing_reason: str
    """Concise operational reason for selected route (logged to LangSmith)."""

    retry_count: int
    """Current count of verification retries (bounded by MAX_INVESTIGATION_RETRIES)."""

    # --- 4. Retrieval & Evidence ---
    search_queries: List[str]
    """Search queries formulated by research agents for vector retrieval."""

    retrieved_chunks: List[Dict[str, Any]]
    """Corpus evidence chunks retrieved from vector store."""

    # --- 5. Draft Synthesis & Citations ---
    draft_response: str
    """Intermediate response drafted by Documentation or Investigation agent."""

    citations: List[Dict[str, Any]]
    """Citations extracted or generated with draft response."""

    # --- 6. Verification & Auditing ---
    verification: Optional[Dict[str, Any]]
    """Structured verdict from Verification Agent."""

    # --- 7. Final Output & State Persistence ---
    final_response: str
    """Verified, citation-grounded response ready for delivery to user."""

    memory_operations_executed: List[str]
    """Log of memory operations (e.g., 'CREATED memory_123', 'FORGOT preference_456')."""

    error: Optional[str]
    """Safe error message in case of unrecoverable failure or provider fault."""

    is_fallback: bool
    """True if deterministic fallback was used instead of live LLM model generation."""

    llm_metadata: Dict[str, Any]
    """Metadata regarding LLM provider execution, model, latency, or fallback error."""

