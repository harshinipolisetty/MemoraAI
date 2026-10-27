# Asterline Research Assistant: Implementation Plan

This document outlines the step-by-step phased roadmap for developing the Asterline Research Assistant. Each phase is scoped to be independent, verifiable, and backed by automated tests before progressing to subsequent steps.

---

## Phase Overview & Dependency Graph

```
[Phase 1: Architecture & Foundation] (Current Phase)
                │
                ▼
[Phase 2: Configuration, State Schemas & SQLite Persistence]
                │
                ▼
[Phase 3: Local Vector Indexing & Hybrid Retrieval Engine]
                │
                ▼
[Phase 4: Agent Core, Temporal Resolution & Guardrails]
                │
                ▼
[Phase 5: LangGraph Multi-Agent Orchestration & Routing]
                │
                ▼
[Phase 6: Interactive Streamlit UI & Memory Inspector]
                │
                ▼
[Phase 7: Evaluation Benchmark, LangSmith Tracing & Failure Analysis]
                │
                ▼
[Phase 8: Documentation, AI Disclosure & Final Polish]
```

---

## Phase 1: Architecture, Requirements & Project Scaffolding *(Completed)*
- **Scope**:
  - Ingest and verify the 75-chunk Asterline corpus.
  - Define system architecture, agent boundaries, state schemas, routing specifications, and conflict-resolution rules.
  - Create project structure, `.gitignore`, and `.env.example`.
- **Files Created / Changed**:
  - `data/corpus.jsonl`
  - `.gitignore`
  - `.env.example`
  - `docs/architecture.md`
  - `docs/implementation_plan.md`
  - `README.md`
- **Dependencies**: None.
- **Acceptance Criteria**:
  - `data/corpus.jsonl` contains exactly 75 records without missing fields.
  - Architectural requirements from the assignment brief are completely mapped.
  - Secrets are excluded from version control.
- **Verification Tests**:
  - Corpus integrity check (valid JSON lines, unique `chunk_id`s, required metadata keys).

---

## Phase 2: Configuration, Typed Schemas & SQLite Storage *(Completed)*
- **Scope**:
  - Environment settings loader with validation (`pydantic-settings`).
  - Structured data schemas (`State`, `RetrievedChunk`, `Citation`, `MemoryItem`, `AuditLog`).
  - SQLite database manager implementing tables for `conversations`, `messages`, `user_memories`, and `memory_audit_log`.
  - Memory service handling selective storage, updates, user isolation, and deletion ("forget").
- **Files Expected**:
  - `src/asterline/config/settings.py`
  - `src/asterline/graph/state.py`
  - `src/asterline/memory/store.py`
  - `src/asterline/memory/service.py`
  - `src/asterline/memory/policy.py`
  - `tests/test_memory.py`
- **Dependencies**: Phase 1.
- **Acceptance Criteria**:
  - Memory read/writes are strictly partitioned by `user_id`.
  - Sensitive tokens (API keys) are rejected by memory policy.
  - Deletion completely purges active records and logs an audit entry.
- **Verification Tests**:
  - `tests/test_memory.py` (test user isolation, test forget request, test update conflict).

---

## Phase 3: Local Vector Indexing & Hybrid Retrieval Engine *(Completed)*
- **Scope**:
  - Corpus ingestion pipeline parsing `data/corpus.jsonl` without data loss.
  - Local embedding integration using FastEmbed (`BAAI/bge-small-en-v1.5`).
  - Local vector store (ChromaDB / in-memory vector index) storing document chunks with complete metadata (`doc_id`, `title`, `category`, `published`, `version`, `source_url`).
  - Hybrid / metadata-aware retriever supporting semantic queries and exact token searches (e.g., incident codes `INC-2025-08`).
- **Files Expected**:
  - `src/asterline/rag/ingestion.py`
  - `src/asterline/rag/retriever.py`
  - `src/asterline/rag/metadata.py`
  - `tests/test_retrieval.py`
- **Dependencies**: Phase 2.
- **Acceptance Criteria**:
  - Ingestion indexes all 75 chunks cleanly with zero external API calls.
  - Retrieval achieves $>90\%$ Recall@5 on benchmark queries (pricing, incidents, SLAs).
  - Preserves exact `chunk_id` and `title` in all query results.
- **Verification Tests**:
  - `tests/test_retrieval.py` (test index count = 75, test incident lookup, test plan limit lookup).

---

## Phase 4: Agent Core, Temporal Resolution & Guardrails *(Completed)*
- **Scope**:
  - Input guardrails: character length limit, prompt injection sanitizer, API key scrubber.
  - Output guardrails: citation verifier against retrieved chunks, privacy auditor.
  - Documentation Research Agent: answers direct queries with precise chunk citations.
  - Investigation Research Agent: synthesizes multi-document incidents, reconciles superseded policies, and evaluates version differences.
  - Memory Manager Agent: parses conversation turns to identify durable user context.
  - Verification Agent: structured fact-checker evaluating groundedness, citation validity, and privacy.
- **Files Expected**:
  - `src/asterline/guardrails/input.py`
  - `src/asterline/guardrails/output.py`
  - `src/asterline/agents/documentation.py`
  - `src/asterline/agents/investigation.py`
  - `src/asterline/agents/conflict_resolution.py`
  - `src/asterline/agents/memory_manager.py`
  - `src/asterline/agents/verification.py`
  - `tests/test_guardrails.py`
  - `tests/test_agents.py`
  - `tests/test_verification.py`
- **Dependencies**: Phase 3.
- **Acceptance Criteria**:
  - Verifier detects fabricated citations and flags ungrounded claims.
  - Investigation agent correctly resolves known corpus supersessions (cold storage retention: 20d vs 60d; on-call stipend: $175 vs $140).
  - Memory manager ignores transient remarks and rejects secrets.
- **Verification Tests**:
  - `tests/test_guardrails.py`, `tests/test_agents.py`, `tests/test_verification.py`.

---

## Phase 5: Evaluation, LangSmith Observability & System Hardening *(Completed)*
- **Scope**:
  - Offline reproducible evaluation harness (`evaluation/evaluators.py`, `evaluation/run_evals.py`).
  - Comprehensive evaluation datasets:
    - `evaluation/datasets/research.jsonl`: 20 research cases (grounding, pricing, SLAs, incidents, temporal supersession, and abstention).
    - `evaluation/datasets/routing.jsonl`: 20 intent routing cases across 5 operational paths.
    - `evaluation/datasets/memory.jsonl`: 10 multi-turn scenarios covering persistence, Alice/Bob/Charlie isolation, forget purges, and policy rejections.
  - Metrics evaluation: Hit Rate@k, Recall@k, MRR, Citation Validity (100%), Grounded Response (100%), Abstention (100%), Routing (100%), Memory Isolation (100%).
  - LangSmith Observability & Redaction (`src/asterline/observability/tracing.py`):
    - Recursive PII, token, and API key scrubbing before trace dispatch.
    - Sanitized trace metadata omitting raw memory contents.
    - Graceful non-blocking fallback when credentials are absent or tracing is disabled.
  - Hardening fixes:
    - Bounded retry routing returning to original researcher.
    - True cryptographic/physical deletion (`DELETE FROM user_memories`) upon forget operations.
    - Explicit fallback reporting with structured metadata.
  - Documentation: `docs/evaluation.md`, `docs/failure_analysis.md`, `evaluation/results/evaluation_report.json`.
- **Files Created / Updated**:
  - `evaluation/evaluators.py`
  - `evaluation/run_evals.py`
  - `evaluation/datasets/research.jsonl`
  - `evaluation/datasets/routing.jsonl`
  - `evaluation/datasets/memory.jsonl`
  - `src/asterline/observability/tracing.py`
  - `tests/test_evaluation.py`
  - `tests/test_tracing.py`
  - `docs/evaluation.md`
  - `docs/failure_analysis.md`
- **Dependencies**: Phase 4.
- **Acceptance Criteria**:
  - Zero paid API calls required for offline evaluations.
  - Corpus integrity preserved (SHA256: `F63FC05CF43204485C98BD8FB6D89860A32C16A7E906BC90975E7C3954FE9FB1`).
  - All 110 automated tests passing.
- **Verification Tests**:
  - `python -m evaluation.run_evals`
  - `python -m pytest`

---

## Phase 6: Interactive Streamlit UI & Memory Inspector *(Completed)*
- **Scope**:
  - Multi-user simulation selector (Alice - DevOps Lead, Bob - Backend Eng, Charlie - Support) with strict isolation.
  - Multi-turn chat interface with live SQLite session loading, thread switching, and auto-titling.
  - Collapsible citation drawer displaying cited `chunk_id`, document `title`, `source_url`, and chunk excerpts.
  - Complete Long-Term Memory Management panel allowing users to inspect, add, edit, or forget records with confirmation.
  - Execution trace viewer indicating routing decisions, fallback status, verifier verdicts, and LangSmith status.
  - Modular component architecture (`app/components.py`, `app/main.py`) with cached runtime (`@st.cache_resource`).
- **Files Created / Updated**:
  - `app/main.py`
  - `app/components.py`
  - `tests/test_ui.py`
  - `pyproject.toml`
  - `src/asterline/memory/service.py`
  - `src/asterline/memory/store.py`
- **Dependencies**: Phase 5.
- **Acceptance Criteria**:
  - Restarting the app preserves conversation history and stored memories for the selected user.
  - Switching between users completely re-scopes view with zero cross-user context leakage.
  - One-click launch via single command (`streamlit run app/main.py`).
  - Streamlit boot verified via HTTP 200 smoke test on port 8505.
- **Verification Tests**:
  - `tests/test_ui.py` (6 tests passed).
  - Full test suite: 116 tests passing.
  - Manual UI workflow walk-through and automated component smoke test.

---

## Phase 8: Project Polish, AI Disclosure & Final Deliverables
- **Scope**:
  - Finalize `README.md` with verified launch commands and architecture diagrams.
  - Complete `AI_DISCLOSURE.md` documenting tool usage and human verification.
  - Verify complete repository cleanliness (no keys, no local db or cache committed).
- **Files Expected**:
  - `README.md`
  - `AI_DISCLOSURE.md`
  - `pyproject.toml`
- **Dependencies**: Phase 7.
- **Acceptance Criteria**:
  - Clean `git status`.
  - All test suites passing.
  - Accessible LangSmith project link documented.
