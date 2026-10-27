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

## Phase 4: Agent Core, Temporal Resolution & Guardrails
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
  - `src/asterline/agents/memory_manager.py`
  - `src/asterline/agents/verification.py`
  - `tests/test_guardrails.py`
  - `tests/test_verification.py`
- **Dependencies**: Phase 3.
- **Acceptance Criteria**:
  - Verifier detects fabricated citations and flags ungrounded claims.
  - Investigation agent correctly resolves known corpus supersessions (cold storage retention: 20d vs 60d; on-call stipend: $175 vs $140).
  - Memory manager ignores transient remarks and rejects secrets.
- **Verification Tests**:
  - `tests/test_guardrails.py`, `tests/test_verification.py`.

---

## Phase 5: LangGraph Multi-Agent Orchestration & Routing
- **Scope**:
  - StateGraph setup connecting nodes with typed `ResearchAssistantState`.
  - Conditional router evaluating user intent and selecting minimal execution path.
  - Bounded retry loop (max 2 retries) between Verifier and Research agents.
  - Observability integration attaching metadata, node timings, and tags to LangSmith traces.
- **Files Expected**:
  - `src/asterline/graph/routing.py`
  - `src/asterline/graph/workflow.py`
  - `src/asterline/observability/tracing.py`
  - `tests/test_routing.py`
  - `tests/test_workflow.py`
- **Dependencies**: Phase 4.
- **Acceptance Criteria**:
  - Direct greetings bypass the RAG pipeline.
  - Factual questions route to Documentation Agent; incident/conflict questions route to Investigation Agent.
  - LangSmith traces clearly show agent collaboration sequence.
- **Verification Tests**:
  - `tests/test_routing.py`, `tests/test_workflow.py`.

---

## Phase 6: Interactive Streamlit UI & Memory Inspector
- **Scope**:
  - Multi-user simulation selector (e.g., Alice - DevOps Lead, Bob - Backend Eng, Charlie - Support).
  - Multi-turn chat interface with live session loading and thread persistence.
  - Collapsible citation drawer displaying cited `chunk_id`, document `title`, and source excerpt.
  - Long-Term Memory Management panel allowing users to inspect what is remembered, edit entries, or trigger "forget" operations.
  - Execution trace viewer / status badge indicating routing decisions and verification verdicts.
- **Files Expected**:
  - `app/main.py`
  - `app/components.py`
- **Dependencies**: Phase 5.
- **Acceptance Criteria**:
  - Restarting the app preserves conversation history and stored memories for the selected user.
  - Switching between users completely updates the view with zero context leakage.
  - One-click launch via single command (`streamlit run app/main.py`).
- **Verification Tests**:
  - Manual UI workflow walk-through and automated component smoke test.

---

## Phase 7: Evaluation Suite, LangSmith Benchmarking & Failure Analysis
- **Scope**:
  - Curated evaluation datasets:
    - `evaluation/datasets/research.jsonl`: 20+ grounded questions with ground truth chunk IDs.
    - `evaluation/datasets/memory.jsonl`: Multi-turn memory recall, correction, and isolation tests.
    - `evaluation/datasets/routing.jsonl`: Intent classification test cases.
  - Automated evaluation harness computing Groundedness, Citation Precision/Recall, and Memory Safety.
  - Representative failure analysis documenting edge cases, model limitations, and iterative improvements.
- **Files Expected**:
  - `evaluation/evaluators.py`
  - `evaluation/run_evals.py`
  - `docs/evaluation.md`
  - `docs/failure_analysis.md`
- **Dependencies**: Phase 6.
- **Acceptance Criteria**:
  - $>90\%$ Citation validity score.
  - $100\%$ Isolation across synthetic user test cases.
  - Documented failure cases with analysis and corrective prompt/logic adjustments.
- **Verification Tests**:
  - `python -m evaluation.run_evals`.

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
