# MemoraAI — An Enterprise Research Assistant That Remembers

[![Python Version](https://img.shields.io/badge/python-3.10%20%7C%203.11%20%7C%203.12%20%7C%203.14-blue.svg)](https://www.python.org/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
[![Tests](https://img.shields.io/badge/pytest-134%20passed-brightgreen.svg)](tests/)
[![Corpus](https://img.shields.io/badge/corpus-75%20chunks%20verified-purple.svg)](data/corpus.jsonl)
[![Tracing](https://img.shields.io/badge/observability-LangSmith%20ready-orange.svg)](https://smith.langchain.com/)

> **MemoraAI** is a dependable, high-density enterprise research assistant developed for Asterline Labs. It produces strictly grounded, citation-backed answers from company technical documentation, automatically resolves conflicting policy versions, and maintains user-isolated persistent memory across multi-turn investigations.

---

## Table of Contents

1. [Project Overview & Problem Statement](#1-project-overview--problem-statement)
2. [Core Architecture & Workflow](#2-core-architecture--workflow)
3. [Key Implemented Features](#3-key-implemented-features)
   - [3.1 Multi-Agent System & Responsibilities](#31-multi-agent-system--responsibilities)
   - [3.2 Grounded Research & Temporal Supersession](#32-grounded-research--temporal-supersession)
   - [3.3 Selective Durable Memory & Privacy Isolation](#33-selective-durable-memory--privacy-isolation)
   - [3.4 Hybrid Retrieval Engine & Answer Caching](#34-hybrid-retrieval-engine--answer-caching)
   - [3.5 Enterprise Developer UI](#35-enterprise-developer-ui)
   - [3.6 Observability, Telemetry & Guardrails](#36-observability-telemetry--guardrails)
4. [Technology Stack](#4-technology-stack)
5. [Repository Structure](#5-repository-structure)
6. [Prerequisites](#6-prerequisites)
7. [Installation & Setup](#7-installation--setup)
8. [Environment Configuration Reference](#8-environment-configuration-reference)
9. [Running the Application](#9-running-the-application)
10. [Testing & Verification](#10-testing--verification)
11. [Evaluation Results & Benchmarks](#11-evaluation-results--benchmarks)
12. [Security, Privacy & Compliance Safeguards](#12-security-privacy--compliance-safeguards)
13. [Troubleshooting & Common Failure Modes](#13-troubleshooting--common-failure-modes)
14. [Limitations & Future Roadmap](#14-limitations--future-roadmap)
15. [AI Usage Disclosure](#15-ai-usage-disclosure)

---

## 1. Project Overview & Problem Statement

### The Problem
In modern software enterprises, technical knowledge is fragmented across runbooks, release notes, SLA agreements, and operational policies. When engineers investigate production incidents or query infrastructure parameters, traditional search engines and generic AI chatbots suffer from critical failure modes:
1. **Hallucinated Documentation**: Emitting plausible-sounding but non-existent endpoints, limits, or configurations.
2. **Version Conflicts & Temporal Blindness**: Failing to recognize that a newer policy (e.g., 2026 cold-storage retention of 20 days) explicitly supersedes an older overview (e.g., 2025 security guide stating 60 days).
3. **Amnesia Across Sessions**: Requiring engineers to repeatedly re-explain their role, toolset, warehouse configurations, and past incident context.
4. **Data Bleed Between Users**: Leaking private workspace context, sensitive tokens, or operational notes between different team members.

### The Solution: MemoraAI
MemoraAI solves these challenges through a deterministic, inspectable multi-agent architecture orchestrated by **LangGraph**:
- **Authoritative Grounding**: All factual claims must be strictly anchored to verified chunks from the immutable 75-chunk Asterline knowledge corpus (`data/corpus.jsonl`).
- **Independent Verification**: An autonomous Verification Agent fact-checks every draft against retrieved evidence, flags ungrounded claims or invalid citation IDs, and commands bounded revisions before output delivery.
- **Three-Tier Data Separation**: Clean partitioning between read-only company documentation, ephemeral multi-turn session threads, and durable user preferences.
- **Strict User Isolation**: All memory and conversation records in SQLite are partitioned by `user_id`. Alice, Bob, and Charlie can never view, update, or purge each other's data.
- **Deterministic Offline Capability**: Operates 100% offline with zero paid API dependencies, while seamlessly activating live high-throughput generation (Groq Llama-3.3-70B) when configured.

---

## 2. Core Architecture & Workflow

MemoraAI models research execution as a typed state graph using **LangGraph**. Requests enter through strict input guardrails, pass through a conditional router, fan out to specialized agents, undergo verification fact-checking with bounded retry loops, and commit to isolated SQLite persistence.

```mermaid
flowchart TD
    subgraph UI ["Client Layer (Streamlit Dark Workspace)"]
        User(["👤 User Query"])
        UIRender["Chat Stream & Technical Header"]
    end

    subgraph Guardrails ["Boundary Layer"]
        InGuard["🛡️ Input Guardrails<br/>• Injection detection<br/>• Session validation<br/>• Context sanitization"]
        OutGuard["🛡️ Output Guardrails<br/>• Secret scrubbing (ast_live_*, sk-*)<br/>• PII & credential masking<br/>• Citation integrity check"]
    end

    subgraph Orchestrator ["LangGraph Multi-Agent Workflow"]
        Router{"🧭 Intent Router<br/>(Deterministic Classifier)"}
        
        ConvNode["👋 Conversational Node<br/>Fast path (greetings/identity)"]
        ClarNode["❓ Clarification Node<br/>Underspecified queries"]
        MemNode["🧠 Memory Manager Agent<br/>Extraction, policy check & audit"]
        DocNode["📖 Documentation Agent<br/>Single-topic factual research"]
        InvestNode["🔬 Investigation Agent<br/>Multi-doc synthesis & supersession"]
        
        VerifyNode{"⚖️ Verification Agent<br/>(Hallucination & Citation Check)"}
        PersistNode["💾 Persistence Node<br/>SQLite thread & message commit"]
    end

    subgraph Storage ["Retrieval & Data Layer"]
        RAGCache[("⚡ In-Memory RAG Cache<br/>TTL Query & Embedding Cache")]
        DenseStore[("🔵 ChromaDB Vector Index<br/>FastEmbed BAAI/bge-small-en-v1.5")]
        BM25Store[("🟠 BM25 Lexical Engine<br/>Exact token & identifier matching")]
        CorpusDoc[("📄 Immutable Corpus<br/>75 Chunks (data/corpus.jsonl)")]
        SQLiteDB[("🗄️ SQLite Database<br/>memory.db (Isolated user partitions)")]
    end

    subgraph External ["Telemetry & Remote Providers (Optional)"]
        GroqLLM["⚡ Groq API<br/>llama-3.3-70b-versatile"]
        LangSmith["📊 LangSmith Cloud<br/>Non-blocking tracing & metrics"]
    end

    %% Workflow Edges
    User --> InGuard
    InGuard -->|Valid| Router
    InGuard -->|Blocked| OutGuard
    
    Router -->|conversational| ConvNode
    Router -->|clarification| ClarNode
    Router -->|memory_ops| MemNode
    Router -->|factual| DocNode
    Router -->|investigation| InvestNode
    
    ConvNode --> OutGuard
    ClarNode --> OutGuard
    MemNode <--> SQLiteDB
    MemNode --> OutGuard
    
    DocNode <--> RAGCache
    InvestNode <--> RAGCache
    RAGCache <--> DenseStore
    RAGCache <--> BM25Store
    DenseStore -.-> CorpusDoc
    BM25Store -.-> CorpusDoc
    
    DocNode -.-> GroqLLM
    InvestNode -.-> GroqLLM
    
    DocNode --> VerifyNode
    InvestNode --> VerifyNode
    
    VerifyNode -->|Pass / Abstain| OutGuard
    VerifyNode -->|Revise (Retry <= 2)| InvestNode
    
    OutGuard --> PersistNode
    PersistNode --> SQLiteDB
    PersistNode --> UIRender
    UIRender --> User

    Orchestrator -.->|Sanitized traces| LangSmith
```

### Execution Flow Step-by-Step
1. **Input Guardrails**: Validates message length, strips control sequences, flags prompt injection attacks, and loads the active user's approved long-term memories from SQLite.
2. **Intent Classification**: Evaluates query semantics to choose the optimal path:
   - `conversational`: Immediate response without RAG latency.
   - `clarification`: Prompts user when query is underspecified.
   - `memory_ops`: Invokes memory manager to record, update, or purge user preferences.
   - `factual`: Dispatches to Documentation Agent for direct factual lookup.
   - `investigation`: Dispatches to Investigation Agent for multi-document synthesis and conflict resolution.
3. **Hybrid Retrieval**: Queries ChromaDB (dense semantic similarity via FastEmbed) and BM25 (lexical matching), fusing scores via Reciprocal Rank Fusion (RRF, $k=60$) with an in-memory TTL query cache.
4. **Agent Generation**: Generates draft answers citing explicit chunk IDs (`[chunk_id]`). If evidence is missing, emits an honest evidence-based abstention.
5. **Verification & Bounded Retry**: Fact-checks claims against retrieved text. If ungrounded assertions or fabricated chunk IDs are detected, triggers a bounded revision loop (up to 2 retries).
6. **Output Guardrails & Persistence**: Redacts credentials or API keys, verifies citation syntax, commits conversation turns to isolated SQLite tables, and dispatches sanitized traces to LangSmith.

---

## 3. Key Implemented Features

### 3.1 Multi-Agent System & Responsibilities

| Component | Implementation File | Architectural Role | Core Responsibility |
| :--- | :--- | :--- | :--- |
| **Input Guardrails** | [`src/asterline/guardrails/input.py`](src/asterline/guardrails/input.py) | Workflow Node | Sanitizes input text, prevents prompt injection, checks context length, and loads user memories. |
| **Conditional Router** | [`src/asterline/graph/routing.py`](src/asterline/graph/routing.py) | Workflow Node | Classifies query intent into 5 deterministic paths (`conversational`, `clarification`, `memory_ops`, `factual`, `investigation`). |
| **Documentation Agent** | [`src/asterline/agents/documentation.py`](src/asterline/agents/documentation.py) | Specialized Agent | Answers single-topic factual queries (APIs, quotas, billing). Emits verified chunk citations or abstains. |
| **Investigation Agent** | [`src/asterline/agents/investigation.py`](src/asterline/agents/investigation.py) | Specialized Agent | Reconciles multi-document incidents (`INC-2025-08`, `INC-2025-12`), compares version differences, and resolves supersessions. |
| **Memory Manager Agent** | [`src/asterline/agents/memory_manager.py`](src/asterline/agents/memory_manager.py) | Specialized Agent | Evaluates memory safety, records durable workspace preferences, updates active records, and executes permanent purges. |
| **Verification Agent** | [`src/asterline/agents/verification.py`](src/asterline/agents/verification.py) | Specialized Agent | Audits citations against retrieved evidence, flags hallucinations, enforces privacy boundaries, and triggers retries. |
| **Output Guardrails** | [`src/asterline/guardrails/output.py`](src/asterline/guardrails/output.py) | Workflow Node | Scrubs accidental tokens (`ast_live_*`, `sk-*`, passwords) and validates bracketed citation references. |
| **Persistence Node** | [`src/asterline/graph/workflow.py`](src/asterline/graph/workflow.py) | Workflow Node | Transactionally commits conversation turns and citation metadata to isolated SQLite stores. |

### 3.2 Grounded Research & Temporal Supersession
- **Exact Document Grounding**: Every answer links to authoritative source chunks with explicit `chunk_id` and document `title`.
- **Temporal Policy Supersession**: Automatically detects when a newer document overrides an older rule. For example:
  - Cold recovery storage retention: Updated from 60 days (`security:1`, 2025-03-15) to 20 days (`retention:0`, 2026-01-15).
  - On-call engineer stipend: Increased from $140/week (`runbook:4`) to $175/week (`handbook:1`, 2026-01-01).
- **Honest Abstention**: When a topic is not present in the documentation (e.g., webhook signing key rotation), MemoraAI honestly abstains rather than hallucinating plausible policies.

### 3.3 Selective Durable Memory & Privacy Isolation
- **Policy Enforcement**: Distinguishes durable workspace preferences from transient conversation chatter.
  - *Allowed*: Tool preferences (`"We use Snowflake for warehousing"`), deployment targets (`"Primary cluster is us-east-1"`), team roles.
  - *Rejected*: Passwords, API keys (`ast_live_*`, `sk-*`, `ghp_*`), bearer tokens, and broad company assertions.
- **Physical "Right to Be Forgotten"**: Memory deletion executes a true physical `DELETE FROM user_memories WHERE memory_id = ? AND user_id = ?`. Audit logs preserve only anonymized timestamps and action records (`FORGET`), with zero residual PII.
- **Strict User Isolation**: All database operations are filtered by `user_id`. Alice, Bob, and Charlie operate in completely isolated SQLite data spaces.

### 3.4 Hybrid Retrieval Engine & Answer Caching
- **FastEmbed Local Embeddings**: Uses `BAAI/bge-small-en-v1.5` running locally on CPU (384 dimensions) with zero external embedding API cost.
- **ChromaDB Vector Store**: Persistent local vector index (`data/chroma`) storing all 75 corpus chunks with metadata preservation.
- **BM25 Lexical Search**: Keyword search specifically tuned for exact incident IDs (`INC-2025-12`), HTTP routes (`POST /v2/runs`), and version tokens (`v2.2`, `v2.3`).
- **Reciprocal Rank Fusion (RRF)**: Combines dense vector similarity and lexical ranking using reciprocal rank fusion ($k=60$):
  $$\text{RRF Score}(d) = \sum_{m \in \{\text{dense}, \text{bm25}\}} \frac{1}{60 + \text{rank}_m(d)}$$
- **In-Memory TTL Query Cache**: Deduplicates identical retrieval calls and repeated factual questions, reducing latency from ~3s to under 70ms.

### 3.5 Enterprise Developer UI
- **Design Philosophy**: High-density dark interface inspired by Linear, Datadog, and Stripe (`#0B0D10` canvas, `#101318` sidebar, `#7DA5FF` primary blue).
- **Content-Fitted User Bubbles**: Sleek, right-aligned message bubbles with dynamic width fitting (`width: fit-content`) preventing oversized empty message blocks.
- **Developer Chip Badges**: Compact indicators above assistant responses (`● Documentation Lookup`, `✓ Grounded`, `llama-3.3-70b`, `📄 1 Source`).
- **Interactive Quick-Starter Cards**: 1-click investigation buttons for payload limits, key deprecation policies, directory lookup, and stored context.
- **Memory Inspector**: Full CRUD interface for durable memory records with category filtering, search, and permanent purge confirmation.
- **Observability Dashboard**: Live metric tiles (`EXECUTION ROUTE`, `VERIFIER VERDICT`, `RETRIEVED / CITED`, `RETRY ATTEMPTS`) and a 5-step vertical workflow execution timeline.

### 3.6 Observability, Telemetry & Guardrails
- **LangSmith Tracing**: Full non-blocking integration via [`src/asterline/observability/tracing.py`](src/asterline/observability/tracing.py).
- **Automatic Secret Scrubbing**: Redacts API keys, passwords, and authorization tokens before dispatching traces to remote dashboards.
- **Privacy Shield**: Excludes personal long-term memories from trace tags, session metadata, and run names.

---

## 4. Technology Stack

| Technology | Purpose in Project |
| :--- | :--- |
| **Python (>=3.10)** | Primary programming language powering the application, agents, and test suite. |
| **LangGraph (>=0.1.0)** | Orchestrates multi-agent state machines, conditional routing, and bounded retry loops. |
| **LangChain Core (>=0.2.0)** | Provides standardized runnables, prompt templates, and LangSmith callback hooks. |
| **Streamlit (>=1.35.0)** | Serves the interactive enterprise research workspace and multi-screen dashboards. |
| **FastEmbed (>=0.3.0)** | Generates local dense embeddings on CPU (`BAAI/bge-small-en-v1.5`) without external API fees. |
| **ChromaDB (>=0.5.0)** | Persistent on-disk vector database storing chunk embeddings and metadata. |
| **rank-bm25 (>=0.2.2)** | Lexical keyword retrieval for exact matches on incident IDs, error codes, and API paths. |
| **SQLite3 (Built-in)** | ACID-compliant relational storage for conversations, user memories, and immutable audit logs. |
| **Groq / Llama-3.3-70B** | Primary high-speed LLM inference provider for factual research and verification. |
| **Pydantic (>=2.8.0)** | Strict schema validation, typed state management, and configuration enforcement. |
| **Pydantic Settings (>=2.4.0)**| Validates `.env` variables and ensures non-blocking defaults for offline runs. |
| **pytest (>=8.0.0)** | Automated test framework executing 134 unit, integration, and regression tests. |
| **LangSmith (>=0.1.0)** | Non-blocking execution tracing, latency profiling, and agent observability. |

---

## 5. Repository Structure

```
Memora/
├── .streamlit/
│   └── config.toml               # Streamlit theme config (dark graphite enforcement)
├── app/
│   ├── __init__.py               # Web app package marker
│   ├── components.py             # UI design system, CSS styling, chip badges, and widgets
│   └── main.py                   # Streamlit multi-tab application entry point
├── data/
│   ├── chroma/                   # Persistent ChromaDB vector index (generated)
│   ├── corpus.jsonl              # Authoritative, immutable 75-chunk Asterline corpus
│   └── memory.db                 # SQLite database for threads, memories, and audit logs
├── docs/
│   ├── architecture.md           # Detailed multi-agent architecture and graph flow
│   ├── evaluation.md             # Benchmark methodology, metric formulas, and datasets
│   ├── failure_analysis.md       # Technical retrospective of failure modes and solutions
│   ├── implementation_plan.md    # Multi-phase engineering roadmap and deliverables
│   └── retrieval.md              # Hybrid retrieval, RRF scoring, and indexing specifications
├── evaluation/
│   ├── datasets/
│   │   ├── memory.jsonl          # 10 curated memory lifecycle and isolation test cases
│   │   ├── research.jsonl        # 20 factual, supersession, and unanswerable test queries
│   │   └── routing.jsonl         # 20 intent classification benchmark queries
│   ├── results/
│   │   ├── evaluation_report.json # Official verified benchmark evaluation report
│   │   ├── latency_benchmark_50.json # Latency metrics across all 50 evaluation cases
│   │   ├── phase4_live_results.json  # Live LLM execution traces and answers
│   │   └── profiling_baseline.json   # Sub-system latency profiling breakdown
│   ├── benchmark.py              # Performance and latency benchmark harness
│   ├── benchmark_50.py           # 50-query end-to-end evaluation runner
│   ├── evaluators.py             # Grounding, citation, routing, and memory evaluators
│   ├── profile_stages.py         # Sub-stage execution timer (retrieval, LLM, verification)
│   ├── run_evals.py              # Top-level offline evaluation benchmark runner
│   └── verify_queries_live.py    # Verification script for live queries and citations
├── src/
│   └── asterline/
│       ├── __init__.py           # Root package initialization
│       ├── agents/
│       │   ├── __init__.py       # Agent package exports
│       │   ├── conflict_resolution.py # Temporal supersession detection and policy overriding
│       │   ├── documentation.py  # Documentation Research Agent (factual queries)
│       │   ├── investigation.py  # Investigation Research Agent (multi-doc incidents)
│       │   ├── llm.py            # LLM client with Groq provider and deterministic fallbacks
│       │   ├── memory_manager.py # Memory Manager Agent (policy enforcement and audit)
│       │   └── verification.py   # Verification Agent (hallucination and citation audit)
│       ├── config/
│       │   ├── __init__.py       # Configuration package exports
│       │   └── settings.py       # Pydantic Settings and environment validation
│       ├── graph/
│       │   ├── __init__.py       # Graph package exports
│       │   ├── routing.py        # Query intent classification and routing rules
│       │   ├── state.py          # TypedDict state schema and Pydantic models
│       │   └── workflow.py       # LangGraph StateGraph builder, nodes, and edges
│       ├── guardrails/
│       │   ├── __init__.py       # Guardrails package exports
│       │   ├── input.py          # Input sanitization, length checks, and injection defense
│       │   └── output.py         # Output credential scrubbing and citation integrity checks
│       ├── memory/
│       │   ├── __init__.py       # Memory package exports
│       │   ├── policy.py         # Memory safety policies (secret and chatter rejection)
│       │   ├── service.py        # High-level memory operations and user isolation logic
│       │   └── store.py          # SQLite relational store and transactional audit logger
│       ├── observability/
│       │   ├── __init__.py       # Observability package exports
│       │   └── tracing.py        # LangSmith setup, callback filters, and secret scrubbing
│       └── rag/
│           ├── __init__.py       # RAG package exports
│           ├── cache.py          # In-memory TTL query cache for retrieval acceleration
│           ├── ingestion.py      # Corpus JSONL validator, checksum verification, and chunking
│           ├── metadata.py       # Metadata extraction and document classification helpers
│           └── retriever.py      # Hybrid retriever (FastEmbed + ChromaDB + BM25 + RRF)
├── tests/
│   ├── test_agents.py            # Unit tests for documentation and investigation agents
│   ├── test_evaluation.py        # Unit tests for evaluation harness and metric calculations
│   ├── test_guardrails.py        # Unit tests for input/output guardrails and secret scrubbing
│   ├── test_memory.py            # Unit tests for SQLite memory, isolation, and hard purges
│   ├── test_retrieval.py         # Unit tests for ChromaDB, BM25, and hybrid RRF retrieval
│   ├── test_state.py             # Unit tests for Pydantic state models and citations
│   ├── test_tracing.py           # Unit tests for LangSmith tracing and secret redaction
│   ├── test_ui.py                # Unit tests for Streamlit UI helpers and demo profiles
│   ├── test_verification.py      # Unit tests for verification verdicts and retry loops
│   └── test_workflow.py          # Integration tests for LangGraph multi-agent execution
├── .env.example                  # Documented environment variable template
├── .gitignore                    # Git exclusion rules (secrets, databases, virtual environments)
├── AI_DISCLOSURE.md              # Transparent disclosure of AI assistance and human verification
├── pyproject.toml                # Project packaging, dependencies, and pytest configuration
└── README.md                     # Comprehensive project documentation (this document)
```

---

## 6. Prerequisites

### Mandatory
- **Operating System**: Windows 10/11, macOS, or Linux.
- **Python**: Version `3.10`, `3.11`, `3.12`, or `3.14` (verified on Python 3.14.3).
- **Package Manager**: `pip` (standard Python package installer).
- **Hardware**: Standard x86_64 or ARM64 processor (at least 4 GB RAM recommended). FastEmbed runs locally on CPU with zero GPU requirements.

### Optional Integrations
- **Live LLM Responses**: A free [Groq API Key](https://console.groq.com/) (defaults to deterministic offline synthesis if omitted).
- **Remote Observability**: A free [LangSmith API Key](https://smith.langchain.com/) (defaults to local non-blocking mode if omitted).

---

## 7. Installation & Setup

### Windows PowerShell (Recommended)

```powershell
# 1. Clone the repository
git clone https://github.com/harshinipolisetty/MemoraAI.git
cd MemoraAI

# 2. Create and activate a Python virtual environment
python -m venv .venv
.venv\Scripts\Activate.ps1

# 3. Upgrade pip and install dependencies
pip install --upgrade pip
pip install -e .
pip install -e ".[test]"

# 4. Create your local environment configuration
Copy-Item .env.example .env

# 5. Launch the application
streamlit run app/main.py
```

### Linux / macOS

```bash
# 1. Clone the repository
git clone https://github.com/harshinipolisetty/MemoraAI.git
cd MemoraAI

# 2. Create and activate a Python virtual environment
python3 -m venv .venv
source .venv/bin/activate

# 3. Upgrade pip and install dependencies
pip install --upgrade pip
pip install -e .
pip install -e ".[test]"

# 4. Create your local environment configuration
cp .env.example .env

# 5. Launch the application
streamlit run app/main.py
```

---

## 8. Environment Configuration Reference

All settings are validated via **Pydantic Settings** in [`src/asterline/config/settings.py`](src/asterline/config/settings.py) and loaded from `.env`.

| Environment Variable | Description | Type / Format | Default Value | Required? |
| :--- | :--- | :--- | :--- | :--- |
| `LLM_PROVIDER` | LLM provider identifier (`groq`, `openai`, `gemini`, `anthropic`, `local`) | String | `groq` | Optional |
| `LLM_MODEL` | Specific model identifier to request | String | `llama-3.3-70b-versatile` | Optional |
| `LLM_API_KEY` | Provider API key for live generation | Secret String | `None` (triggers offline fallback) | Optional |
| `LLM_TEMPERATURE` | Sampling temperature for model responses | Float ($0.0 - 2.0$) | `0.0` (deterministic) | Optional |
| `LLM_MAX_TOKENS` | Maximum tokens allowed per generation turn | Integer | `2048` | Optional |
| `EMBEDDING_PROVIDER` | Embedding engine (`local_fastembed`) | String | `local_fastembed` | Optional |
| `EMBEDDING_MODEL_NAME` | HuggingFace embedding model ID | String | `BAAI/bge-small-en-v1.5` | Optional |
| `CHROMA_PERSIST_DIRECTORY` | Directory where ChromaDB indexes are saved | Path | `./data/chroma` | Optional |
| `CORPUS_PATH` | Path to the immutable 75-chunk documentation corpus | Path | `./data/corpus.jsonl` | Optional |
| `SQLITE_DB_PATH` | File path for SQLite conversation and memory storage | Path | `./data/memory.db` | Optional |
| `LANGCHAIN_TRACING_V2` | Enable/disable LangSmith telemetry dispatch | Boolean (`true`/`false`) | `false` | Optional |
| `LANGCHAIN_ENDPOINT` | LangSmith API ingest URL | URL String | `https://api.smith.langchain.com` | Optional |
| `LANGCHAIN_API_KEY` | Personal LangSmith account API key | Secret String | `None` | Optional |
| `LANGCHAIN_PROJECT` | Target project bucket in LangSmith dashboard | String | `asterline-research-assistant` | Optional |
| `APP_HOST` | Host address to bind the Streamlit app | IP String | `127.0.0.1` | Optional |
| `APP_PORT` | Port number to bind the Streamlit app | Integer | `8501` | Optional |
| `LOG_LEVEL` | Application logging verbosity (`DEBUG`, `INFO`, `WARNING`) | String | `INFO` | Optional |
| `MAX_INVESTIGATION_RETRIES` | Max verification revision retries before forced output | Integer | `2` | Optional |

> [!NOTE]
> **Zero-Cost Local Mode**: If `LLM_API_KEY` is not provided in `.env`, MemoraAI operates in **deterministic offline synthesis mode**, extracting verified facts directly from retrieved chunks without making any network API calls.

---

## 9. Running the Application

### Launch Command
```bash
streamlit run app/main.py
```
Open **http://127.0.0.1:8501** in your browser.

### Application Walkthrough

1. **Active Demonstration Profiles** (Sidebar):
   - **Alice — DevOps Lead**: Focuses on pipelines, warehouse exports (BigQuery/Snowflake), and deployment rollouts.
   - **Bob — Backend Platform Engineer**: Focuses on AQL query optimization, ingestion rate limits, and SDK batching.
   - **Charlie — Technical Support & QA**: Focuses on Enterprise SLAs, incident postmortems, and Signals alert rules.
   - *Switching profiles immediately isolates all past conversations, queries, and stored memory.*

2. **Tab 1: Research Assistant**:
   - Click any of the **Quick-Starter Cards** (e.g., `📋 POST /v2/runs Payload Limits`) or type a custom question in the bottom composer.
   - Observe the **Developer Chip Bar** above the response: route classification (`● Documentation Lookup`), grounding verdict (`✓ Grounded`), model used (`llama-3.3-70b`), and cited sources (`📄 1 Source`).
   - Expand **Authoritative Sources (N)** to inspect the exact document title, chunk ID, and quoted text excerpt.

3. **Tab 2: Memory Inspector**:
   - Search stored preferences or filter by category (`preference`, `workspace_context`, `project_role`, `investigation_note`).
   - Click `+ Add Memory` to manually register durable context.
   - Click `✏️ Edit` to modify existing records, or `🗑️ Forget` to perform a permanent physical deletion from SQLite.

4. **Tab 3: Agent Observability & Traces**:
   - Inspect the **4-tile telemetry strip**: `EXECUTION ROUTE`, `VERIFIER VERDICT`, `RETRIEVED / CITED`, and `RETRY ATTEMPTS`.
   - Review the **Workflow Execution Timeline** detailing every step from input guardrails to SQLite audit commit.
   - Expand the **Sanitized State Payload** to view the redacted state dictionary.

---

## 10. Testing & Verification

The test suite covers unit logic, SQLite isolation, RAG indexing, LangGraph routing, guardrails, and verification retries.

### Running Tests

```bash
# Run the complete test suite (134 tests)
pytest -v

# Run fast tests excluding ChromaDB runtime initialization
pytest -k "not test_ui_runtime_initialization_regression" -v

# Run individual test modules
pytest tests/test_retrieval.py -v       # RAG, ChromaDB & BM25 tests (19 tests)
pytest tests/test_memory.py -v          # SQLite persistence, isolation & hard purge (22 tests)
pytest tests/test_workflow.py -v        # LangGraph routing & agent execution (14 tests)
pytest tests/test_verification.py -v    # Hallucination & citation verification (9 tests)
pytest tests/test_guardrails.py -v      # Input/output security & secret scrubbing (9 tests)
pytest tests/test_ui.py -v              # Streamlit helpers & profile switching (7 tests)
```

### Verified Test Status
- **Total Automated Tests**: **134 passing tests**
- **Failures / Errors**: **0**
- **Execution Speed**: Full suite completes in ~84 seconds (includes local FastEmbed embedding model download and cold vector indexing). Fast suite completes in ~3.2 seconds.

---

## 11. Evaluation Results & Benchmarks

MemoraAI was evaluated against the official 50-case benchmark harness across 4 specialized dimensions. Results are preserved in [`evaluation/results/evaluation_report.json`](evaluation/results/evaluation_report.json).

### Official Evaluation Benchmark Summary (50 Cases)

| Benchmark Dimension | Evaluated Cases | Metric | Observed Score | Target Baseline | Result Status |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Retrieval Engine (Dense)** | 16 queries | Hit Rate @ 5 | **100.0%** (1.000) | $\ge 90.0\%$ | ✅ Exceeds Target |
| | 16 queries | Recall @ 5 | **97.92%** (0.979) | $\ge 85.0\%$ | ✅ Exceeds Target |
| | 16 queries | Mean Reciprocal Rank (MRR) | **0.9062** | $\ge 0.700$ | ✅ Exceeds Target |
| **Retrieval Engine (BM25)** | 16 queries | Hit Rate @ 5 | **100.0%** (1.000) | $\ge 90.0\%$ | ✅ Exceeds Target |
| | 16 queries | Recall @ 5 | **100.0%** (1.000) | $\ge 85.0\%$ | ✅ Exceeds Target |
| | 16 queries | MRR | **0.8854** | $\ge 0.700$ | ✅ Exceeds Target |
| **Retrieval Engine (Hybrid RRF)**| 16 queries | Hit Rate @ 5 | **100.0%** (1.000) | $\ge 95.0\%$ | ✅ Perfect Score |
| | 16 queries | Recall @ 5 | **100.0%** (1.000) | $\ge 90.0\%$ | ✅ Perfect Score |
| | 16 queries | MRR | **0.8438** | $\ge 0.750$ | ✅ Exceeds Target |
| **Grounding & Citation** | 37 citations | Citation Validity Rate | **100.0%** (37/37 valid) | $100.0\%$ | ✅ Zero Hallucinations |
| | 16 queries | Grounded Response Rate | **93.75%** (15/16 grounded) | $\ge 90.0\%$ | ✅ Verified Ceiling* |
| | 4 queries | Correct Abstention Rate | **75.0%** (3/4 abstentions) | $\ge 75.0\%$ | ✅ Passes Target |
| | 4 queries | Temporal Supersession Accuracy| **100.0%** (4/4 resolved) | $100.0\%$ | ✅ Perfect Score |
| **Intent Classification** | 20 queries | Correct Routing Rate | **100.0%** (20/20 correct) | $\ge 95.0\%$ | ✅ Perfect Score |
| | 5 routes | Per-Route Accuracy (All 5) | **100.0%** across all routes| $\ge 90.0\%$ | ✅ Perfect Score |
| **Durable Memory & Privacy** | 10 scenarios | Memory Operation Success | **100.0%** (10/10 success) | $100.0\%$ | ✅ Perfect Score |
| | 10 scenarios | Cross-User Privacy Leaks | **0 Leaks Detected** | $0$ leaks | ✅ Strict Isolation |
| | 10 scenarios | Forget / Hard-Purge Rate | **100.0%** purged from raw DB| $100.0\%$ | ✅ Full Compliance |
| | 10 scenarios | Policy Rejection Accuracy | **100.0%** credentials blocked| $100.0\%$ | ✅ Zero Secret Leakage |

*\*Note on Grounded Response Rate (93.75%)*: As documented in [`docs/failure_analysis.md`](docs/failure_analysis.md), query `eval_16` asks for the "key rotation policy for webhook signing keys" expecting `security:0`. However, chunk `security:0` only documents *project ingestion keys*. MemoraAI correctly refuses to hallucinate that `security:0` applies to webhooks and emits an honest abstention. This 93.75% score represents the mathematical ceiling for an honest, non-hallucinating system.

### Latency Profiling (Warm Sub-System Timing)
From [`evaluation/results/profiling_baseline.json`](evaluation/results/profiling_baseline.json):
- **Input Guardrails**: ~3.38 ms
- **Intent Routing**: ~0.09 ms
- **FastEmbed Local Embedding**: ~12.41 ms
- **ChromaDB Vector Retrieval**: ~5.68 ms
- **BM25 Lexical Scoring**: ~0.48 ms
- **Reciprocal Rank Fusion**: ~0.05 ms
- **Total Retrieval Duration**: ~18.62 ms
- **Repeat / Cached Query Latency**: **68.9 ms** (Total turn time)
- **Cold Generation Latency**: ~1.7s – 3.0s (dominated by remote LLM generation)

---

## 12. Security, Privacy & Compliance Safeguards

1. **Strict User-Scoped Partitions**:
   All conversation threads and memory records in SQLite include a mandatory `user_id` column. Every database select, insert, update, and delete query is parameterized:
   ```sql
   SELECT * FROM user_memories WHERE user_id = ? AND is_active = 1
   ```
2. **Three-Tier Data Separation**:
   Company documentation is immutable and read-only. Private user memories are never written to the corpus, and user preferences are never used as factual citations.
3. **Physical Right to Be Forgotten**:
   Unlike standard applications that implement soft-deletes (`is_active = 0`), MemoraAI executes a true physical delete (`DELETE FROM user_memories`) upon forget commands. The audit log retains only anonymized metadata (`action="FORGET"`, timestamp), scrubbing all raw content.
4. **Secret Scrubbing & Credential Policies**:
   Memory candidate strings and output responses are continuously scanned against regex patterns for API keys (`ast_live_*`, `sk-*`, `ghp_*`), bearer tokens, and passwords. Suspected credentials are automatically rejected from memory persistence and masked in responses.
5. **Demonstration Profile Notice**:
   The user switcher (Alice, Bob, Charlie) is a demonstration convenience for evaluating multi-user isolation without setting up an external identity provider (IdP). In production, this would be wired to OAuth2/OIDC claims.

---

## 13. Troubleshooting & Common Failure Modes

### 1. `Port 8501 is not available`
- **Cause**: A prior Streamlit server instance is still bound to port 8501.
- **Resolution (PowerShell)**:
  ```powershell
  Get-NetTCPConnection -LocalPort 8501 | ForEach-Object { Stop-Process -Id $_.OwningProcess -Force }
  ```

### 2. Slow Startup on First Run (1-2 minutes)
- **Cause**: FastEmbed automatically downloads the local embedding model (`BAAI/bge-small-en-v1.5`, ~130 MB) and indexes the 75 corpus chunks on first initialization.
- **Resolution**: Normal behavior. Subsequent startups load the persistent index from `data/chroma` in under 700 ms.

### 3. `PermissionError: [WinError 32]` During Test Teardown on Windows
- **Cause**: Windows file locks on SQLite database files created in temporary directories.
- **Resolution**: All evaluation runners and stores include `ignore_cleanup_errors=True` and explicit `store.close()` calls to prevent unhandled lock crashes.

### 4. LangSmith Traces Not Appearing
- **Cause**: `LANGCHAIN_TRACING_V2` is set to `false`, or `LANGCHAIN_API_KEY` is missing.
- **Resolution**: Set `LANGCHAIN_TRACING_V2=true` in `.env` and provide a valid key from your [LangSmith Dashboard](https://smith.langchain.com/).

### 5. Deterministic Offline Fallback Triggered
- **Cause**: `LLM_API_KEY` is unset or invalid, or Groq API rate limits were encountered.
- **Resolution**: MemoraAI automatically engages deterministic evidence extraction. The response remains grounded with citations. Supply a valid Groq API key in `.env` to restore live Llama-3.3-70B synthesis.

---

## 14. Limitations & Future Roadmap

### Current Limitations
- **Demonstration Authentication**: User switching uses simulated profile states rather than enterprise SSO (SAML/OAuth2).
- **Corpus Scope**: Limited to the 75-chunk Asterline knowledge base (`data/corpus.jsonl`). Expanding the corpus requires running ingestion indexing.
- **External LLM Latency**: Live multi-document synthesis queries take ~1.8s – 3.0s due to remote Groq LLM inference, while cached repeats take <70ms.

### Future Roadmap
- [ ] Implement enterprise Single Sign-On (SAML 2.0 / Okta OIDC) for production authentication.
- [ ] Add streaming LLM response tokens directly to the Streamlit UI.
- [ ] Enable dynamic document ingestion via S3/Blob storage watchers.
- [ ] Implement cross-encoder reranking (e.g., `bge-reranker-large`) for higher MRR on larger corpora (>10,000 chunks).

---

## 15. AI Usage Disclosure

In compliance with transparent development standards, the use of generative AI tools during this project is fully documented in [**AI_DISCLOSURE.md**](AI_DISCLOSURE.md).

- **Primary AI Assistance**: Antigravity (Google DeepMind Advanced Agentic Coding Assistant).
- **Scope of Assistance**: LangGraph workflow construction, hybrid RRF retrieval engine, SQLite schema with audit logging, 50-case evaluation harness, and Streamlit dark UI redesign.
- **Human Verification**: Every algorithm, prompt, regex pattern, and evaluation result was independently verified through 134 automated unit/regression tests, verified corpus checksums, and live browser testing.

---

## License

This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.
