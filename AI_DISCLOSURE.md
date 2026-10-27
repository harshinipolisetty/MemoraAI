# Asterline Research Assistant — AI Assistance & System Disclosure

This document discloses the use of generative AI tools in the development of the Asterline Labs multi-agent research assistant ("Memora"), architectural decisions, verification methodology, and operational limitations.

---

## 1. Generative AI Tools & Assistance Scope

- **AI Assistance**: Antigravity (Google DeepMind Advanced Agentic Coding Assistant).
- **Assistance Scope**:
  - Implementation of LangGraph workflow orchestration, routing logic, and node definitions.
  - Development of hybrid retrieval engine combining local FastEmbed (`BAAI/bge-small-en-v1.5`) and BM25 with Reciprocal Rank Fusion.
  - Creation of SQLite persistent storage schema, transactional audit logging, and memory safety policies.
  - Development of the 50-case evaluation benchmark harness (`evaluation/evaluators.py`, `evaluation/run_evals.py`).
  - Creation of the Streamlit user interface (`app/main.py`, `app/components.py`) and modular dark-mode components.
  - Writing 116 comprehensive unit and integration tests across all components.

---

## 2. Human Verification & Engineering Oversight

Every component, algorithm, and dataset was independently verified against assignment requirements:
1. **Corpus Immutability**:
   - The authoritative documentation corpus (`data/corpus.jsonl`) was strictly preserved without modification.
   - Checksum verified: `F63FC05CF43204485C98BD8FB6D89860A32C16A7E906BC90975E7C3954FE9FB1`.
2. **Automated Test Validation**:
   - 116 passing pytest automated unit and regression tests.
   - Zero test failures, zero test errors.
3. **Offline Benchmark Execution**:
   - 50 curated evaluation cases executed across retrieval, grounding, citation validity, routing, and memory isolation.
   - Verified 100% citation validity, 100% correct abstention, and 0 cross-user isolation leaks.
4. **Clean Launch Smoke Test**:
   - Verified live Streamlit startup and HTTP 200 response from repository root.
5. **Security & Redaction**:
   - PII and credential scrubbing patterns verified for API keys (`ast_live_*`, `sk-*`, `ghp_*`), bearer tokens, and passwords.
   - Verified complete physical removal (`DELETE FROM user_memories`) on forget operations.

---

## 3. Operational Boundaries & Honest Limitations

1. **Demonstration Profile Selector**:
   - The profile switcher (Alice, Bob, Charlie) is designed for demonstration and evaluation convenience.
   - While database reads, mutations, and conversational contexts are strictly partitioned by `user_id`, this is a demonstration simulation, not production authentication (OAuth2/SAML/OIDC).
2. **Deterministic Offline Synthesis vs. Live Generation**:
   - The assistant operates 100% offline without requiring paid API keys by using deterministic evidence extraction.
   - Live LLM generation (via Groq, OpenAI, Gemini, or Anthropic) requires supplying `LLM_API_KEY` in `.env`.
3. **Observability Distinction**:
   - Tracing, metadata sanitization, and secret scrubbing are verified locally through unit tests (`tests/test_tracing.py`).
   - Remote dashboard transmission requires setting `LANGCHAIN_TRACING_V2=true` and supplying a personal `LANGCHAIN_API_KEY`.
