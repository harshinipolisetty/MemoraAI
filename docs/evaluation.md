# Asterline Research Assistant — Evaluation Framework & Results

This document describes the offline, reproducible evaluation framework for the Asterline Labs multi-agent research assistant. It defines the datasets, evaluation metrics (including formulas, numerators, and denominators), execution commands, and actual benchmark results.

---

## 1. Evaluation Methodology

The evaluation harness evaluates the assistant across four core dimensions:
1. **Retrieval Performance**: Evaluating lexical (BM25), dense (FastEmbed BAAI/bge-small-en-v1.5), and hybrid (Reciprocal Rank Fusion) retrieval against known ground-truth documentation chunks.
2. **Grounding & Citation Integrity**: Measuring citation validity, grounded claim proxies, abstention on absent evidence, and temporal supersession resolution across the controlling documentation corpus.
3. **Intent Routing**: Testing the system router across all 5 operational paths (`conversational`, `clarification`, `memory_ops`, `factual`, `investigation`).
4. **Persistent User Memory & Security**: Evaluating cross-session persistence, strict cross-user isolation (Alice, Bob, Charlie), permanent cryptographic/PII purge on forget operations, and immediate rejection of company secrets/credentials.

The evaluation suite runs **100% offline and deterministically by default**, requiring zero paid LLM API calls.

---

## 2. Evaluation Datasets

All datasets reside in `evaluation/datasets/` in JSONL format:

| Dataset | File Path | Sample Count | Scope & Categories Covered |
| :--- | :--- | :---: | :--- |
| **Research Dataset** | `evaluation/datasets/research.jsonl` | **20 cases** | Factual queries, pricing/seat limits, SLA commitments, incident investigations (`INC-2025-08`, `INC-2025-12`), temporal policy supersessions (cold storage retention 20d vs 60d, on-call stipend $175 vs $140), and out-of-domain abstention queries. |
| **Routing Dataset** | `evaluation/datasets/routing.jsonl` | **20 cases** | 4 cases per route: Greetings/identity (`conversational`), underspecified queries (`clarification`), explicit memory mutations (`memory_ops`), single-document lookups (`factual`), multi-document/temporal investigations (`investigation`). |
| **Memory Dataset** | `evaluation/datasets/memory.jsonl` | **10 scenarios** | User preference persistence, project role persistence, multi-user isolation (Alice vs Bob vs Charlie), permanent forget purge, secret/API key rejection (`ast_live_*`, `sk-*`, `ghp_*`, Bearer tokens, passwords), and product-wide assertion rejection. |

---

## 3. Metric Definitions & Formulas

### A. Retrieval Metrics (Top-K = 5)

1. **Hit Rate@k**:
   - **Definition**: The proportion of test queries where at least one expected ground-truth chunk is retrieved in the top-$k$ results.
   - **Formula**:
     $$\text{Hit Rate@k} = \frac{\sum_{i=1}^{N} \mathbb{I}\left(|\text{Retrieved}_{i, k} \cap \text{Expected}_{i}| > 0\right)}{N}$$
   - **Numerator**: Queries where at least one retrieved chunk is in $\text{Expected}_i$.
   - **Denominator**: Total factual test queries ($N = 16$).

2. **Recall@k**:
   - **Definition**: The average fraction of expected ground-truth chunks retrieved in the top-$k$ results.
   - **Formula**:
     $$\text{Recall@k} = \frac{1}{N} \sum_{i=1}^{N} \frac{|\text{Retrieved}_{i, k} \cap \text{Expected}_{i}|}{|\text{Expected}_{i}|}$$
   - **Numerator**: Total count of expected chunks successfully retrieved across all queries.
   - **Denominator**: Total count of expected chunks across all queries.

3. **Mean Reciprocal Rank (MRR)**:
   - **Definition**: The average of the reciprocal ranks of the first relevant chunk retrieved.
   - **Formula**:
     $$\text{MRR} = \frac{1}{N} \sum_{i=1}^{N} \frac{1}{\text{rank}_i} \quad (\text{where } \text{rank}_i = \min \{r : \text{chunk}_r \in \text{Expected}_i\} \text{ or } 0 \text{ if not found})$$

---

### B. Grounding & Citation Metrics

1. **Citation Validity Rate**:
   - **Definition**: Proportion of cited chunks in assistant responses that exist in the authoritative corpus with exact matching titles.
   - **Formula**:
     $$\text{Citation Validity} = \frac{\text{Valid Citations Cited}}{\text{Total Citations Cited}}$$
   - **Numerator**: Citations where `chunk_id` exists in `data/corpus.jsonl` AND `citation.title == corpus[chunk_id].title`.
   - **Denominator**: Total bracketed citations generated across all responses ($62$).

2. **Grounded Response Rate (Proxy)**:
   - **Definition**: Proportion of standard research queries producing an approved response passing Verification Agent checks (`verdict == "pass"`).
   - **Formula**:
     $$\text{Grounded Response Rate} = \frac{\text{Standard Queries Passing Verification}}{\text{Total Standard Research Queries}}$$
   - **Numerator**: Standard queries receiving `verdict == "pass"`.
   - **Denominator**: Standard factual queries ($16$).

3. **Correct Abstention Rate**:
   - **Definition**: Proportion of ungrounded or out-of-domain queries where the assistant explicitly acknowledges lack of documentation and emits zero citations.
   - **Formula**:
     $$\text{Correct Abstention Rate} = \frac{\text{Queries with Valid Abstention Message and } 0 \text{ Citations}}{\text{Total Abstention Test Queries}}$$
   - **Numerator**: Out-of-domain queries acknowledging lack of evidence without fabricating citations.
   - **Denominator**: Total abstention cases ($4$).

4. **Temporal Supersession Accuracy**:
   - **Definition**: Proportion of conflicting/superseded policy queries where the assistant correctly cites the newer controlling chunk rather than the deprecated historical chunk.
   - **Formula**:
     $$\text{Temporal Accuracy} = \frac{\text{Supersession Queries Citing Controlling Chunk}}{\text{Total Supersession Test Queries}}$$

---

### C. Routing & Agent Collaboration Metrics

1. **Overall Routing Accuracy**:
   - **Definition**: Proportion of user turns assigned to the expected system route.
   - **Formula**:
     $$\text{Routing Accuracy} = \frac{\text{Queries Routed to Expected Route}}{\text{Total Routing Queries}}$$
   - **Numerator**: Correct routing decisions ($20$).
   - **Denominator**: Total queries evaluated ($20$).

2. **Per-Route Accuracy**:
   - Evaluated independently for each of the 5 routes (`conversational`, `clarification`, `memory_ops`, `factual`, `investigation`).

---

### D. Persistent Memory & Security Metrics

1. **Memory Operation Success Rate**:
   - **Definition**: Proportion of memory operations (creation, isolation checks, deletion, policy enforcement) executing as expected.
   - **Formula**:
     $$\text{Operation Success Rate} = \frac{\text{Successful Memory Scenarios}}{\text{Total Evaluated Scenarios}}$$
   - **Numerator**: Scenarios where all assertions pass ($10$).
   - **Denominator**: Total memory scenarios ($10$).

2. **Cross-User Isolation Violations**:
   - **Definition**: Any instance where User A's stored memory is visible, queryable, or mutable by User B or User C.
   - **Target**: **$0$** (Absolute Zero Tolerance).

3. **Forget Permanent Purge Rate**:
   - **Definition**: Proportion of forget operations resulting in complete database deletion (`DELETE FROM user_memories`) rather than soft-delete retention.
   - **Formula**:
     $$\text{Purge Rate} = \frac{\text{Deleted Memories Verified Absent in SQLite}}{\text{Total Forget Operations}}$$

4. **Policy Rejection Accuracy**:
   - **Definition**: Proportion of sensitive inputs (API keys, passwords, bearer tokens, product-wide company assertions) rejected by the memory safety policy.
   - **Target**: **$100\%$**.

---

## 4. Benchmark Execution Command

To execute the offline evaluation benchmark suite and export the JSON report:

```bash
python -m evaluation.run_evals
```

*Note: Execution takes approximately 5–7 seconds on standard hardware, indexing all 75 chunks into ChromaDB and evaluating 50 distinct test cases across retrieval, grounding, routing, and memory.*

---

## 5. Actual Evaluation Results

*Benchmark executed on 2026-10-27 against corpus SHA256 `F63FC05CF43204485C98BD8FB6D89860A32C16A7E906BC90975E7C3954FE9FB1`:*

### Summary Table

```
======================================================================
ASTERLINE RESEARCH ASSISTANT — EVALUATION SUMMARY REPORT
======================================================================
Corpus Chunks: 75 (SHA256: F63FC05CF4320448...)

[1] RETRIEVAL BENCHMARK (Top-K = 5)
--------------------------------------------------
Mode         | Hit Rate@5   | Recall@5     | MRR       
--------------------------------------------------
Dense        | 1.0000       | 0.9792       | 0.9062    
Keyword      | 1.0000       | 1.0000       | 0.8854    
Hybrid       | 1.0000       | 1.0000       | 0.8438    

[2] GROUNDING & CITATION BENCHMARK
--------------------------------------------------
Total Research Cases:            20
Citation Validity Rate:          100.00% (62/62)
Grounded Response Rate:          100.00%
Correct Abstention Rate:         100.00%
Temporal Supersession Accuracy:  100.00%

[3] INTENT ROUTING BENCHMARK
--------------------------------------------------
Total Routing Queries:           20
Overall Correct Routing Rate:    100.00%
  - Route 'conversational': 100.0%
  - Route 'clarification': 100.0%
  - Route 'memory_ops': 100.0%
  - Route 'factual': 100.0%
  - Route 'investigation': 100.0%

[4] PERSISTENT MEMORY & ISOLATION BENCHMARK
--------------------------------------------------
Total Scenarios Evaluated:       10
Operation Success Rate:          100.00%
Cross-User Isolation Leaks:      0 (Expected: 0)
Forget Permanent Purge Rate:     100.00%
Policy Rejection Accuracy:       100.00%
======================================================================
```

Results are saved to: `evaluation/results/evaluation_report.json`.
