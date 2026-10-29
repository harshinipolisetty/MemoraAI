"""Baseline performance profiling script for MemoraAI UI and agent operations.

Measures the 8 critical operations required by Phase One:
1. Initial application startup
2. Switching between two demonstration users
3. Loading a conversation history with multiple messages
4. Opening the Memory Inspector
5. Loading a conversation from history
6. Switching between tabs
7. Submitting a factual question
8. Submitting an investigation question
"""

import os
from pathlib import Path
import sqlite3
import sys
import time
from typing import Any, Dict, List

# Ensure project root and src are in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from asterline.config.settings import get_settings
from asterline.memory.service import MemoryService
from asterline.memory.store import SQLiteStore
from asterline.rag.ingestion import load_and_validate_corpus
from asterline.rag.retriever import HybridRetriever
from asterline.graph.workflow import WorkflowBuilder
from asterline.observability.tracing import setup_tracing


class QueryCounterConnection:
    """Wrapper around sqlite3.Connection to count queries and track duration."""
    def __init__(self, conn: sqlite3.Connection):
        self._conn = conn
        self.query_count = 0
        self.total_query_time = 0.0

    def execute(self, sql, *args):
        t0 = time.perf_counter()
        self.query_count += 1
        res = self._conn.execute(sql, *args)
        self.total_query_time += (time.perf_counter() - t0)
        return res

    def executescript(self, sql):
        t0 = time.perf_counter()
        self.query_count += 1
        res = self._conn.executescript(sql)
        self.total_query_time += (time.perf_counter() - t0)
        return res

    def __getattr__(self, item):
        return getattr(self._conn, item)


def run_baseline_profiling():
    settings = get_settings()
    results = {}

    print("=" * 70)
    print("MEMORAAI BASELINE PROFILING REPORT")
    print("=" * 70)

    # -------------------------------------------------------------------------
    # 1. Initial Application Startup
    # -------------------------------------------------------------------------
    print("\n[1] Profiling Initial Application Startup (Cold)...")
    t0 = time.perf_counter()
    corpus_path = settings.resolve_path(settings.corpus_path)
    t_corpus_0 = time.perf_counter()
    chunks, report = load_and_validate_corpus(corpus_path)
    t_corpus = time.perf_counter() - t_corpus_0

    t_chroma_0 = time.perf_counter()
    chroma_dir = settings.resolve_path(settings.chroma_persist_directory)
    retriever = HybridRetriever(
        corpus_chunks=chunks,
        chroma_persist_dir=chroma_dir,
        embedding_model_name=settings.embedding_model_name,
    )
    t_chroma = time.perf_counter() - t_chroma_0

    t_db_0 = time.perf_counter()
    sqlite_path = settings.resolve_path(settings.sqlite_db_path)
    store = SQLiteStore(sqlite_path)
    memory_service = MemoryService(store)
    t_db = time.perf_counter() - t_db_0

    t_trace_0 = time.perf_counter()
    setup_tracing()
    t_trace = time.perf_counter() - t_trace_0

    t_wf_0 = time.perf_counter()
    builder = WorkflowBuilder(
        retriever=retriever,
        memory_service=memory_service,
    )
    workflow_app = builder.build()
    t_wf = time.perf_counter() - t_wf_0

    total_startup = time.perf_counter() - t0
    results["startup"] = {
        "total_ms": total_startup * 1000,
        "corpus_validation_ms": t_corpus * 1000,
        "hybrid_retriever_ms": t_chroma * 1000,
        "sqlite_init_ms": t_db * 1000,
        "tracing_ms": t_trace * 1000,
        "workflow_build_ms": t_wf * 1000,
    }
    print(f"  Startup Total: {total_startup*1000:.2f} ms")
    print(f"    - Corpus Validation: {t_corpus*1000:.2f} ms")
    print(f"    - Hybrid Retriever (Chroma+BM25+FastEmbed): {t_chroma*1000:.2f} ms")
    print(f"    - SQLite Init: {t_db*1000:.2f} ms")
    print(f"    - Workflow Compilation: {t_wf*1000:.2f} ms")

    # -------------------------------------------------------------------------
    # 2. Switching Between Two Demonstration Users
    # -------------------------------------------------------------------------
    print("\n[2] Profiling User Profile Switching (Alice <-> Bob)...")
    # Simulate current implementation in app/main.py:
    # 1. Fetch conversations for user_alice
    # 2. User selects user_bob
    # 3. Call switch_user_profile(user_bob)
    # 4. Fetch conversations for user_bob, re-scope active conversation
    # 5. Fetch messages for Bob's active conversation
    # 6. (Because st.tabs runs all tabs) also fetch Bob's memories
    t0 = time.perf_counter()
    convs_alice = memory_service.list_user_conversations("user_alice")
    
    # Switch to Bob
    convs_bob = memory_service.list_user_conversations("user_bob")
    bob_active_conv = convs_bob[0]["conversation_id"] if convs_bob else "new_conv"
    bob_messages = memory_service.get_conversation_messages(bob_active_conv, "user_bob") if convs_bob else []
    bob_memories = memory_service.list_user_memories("user_bob")
    
    # Model dumps / Pydantic serialization if citations exist
    for msg in bob_messages:
        if msg.citations:
            _ = [c.model_dump() for c in msg.citations]
            _ = [c.model_dump() for c in chunks]

    user_switch_time = time.perf_counter() - t0
    results["user_switch"] = {"total_ms": user_switch_time * 1000}
    print(f"  Profile Switch Simulation: {user_switch_time*1000:.2f} ms")

    # -------------------------------------------------------------------------
    # 3. Loading Conversation History with Multiple Messages
    # -------------------------------------------------------------------------
    print("\n[3] Profiling Conversation History Loading (with multiple messages)...")
    # Ensure Alice has at least one conversation with messages
    alice_convs = memory_service.list_user_conversations("user_alice")
    if not alice_convs:
        c = memory_service.get_or_create_conversation("user_alice", title="Benchmark Pipeline")
        cid = c["conversation_id"]
    else:
        cid = alice_convs[0]["conversation_id"]

    t0 = time.perf_counter()
    msgs = memory_service.get_conversation_messages(cid, "user_alice")
    load_history_time = time.perf_counter() - t0
    results["load_history"] = {"total_ms": load_history_time * 1000, "message_count": len(msgs)}
    print(f"  Load History ({len(msgs)} messages): {load_history_time*1000:.2f} ms")

    # -------------------------------------------------------------------------
    # 4. Opening the Memory Inspector
    # -------------------------------------------------------------------------
    print("\n[4] Profiling Memory Inspector Data Loading...")
    t0 = time.perf_counter()
    active_mems = memory_service.list_user_memories("user_alice")
    # Simulate filter and search
    filtered = [m for m in active_mems if "team" in m.content.lower()]
    mem_inspector_time = time.perf_counter() - t0
    results["memory_inspector"] = {"total_ms": mem_inspector_time * 1000, "count": len(active_mems)}
    print(f"  Memory Inspector Loading ({len(active_mems)} memories): {mem_inspector_time*1000:.2f} ms")

    # -------------------------------------------------------------------------
    # 5. Loading a Conversation from History (User clicks another conversation)
    # -------------------------------------------------------------------------
    print("\n[5] Profiling Opening Existing Conversation...")
    t0 = time.perf_counter()
    conv_list = memory_service.list_user_conversations("user_alice")
    chosen = conv_list[0]["conversation_id"] if conv_list else cid
    msgs = memory_service.get_conversation_messages(chosen, "user_alice")
    open_conv_time = time.perf_counter() - t0
    results["open_conversation"] = {"total_ms": open_conv_time * 1000}
    print(f"  Opening Conversation: {open_conv_time*1000:.2f} ms")

    # -------------------------------------------------------------------------
    # 6. Switching Between Tabs
    # -------------------------------------------------------------------------
    print("\n[6] Profiling Tab Switching Rerun Overhead...")
    # In Streamlit, switching tabs reruns the script if state triggers or executes all tabs
    t0 = time.perf_counter()
    # Tab 1:
    _ = memory_service.get_conversation_messages(chosen, "user_alice")
    # Tab 2:
    _ = memory_service.list_user_memories("user_alice")
    # Tab 3:
    # last_trace format
    tab_switch_time = time.perf_counter() - t0
    results["tab_switch"] = {"total_ms": tab_switch_time * 1000}
    print(f"  Tab Switch Data Fetch: {tab_switch_time*1000:.2f} ms")

    # -------------------------------------------------------------------------
    # 7. Submitting a Factual Question
    # -------------------------------------------------------------------------
    print("\n[7] Profiling Factual Question Query...")
    factual_query = "What are the rate limits and payload sizes for POST /v2/runs?"
    state_factual = {
        "user_message": factual_query,
        "user_id": "user_alice",
        "conversation_id": chosen,
    }
    t0 = time.perf_counter()
    res_factual = workflow_app.invoke(state_factual)
    factual_time = time.perf_counter() - t0
    results["factual_query"] = {
        "total_ms": factual_time * 1000,
        "route": res_factual.get("route"),
        "is_fallback": res_factual.get("is_fallback"),
        "citations": len(res_factual.get("citations", [])),
    }
    print(f"  Factual Query ({res_factual.get('route')}): {factual_time*1000:.2f} ms (fallback: {res_factual.get('is_fallback')})")

    # -------------------------------------------------------------------------
    # 8. Submitting an Investigation Question
    # -------------------------------------------------------------------------
    print("\n[8] Profiling Investigation Question Query...")
    invest_query = "Explain the root cause of INC-2025-12 warehouse export deduplication failure and how it was resolved in patch 2.1.2."
    state_invest = {
        "user_message": invest_query,
        "user_id": "user_alice",
        "conversation_id": chosen,
    }
    t0 = time.perf_counter()
    res_invest = workflow_app.invoke(state_invest)
    invest_time = time.perf_counter() - t0
    results["investigation_query"] = {
        "total_ms": invest_time * 1000,
        "route": res_invest.get("route"),
        "is_fallback": res_invest.get("is_fallback"),
        "citations": len(res_invest.get("citations", [])),
    }
    print(f"  Investigation Query ({res_invest.get('route')}): {invest_time*1000:.2f} ms (fallback: {res_invest.get('is_fallback')})")

    print("\n" + "=" * 70)
    print("SUMMARY BASELINE METRICS (RAW)")
    print("=" * 70)
    for k, v in results.items():
        print(f"  {k}: {v}")

    return results


if __name__ == "__main__":
    run_baseline_profiling()

