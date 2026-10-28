"""Live execution and LangSmith trace verification for:
1. "Who is Amaresh?"
"""

import sys
from pathlib import Path
import time

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from asterline.config.settings import get_settings
from asterline.graph.workflow import WorkflowBuilder
from asterline.memory.service import MemoryService
from asterline.memory.store import SQLiteStore
from asterline.observability.tracing import get_safe_trace_metadata, setup_tracing
from asterline.rag.ingestion import load_and_validate_corpus
from asterline.rag.retriever import HybridRetriever


def test_query(app, user_id, conversation_id, query_text):
    print(f"\n=======================================================")
    print(f"QUERY: '{query_text}' (User: {user_id})")
    print(f"=======================================================")

    # Run 1: Cold
    t0 = time.perf_counter()
    state_cold = {
        "user_id": user_id,
        "conversation_id": f"{conversation_id}_cold",
        "user_message": query_text,
    }
    res_cold = app.invoke(state_cold)
    t_cold = (time.perf_counter() - t0) * 1000.0

    print(f"\n[COLD RUN]")
    print(f"  Latency:    {t_cold:.1f} ms ({t_cold/1000.0:.2f} s)")
    print(f"  Route:      {res_cold.get('route')}")
    print(f"  Cached:     {res_cold.get('llm_metadata', {}).get('cached', False)}")
    print(f"  LLM Calls:  {res_cold.get('llm_metadata', {}).get('llm_calls', 1)}")
    print(f"  Citations:  {[c['chunk_id'] for c in res_cold.get('citations', [])]}")
    print(f"  Response:   {res_cold.get('final_response', '')[:160]}...")

    # Run 2: Warm
    t1 = time.perf_counter()
    state_warm = {
        "user_id": user_id,
        "conversation_id": f"{conversation_id}_warm",
        "user_message": query_text,
    }
    res_warm = app.invoke(state_warm)
    t_warm = (time.perf_counter() - t1) * 1000.0

    print(f"\n[WARM RUN]")
    print(f"  Latency:    {t_warm:.1f} ms ({t_warm/1000.0:.3f} s)")
    print(f"  Route:      {res_warm.get('route')}")
    print(f"  Cached:     {res_warm.get('llm_metadata', {}).get('cached', False)}")
    print(f"  LLM Calls:  {res_warm.get('llm_metadata', {}).get('llm_calls', 0)}")
    print(f"  Citations:  {[c['chunk_id'] for c in res_warm.get('citations', [])]}")
    print(f"  Response:   {res_warm.get('final_response', '')[:160]}...")


def main():
    tracing_active = setup_tracing()
    print(f"LangSmith Tracing Active: {tracing_active}")

    settings = get_settings()
    chunks, _ = load_and_validate_corpus(settings.get_resolved_corpus_path())
    db_path = Path("data/verify_live.db")
    if db_path.exists():
        db_path.unlink()
    store = SQLiteStore(db_path)
    service = MemoryService(store)
    retriever = HybridRetriever(corpus_chunks=chunks)
    builder = WorkflowBuilder(retriever=retriever, memory_service=service)
    app = builder.build()

    # Clear cache at start
    builder.doc_agent.cache.clear()

    # Query 1: Out-of-domain unanswerable question
    test_query(app, "user_demo", "conv_amaresh", "Who is Amaresh?")

    # Query 2: Factual payload limits question
    test_query(
        app,
        "user_demo",
        "conv_payload",
        "What is the maximum ingestion payload size for the /v2/runs endpoint?",
    )


if __name__ == "__main__":
    main()
