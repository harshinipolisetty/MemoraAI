"""Asterline Research Assistant — Streamlit Multi-Agent Research Interface.

Features:
- Enterprise dark-mode design with accessible typography and comfortable spacing.
- Multi-user demo profile selector (Alice, Bob, Charlie) with strict isolation.
- Persistent multi-turn conversations stored in SQLite.
- Grounded documentation responses with expandable authoritative citations.
- Complete Memory Inspector: add, edit, and forget operations with policy checks.
- Execution observability trace panel: routing, node graph, verifier verdicts, retries, and LangSmith status.
"""

from pathlib import Path
import sys
from typing import Any, Dict, List, Optional
import uuid

# Ensure 'src' is in python path
SRC_DIR = Path(__file__).resolve().parent.parent / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

import streamlit as st

from asterline.config.settings import Settings, get_settings
from asterline.graph.workflow import WorkflowBuilder
from asterline.memory.service import MemoryService
from asterline.memory.store import SQLiteStore
from asterline.observability.tracing import check_langsmith_status
from asterline.rag.ingestion import load_and_validate_corpus
from asterline.rag.retriever import HybridRetriever

from app.components import (
    DEMO_PROFILES,
    format_timestamp,
    inject_custom_css,
    render_citations_expander,
    render_fallback_pill,
    render_observability_card,
    render_route_pill,
    render_user_profile_header,
)


# =============================================================================
# Cached Application Runtime (Single Initialization across Streamlit Reruns)
# =============================================================================

@st.cache_resource(show_spinner="Initializing Asterline Hybrid Index & Multi-Agent Runtime...")
def get_runtime() -> Dict[str, Any]:
    """Initialize and cache the knowledge base, vector index, SQLite store, and LangGraph workflow.

    Guarantees that index building and database connections are not recreated on every Streamlit rerun.
    """
    settings = get_settings()

    corpus_path = settings.resolve_path(settings.corpus_path)
    if not corpus_path.exists():
        st.error(f"Critical error: Documentation corpus not found at {corpus_path}")
        st.stop()

    # 1. Ingest & validate 75-chunk corpus
    chunks, report = load_and_validate_corpus(corpus_path)

    # 2. Build local hybrid retriever
    chroma_dir = settings.resolve_path(settings.chroma_persist_directory)
    retriever = HybridRetriever(
        corpus_chunks=chunks,
        chroma_persist_directory=chroma_dir,
        embedding_model_name=settings.embedding_model,
    )

    # 3. Persistent SQLite Memory Store & Application Service
    sqlite_path = settings.resolve_path(settings.sqlite_db_path)
    store = SQLiteStore(sqlite_path)
    memory_service = MemoryService(store)

    # 4. Compile LangGraph Multi-Agent Workflow
    builder = WorkflowBuilder(
        retriever=retriever,
        memory_service=memory_service,
    )
    workflow_app = builder.build()

    return {
        "settings": settings,
        "chunks": chunks,
        "corpus_report": report,
        "retriever": retriever,
        "store": store,
        "memory_service": memory_service,
        "workflow": workflow_app,
        "chunk_map": {c.chunk_id: c for c in chunks},
    }


# =============================================================================
# Streamlit Application Setup & State Management
# =============================================================================

def init_session_state(memory_service: MemoryService) -> None:
    """Initialize persistent session state variables."""
    if "user_id" not in st.session_state:
        st.session_state["user_id"] = "user_alice"

    user_id = st.session_state["user_id"]

    # Initialize active conversation for the user
    if "active_conv_id" not in st.session_state or not st.session_state["active_conv_id"]:
        user_convs = memory_service.list_user_conversations(user_id)
        if user_convs:
            st.session_state["active_conv_id"] = user_convs[0]["conversation_id"]
        else:
            new_id = f"conv_{uuid.uuid4().hex[:12]}"
            memory_service.get_or_create_conversation(user_id, new_id, "New Investigation")
            st.session_state["active_conv_id"] = new_id

    if "last_trace_info" not in st.session_state:
        st.session_state["last_trace_info"] = None

    if "editing_memory_id" not in st.session_state:
        st.session_state["editing_memory_id"] = None

    if "confirm_delete_id" not in st.session_state:
        st.session_state["confirm_delete_id"] = None


def switch_user_profile(new_user_id: str, memory_service: MemoryService) -> None:
    """Safely switch the active demonstration user identity and re-scope all conversation state."""
    if st.session_state.get("user_id") != new_user_id:
        st.session_state["user_id"] = new_user_id
        # Re-scope conversation to selected user's latest conversation or create new
        user_convs = memory_service.list_user_conversations(new_user_id)
        if user_convs:
            st.session_state["active_conv_id"] = user_convs[0]["conversation_id"]
        else:
            new_id = f"conv_{uuid.uuid4().hex[:12]}"
            memory_service.get_or_create_conversation(new_user_id, new_id, "New Investigation")
            st.session_state["active_conv_id"] = new_id

        # Clear transient UI state
        st.session_state["last_trace_info"] = None
        st.session_state["editing_memory_id"] = None
        st.session_state["confirm_delete_id"] = None
        st.rerun()


# =============================================================================
# Main Application Layout
# =============================================================================

def main() -> None:
    st.set_page_config(
        page_title="Asterline Research Assistant",
        page_icon="🔬",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    inject_custom_css()

    # Load cached runtime components
    try:
        runtime = get_runtime()
    except Exception as e:
        st.error(f"Failed to initialize application runtime: {e}")
        st.stop()

    memory_service: MemoryService = runtime["memory_service"]
    workflow = runtime["workflow"]
    init_session_state(memory_service)

    active_user_id = st.session_state["user_id"]
    active_conv_id = st.session_state["active_conv_id"]

    # =========================================================================
    # Sidebar: Demonstration Profile Selector & Thread Management
    # =========================================================================
    with st.sidebar:
        st.markdown(
            """
            <div class="app-title-container">
                <div class="app-title">🔬 Asterline Labs</div>
                <div class="app-subtitle">Multi-Agent Research Assistant with Persistent Memory</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.subheader("👤 User Profile")
        selected_user = st.selectbox(
            label="Active User Profile (Demonstration Identity)",
            options=list(DEMO_PROFILES.keys()),
            format_func=lambda uid: DEMO_PROFILES[uid]["name"],
            index=list(DEMO_PROFILES.keys()).index(active_user_id),
            help="Strictly partitions all SQLite conversation threads, queries, and persistent memories.",
        )
        if selected_user != active_user_id:
            switch_user_profile(selected_user, memory_service)

        profile = DEMO_PROFILES[selected_user]
        st.caption(
            f"**Role:** {profile['role']}\n\n"
            f"**Workspace Focus:** {profile['focus']}\n\n"
            "*Notice: Demonstration identity mechanism — strictly scopes all database reads, mutations, and agent context to this user ID.*"
        )

        st.divider()

        # Conversation Management
        st.subheader("💬 Conversations")
        if st.button("➕ New Investigation", use_container_width=True, type="primary"):
            new_conv_id = f"conv_{uuid.uuid4().hex[:12]}"
            memory_service.get_or_create_conversation(active_user_id, new_conv_id, "New Investigation")
            st.session_state["active_conv_id"] = new_conv_id
            st.session_state["last_trace_info"] = None
            st.rerun()

        # List existing conversation threads for this user
        user_convs = memory_service.list_user_conversations(active_user_id)
        if user_convs:
            conv_options = [c["conversation_id"] for c in user_convs]
            conv_titles = {
                c["conversation_id"]: f"{c['title']} ({format_timestamp(str(c['updated_at']))})"
                for c in user_convs
            }
            current_index = conv_options.index(active_conv_id) if active_conv_id in conv_options else 0
            chosen_conv = st.selectbox(
                label="Select Conversation Thread",
                options=conv_options,
                format_func=lambda cid: conv_titles.get(cid, cid),
                index=current_index,
                label_visibility="collapsed",
            )
            if chosen_conv != active_conv_id:
                st.session_state["active_conv_id"] = chosen_conv
                st.session_state["last_trace_info"] = None
                st.rerun()
        else:
            st.caption("No saved conversations yet for this profile.")

        st.divider()

        # System & LangSmith Observability Diagnostics
        with st.expander("⚙️ System & Observability Status", expanded=False):
            st.write(f"**Corpus Integrity:** 75 Chunks Verified")
            st.code(runtime["corpus_report"].corpus_sha256[:20] + "...", language="text")
            st.write(f"**Retrieval Index:** Hybrid (ChromaDB + BM25)")

            # LangSmith Check
            ls_status = check_langsmith_status()
            if ls_status["tracing_enabled"]:
                st.success(f"LangSmith Tracing: Active ({ls_status['project']})")
            else:
                st.info("LangSmith Tracing: Offline / Disabled (Safe local execution)")
            st.caption("All traces are automatically scrubbed of credentials, API keys, and sensitive tokens.")

    # =========================================================================
    # Main View: Tabs for Chat, Memory Inspector, and Observability Details
    # =========================================================================
    render_user_profile_header(active_user_id)

    tab_chat, tab_memory, tab_trace = st.tabs([
        "💬 Research Assistant",
        "🧠 Memory Inspector",
        "🔍 Agent Observability & Traces",
    ])

    # -------------------------------------------------------------------------
    # Tab 1: Chat Interface
    # -------------------------------------------------------------------------
    with tab_chat:
        # Load chronological messages from authoritative SQLite store
        messages = memory_service.get_conversation_messages(active_conv_id, active_user_id)

        if not messages:
            st.markdown(
                """
                <div style="background-color: #161b22; border: 1px dashed #30363d; border-radius: 8px; padding: 1.5rem; text-align: center; margin-bottom: 1rem;">
                    <h4 style="color: #f0f6fc; margin-bottom: 0.5rem;">Start an Authoritative Research Investigation</h4>
                    <p style="color: #8b949e; font-size: 0.9rem; margin-bottom: 1rem;">
                        Ask factual questions grounded in Asterline technical documentation, investigate multi-document incidents, or set durable workspace context.
                    </p>
                    <div style="display: flex; gap: 0.5rem; justify-content: center; flex-wrap: wrap;">
                        <span class="pill-badge" style="background-color: #21262d; color: #c9d1d9;">💡 "What are the payload limits for POST /v2/runs?"</span>
                        <span class="pill-badge" style="background-color: #21262d; color: #c9d1d9;">💡 "Explain what caused incident INC-2025-12."</span>
                        <span class="pill-badge" style="background-color: #21262d; color: #c9d1d9;">💡 "Remember that our team uses Snowflake."</span>
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        else:
            for msg in messages:
                if msg.role == "user":
                    with st.chat_message("user", avatar=DEMO_PROFILES[active_user_id]["avatar"]):
                        st.markdown(msg.content)
                else:
                    with st.chat_message("assistant", avatar="🔬"):
                        # Badges for route and fallback status
                        pills_html = []
                        if msg.route_taken:
                            pills_html.append(render_route_pill(msg.route_taken))
                        if pills_html:
                            st.markdown(" ".join(pills_html), unsafe_allow_html=True)

                        st.markdown(msg.content)

                        # Render expandable citations if present
                        if msg.citations:
                            render_citations_expander(
                                citations=[c.model_dump() for c in msg.citations],
                                retrieved_chunks=[c.model_dump() for c in runtime["chunks"]],
                            )

        # Chat Input
        user_prompt = st.chat_input("Ask a technical documentation question, investigate an incident, or manage memory...")
        if user_prompt:
            if not user_prompt.strip():
                st.warning("Please enter a valid message.")
            else:
                # 1. Render immediate user message
                with st.chat_message("user", avatar=DEMO_PROFILES[active_user_id]["avatar"]):
                    st.markdown(user_prompt)

                # 2. Execute multi-agent LangGraph workflow
                with st.chat_message("assistant", avatar="🔬"):
                    with st.spinner("Analyzing query, routing agents, and verifying documentation evidence..."):
                        try:
                            state_input = {
                                "user_message": user_prompt,
                                "user_id": active_user_id,
                                "conversation_id": active_conv_id,
                            }
                            result = workflow.invoke(state_input)

                            final_resp = result.get("final_response", "")
                            citations = result.get("citations", [])
                            route = result.get("route", "factual")
                            is_fallback = result.get("is_fallback", False)
                            llm_metadata = result.get("llm_metadata", {})

                            # Save trace info in session state for tab 3
                            st.session_state["last_trace_info"] = result

                            # Auto-update conversation title if it is a fresh conversation
                            if len(messages) == 0:
                                title_summary = user_prompt.strip()[:35] + ("..." if len(user_prompt) > 35 else "")
                                memory_service.update_conversation_title(active_conv_id, active_user_id, title_summary)

                            # Rerun to cleanly re-render persistent history from SQLite
                            st.rerun()

                        except Exception as e:
                            st.error(f"Execution Error: {str(e)}")

    # -------------------------------------------------------------------------
    # Tab 2: Memory Inspector
    # -------------------------------------------------------------------------
    with tab_memory:
        st.markdown(
            f"""
            <div style="margin-bottom: 1rem;">
                <h3 style="color: #f0f6fc; margin-bottom: 0.25rem;">Authoritative User Memory for {profile['name']}</h3>
                <p style="color: #8b949e; font-size: 0.85rem; margin: 0;">
                    Personal preferences and workspace contexts are strictly isolated to this profile.
                    Company documentation is never treated as personal memory, and company secrets are rejected by policy.
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )

        # Form to add a new memory
        with st.expander("➕ Add New Long-Term Memory", expanded=False):
            with st.form("add_memory_form", clear_on_submit=True):
                new_content = st.text_area(
                    "Memory Content",
                    placeholder="e.g., We use BigQuery for cold archive storage exports.",
                    help="Describe a durable workspace preference or configuration.",
                )
                new_cat = st.selectbox(
                    "Category",
                    options=["preference", "workspace_context", "project_role", "investigation_note", "general"],
                    index=0,
                )
                submitted = st.form_submit_button("Save Memory to Profile", type="primary")
                if submitted:
                    if not new_content.strip():
                        st.warning("Memory content cannot be empty.")
                    else:
                        item, decision = memory_service.create_memory(
                            user_id=active_user_id,
                            content=new_content,
                            category=new_cat,
                            reason="Added via Streamlit Memory Inspector",
                        )
                        if decision.allowed and item:
                            st.success("Memory persisted successfully to SQLite store.")
                            st.rerun()
                        else:
                            st.error(f"Policy Rejection: {decision.message} (Reason Code: {decision.reason_code})")

        # List active memories
        active_memories = memory_service.list_user_memories(active_user_id)
        if not active_memories:
            st.info(f"No active memories currently stored for {profile['name']}. You can add one above or tell the assistant in chat (e.g., 'Remember that I use Snowflake').")
        else:
            st.write(f"**Active Stored Memories ({len(active_memories)}):**")
            for mem in active_memories:
                mid = mem.memory_id
                with st.container():
                    st.markdown(
                        f"""
                        <div class="memory-card">
                            <div style="display: flex; justify-content: space-between; align-items: center;">
                                <span class="memory-category">🏷️ {mem.category.upper()}</span>
                                <span class="memory-meta">Created: {format_timestamp(mem.created_at)}</span>
                            </div>
                            <div class="memory-content">{mem.content}</div>
                            <div class="memory-meta">Memory ID: <code>{mem.memory_id}</code> • Confidence: {mem.confidence}</div>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )

                    col_edit, col_del, col_space = st.columns([1, 1, 4])

                    # Edit toggle
                    with col_edit:
                        if st.button("✏️ Edit", key=f"btn_edit_{mid}"):
                            st.session_state["editing_memory_id"] = mid
                            st.session_state["confirm_delete_id"] = None
                            st.rerun()

                    # Forget button
                    with col_del:
                        if st.button("🗑️ Forget", key=f"btn_del_{mid}"):
                            st.session_state["confirm_delete_id"] = mid
                            st.session_state["editing_memory_id"] = None
                            st.rerun()

                    # Inline Edit Form if active
                    if st.session_state.get("editing_memory_id") == mid:
                        with st.form(f"form_edit_{mid}"):
                            updated_text = st.text_area("Updated Content", value=mem.content)
                            save_btn = st.form_submit_button("Update Memory")
                            cancel_btn = st.form_submit_button("Cancel")
                            if save_btn:
                                updated_item, dec = memory_service.update_memory(
                                    user_id=active_user_id,
                                    memory_id=mid,
                                    new_content=updated_text,
                                    reason="Updated via Streamlit Memory Inspector",
                                )
                                if dec.allowed and updated_item:
                                    st.success("Memory updated successfully.")
                                    st.session_state["editing_memory_id"] = None
                                    st.rerun()
                                else:
                                    st.error(f"Policy Rejected: {dec.message}")
                            if cancel_btn:
                                st.session_state["editing_memory_id"] = None
                                st.rerun()

                    # Inline Forget Confirmation if active
                    if st.session_state.get("confirm_delete_id") == mid:
                        st.warning("⚠️ **Confirm Forget:** This action permanently purges this memory record from SQLite persistent storage.")
                        col_yes, col_no = st.columns([1, 1])
                        with col_yes:
                            if st.button("Confirm Permanent Purge", key=f"conf_yes_{mid}", type="primary"):
                                deleted = memory_service.delete_memory(active_user_id, mid)
                                if deleted:
                                    st.success("Memory permanently purged from database.")
                                    st.session_state["confirm_delete_id"] = None
                                    st.rerun()
                                else:
                                    st.error("Failed to delete memory.")
                        with col_no:
                            if st.button("Cancel", key=f"conf_no_{mid}"):
                                st.session_state["confirm_delete_id"] = None
                                st.rerun()

                    st.markdown("<div style='margin-bottom: 0.5rem;'></div>", unsafe_allow_html=True)

    # -------------------------------------------------------------------------
    # Tab 3: Observability & Trace Details
    # -------------------------------------------------------------------------
    with tab_trace:
        st.subheader("🔍 Execution Trace & Agent Metadata")
        last_trace = st.session_state.get("last_trace_info")

        if not last_trace:
            st.info("No query has been executed in the current session yet. Ask a question in the Research tab to view the live multi-agent execution trace.")
        else:
            render_observability_card(last_trace)

            with st.expander("📄 Raw Sanitized Execution State (Redacted)", expanded=False):
                # Clean up state dictionary for display
                safe_state = {
                    "user_id": last_trace.get("user_id"),
                    "conversation_id": last_trace.get("conversation_id"),
                    "route": last_trace.get("route"),
                    "routing_reason": last_trace.get("routing_reason"),
                    "is_fallback": last_trace.get("is_fallback"),
                    "llm_metadata": last_trace.get("llm_metadata"),
                    "retry_count": last_trace.get("retry_count"),
                    "verification": last_trace.get("verification"),
                    "citations_count": len(last_trace.get("citations", [])),
                    "retrieved_chunks_count": len(last_trace.get("retrieved_chunks", [])),
                }
                st.json(safe_state)


if __name__ == "__main__":
    main()
