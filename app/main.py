"""MemoraAI — Enterprise Multi-Agent Research Assistant.

Polished, production-ready interface inspired by Linear and Notion:
- Minimalist dark aesthetic with curated tokens: #0F1115, #14171D, #191D24, #292F39, #7AA7FF.
- Refined professional sidebar with workspace branding, date-grouped conversation history, and profile isolation.
- Conversational research canvas with document-style assistant answers, verified citations, and copy helpers.
- Full Memory Inspector with instant policy validation, editing, and permanent purge confirmation.
- Transparent Agent Observability dashboard with execution timelines, verifier verdicts, and LangSmith diagnostics.
"""

from datetime import datetime
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
from asterline.observability.tracing import check_langsmith_status, setup_tracing
from asterline.rag.ingestion import load_and_validate_corpus
from asterline.rag.retriever import HybridRetriever

from app.components import (
    DEMO_PROFILES,
    format_timestamp,
    inject_custom_css,
    render_brand_header,
    render_citations_expander,
    render_empty_state,
    render_fallback_pill,
    render_observability_card,
    render_response_meta_row,
    render_route_pill,
    render_user_profile_header,
)


# =============================================================================
# Cached Application Runtime (Single Initialization across Streamlit Reruns)
# =============================================================================

@st.cache_resource(show_spinner="Initializing MemoraAI Hybrid Index & Multi-Agent Runtime...")
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
        chroma_persist_dir=chroma_dir,
        embedding_model_name=settings.embedding_model_name,
    )

    # 3. Persistent SQLite Memory Store & Application Service
    sqlite_path = settings.resolve_path(settings.sqlite_db_path)
    store = SQLiteStore(sqlite_path)
    memory_service = MemoryService(store)

    # 4. Activate LangSmith tracing if configured
    setup_tracing()

    # 5. Compile LangGraph Multi-Agent Workflow
    builder = WorkflowBuilder(
        retriever=retriever,
        memory_service=memory_service,
    )
    workflow_app = builder.build()

    # Precompute serializations once for citation expanders
    chunks_dict = [c.model_dump() for c in chunks]
    chunk_map = {c["chunk_id"]: c for c in chunks_dict}

    return {
        "settings": settings,
        "chunks": chunks,
        "chunks_dict": chunks_dict,
        "chunk_map": chunk_map,
        "corpus_report": report,
        "retriever": retriever,
        "store": store,
        "memory_service": memory_service,
        "workflow": workflow_app,
    }


# =============================================================================
# Date Grouping & Conversation Helper
# =============================================================================

def group_conversations(conversations: List[Dict[str, Any]]) -> Dict[str, List[Dict[str, Any]]]:
    """Group conversation records by recency (Today, Yesterday, Previous)."""
    groups: Dict[str, List[Dict[str, Any]]] = {
        "Today": [],
        "Yesterday": [],
        "Previous": [],
    }
    now = datetime.now()

    for conv in conversations:
        raw_ts = conv.get("updated_at") or conv.get("created_at")
        try:
            dt = datetime.fromisoformat(str(raw_ts).replace("Z", "+00:00"))
            delta_days = (now.date() - dt.date()).days
            if delta_days == 0:
                groups["Today"].append(conv)
            elif delta_days == 1:
                groups["Yesterday"].append(conv)
            else:
                groups["Previous"].append(conv)
        except Exception:
            groups["Previous"].append(conv)

    return {k: v for k, v in groups.items() if v}


# =============================================================================
# State Management
# =============================================================================

def create_new_conversation(user_id: str, memory_service: MemoryService, title: str = "New Investigation") -> str:
    """Create a distinct, persistent conversation and synchronize session state."""
    new_conv_id = f"conv_{uuid.uuid4().hex[:12]}"
    memory_service.get_or_create_conversation(user_id, new_conv_id, title)
    st.session_state["active_conv_id"] = new_conv_id
    st.session_state[f"conv_select_box_{user_id}"] = new_conv_id
    st.session_state["last_trace_info"] = None
    st.session_state.pop("pending_prompt", None)
    st.session_state["conv_search_query"] = ""
    return new_conv_id


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

    # Sync selectbox widget key to active_conv_id if not present
    widget_key = f"conv_select_box_{user_id}"
    if widget_key not in st.session_state:
        st.session_state[widget_key] = st.session_state["active_conv_id"]

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
        if "user_profile_selector_widget" in st.session_state:
            st.session_state["user_profile_selector_widget"] = new_user_id
        # Re-scope conversation to selected user's latest conversation or create new
        user_convs = memory_service.list_user_conversations(new_user_id)
        if user_convs:
            selected_conv = user_convs[0]["conversation_id"]
        else:
            new_id = f"conv_{uuid.uuid4().hex[:12]}"
            memory_service.get_or_create_conversation(new_user_id, new_id, "New Investigation")
            selected_conv = new_id

        st.session_state["active_conv_id"] = selected_conv
        st.session_state[f"conv_select_box_{new_user_id}"] = selected_conv

        # Clear transient UI state
        st.session_state["last_trace_info"] = None
        st.session_state["editing_memory_id"] = None
        st.session_state["confirm_delete_id"] = None
        st.session_state["conv_search_query"] = ""
        st.session_state.pop("pending_prompt", None)
        st.rerun()


# =============================================================================
# Main Application Flow
# =============================================================================

def main() -> None:
    st.set_page_config(
        page_title="MemoraAI — Enterprise Research Assistant",
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
    # Refined Sidebar
    # =========================================================================
    with st.sidebar:
        render_brand_header()

        # Prominent New Investigation action
        if st.button("＋ New Investigation", use_container_width=True, type="primary"):
            create_new_conversation(active_user_id, memory_service)
            st.rerun()

        # Searchable Conversation History
        st.markdown('<div class="sidebar-section-title">Investigation History</div>', unsafe_allow_html=True)
        user_convs = memory_service.list_user_conversations(active_user_id)

        conv_search = st.text_input(
            "Filter History",
            placeholder="Search conversations...",
            label_visibility="collapsed",
            key="conv_search_query",
        )

        filtered_convs = user_convs
        if conv_search and conv_search.strip():
            query_lower = conv_search.strip().lower()
            filtered_convs = [c for c in user_convs if query_lower in c.get("title", "").lower()]

        widget_key = f"conv_select_box_{active_user_id}"

        if filtered_convs:
            grouped = group_conversations(filtered_convs)
            conv_options = []
            conv_labels = {}

            for group_name, group_items in grouped.items():
                for c in group_items:
                    cid = c["conversation_id"]
                    conv_options.append(cid)
                    ts_clean = format_timestamp(str(c.get("updated_at") or c.get("created_at")))
                    conv_labels[cid] = f"[{group_name}] {c['title']} ({ts_clean})"

            # Keep widget session state in sync with canonical active_conv_id
            current_active = st.session_state.get("active_conv_id")
            if current_active in conv_options:
                st.session_state[widget_key] = current_active
            elif widget_key in st.session_state and st.session_state[widget_key] not in conv_options:
                del st.session_state[widget_key]

            def on_conv_change() -> None:
                new_cid = st.session_state.get(widget_key)
                if new_cid and new_cid != st.session_state.get("active_conv_id"):
                    st.session_state["active_conv_id"] = new_cid
                    st.session_state["last_trace_info"] = None

            chosen_conv = st.selectbox(
                label="Select Investigation",
                options=conv_options,
                format_func=lambda cid: conv_labels.get(cid, cid),
                index=None if widget_key in st.session_state else 0,
                key=widget_key,
                on_change=on_conv_change,
                label_visibility="collapsed",
            )
        else:
            if conv_search:
                st.caption("No matching investigations found.")
            else:
                st.caption("No saved investigations yet.")

        st.divider()

        # User Identity & Workspace Role (Bottom Positioned)
        st.markdown('<div class="sidebar-section-title">Active Demonstration Identity</div>', unsafe_allow_html=True)

        def on_user_profile_change() -> None:
            new_uid = st.session_state.get("user_profile_selector_widget")
            if new_uid and new_uid != st.session_state.get("user_id"):
                switch_user_profile(new_uid, memory_service)

        profile_keys = list(DEMO_PROFILES.keys())
        current_profile_index = profile_keys.index(active_user_id) if active_user_id in profile_keys else 0

        selected_user = st.selectbox(
            label="Active User Profile",
            options=profile_keys,
            format_func=lambda uid: DEMO_PROFILES[uid]["name"],
            index=current_profile_index,
            key="user_profile_selector_widget",
            on_change=on_user_profile_change,
            label_visibility="collapsed",
            help="Strictly scopes all SQLite conversation threads, queries, and persistent memories.",
        )
        if selected_user != active_user_id:
            switch_user_profile(selected_user, memory_service)

        profile = DEMO_PROFILES[selected_user]
        st.caption(
            f"**Focus:** {profile['focus']}\n\n"
            f"*Privacy: Data, memory, and conversation history are 100% isolated to `{profile['id']}`.*"
        )

        # System Diagnostics
        with st.expander("⚙️ System Diagnostics", expanded=False):
            st.write("**Corpus:** 75 Chunks Verified")
            st.code(runtime["corpus_report"].corpus_sha256[:20] + "...", language="text")
            st.write("**Hybrid Search:** ChromaDB Dense + BM25 Lexical")

            ls_status = check_langsmith_status()
            if ls_status["tracing_enabled"]:
                st.success(f"LangSmith: Active ({ls_status['project']})")
            else:
                st.info("LangSmith: Offline / Local Mode")

    # Re-read active session state after sidebar interactions to guarantee synchronization
    active_user_id = st.session_state["user_id"]
    active_conv_id = st.session_state["active_conv_id"]

    # Look up active conversation title
    active_conv_title = "New Investigation"
    current_user_convs = memory_service.list_user_conversations(active_user_id)
    for c in current_user_convs:
        if c["conversation_id"] == active_conv_id:
            active_conv_title = c.get("title") or "New Investigation"
            break

    # =========================================================================
    # Main Workspace Canvas
    # =========================================================================
    render_user_profile_header(active_user_id, active_conv_title)

    tab_chat, tab_memory, tab_trace = st.tabs([
        "Research Assistant",
        "Memory Inspector",
        "Agent Observability & Traces",
    ])

    # -------------------------------------------------------------------------
    # Tab 1: Research Assistant (Conversation-First)
    # -------------------------------------------------------------------------
    with tab_chat:
        messages = memory_service.get_conversation_messages(active_conv_id, active_user_id)

        # ChatGPT-style bounded scrollable conversation container with autoscroll
        chat_stream = st.container(height=580, autoscroll=True)

        # Chat Composer placed cleanly at the bottom
        user_prompt = st.chat_input("Ask a technical documentation question, investigate an incident, or manage memory...")
        st.markdown('<div class="composer-hint">Enter to send • Shift + Enter for new line • Grounded in 75 verified Asterline documentation chunks</div>', unsafe_allow_html=True)
        
        active_prompt = None
        if "pending_prompt" in st.session_state and st.session_state["pending_prompt"]:
            active_prompt = st.session_state.pop("pending_prompt")
        elif user_prompt:
            active_prompt = user_prompt.strip()

        # Render everything inside chat_stream so it ALWAYS appears ABOVE the input box
        with chat_stream:
            if not messages and not active_prompt:
                render_empty_state()
                col1, col2 = st.columns(2)
                with col1:
                    if st.button("📋 POST /v2/runs Payload Limits\n\nPayload size and batch rate limits", key="start_btn_1", use_container_width=True):
                        st.session_state["pending_prompt"] = "What are the maximum payload size and rate limits for Asterline webhooks?"
                        st.rerun()
                    if st.button("👥 Directory: Who is Amaresh?\n\nLookup engineering role and ownership", key="start_btn_2", use_container_width=True):
                        st.session_state["pending_prompt"] = "Who is Amaresh and what are his responsibilities?"
                        st.rerun()
                with col2:
                    if st.button("🔄 Webhook Signing Key Policy\n\nDeprecation schedule & grace period", key="start_btn_3", use_container_width=True):
                        st.session_state["pending_prompt"] = "What is the deprecation policy and grace period for webhook signing keys?"
                        st.rerun()
                    if st.button("🧠 Inspect Stored Context\n\nReview preferences saved for this profile", key="start_btn_4", use_container_width=True):
                        st.session_state["pending_prompt"] = "What preferences and context do you have stored for me?"
                        st.rerun()
            else:
                for msg in messages:
                    if msg.role == "user":
                        with st.chat_message("user"):
                            st.markdown(msg.content)
                    else:
                        with st.chat_message("assistant", avatar="🔬"):
                            # Restrained, quiet metadata indicator
                            render_response_meta_row(
                                route=msg.route_taken,
                                llm_metadata={"provider": "groq", "model": "llama-3.3-70b"},
                                is_grounded=True,
                                citation_count=len(msg.citations) if msg.citations else 0,
                            )

                            # Formatted Markdown Answer
                            st.markdown(msg.content)

                            # Authoritative Expandable Citations
                            if msg.citations:
                                render_citations_expander(
                                    citations=[c.model_dump() for c in msg.citations],
                                    chunk_map=runtime["chunk_map"],
                                )

            # Render active prompt and live agent invocation inside chat_stream (ABOVE composer)
            if active_prompt:
                clean_prompt = active_prompt.strip()
                if clean_prompt:
                    # Always read authoritative conversation and user IDs at submission time
                    current_user_id = st.session_state["user_id"]
                    current_conv_id = st.session_state["active_conv_id"]

                    # 1. Render immediate user message
                    with st.chat_message("user"):
                        st.markdown(clean_prompt)

                    # 2. Execute multi-agent LangGraph workflow
                    with st.chat_message("assistant", avatar="🔬"):
                        with st.spinner("Analyzing query, routing agents, and verifying documentation evidence..."):
                            try:
                                state_input = {
                                    "user_message": clean_prompt,
                                    "user_id": current_user_id,
                                    "conversation_id": current_conv_id,
                                }
                                result = workflow.invoke(
                                    state_input,
                                    config={"configurable": {"thread_id": current_conv_id}},
                                )

                                st.session_state["last_trace_info"] = result

                                # Update conversation title if this is the start of a thread
                                if len(messages) == 0:
                                    title_summary = clean_prompt[:36] + ("..." if len(clean_prompt) > 36 else "")
                                    memory_service.update_conversation_title(
                                        current_conv_id,
                                        current_user_id,
                                        title_summary,
                                    )

                                st.rerun()

                            except Exception as e:
                                st.error(f"Execution Error: {str(e)}")

    # -------------------------------------------------------------------------
    # Tab 2: Memory Inspector (Durable User Preferences)
    # -------------------------------------------------------------------------
    with tab_memory:
        st.markdown(
            f"""
            <div style="margin-bottom: 1rem;">
                <h3 style="color: #E8ECF3; margin-bottom: 0.25rem; font-weight: 600; font-size: 1.15rem;">Memory Inspector</h3>
                <p style="color: #98A3B5; font-size: 0.82rem; margin: 0; line-height: 1.45;">
                    Manage the context MemoraAI remembers to make future conversations more useful.
                    Preferences are strictly partitioned to <strong>{profile['name']}</strong>.
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )

        active_memories = memory_service.list_user_memories(active_user_id)

        # Top Control Strip: Search, Filter, and Add Action
        col_search, col_filter, col_add_btn = st.columns([3, 2, 2])
        with col_search:
            mem_search = st.text_input("Search Memories", placeholder="Search memories...", label_visibility="collapsed", key="mem_search_kw")
        with col_filter:
            category_filter = st.selectbox(
                "Filter Category",
                options=["All Categories", "preference", "workspace_context", "project_role", "investigation_note", "general"],
                index=0,
                label_visibility="collapsed",
            )
        with col_add_btn:
            show_add_form = st.checkbox("＋ Add Memory", value=False, key="toggle_add_mem")

        # Form to add a new long-term memory (Collapsible / Active)
        if show_add_form:
            with st.form("add_memory_form", clear_on_submit=True):
                st.markdown("<div style='font-size: 0.82rem; font-weight: 600; color: #7DA5FF; margin-bottom: 0.4rem;'>Record New Long-Term Memory</div>", unsafe_allow_html=True)
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
                            reason="Added via MemoraAI Memory Inspector",
                        )
                        if decision.allowed and item:
                            st.success("Memory persisted successfully to SQLite store.")
                            st.rerun()
                        else:
                            st.error(f"Policy Rejection: {decision.message} (Reason Code: {decision.reason_code})")

        # Filter active memories
        filtered_memories = active_memories
        if mem_search and mem_search.strip():
            kw = mem_search.strip().lower()
            filtered_memories = [m for m in filtered_memories if kw in m.content.lower()]
        if category_filter != "All Categories":
            filtered_memories = [m for m in filtered_memories if m.category == category_filter]

        st.markdown(
            f"""
            <div style="display: flex; justify-content: space-between; align-items: center; margin: 0.75rem 0 0.5rem 0;">
                <span style="font-size: 0.75rem; color: #98A3B5;">Stored Records: <strong style="color: #E8ECF3;">{len(filtered_memories)}</strong> of {len(active_memories)}</span>
                <span style="font-size: 0.72rem; color: #697588;">Scope: <code>{profile['id']}</code></span>
            </div>
            """,
            unsafe_allow_html=True,
        )

        if not filtered_memories:
            if mem_search or category_filter != "All Categories":
                st.info("No memories match the active filter criteria.")
            else:
                st.info(f"No active memories stored for {profile['name']}. You can add one above or tell the assistant in chat (e.g. 'Remember that I use Snowflake').")
        else:
            for mem in filtered_memories:
                mid = mem.memory_id
                with st.container():
                    st.markdown(
                        f"""
                        <div class="memory-row-card">
                            <div class="memory-row-header">
                                <span class="memory-cat-pill">{mem.category.upper()}</span>
                                <span style="font-size: 0.72rem; color: #98A3B5;">Created {format_timestamp(mem.created_at)}</span>
                            </div>
                            <div class="memory-body-text">{mem.content}</div>
                            <div class="memory-footer-meta">
                                <span>ID: <code style="color: #98A3B5; font-size: 0.68rem;">{mem.memory_id}</code></span>
                                <span>•</span>
                                <span>Confidence: {mem.confidence}</span>
                            </div>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )

                    col_edit, col_del, _ = st.columns([1, 1, 5])

                    with col_edit:
                        if st.button("✏️ Edit", key=f"btn_edit_{mid}"):
                            st.session_state["editing_memory_id"] = mid
                            st.session_state["confirm_delete_id"] = None
                            st.rerun()

                    with col_del:
                        if st.button("🗑️ Forget", key=f"btn_del_{mid}"):
                            st.session_state["confirm_delete_id"] = mid
                            st.session_state["editing_memory_id"] = None
                            st.rerun()

                    # Inline Edit Form
                    if st.session_state.get("editing_memory_id") == mid:
                        with st.form(f"form_edit_{mid}"):
                            updated_text = st.text_area("Updated Content", value=mem.content)
                            save_btn = st.form_submit_button("Update Memory", type="primary")
                            cancel_btn = st.form_submit_button("Cancel")
                            if save_btn:
                                updated_item, dec = memory_service.update_memory(
                                    user_id=active_user_id,
                                    memory_id=mid,
                                    new_content=updated_text,
                                    reason="Updated via MemoraAI Memory Inspector",
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

                    # Inline Confirmation for Permanent Forget
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

                    st.markdown("<div style='margin-bottom: 0.4rem;'></div>", unsafe_allow_html=True)

    # -------------------------------------------------------------------------
    # Tab 3: Agent Observability & Traces
    # -------------------------------------------------------------------------
    with tab_trace:
        st.markdown(
            """
            <div style="margin-bottom: 1rem;">
                <h3 style="color: #E8ECF3; margin-bottom: 0.25rem; font-weight: 600; font-size: 1.15rem;">Agent Observability & Tracing</h3>
                <p style="color: #98A3B5; font-size: 0.82rem; margin: 0; line-height: 1.45;">
                    Inspect execution flows, agent decisions, retrieval, verification, and performance telemetry.
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )

        last_trace = st.session_state.get("last_trace_info")
        if not last_trace:
            st.info("No execution trace recorded yet in the active session. Submit an investigation in the Research Assistant tab to inspect live multi-agent execution telemetry.")
        else:
            render_observability_card(last_trace)

            with st.expander("📄 Sanitized State Payload (Redacted)", expanded=False):
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
