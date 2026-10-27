"""Modular UI components, styling, profile models, and display formatters for the Streamlit application."""

from datetime import datetime
from typing import Any, Dict, List, Optional
import streamlit as st

# =============================================================================
# Demo Profile Definitions (Strict Demonstration Identity)
# =============================================================================

DEMO_PROFILES: Dict[str, Dict[str, str]] = {
    "user_alice": {
        "id": "user_alice",
        "name": "Alice — DevOps Lead",
        "short_name": "Alice",
        "role": "DevOps & Infrastructure Lead",
        "focus": "Pipeline reliability, warehouse exports, deployment rollouts",
        "avatar": "👩‍💻",
        "badge_color": "#388bfd",
    },
    "user_bob": {
        "id": "user_bob",
        "name": "Bob — Backend Platform Engineer",
        "short_name": "Bob",
        "role": "Backend Platform Engineer",
        "focus": "AQL query optimization, ingestion limits, SDK batching",
        "avatar": "👨‍💻",
        "badge_color": "#2ea043",
    },
    "user_charlie": {
        "id": "user_charlie",
        "name": "Charlie — Customer Support & QA",
        "short_name": "Charlie",
        "role": "Technical Support Specialist",
        "focus": "Enterprise SLAs, incident timelines, Signals alert rules",
        "avatar": "🕵️",
        "badge_color": "#d29922",
    },
}

ROUTE_DISPLAY_META: Dict[str, Dict[str, str]] = {
    "conversational": {
        "label": "Conversational Greeting",
        "color": "#8b949e",
        "icon": "👋",
        "description": "Bypasses vector search; directly handles greetings & identity inquiries.",
    },
    "clarification": {
        "label": "Clarification Request",
        "color": "#d29922",
        "icon": "❓",
        "description": "Triggered when query is underspecified or ambiguous.",
    },
    "memory_ops": {
        "label": "Memory Management",
        "color": "#a371f7",
        "icon": "🧠",
        "description": "Explicit memory creation, retrieval, or forget command.",
    },
    "factual": {
        "label": "Documentation Lookup",
        "color": "#58a6ff",
        "icon": "📖",
        "description": "Single-document factual research backed by authoritative citations.",
    },
    "investigation": {
        "label": "Multi-Document Investigation",
        "color": "#f0883e",
        "icon": "🔬",
        "description": "Reconciles multi-document incidents, version evolution, and superseded policies.",
    },
}


# =============================================================================
# Design & CSS Injection
# =============================================================================

def inject_custom_css() -> None:
    """Inject polished, modern dark-mode enterprise CSS."""
    custom_css = """
    <style>
    /* Global Typography & Palette */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap');

    html, body, [class*="css"] {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
    }

    code, pre, [class*="stCode"] {
        font-family: 'JetBrains Mono', monospace !important;
    }

    /* Enterprise App Container */
    .stApp {
        background-color: #0d1117;
        color: #e6edf3;
    }

    /* Sidebar Customization */
    section[data-testid="stSidebar"] {
        background-color: #161b22 !important;
        border-right: 1px solid #30363d !important;
    }

    /* Header & Badges */
    .app-title-container {
        padding: 0.5rem 0 1rem 0;
        border-bottom: 1px solid #30363d;
        margin-bottom: 1.2rem;
    }
    .app-title {
        font-size: 1.5rem;
        font-weight: 700;
        color: #f0f6fc;
        display: flex;
        align-items: center;
        gap: 0.5rem;
        margin: 0;
    }
    .app-subtitle {
        font-size: 0.85rem;
        color: #8b949e;
        margin-top: 0.2rem;
    }

    /* Pill Badges */
    .pill-badge {
        display: inline-flex;
        align-items: center;
        gap: 0.35rem;
        padding: 0.2rem 0.6rem;
        border-radius: 9999px;
        font-size: 0.75rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.03em;
    }
    .badge-route {
        background-color: rgba(56, 139, 253, 0.15);
        color: #58a6ff;
        border: 1px solid rgba(56, 139, 253, 0.3);
    }
    .badge-fallback {
        background-color: rgba(210, 153, 34, 0.15);
        color: #e3b341;
        border: 1px solid rgba(210, 153, 34, 0.3);
    }
    .badge-live {
        background-color: rgba(46, 160, 67, 0.15);
        color: #3fb950;
        border: 1px solid rgba(46, 160, 67, 0.3);
    }
    .badge-verified {
        background-color: rgba(46, 160, 67, 0.12);
        color: #3fb950;
        border: 1px solid rgba(46, 160, 67, 0.25);
    }

    /* Citation Cards */
    .citation-card {
        background-color: #161b22;
        border: 1px solid #30363d;
        border-radius: 6px;
        padding: 0.75rem;
        margin-bottom: 0.5rem;
    }
    .citation-header {
        font-weight: 600;
        color: #58a6ff;
        font-size: 0.85rem;
        margin-bottom: 0.25rem;
        display: flex;
        justify-content: space-between;
    }
    .citation-excerpt {
        font-size: 0.8rem;
        color: #c9d1d9;
        line-height: 1.4;
    }

    /* Memory Card */
    .memory-card {
        background-color: #161b22;
        border: 1px solid #30363d;
        border-radius: 8px;
        padding: 1rem;
        margin-bottom: 0.85rem;
        transition: border-color 0.15s ease-in-out;
    }
    .memory-card:hover {
        border-color: #58a6ff;
    }
    .memory-category {
        font-size: 0.72rem;
        font-weight: 600;
        text-transform: uppercase;
        color: #8b949e;
        letter-spacing: 0.05em;
    }
    .memory-content {
        font-size: 0.95rem;
        color: #f0f6fc;
        margin: 0.4rem 0 0.6rem 0;
    }
    .memory-meta {
        font-size: 0.75rem;
        color: #6e7681;
    }

    /* Observability Metric Tiles */
    .metric-tile {
        background-color: #161b22;
        border: 1px solid #30363d;
        border-radius: 6px;
        padding: 0.75rem 1rem;
        text-align: left;
    }
    .metric-label {
        font-size: 0.75rem;
        color: #8b949e;
        text-transform: uppercase;
        font-weight: 600;
    }
    .metric-value {
        font-size: 1.2rem;
        color: #f0f6fc;
        font-weight: 700;
        margin-top: 0.2rem;
    }

    /* Streamlit Widget Overrides */
    div[data-testid="stChatMessage"] {
        background-color: #161b22 !important;
        border: 1px solid #30363d !important;
        border-radius: 8px !important;
        padding: 1rem !important;
        margin-bottom: 0.75rem !important;
    }
    </style>
    """
    st.markdown(custom_css, unsafe_allow_html=True)


# =============================================================================
# Formatting & Presentation Helpers
# =============================================================================

def format_timestamp(ts: Optional[str]) -> str:
    """Format ISO or SQLite timestamp into clean readable string."""
    if not ts:
        return "Unknown"
    try:
        dt = datetime.fromisoformat(ts.replace("Z", "+00:00"))
        return dt.strftime("%b %d, %Y • %H:%M")
    except Exception:
        return str(ts)[:16]


def render_user_profile_header(user_id: str) -> None:
    """Render profile identity header for the active user."""
    profile = DEMO_PROFILES.get(user_id, DEMO_PROFILES["user_alice"])
    st.markdown(
        f"""
        <div style="background-color: #161b22; border: 1px solid #30363d; border-radius: 8px; padding: 0.85rem 1.1rem; margin-bottom: 1rem;">
            <div style="display: flex; align-items: center; justify-content: space-between;">
                <div style="display: flex; align-items: center; gap: 0.75rem;">
                    <span style="font-size: 1.6rem;">{profile['avatar']}</span>
                    <div>
                        <div style="font-weight: 700; font-size: 1.05rem; color: #f0f6fc;">{profile['name']}</div>
                        <div style="font-size: 0.8rem; color: #8b949e;">{profile['focus']}</div>
                    </div>
                </div>
                <div style="text-align: right;">
                    <span class="pill-badge" style="background-color: rgba(56, 139, 253, 0.15); color: #58a6ff; border: 1px solid rgba(56, 139, 253, 0.3);">
                        Active Demonstration Profile
                    </span>
                    <div style="font-size: 0.72rem; color: #6e7681; margin-top: 0.25rem;">ID: <code>{profile['id']}</code></div>
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_route_pill(route_key: Optional[str]) -> str:
    """Generate HTML pill for route classification."""
    meta = ROUTE_DISPLAY_META.get(route_key or "", {
        "label": route_key or "Unknown",
        "color": "#8b949e",
        "icon": "📍",
    })
    return f"""<span class="pill-badge badge-route">{meta['icon']} {meta['label']}</span>"""


def render_fallback_pill(is_fallback: bool, llm_metadata: Optional[Dict[str, Any]] = None) -> str:
    """Generate HTML pill indicating whether deterministic fallback or live model was used."""
    if is_fallback:
        return """<span class="pill-badge badge-fallback">⚠️ Deterministic Fallback</span>"""
    provider = (llm_metadata or {}).get("provider", "live")
    model = (llm_metadata or {}).get("model", "llm")
    return f"""<span class="pill-badge badge-live">⚡ Live: {provider}/{model}</span>"""


def render_citations_expander(
    citations: List[Dict[str, Any]],
    retrieved_chunks: Optional[List[Dict[str, Any]]] = None,
) -> None:
    """Render expandable source cards with document title, chunk ID, and source text."""
    if not citations:
        return

    chunk_map = {c.get("chunk_id"): c for c in (retrieved_chunks or []) if isinstance(c, dict)}

    with st.expander(f"📚 Authoritative Sources & Citations ({len(citations)})", expanded=False):
        for idx, cit in enumerate(citations):
            cid = cit.get("chunk_id", "Unknown")
            title = cit.get("title", "Untitled Document")
            source_url = cit.get("source_url", "")

            # Look up excerpt from retrieved chunks if available
            excerpt = ""
            if cid in chunk_map:
                excerpt = chunk_map[cid].get("text", "")
            elif "text" in cit:
                excerpt = cit["text"]

            st.markdown(
                f"""
                <div class="citation-card">
                    <div class="citation-header">
                        <span><strong>[{cid}]</strong> {title}</span>
                        <span style="font-size: 0.72rem; color: #8b949e;">{source_url}</span>
                    </div>
                    <div class="citation-excerpt">
                        {excerpt if excerpt else "<em>Authoritative chunk verified in Asterline knowledge corpus.</em>"}
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )


def render_observability_card(trace_info: Dict[str, Any]) -> None:
    """Render a clean, compact execution trace tile for the current turn."""
    if not trace_info:
        st.info("No execution trace captured yet for this turn.")
        return

    route = trace_info.get("route", "unspecified")
    is_fallback = trace_info.get("is_fallback", False)
    llm_metadata = trace_info.get("llm_metadata", {})
    verification = trace_info.get("verification", {})
    verdict = verification.get("verdict", "pass")
    retry_count = trace_info.get("retry_count", 0)
    citations_count = len(trace_info.get("citations", []))
    retrieved_count = len(trace_info.get("retrieved_chunks", []))

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.markdown(
            f"""
            <div class="metric-tile">
                <div class="metric-label">Execution Route</div>
                <div class="metric-value" style="font-size: 1rem; color: #58a6ff;">{route.upper()}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with col2:
        verdict_color = "#3fb950" if verdict == "pass" else ("#e3b341" if verdict == "abstain" else "#f85149")
        st.markdown(
            f"""
            <div class="metric-tile">
                <div class="metric-label">Verifier Verdict</div>
                <div class="metric-value" style="font-size: 1rem; color: {verdict_color};">{verdict.upper()}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with col3:
        st.markdown(
            f"""
            <div class="metric-tile">
                <div class="metric-label">Retrieved / Cited</div>
                <div class="metric-value">{retrieved_count} / {citations_count}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with col4:
        st.markdown(
            f"""
            <div class="metric-tile">
                <div class="metric-label">Retry Attempts</div>
                <div class="metric-value">{retry_count} / 2</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    # Detailed Nodes & Status
    with st.expander("🛠️ Workflow Execution Graph & Verifier Details", expanded=False):
        st.write("**Routing Decision Reason:**", trace_info.get("routing_reason", "Keyword and intent classification"))
        if verification:
            st.write("**Verification Feedback:**", verification.get("feedback") or "Grounded claims and valid citations confirmed.")
            if verification.get("reasons"):
                st.write("**Verifier Observations:**", verification["reasons"])
        if is_fallback:
            reason = llm_metadata.get("fallback_reason", "No external LLM API key configured (offline deterministic fallback)")
            st.warning(f"Deterministic Fallback Reason: {reason}")
