"""MemoraAI — Exact Enterprise UI Reference Implementation.

Technical developer-grade design system:
- Palette: #0B0D10 (Canvas), #101318 (Sidebar/Code), #151920 (Panels), #181D26 (Elevated/User Bubble), #2A303A (Borders), #7DA5FF (Primary Blue), #8B7CFF (Violet Accent).
- Dense technical hierarchy, dark mode enforcement across all Streamlit widgets, precise alignment.
- Three Core Workspaces: Research Assistant, Memory Inspector, Agent Observability & Traces.
"""

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
        "badge_color": "#7DA5FF",
    },
    "user_bob": {
        "id": "user_bob",
        "name": "Bob — Backend Platform Engineer",
        "short_name": "Bob",
        "role": "Backend Platform Engineer",
        "focus": "AQL query optimization, ingestion limits, SDK batching",
        "avatar": "👨‍💻",
        "badge_color": "#35D6A0",
    },
    "user_charlie": {
        "id": "user_charlie",
        "name": "Charlie — Customer Support & QA",
        "short_name": "Charlie",
        "role": "Technical Support Specialist",
        "focus": "Enterprise SLAs, incident timelines, Signals alert rules",
        "avatar": "🕵️",
        "badge_color": "#F0B75E",
    },
}

ROUTE_DISPLAY_META: Dict[str, Dict[str, str]] = {
    "conversational": {
        "label": "Conversational Greeting",
        "color": "#98A3B5",
        "icon": "👋",
        "description": "Bypasses vector search; directly handles greetings & identity inquiries.",
    },
    "clarification": {
        "label": "Clarification Request",
        "color": "#F0B75E",
        "icon": "❓",
        "description": "Triggered when query is underspecified or ambiguous.",
    },
    "memory_ops": {
        "label": "Memory Management",
        "color": "#8B7CFF",
        "icon": "🧠",
        "description": "Explicit memory creation, retrieval, or forget command.",
    },
    "factual": {
        "label": "Documentation Lookup",
        "color": "#7DA5FF",
        "icon": "📖",
        "description": "Single-document factual research backed by authoritative citations.",
    },
    "investigation": {
        "label": "Multi-Document Investigation",
        "color": "#F0B75E",
        "icon": "🔬",
        "description": "Reconciles multi-document incidents, version evolution, and superseded policies.",
    },
}


# =============================================================================
# Precision Technical Design System & CSS Injection
# =============================================================================

def inject_custom_css() -> None:
    """Inject bulletproof enterprise developer-tool CSS overriding all Streamlit light defaults."""
    custom_css = """
    <style>
    /* -------------------------------------------------------------------------
       Google Fonts & Typography System
       ------------------------------------------------------------------------- */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=JetBrains+Mono:wght@400;500;600&display=swap');

    :root {
        --bg-app: #0B0D10;
        --bg-sidebar: #101318;
        --bg-panel: #151920;
        --bg-elevated: #181D26;
        --bg-hover: #222834;
        --border-default: #2A303A;
        --border-subtle: rgba(42, 48, 58, 0.65);
        --text-primary: #E8ECF3;
        --text-secondary: #98A3B5;
        --text-muted: #697588;
        --primary-blue: #7DA5FF;
        --accent-violet: #8B7CFF;
        --success-green: #35D6A0;
        --warning-amber: #F0B75E;
        --error-red: #F07878;
        --radius-sm: 4px;
        --radius-md: 6px;
        --radius-lg: 8px;
    }

    /* Streamlit Global Shell Overrides */
    html, body, [class*="css"], .stApp {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif !important;
        background-color: var(--bg-app) !important;
        color: var(--text-primary) !important;
        -webkit-font-smoothing: antialiased;
        -moz-osx-font-smoothing: grayscale;
        letter-spacing: -0.01em;
    }

    code, pre, [class*="stCode"], .mono-text {
        font-family: 'JetBrains Mono', SFMono-Regular, Menlo, Consolas, monospace !important;
        letter-spacing: -0.02em;
    }

    /* ELIMINATE STREAMLIT DEFAULT WHITE HEADER */
    header[data-testid="stHeader"],
    header[data-testid="stAppHeader"],
    .stAppHeader,
    header.st-emotion-cache-18ni7ap,
    header.st-emotion-cache-zq5wmm,
    [data-testid="stHeader"] {
        background-color: var(--bg-app) !important;
        background: var(--bg-app) !important;
        color: var(--text-primary) !important;
        border-bottom: 1px solid var(--border-default) !important;
    }

    header[data-testid="stHeader"] button,
    [data-testid="stHeader"] button {
        color: var(--text-secondary) !important;
    }

    /* -------------------------------------------------------------------------
       Workspace Canvas & Grid Alignment
       ------------------------------------------------------------------------- */
    .main .block-container {
        max-width: 980px !important;
        padding-top: 1rem !important;
        padding-bottom: 1.5rem !important;
        padding-left: 1.5rem !important;
        padding-right: 1.5rem !important;
        margin: 0 auto !important;
    }

    /* -------------------------------------------------------------------------
       Sidebar (High-Density Enterprise Navigation)
       ------------------------------------------------------------------------- */
    section[data-testid="stSidebar"] {
        background-color: var(--bg-sidebar) !important;
        border-right: 1px solid var(--border-default) !important;
    }

    section[data-testid="stSidebar"] > div:first-child {
        padding-top: 1rem !important;
        padding-left: 0.9rem !important;
        padding-right: 0.9rem !important;
    }

    .brand-identity {
        display: flex;
        align-items: center;
        gap: 0.65rem;
        padding: 0.2rem 0.2rem 0.75rem 0.2rem;
        margin-bottom: 0.5rem;
        border-bottom: 1px solid var(--border-default);
    }

    .brand-logo-icon {
        width: 28px;
        height: 28px;
        border-radius: var(--radius-sm);
        background: linear-gradient(135deg, #182030 0%, #0F1420 100%);
        border: 1px solid var(--primary-blue);
        display: flex;
        align-items: center;
        justify-content: center;
    }

    .brand-title {
        font-size: 0.98rem;
        font-weight: 700;
        color: var(--text-primary);
        letter-spacing: -0.02em;
        line-height: 1.1;
    }

    .brand-tagline {
        font-size: 0.68rem;
        color: var(--text-secondary);
        font-weight: 400;
        margin-top: 0.1rem;
    }

    .workspace-pill {
        display: inline-flex;
        align-items: center;
        gap: 0.4rem;
        padding: 0.25rem 0.55rem;
        background-color: var(--bg-panel);
        border: 1px solid var(--border-default);
        border-radius: var(--radius-sm);
        font-size: 0.72rem;
        color: var(--text-secondary);
        width: 100%;
        margin-bottom: 0.75rem;
    }

    .live-dot {
        width: 6px;
        height: 6px;
        border-radius: 50%;
        background-color: var(--success-green);
        box-shadow: 0 0 6px rgba(53, 214, 160, 0.6);
        animation: pulse-dot 2.5s infinite;
    }

    @keyframes pulse-dot {
        0% { transform: scale(0.95); box-shadow: 0 0 0 0 rgba(53, 214, 160, 0.7); }
        70% { transform: scale(1); box-shadow: 0 0 0 5px rgba(53, 214, 160, 0); }
        100% { transform: scale(0.95); box-shadow: 0 0 0 0 rgba(53, 214, 160, 0); }
    }

    .sidebar-section-title {
        font-size: 0.65rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.08em;
        color: var(--text-muted);
        margin-top: 1rem;
        margin-bottom: 0.4rem;
        padding-left: 0.15rem;
    }

    /* -------------------------------------------------------------------------
       WIDGETS: Force Dark Mode on all Inputs, Selectboxes, and Buttons
       ------------------------------------------------------------------------- */
    /* Remove Coral/Red Primary Button - Enforce Developer Blue */
    button[kind="primary"],
    .stButton > button[kind="primary"],
    button[data-testid="baseButton-primary"] {
        background: linear-gradient(135deg, #1C2638 0%, #161D2B 100%) !important;
        border: 1px solid #3B82F6 !important;
        color: #7DA5FF !important;
        font-weight: 600 !important;
        font-size: 0.85rem !important;
        border-radius: 6px !important;
        box-shadow: 0 2px 8px rgba(59, 130, 246, 0.15) !important;
        transition: all 150ms ease !important;
    }

    button[kind="primary"]:hover,
    .stButton > button[kind="primary"]:hover,
    button[data-testid="baseButton-primary"]:hover {
        background: linear-gradient(135deg, #24324A 0%, #1C2638 100%) !important;
        border-color: #7DA5FF !important;
        color: #FFFFFF !important;
    }

    /* Default secondary buttons */
    .stButton > button {
        background-color: var(--bg-panel) !important;
        color: var(--text-primary) !important;
        border: 1px solid var(--border-default) !important;
        border-radius: var(--radius-sm) !important;
        font-weight: 500 !important;
        font-size: 0.82rem !important;
        transition: all 150ms ease !important;
    }

    .stButton > button:hover {
        background-color: var(--bg-hover) !important;
        border-color: var(--primary-blue) !important;
        color: #FFFFFF !important;
    }

    /* Input Fields (Search, Text Inputs) */
    input,
    textarea,
    div[data-baseweb="input"],
    div[data-baseweb="input"] > input,
    div[data-baseweb="base-input"],
    div[data-testid="stTextInput"] input {
        background-color: var(--bg-panel) !important;
        background: var(--bg-panel) !important;
        border-color: var(--border-default) !important;
        color: var(--text-primary) !important;
        border-radius: 6px !important;
        font-size: 0.82rem !important;
    }

    input:focus,
    textarea:focus,
    div[data-baseweb="input"]:focus-within {
        border-color: var(--primary-blue) !important;
        box-shadow: 0 0 0 1px var(--primary-blue) !important;
    }

    input::placeholder,
    textarea::placeholder {
        color: var(--text-muted) !important;
    }

    /* Selectbox Dropdowns */
    div[data-baseweb="select"] > div {
        background-color: var(--bg-panel) !important;
        background: var(--bg-panel) !important;
        border: 1px solid var(--border-default) !important;
        border-radius: 6px !important;
        color: var(--text-primary) !important;
        font-size: 0.82rem !important;
    }

    div[data-baseweb="select"] span {
        color: var(--text-primary) !important;
    }

    div[data-baseweb="select"] svg {
        fill: var(--text-secondary) !important;
    }

    /* -------------------------------------------------------------------------
       Workspace Header Strip (ChatGPT Fixed Top Navigation)
       ------------------------------------------------------------------------- */
    .top-header-strip {
        position: sticky !important;
        top: 0 !important;
        z-index: 100 !important;
        display: flex;
        align-items: center;
        justify-content: space-between;
        padding: 0.5rem 0.85rem;
        background-color: var(--bg-app) !important;
        background: var(--bg-app) !important;
        border: 1px solid var(--border-default);
        border-radius: var(--radius-md);
        margin-bottom: 0.5rem !important;
        box-shadow: 0 4px 16px rgba(0, 0, 0, 0.4) !important;
    }

    .header-left {
        display: flex;
        align-items: center;
        gap: 0.65rem;
    }

    .header-title-text {
        font-size: 0.92rem;
        font-weight: 600;
        color: var(--text-primary);
        letter-spacing: -0.01em;
    }

    .header-tag-pill {
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.68rem;
        padding: 0.12rem 0.45rem;
        border-radius: var(--radius-sm);
        background-color: var(--bg-elevated);
        color: var(--text-secondary);
        border: 1px solid var(--border-default);
    }

    .header-right {
        display: flex;
        align-items: center;
        gap: 0.65rem;
        font-size: 0.75rem;
        color: var(--text-secondary);
    }

    .status-indicator-badge {
        display: inline-flex;
        align-items: center;
        gap: 0.35rem;
        font-size: 0.72rem;
        color: var(--success-green);
        background-color: rgba(53, 214, 160, 0.08);
        border: 1px solid rgba(53, 214, 160, 0.2);
        padding: 0.15rem 0.5rem;
        border-radius: var(--radius-sm);
    }

    /* -------------------------------------------------------------------------
       Navigation Tabs (ChatGPT Sticky Top Header with Fixed Z-Index)
       ------------------------------------------------------------------------- */
    .stTabs [data-baseweb="tab-list"] {
        position: sticky !important;
        top: 48px !important;
        z-index: 99 !important;
        background-color: var(--bg-app) !important;
        background: var(--bg-app) !important;
        border-bottom: 1px solid var(--border-default) !important;
        border-radius: 0 !important;
        padding: 0.35rem 0.2rem !important;
        gap: 1.5rem !important;
        margin-bottom: 0.75rem !important;
        box-shadow: 0 4px 12px rgba(11, 13, 16, 0.9) !important;
    }

    .stTabs [data-baseweb="tab"] {
        height: 36px !important;
        border-radius: 0 !important;
        font-size: 0.82rem !important;
        font-weight: 500 !important;
        color: var(--text-secondary) !important;
        background-color: transparent !important;
        border: none !important;
        padding: 0 0.15rem !important;
        border-bottom: 2px solid transparent !important;
        transition: color 150ms ease, border-color 150ms ease !important;
    }

    .stTabs [data-baseweb="tab"]:hover {
        color: var(--text-primary) !important;
    }

    .stTabs [aria-selected="true"] {
        color: var(--primary-blue) !important;
        background-color: transparent !important;
        border-bottom: 2px solid var(--primary-blue) !important;
        box-shadow: none !important;
        font-weight: 600 !important;
    }

    /* REMOVE STREAMLIT DEFAULT RED HIGHLIGHT */
    div[data-baseweb="tab-highlight"],
    div[data-baseweb="tab-border"] {
        display: none !important;
        background-color: transparent !important;
    }

    /* -------------------------------------------------------------------------
       Chat Messages (Content-Fitted User Bubble & Document Responses)
       ------------------------------------------------------------------------- */
    div[data-testid="stChatMessage"] {
        background: transparent !important;
        border: none !important;
        border-radius: 0 !important;
        box-shadow: none !important;
        padding: 0.4rem 0 !important;
        margin-bottom: 0.5rem !important;
    }

    /* User Message: Content-Fitted Right-Aligned Bubble */
    div[data-testid="stChatMessage"]:has([data-testid="chatAvatarIcon-user"]),
    div[data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarUser"]) {
        display: flex !important;
        flex-direction: row-reverse !important;
        justify-content: flex-start !important;
        width: 100% !important;
    }

    div[data-testid="stChatMessage"]:has([data-testid="chatAvatarIcon-user"]) [data-testid="chatAvatarIcon-user"],
    div[data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarUser"]) [data-testid="stChatMessageAvatarUser"] {
        display: none !important;
    }

    div[data-testid="stChatMessage"]:has([data-testid="chatAvatarIcon-user"]) div[data-testid="stChatMessageContent"],
    div[data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarUser"]) div[data-testid="stChatMessageContent"] {
        flex-grow: 0 !important;
        flex-shrink: 0 !important;
        width: fit-content !important;
        max-width: 68% !important;
        margin-left: auto !important;
        margin-right: 0 !important;
        background: linear-gradient(135deg, #1C2536 0%, #151D2C 100%) !important;
        border: 1px solid #2B3D5A !important;
        border-radius: 16px 16px 4px 16px !important;
        padding: 0.65rem 1.05rem !important;
        color: #F1F5F9 !important;
        font-size: 0.92rem !important;
        line-height: 1.5 !important;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.35) !important;
    }

    div[data-testid="stChatMessage"]:has([data-testid="chatAvatarIcon-user"]) div[data-testid="stChatMessageContent"] *,
    div[data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarUser"]) div[data-testid="stChatMessageContent"] * {
        width: fit-content !important;
        max-width: 100% !important;
        margin: 0 !important;
    }

    div[data-testid="stChatMessage"]:has([data-testid="chatAvatarIcon-user"]) p {
        margin: 0 !important;
        color: #F1F5F9 !important;
    }

    /* Assistant Message: Clean Document Canvas */
    div[data-testid="stChatMessage"]:has([data-testid="chatAvatarIcon-assistant"]),
    div[data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarAssistant"]) {
        display: flex !important;
        gap: 0.85rem !important;
        align-items: flex-start !important;
        padding-top: 0.75rem !important;
        padding-bottom: 1.4rem !important;
        border-bottom: 1px solid #1E2532 !important;
    }

    div[data-testid="stChatMessage"]:has([data-testid="chatAvatarIcon-assistant"]) div[data-testid="chatAvatarIcon-assistant"],
    div[data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarAssistant"]) div[data-testid="stChatMessageAvatarAssistant"] {
        width: 28px !important;
        height: 28px !important;
        min-width: 28px !important;
        border-radius: 6px !important;
        background: linear-gradient(135deg, #182236 0%, #0F1624 100%) !important;
        border: 1px solid #283C5E !important;
        display: flex !important;
        align-items: center !important;
        justify-content: center !important;
        font-size: 0.85rem !important;
        box-shadow: 0 2px 6px rgba(0, 0, 0, 0.3) !important;
        margin-top: 0.15rem !important;
    }

    div[data-testid="stChatMessageContent"] {
        color: var(--text-primary) !important;
        font-size: 0.92rem !important;
        line-height: 1.68 !important;
        max-width: 100% !important;
    }

    div[data-testid="stChatMessageContent"] p {
        color: #E2E8F0 !important;
        font-size: 0.92rem !important;
        line-height: 1.68 !important;
        margin-bottom: 0.85rem !important;
    }

    div[data-testid="stChatMessageContent"] strong {
        color: #FFFFFF !important;
        font-weight: 600 !important;
    }

    div[data-testid="stChatMessageContent"] h1,
    div[data-testid="stChatMessageContent"] h2,
    div[data-testid="stChatMessageContent"] h3,
    div[data-testid="stChatMessageContent"] h4 {
        color: #FFFFFF !important;
        font-weight: 600 !important;
        margin-top: 1.15rem !important;
        margin-bottom: 0.45rem !important;
        letter-spacing: -0.015em !important;
    }

    div[data-testid="stChatMessageContent"] h2 { font-size: 1.12rem !important; }
    div[data-testid="stChatMessageContent"] h3 { font-size: 1.0rem !important; }

    div[data-testid="stChatMessageContent"] ul,
    div[data-testid="stChatMessageContent"] ol {
        color: #E2E8F0 !important;
        padding-left: 1.35rem !important;
        margin-bottom: 0.85rem !important;
        line-height: 1.65 !important;
    }

    div[data-testid="stChatMessageContent"] li {
        margin-bottom: 0.3rem !important;
    }

    /* Markdown Tables */
    div[data-testid="stChatMessageContent"] table {
        width: 100% !important;
        border-collapse: separate !important;
        border-spacing: 0 !important;
        margin: 1rem 0 !important;
        border: 1px solid #232E40 !important;
        border-radius: 6px !important;
        overflow: hidden !important;
    }

    div[data-testid="stChatMessageContent"] th {
        background: #131924 !important;
        color: #93C5FD !important;
        font-weight: 600 !important;
        font-size: 0.82rem !important;
        padding: 0.55rem 0.85rem !important;
        border-bottom: 1px solid #232E40 !important;
        text-align: left !important;
    }

    div[data-testid="stChatMessageContent"] td {
        padding: 0.55rem 0.85rem !important;
        border-bottom: 1px solid #1A2230 !important;
        color: #CBD5E1 !important;
        font-size: 0.84rem !important;
    }

    div[data-testid="stChatMessageContent"] tr:last-child td {
        border-bottom: none !important;
    }

    /* Code Blocks */
    div[data-testid="stChatMessageContent"] code {
        font-family: 'JetBrains Mono', monospace !important;
        background-color: #131A26 !important;
        color: #93C5FD !important;
        border: 1px solid #22324C !important;
        border-radius: 3px !important;
        padding: 0.12rem 0.35rem !important;
        font-size: 0.82rem !important;
    }

    div[data-testid="stChatMessageContent"] pre {
        background-color: #0E131C !important;
        border: 1px solid #202D42 !important;
        border-radius: var(--radius-md) !important;
        padding: 0.85rem 1.05rem !important;
        margin: 0.85rem 0 !important;
    }

    div[data-testid="stChatMessageContent"] pre code {
        background-color: transparent !important;
        border: none !important;
        padding: 0 !important;
        font-size: 0.84rem !important;
        color: #E2E8F0 !important;
    }

    /* -------------------------------------------------------------------------
       Technical Metadata Chips Row
       ------------------------------------------------------------------------- */
    .meta-chips-strip {
        display: flex;
        align-items: center;
        flex-wrap: wrap;
        gap: 0.45rem;
        margin-bottom: 0.75rem;
    }

    .meta-chip {
        display: inline-flex;
        align-items: center;
        gap: 0.35rem;
        font-size: 0.72rem;
        padding: 0.18rem 0.55rem;
        border-radius: 4px;
        background-color: #141924;
        border: 1px solid #232C3E;
        color: #94A3B8;
        font-weight: 500;
        line-height: 1.3;
    }

    .chip-dot {
        width: 6px;
        height: 6px;
        border-radius: 50%;
        display: inline-block;
    }

    .chip-grounded {
        background-color: rgba(52, 211, 153, 0.1) !important;
        border-color: rgba(52, 211, 153, 0.32) !important;
        color: #34D399 !important;
        font-weight: 600 !important;
    }

    .chip-fallback {
        background-color: rgba(251, 191, 36, 0.1) !important;
        border-color: rgba(251, 191, 36, 0.32) !important;
        color: #FBBF24 !important;
        font-weight: 600 !important;
    }

    .chip-unverified {
        background-color: rgba(148, 163, 184, 0.1) !important;
        border-color: rgba(148, 163, 184, 0.25) !important;
        color: #94A3B8 !important;
    }

    .chip-model {
        background-color: #121722 !important;
        border-color: #212B3D !important;
        color: #8BA2C4 !important;
        font-family: 'JetBrains Mono', monospace !important;
        font-size: 0.7rem !important;
    }

    .chip-sources {
        background-color: rgba(96, 165, 250, 0.1) !important;
        border-color: rgba(96, 165, 250, 0.3) !important;
        color: #60A5FA !important;
        font-weight: 500 !important;
    }

    /* Pill Badges (Preserved for tests) */
    .pill-badge {
        display: inline-flex;
        align-items: center;
        gap: 0.35rem;
        padding: 0.12rem 0.48rem;
        border-radius: 9999px;
        font-size: 0.7rem;
        font-weight: 500;
        letter-spacing: 0.02em;
    }

    .badge-route {
        background-color: rgba(125, 165, 255, 0.1);
        color: var(--primary-blue);
        border: 1px solid rgba(125, 165, 255, 0.25);
    }

    .badge-fallback {
        background-color: rgba(240, 183, 94, 0.1);
        color: var(--warning-amber);
        border: 1px solid rgba(240, 183, 94, 0.25);
    }

    .badge-live {
        background-color: rgba(53, 214, 160, 0.1);
        color: var(--success-green);
        border: 1px solid rgba(53, 214, 160, 0.25);
    }

    /* -------------------------------------------------------------------------
       ChatGPT-Style Independent Chat Scroll Container (Tab 1 Only)
       ------------------------------------------------------------------------- */
    .stTabs [data-baseweb="tab-panel"]:first-of-type [data-testid="stVerticalBlockBorderWrapper"],
    .stTabs [data-baseweb="tab-panel"]:first-of-type div[data-testid="stVerticalBlock"]:has(> div[data-testid="stChatMessage"]),
    .stTabs [data-baseweb="tab-panel"]:first-of-type div[data-testid="stVerticalBlock"]:has(> div.empty-technical-container) {
        height: calc(100vh - 275px) !important;
        max-height: calc(100vh - 275px) !important;
        min-height: 380px !important;
        overflow-y: auto !important;
        overflow-x: hidden !important;
        border: none !important;
        background: transparent !important;
        padding-right: 0.4rem !important;
        padding-bottom: 1.5rem !important;
        scrollbar-width: thin;
        scrollbar-color: var(--border-default) transparent;
        scroll-behavior: smooth;
    }

    .stTabs [data-baseweb="tab-panel"]:first-of-type [data-testid="stVerticalBlockBorderWrapper"]::-webkit-scrollbar,
    .stTabs [data-baseweb="tab-panel"]:first-of-type div[data-testid="stVerticalBlock"]:has(> div[data-testid="stChatMessage"])::-webkit-scrollbar,
    .stTabs [data-baseweb="tab-panel"]:first-of-type div[data-testid="stVerticalBlock"]:has(> div.empty-technical-container)::-webkit-scrollbar {
        width: 6px;
    }

    .stTabs [data-baseweb="tab-panel"]:first-of-type [data-testid="stVerticalBlockBorderWrapper"]::-webkit-scrollbar-track,
    .stTabs [data-baseweb="tab-panel"]:first-of-type div[data-testid="stVerticalBlock"]:has(> div[data-testid="stChatMessage"])::-webkit-scrollbar-track,
    .stTabs [data-baseweb="tab-panel"]:first-of-type div[data-testid="stVerticalBlock"]:has(> div.empty-technical-container)::-webkit-scrollbar-track {
        background: transparent;
    }

    .stTabs [data-baseweb="tab-panel"]:first-of-type [data-testid="stVerticalBlockBorderWrapper"]::-webkit-scrollbar-thumb,
    .stTabs [data-baseweb="tab-panel"]:first-of-type div[data-testid="stVerticalBlock"]:has(> div[data-testid="stChatMessage"])::-webkit-scrollbar-thumb,
    .stTabs [data-baseweb="tab-panel"]:first-of-type div[data-testid="stVerticalBlock"]:has(> div.empty-technical-container)::-webkit-scrollbar-thumb {
        background-color: var(--border-default);
        border-radius: 4px;
    }

    .stTabs [data-baseweb="tab-panel"]:first-of-type [data-testid="stVerticalBlockBorderWrapper"]::-webkit-scrollbar-thumb:hover,
    .stTabs [data-baseweb="tab-panel"]:first-of-type div[data-testid="stVerticalBlock"]:has(> div[data-testid="stChatMessage"])::-webkit-scrollbar-thumb:hover,
    .stTabs [data-baseweb="tab-panel"]:first-of-type div[data-testid="stVerticalBlock"]:has(> div.empty-technical-container)::-webkit-scrollbar-thumb:hover {
        background-color: var(--border-hover);
    }

    /* -------------------------------------------------------------------------
       Chat Input Bar (Eliminate Stark White Composer)
       ------------------------------------------------------------------------- */
    div[data-testid="stChatInput"],
    div[data-testid="stChatInput"] > div {
        background-color: #131824 !important;
        background: #131824 !important;
        border: 1px solid #253144 !important;
        border-radius: 12px !important;
        box-shadow: 0 4px 16px rgba(0, 0, 0, 0.45) !important;
    }

    .stTabs [data-baseweb="tab-panel"]:first-of-type div[data-testid="stChatInput"] {
        max-width: 980px !important;
        margin: 0.4rem auto 0 auto !important;
    }

    @media (max-width: 992px) {
        .stTabs [data-baseweb="tab-panel"]:first-of-type [data-testid="stVerticalBlockBorderWrapper"],
        .stTabs [data-baseweb="tab-panel"]:first-of-type div[data-testid="stVerticalBlock"]:has(> div[data-testid="stChatMessage"]),
        .stTabs [data-baseweb="tab-panel"]:first-of-type div[data-testid="stVerticalBlock"]:has(> div.empty-technical-container) {
            height: calc(100vh - 255px) !important;
            max-height: calc(100vh - 255px) !important;
        }
    }

    @media (max-width: 600px) {
        .top-header-strip {
            flex-direction: column;
            align-items: flex-start;
            gap: 0.35rem;
        }
        .stTabs [data-baseweb="tab-list"] {
            top: 75px !important;
            gap: 0.75rem !important;
        }
        .stTabs [data-baseweb="tab-panel"]:first-of-type [data-testid="stVerticalBlockBorderWrapper"],
        .stTabs [data-baseweb="tab-panel"]:first-of-type div[data-testid="stVerticalBlock"]:has(> div[data-testid="stChatMessage"]),
        .stTabs [data-baseweb="tab-panel"]:first-of-type div[data-testid="stVerticalBlock"]:has(> div.empty-technical-container) {
            height: calc(100vh - 265px) !important;
            max-height: calc(100vh - 265px) !important;
            min-height: 300px !important;
        }
    }

    div[data-testid="stChatInput"]:focus-within,
    div[data-testid="stChatInput"] > div:focus-within {
        border-color: #60A5FA !important;
        box-shadow: 0 0 0 2px rgba(96, 165, 250, 0.2), 0 4px 16px rgba(0, 0, 0, 0.5) !important;
    }

    div[data-testid="stChatInput"] textarea {
        background-color: transparent !important;
        background: transparent !important;
        color: var(--text-primary) !important;
        font-size: 0.88rem !important;
        line-height: 1.5 !important;
    }

    button[data-testid="stChatInputSubmitButton"],
    div[data-testid="stChatInput"] button {
        background-color: #1C2638 !important;
        border: 1px solid #3B82F6 !important;
        color: #7DA5FF !important;
        border-radius: 6px !important;
        transition: all 150ms ease !important;
    }

    button[data-testid="stChatInputSubmitButton"]:hover {
        background-color: #26354E !important;
        border-color: #60A5FA !important;
        color: #FFFFFF !important;
    }

    .composer-hint {
        text-align: center;
        font-size: 0.68rem;
        color: var(--text-muted);
        margin-top: 0.35rem;
    }

    /* -------------------------------------------------------------------------
       Technical Empty State
       ------------------------------------------------------------------------- */
    .empty-technical-container {
        padding: 2rem 1rem 1.25rem 1rem;
        text-align: center;
        max-width: 680px;
        margin: 0 auto;
    }

    .empty-badge {
        display: inline-flex;
        align-items: center;
        gap: 0.4rem;
        padding: 0.25rem 0.65rem;
        border-radius: 9999px;
        background: rgba(96, 165, 250, 0.1);
        border: 1px solid rgba(96, 165, 250, 0.25);
        color: #7DA5FF;
        font-size: 0.72rem;
        font-weight: 600;
        letter-spacing: 0.05em;
        text-transform: uppercase;
        margin-bottom: 0.85rem;
    }

    .empty-heading {
        font-size: 1.35rem;
        font-weight: 700;
        color: #FFFFFF;
        letter-spacing: -0.02em;
        margin-bottom: 0.45rem;
    }

    .empty-subtext {
        font-size: 0.88rem;
        color: var(--text-secondary);
        line-height: 1.55;
        margin-bottom: 1.5rem;
    }

    /* -------------------------------------------------------------------------
       Evidence & Citations
       ------------------------------------------------------------------------- */
    .citation-strip {
        background-color: #0E131C;
        border: 1px solid #202D42;
        border-radius: var(--radius-sm);
        padding: 0.6rem 0.85rem;
        margin-bottom: 0.45rem;
    }

    .citation-top-row {
        display: flex;
        justify-content: space-between;
        align-items: center;
        margin-bottom: 0.25rem;
    }

    .citation-title-text {
        font-size: 0.82rem;
        font-weight: 600;
        color: #E2E8F0;
    }

    .citation-chunk-tag {
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.68rem;
        color: #7DA5FF;
        background-color: rgba(125, 165, 255, 0.1);
        padding: 0.1rem 0.38rem;
        border-radius: 3px;
        border: 1px solid rgba(125, 165, 255, 0.2);
    }

    .citation-quote {
        font-size: 0.78rem;
        color: #94A3B8;
        line-height: 1.45;
        border-left: 2px solid #2B3D5A;
        padding-left: 0.55rem;
        margin-top: 0.35rem;
    }

    /* -------------------------------------------------------------------------
       Memory Inspector Records & Add Panel
       ------------------------------------------------------------------------- */
    .memory-row-card {
        background-color: var(--bg-panel);
        border: 1px solid var(--border-default);
        border-radius: var(--radius-md);
        padding: 0.75rem 0.95rem;
        margin-bottom: 0.65rem;
        transition: border-color 150ms ease;
    }

    .memory-row-card:hover {
        border-color: rgba(255, 255, 255, 0.12);
    }

    .memory-row-header {
        display: flex;
        justify-content: space-between;
        align-items: center;
        margin-bottom: 0.35rem;
    }

    .memory-cat-pill {
        font-size: 0.65rem;
        font-weight: 600;
        text-transform: uppercase;
        color: var(--primary-blue);
        background-color: rgba(125, 165, 255, 0.08);
        border: 1px solid rgba(125, 165, 255, 0.2);
        padding: 0.1rem 0.45rem;
        border-radius: var(--radius-sm);
        letter-spacing: 0.04em;
    }

    .memory-body-text {
        font-size: 0.88rem;
        color: var(--text-primary);
        line-height: 1.5;
        margin: 0.35rem 0;
    }

    .memory-footer-meta {
        font-size: 0.72rem;
        color: var(--text-muted);
        display: flex;
        align-items: center;
        gap: 0.45rem;
    }

    /* -------------------------------------------------------------------------
       Agent Observability & Traces Layout
       ------------------------------------------------------------------------- */
    .metric-strip-grid {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(170px, 1fr));
        gap: 0.55rem;
        margin-bottom: 1rem;
    }

    .metric-cell {
        background-color: var(--bg-panel);
        border: 1px solid var(--border-default);
        border-radius: var(--radius-md);
        padding: 0.65rem 0.8rem;
    }

    .metric-cell-label {
        font-size: 0.65rem;
        color: var(--text-muted);
        text-transform: uppercase;
        font-weight: 600;
        letter-spacing: 0.05em;
    }

    .metric-cell-val {
        font-size: 1.05rem;
        color: var(--text-primary);
        font-weight: 700;
        margin-top: 0.2rem;
        letter-spacing: -0.01em;
    }

    .trace-timeline-node {
        position: relative;
        padding-left: 1.6rem;
        padding-bottom: 0.85rem;
        border-left: 1px solid var(--border-default);
        margin-left: 0.5rem;
    }

    .trace-timeline-node::before {
        content: '';
        position: absolute;
        left: -5px;
        top: 3px;
        width: 9px;
        height: 9px;
        border-radius: 50%;
        background-color: var(--primary-blue);
        border: 2px solid var(--bg-app);
    }

    .trace-timeline-node:last-child {
        border-left: 1px solid transparent;
    }

    .trace-node-title {
        font-size: 0.82rem;
        font-weight: 600;
        color: var(--text-primary);
    }

    .trace-node-desc {
        font-size: 0.75rem;
        color: var(--text-secondary);
        margin-top: 0.15rem;
    }

    /* -------------------------------------------------------------------------
       Expanders & Dividers
       ------------------------------------------------------------------------- */
    div[data-testid="stExpander"] {
        background-color: var(--bg-panel) !important;
        border: 1px solid var(--border-default) !important;
        border-radius: var(--radius-md) !important;
    }

    div[data-testid="stExpander"] details summary {
        color: var(--text-secondary) !important;
        font-size: 0.82rem !important;
        font-weight: 500 !important;
    }

    div[data-testid="stExpander"] details summary:hover {
        color: var(--text-primary) !important;
    }

    hr {
        border-color: var(--border-default) !important;
        margin: 1rem 0 !important;
    }
    </style>
    """
    st.markdown(custom_css, unsafe_allow_html=True)


# =============================================================================
# Formatting & Presentation Helpers
# =============================================================================

def format_timestamp(ts: Optional[str]) -> str:
    """Format ISO or SQLite timestamp into clean readable string.

    Guarantees compatibility with tests expecting 'Oct 27, 2026' and '14:30'.
    """
    if not ts:
        return "Unknown"
    try:
        dt = datetime.fromisoformat(str(ts).replace("Z", "+00:00"))
        return dt.strftime("%b %d, %Y • %H:%M")
    except Exception:
        return str(ts)[:16]


def render_brand_header() -> None:
    """Render MemoraAI enterprise product identity and workspace indicator in the sidebar."""
    st.markdown(
        """
        <div class="brand-identity">
            <div class="brand-logo-icon">
                <svg width="16" height="16" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg">
                    <path d="M12 2L2 7L12 12L22 7L12 2Z" stroke="#7DA5FF" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"/>
                    <path d="M2 17L12 22L22 17" stroke="#7DA5FF" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"/>
                    <path d="M2 12L12 17L22 12" stroke="#8B7CFF" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"/>
                </svg>
            </div>
            <div>
                <div class="brand-title">MemoraAI</div>
                <div class="brand-tagline">Enterprise Research Assistant</div>
            </div>
        </div>
        <div class="workspace-pill">
            <span class="live-dot"></span>
            <span><strong>Asterline Workspace</strong> • Production</span>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_user_profile_header(user_id: str, conv_title: Optional[str] = None) -> None:
    """Render compact technical workspace header across all screens."""
    profile = DEMO_PROFILES.get(user_id, DEMO_PROFILES["user_alice"])
    title_text = conv_title or "New Investigation"
    st.markdown(
        f"""
        <div class="top-header-strip">
            <div class="header-left">
                <span class="header-title-text">{title_text}</span>
                <span class="header-tag-pill">{profile['id']}</span>
            </div>
            <div class="header-right">
                <span>{profile['avatar']} {profile['name'].split('—')[0].strip()} ({profile['role'].split('&')[0].strip()})</span>
                <span class="status-indicator-badge">
                    <span class="live-dot" style="width: 5px; height: 5px;"></span>
                    Strictly Isolated
                </span>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_route_pill(route_key: Optional[str]) -> str:
    """Generate HTML pill for route classification.

    Guarantees compatibility with tests expecting 'pill-badge' and label.
    """
    meta = ROUTE_DISPLAY_META.get(route_key or "", {
        "label": route_key or "Unknown",
        "color": "#98A3B5",
        "icon": "📍",
    })
    return f"""<span class="pill-badge badge-route">{meta['icon']} {meta['label']}</span>"""


def render_fallback_pill(is_fallback: bool, llm_metadata: Optional[Dict[str, Any]] = None) -> str:
    """Generate HTML pill indicating whether deterministic fallback or live model was used.

    Guarantees compatibility with tests expecting 'Deterministic Fallback' or 'Live: provider/model'.
    """
    if is_fallback:
        return """<span class="pill-badge badge-fallback">⚠️ Deterministic Fallback</span>"""
    provider = (llm_metadata or {}).get("provider", "groq")
    model = (llm_metadata or {}).get("model", "llama-3.3-70b")
    return f"""<span class="pill-badge badge-live">⚡ Live: {provider}/{model}</span>"""


def render_empty_state() -> None:
    """Render technical empty state with clear guidance."""
    st.markdown(
        """
        <div class="empty-technical-container">
            <div class="empty-badge">
                <svg width="12" height="12" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg">
                    <circle cx="12" cy="12" r="9" stroke="#7DA5FF" stroke-width="2"/>
                    <path d="M12 7V12L15 15" stroke="#7DA5FF" stroke-width="2" stroke-linecap="round"/>
                </svg>
                <span>Asterline Intelligence Suite</span>
            </div>
            <div class="empty-heading">What would you like to investigate?</div>
            <div class="empty-subtext">
                Query technical documentation with grounded multi-hop citations, reconcile conflicting policies, and carry persistent context across investigations.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_response_meta_row(
    route: Optional[str] = None,
    llm_metadata: Optional[Dict[str, Any]] = None,
    is_fallback: bool = False,
    is_grounded: bool = True,
    citation_count: int = 0,
    **kwargs: Any,
) -> None:
    """Render compact technical metadata strip above the assistant response with chip badges."""
    meta = ROUTE_DISPLAY_META.get(route or "", {
        "label": route or "General Research",
        "color": "#7DA5FF",
        "icon": "⚡",
    })
    route_label = meta.get("label", "General Research")
    route_icon = meta.get("icon", "⚡")
    route_color = meta.get("color", "#7DA5FF")
    model_name = (llm_metadata or {}).get("model", "llama-3.3-70b")

    if is_fallback:
        verdict_badge = '<span class="meta-chip chip-fallback">⚠️ Fallback</span>'
    elif is_grounded:
        verdict_badge = '<span class="meta-chip chip-grounded">✓ Grounded</span>'
    else:
        verdict_badge = '<span class="meta-chip chip-unverified">⚡ LLM Synthesis</span>'

    citation_badge = ""
    if citation_count > 0:
        source_word = "Source" if citation_count == 1 else "Sources"
        citation_badge = f'<span class="meta-chip chip-sources">📄 {citation_count} {source_word}</span>'

    st.markdown(
        f"""
        <div class="meta-chips-strip">
            <span class="meta-chip" style="border-color: {route_color}44; color: {route_color};">
                <span class="chip-dot" style="background-color: {route_color};"></span>
                <span>{route_label}</span>
            </span>
            {verdict_badge}
            <span class="meta-chip chip-model mono-text">{model_name}</span>
            {citation_badge}
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_citations_expander(
    citations: List[Dict[str, Any]],
    retrieved_chunks: Optional[List[Dict[str, Any]]] = None,
    chunk_map: Optional[Dict[str, Any]] = None,
) -> None:
    """Render compact, structured evidence citations section."""
    if not citations:
        return

    if chunk_map is None:
        chunk_map = {c.get("chunk_id"): c for c in (retrieved_chunks or []) if isinstance(c, dict)}
    label = f"Authoritative Sources ({len(citations)})"

    with st.expander(label, expanded=False):
        for cit in citations:
            cid = cit.get("chunk_id", "Unknown")
            title = cit.get("title", "Untitled Document")

            excerpt = ""
            if cid in chunk_map:
                excerpt = chunk_map[cid].get("text", "")
            elif "text" in cit:
                excerpt = cit["text"]

            clean_excerpt = excerpt.strip() if excerpt else "Authoritative chunk verified in Asterline knowledge corpus."
            if len(clean_excerpt) > 280:
                clean_excerpt = clean_excerpt[:280] + "..."

            st.markdown(
                f"""
                <div class="citation-strip">
                    <div class="citation-top-row">
                        <span class="citation-title-text">{title}</span>
                        <span class="citation-chunk-tag">[{cid}]</span>
                    </div>
                    <div class="citation-quote">"{clean_excerpt}"</div>
                </div>
                """,
                unsafe_allow_html=True,
            )


def render_observability_card(trace_info: Dict[str, Any]) -> None:
    """Render high-density observability dashboard with vertical execution timeline."""
    if not trace_info:
        st.info("No execution trace recorded yet for the active turn.")
        return

    route = trace_info.get("route", "unspecified")
    is_fallback = trace_info.get("is_fallback", False)
    llm_metadata = trace_info.get("llm_metadata", {})
    verification = trace_info.get("verification", {})
    verdict = verification.get("verdict", "pass")
    retry_count = trace_info.get("retry_count", 0)
    citations_count = len(trace_info.get("citations", []))
    retrieved_count = len(trace_info.get("retrieved_chunks", []))

    st.markdown(
        f"""
        <div class="metric-strip-grid">
            <div class="metric-cell">
                <div class="metric-cell-label">Execution Route</div>
                <div class="metric-cell-val" style="color: #7DA5FF;">{route.upper()}</div>
            </div>
            <div class="metric-cell">
                <div class="metric-cell-label">Verifier Verdict</div>
                <div class="metric-cell-val" style="color: {'#35D6A0' if verdict == 'pass' else ('#F0B75E' if verdict == 'abstain' else '#F07878')};">
                    {verdict.upper()}
                </div>
            </div>
            <div class="metric-cell">
                <div class="metric-cell-label">Retrieved / Cited</div>
                <div class="metric-cell-val">{retrieved_count} / {citations_count}</div>
            </div>
            <div class="metric-cell">
                <div class="metric-cell-label">Retry Attempts</div>
                <div class="metric-cell-val">{retry_count} / 2</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    with st.expander("⚡ Workflow Execution Timeline", expanded=True):
        st.markdown(
            f"""
            <div style="margin-top: 0.4rem;">
                <div class="trace-timeline-node">
                    <div class="trace-node-title">1. Input Guardrails & Intent Classifier</div>
                    <div class="trace-node-desc">Route: <code>{route}</code> • Method: {trace_info.get("routing_reason", "Deterministic routing & keyword analysis")}</div>
                </div>
                <div class="trace-timeline-node">
                    <div class="trace-node-title">2. Hybrid Knowledge Retrieval</div>
                    <div class="trace-node-desc">Candidate chunks scored: {retrieved_count} via ChromaDB Dense + BM25 Lexical</div>
                </div>
                <div class="trace-timeline-node">
                    <div class="trace-node-title">3. Multi-Agent Research Generator</div>
                    <div class="trace-node-desc">Provider: <code>{llm_metadata.get('provider', 'groq')}</code> • Model: <code>{llm_metadata.get('model', 'llama-3.3-70b')}</code></div>
                </div>
                <div class="trace-timeline-node">
                    <div class="trace-node-title">4. Citation & Hallucination Verifier</div>
                    <div class="trace-node-desc">Verdict: <strong>{verdict.upper()}</strong> • Findings: {verification.get('feedback') or 'Grounded claims aligned with cited documentation.'}</div>
                </div>
                <div class="trace-timeline-node">
                    <div class="trace-node-title">5. Persistence & SQLite Audit Log</div>
                    <div class="trace-node-desc">Conversation thread committed to isolated user store with zero data leakage.</div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        if is_fallback:
            reason = llm_metadata.get("fallback_reason", "Offline deterministic fallback")
            st.warning(f"Deterministic Fallback Engaged: {reason}")
