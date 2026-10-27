"""Conditional router for Asterline Research Assistant.

Directs incoming user queries to the minimum necessary execution workflow:
- 'conversational': Greetings, bot identity (bypasses RAG and model calls)
- 'clarification': Underspecified or ambiguous queries
- 'memory_ops': Commands to remember, view, or forget personal context
- 'investigation': Incidents (INC-*), multi-version comparisons, rollouts, conflicts
- 'factual': Single-topic specifications, pricing, limits, policies
"""

from dataclasses import dataclass
import re
from typing import Literal, Tuple

RouteType = Literal["factual", "investigation", "memory_ops", "conversational", "clarification"]


@dataclass
class RoutingDecision:
    """Routing choice and operational justification."""

    route: RouteType
    reason: str


# Conversational greeting keywords
GREETING_PATTERNS = [
    re.compile(r"^\s*(?:hi|hello|hey|good\s+morning|good\s+evening|howdy|greetings)\b", re.IGNORECASE),
    re.compile(r"\b(?:who\s+are\s+you|what\s+can\s+you\s+do|what\s+is\s+your\s+name|how\s+can\s+you\s+help)\b", re.IGNORECASE),
    re.compile(r"^\s*help\s*\??$", re.IGNORECASE),
]

# Explicit memory command patterns
MEMORY_COMMAND_PATTERNS = [
    re.compile(r"\b(?:remember|please\s+remember|take\s+note|note\s+that)\b", re.IGNORECASE),
    re.compile(r"\b(?:forget|delete\s+memory|remove\s+memory|erase\s+memory)\b", re.IGNORECASE),
    re.compile(r"\b(?:what\s+do\s+you\s+(?:know|remember)\s+about\s+me|list\s+my\s+memories|my\s+preferences)\b", re.IGNORECASE),
]

# Investigation and multi-document query triggers
INVESTIGATION_PATTERNS = [
    re.compile(r"\binc-\d{4}-\d{2}\b", re.IGNORECASE),  # e.g. INC-2025-12, INC-2025-08
    re.compile(r"\b(?:incident|outage|root\s+cause|post-mortem|postmortem|downtime)\b", re.IGNORECASE),
    re.compile(r"\b(?:supersede|superseded|supersedes|conflict|discrepancy|differs|differ|difference|differences)\b", re.IGNORECASE),
    re.compile(r"\b(?:compare|comparison|timeline|historical|history|versus|vs\.?|evolution)\b", re.IGNORECASE),
    re.compile(r"\b(?:why\s+did|what\s+caused|how\s+was\s+it\s+resolved|patch\s+\d+\.\d+)\b", re.IGNORECASE),
]


def determine_route(user_message: str) -> RoutingDecision:
    """Analyze query and select the minimal necessary execution branch."""
    cleaned = user_message.strip()

    # 1. Conversational greetings (checked before short length check to allow "hi", "hey")
    for p in GREETING_PATTERNS:
        if p.search(cleaned):
            return RoutingDecision(
                route="conversational",
                reason="Direct conversational greeting; bypassing RAG and documentation research.",
            )

    # 2. Ambiguous or empty
    if len(cleaned) < 3 or cleaned in ("?", "help me", "..."):
        return RoutingDecision(
            route="clarification",
            reason="Query is ambiguous or underspecified; requesting clarification.",
        )

    # 3. Explicit memory commands
    for p in MEMORY_COMMAND_PATTERNS:
        if p.search(cleaned):
            return RoutingDecision(
                route="memory_ops",
                reason="User initiated a memory lifecycle command (remember, list, or forget).",
            )

    # 4. Complex Investigation triggers
    for p in INVESTIGATION_PATTERNS:
        if p.search(cleaned):
            return RoutingDecision(
                route="investigation",
                reason="Query involves incident root cause, multi-document synthesis, or temporal conflict.",
            )

    # 5. Default: Focused documentation lookup
    return RoutingDecision(
        route="factual",
        reason="Direct documentation inquiry regarding company policies, pricing, APIs, or limits.",
    )
