"""LangGraph multi-agent workflow orchestration for Asterline Research Assistant.

Coordinates:
- Input guardrails & trusted session validation
- Memory retrieval & conditional routing
- Documentation & Investigation research agents
- Verification & bounded retry loop
- Output guardrails & SQLite conversation persistence
"""

from typing import Any, Dict, List, Literal, Optional
from langgraph.graph import END, START, StateGraph
from asterline.agents.documentation import DocumentationAgent
from asterline.agents.investigation import InvestigationAgent
from asterline.agents.llm import LLMClient
from asterline.agents.memory_manager import MemoryManagerAgent
from asterline.agents.verification import VerificationAgent
from asterline.config.settings import get_settings
from asterline.graph.routing import determine_route
from asterline.graph.state import Citation, ResearchAssistantState, RetrievedChunk, VerificationResult
from asterline.guardrails.input import validate_input
from asterline.guardrails.output import validate_output
from asterline.memory.service import MemoryService
from asterline.rag.retriever import HybridRetriever


class WorkflowBuilder:
    """Builds and compiles the multi-agent LangGraph execution graph."""

    def __init__(
        self,
        retriever: HybridRetriever,
        memory_service: MemoryService,
        llm_client: Optional[LLMClient] = None,
    ):
        self.retriever = retriever
        self.memory_service = memory_service
        self.llm = llm_client or LLMClient()
        self.settings = get_settings()

        # Agents
        self.doc_agent = DocumentationAgent(self.retriever, self.llm)
        self.invest_agent = InvestigationAgent(self.retriever, self.llm)
        self.mem_agent = MemoryManagerAgent(self.memory_service)
        self.verification_agent = VerificationAgent()

    # -------------------------------------------------------------------------
    # Node Callables
    # -------------------------------------------------------------------------

    def input_guardrails_node(self, state: ResearchAssistantState) -> Dict[str, Any]:
        """Validate untrusted input and session context."""
        user_msg = state.get("user_message", "")
        uid = state.get("user_id", "")
        cid = state.get("conversation_id", "")

        result = validate_input(user_msg, user_id=uid, conversation_id=cid)
        if not result.is_valid:
            return {
                "error": result.error_message,
                "final_response": result.error_message or "Invalid request.",
            }

        # Load active user memories if not already loaded
        active_memories = state.get("active_user_memories", [])
        if not active_memories:
            raw_mems = self.memory_service.list_user_memories(uid)
            active_memories = [m.model_dump() for m in raw_mems]

        return {
            "user_message": result.sanitized_message,
            "active_user_memories": active_memories,
            "retry_count": state.get("retry_count", 0),
        }

    def router_node(self, state: ResearchAssistantState) -> Dict[str, Any]:
        """Determine the minimum necessary execution workflow."""
        decision = determine_route(state["user_message"])
        return {
            "route": decision.route,
            "routing_reason": decision.reason,
        }

    def conversational_node(self, state: ResearchAssistantState) -> Dict[str, Any]:
        """Handle greetings or bot identity inquiries without RAG overhead."""
        greeting_text = (
            "Hello! I am the Asterline Labs Research Assistant. "
            "I can help you look up product specifications, pricing, APIs, investigate incidents (e.g., INC-2025-12), "
            "or remember your workspace context across sessions."
        )
        return {
            "draft_response": greeting_text,
            "final_response": greeting_text,
            "citations": [],
            "retrieved_chunks": [],
        }

    def clarification_node(self, state: ResearchAssistantState) -> Dict[str, Any]:
        """Ask for clarification on ambiguous or underspecified messages."""
        msg = (
            "Your question is underspecified. Could you please provide more details? "
            "For example, you can ask about pricing plans, ingestion limits, alert rules, or incident timelines."
        )
        return {
            "draft_response": msg,
            "final_response": msg,
            "citations": [],
            "retrieved_chunks": [],
        }

    def memory_manager_node(self, state: ResearchAssistantState) -> Dict[str, Any]:
        """Execute user memory lifecycle operation."""
        result = self.mem_agent.handle_request(
            user_id=state["user_id"],
            user_message=state["user_message"],
        )
        return {
            "draft_response": result["response"],
            "final_response": result["response"],
            "memory_operations_executed": [result["operation"]],
            "citations": [],
            "retrieved_chunks": [],
        }

    def documentation_agent_node(self, state: ResearchAssistantState) -> Dict[str, Any]:
        """Execute documentation research."""
        feedback = None
        if state.get("verification") and state["verification"].get("feedback"):
            feedback = state["verification"]["feedback"]

        res = self.doc_agent.research(
            query=state["user_message"],
            user_memories=state.get("active_user_memories"),
            feedback=feedback,
        )
        return {
            "draft_response": res["draft_response"],
            "citations": [c.model_dump() for c in res["citations"]],
            "retrieved_chunks": [c.model_dump() for c in res["retrieved_chunks"]],
        }

    def investigation_agent_node(self, state: ResearchAssistantState) -> Dict[str, Any]:
        """Execute multi-document investigation."""
        feedback = None
        if state.get("verification") and state["verification"].get("feedback"):
            feedback = state["verification"]["feedback"]

        res = self.invest_agent.investigate(
            query=state["user_message"],
            user_memories=state.get("active_user_memories"),
            feedback=feedback,
        )
        return {
            "draft_response": res["draft_response"],
            "citations": [c.model_dump() for c in res["citations"]],
            "retrieved_chunks": [c.model_dump() for c in res["retrieved_chunks"]],
        }

    def verification_node(self, state: ResearchAssistantState) -> Dict[str, Any]:
        """Audit draft response and citations against evidence."""
        cits = [Citation(**c) for c in state.get("citations", [])]
        chunks = [RetrievedChunk(**c) for c in state.get("retrieved_chunks", [])]

        verdict_obj = self.verification_agent.verify(
            draft_response=state.get("draft_response", ""),
            citations=cits,
            retrieved_chunks=chunks,
            user_id=state["user_id"],
            is_investigation=(state.get("route") == "investigation"),
        )
        return {"verification": verdict_obj.model_dump()}

    def output_guardrails_node(self, state: ResearchAssistantState) -> Dict[str, Any]:
        """Apply output sanitization, citation checks, and final response formatting."""
        # If already errored or direct conversational/memory response
        if state.get("error"):
            return {"final_response": state.get("final_response", state["error"])}

        draft = state.get("draft_response", "")
        cits = [Citation(**c) for c in state.get("citations", [])]
        chunks = [RetrievedChunk(**c) for c in state.get("retrieved_chunks", [])]

        guard_res = validate_output(
            response_text=draft,
            citations=cits,
            retrieved_chunks=chunks,
            current_user_id=state["user_id"],
        )

        if not guard_res.is_valid:
            # If citations failed guardrails and retries exhausted, deliver honest fallback
            final_ans = f"Verification note: {guard_res.error_message}\n\nEvidence Summary:\n" + "\n".join(
                f"- [{c.chunk_id}] {c.title}: {c.text[:180]}..." for c in chunks[:3]
            )
            return {
                "final_response": final_ans,
                "citations": [],
            }

        return {
            "final_response": guard_res.validated_response,
            "citations": [c.model_dump() for c in guard_res.validated_citations],
        }

    def persistence_node(self, state: ResearchAssistantState) -> Dict[str, Any]:
        """Persist user turn and assistant response into SQLite conversation storage."""
        uid = state["user_id"]
        cid = state["conversation_id"]

        try:
            # Ensure conversation exists
            self.memory_service.get_or_create_conversation(uid, cid)

            # Save user turn
            self.memory_service.append_message(
                conversation_id=cid,
                user_id=uid,
                role="user",
                content=state["user_message"],
            )

            # Save assistant turn
            self.memory_service.append_message(
                conversation_id=cid,
                user_id=uid,
                role="assistant",
                content=state["final_response"],
                citations=state.get("citations"),
                route_taken=state.get("route"),
            )
        except Exception as e:
            # Persistence logging error should not corrupt state
            return {"error": f"Persistence warning: {str(e)}"}

        return {}

    # -------------------------------------------------------------------------
    # Conditional Edge Routing Functions
    # -------------------------------------------------------------------------

    def check_input_guardrail(
        self, state: ResearchAssistantState
    ) -> Literal["router_node", "output_guardrails_node"]:
        """Route to router if input valid; jump to output if invalid."""
        if state.get("error"):
            return "output_guardrails_node"
        return "router_node"

    def route_decision(
        self, state: ResearchAssistantState
    ) -> Literal[
        "conversational_node",
        "clarification_node",
        "memory_manager_node",
        "documentation_agent_node",
        "investigation_agent_node",
    ]:
        """Route to chosen node based on state['route']."""
        route = state.get("route", "factual")
        if route == "conversational":
            return "conversational_node"
        elif route == "clarification":
            return "clarification_node"
        elif route == "memory_ops":
            return "memory_manager_node"
        elif route == "investigation":
            return "investigation_agent_node"
        return "documentation_agent_node"

    def check_verification(
        self, state: ResearchAssistantState
    ) -> Literal[
        "output_guardrails_node",
        "documentation_agent_node",
        "investigation_agent_node",
    ]:
        """Evaluate verification verdict and apply bounded retries."""
        verdict = state.get("verification", {}).get("verdict", "pass")
        retry_count = state.get("retry_count", 0)
        max_retries = self.settings.max_investigation_retries

        if verdict == "revise" and retry_count < max_retries:
            # Increment retry count in state and loop back to the agent
            state["retry_count"] = retry_count + 1
            if state.get("route") == "investigation":
                return "investigation_agent_node"
            return "documentation_agent_node"

        # Pass or exhausted retries
        return "output_guardrails_node"

    def build(self) -> Any:
        """Assemble and compile the StateGraph."""
        workflow = StateGraph(ResearchAssistantState)

        # Add Nodes
        workflow.add_node("input_guardrails_node", self.input_guardrails_node)
        workflow.add_node("router_node", self.router_node)
        workflow.add_node("conversational_node", self.conversational_node)
        workflow.add_node("clarification_node", self.clarification_node)
        workflow.add_node("memory_manager_node", self.memory_manager_node)
        workflow.add_node("documentation_agent_node", self.documentation_agent_node)
        workflow.add_node("investigation_agent_node", self.investigation_agent_node)
        workflow.add_node("verification_node", self.verification_node)
        workflow.add_node("output_guardrails_node", self.output_guardrails_node)
        workflow.add_node("persistence_node", self.persistence_node)

        # Add Edges
        workflow.add_edge(START, "input_guardrails_node")

        workflow.add_conditional_edges(
            "input_guardrails_node",
            self.check_input_guardrail,
            {
                "router_node": "router_node",
                "output_guardrails_node": "output_guardrails_node",
            },
        )

        workflow.add_conditional_edges(
            "router_node",
            self.route_decision,
            {
                "conversational_node": "conversational_node",
                "clarification_node": "clarification_node",
                "memory_manager_node": "memory_manager_node",
                "documentation_agent_node": "documentation_agent_node",
                "investigation_agent_node": "investigation_agent_node",
            },
        )

        # Fast paths to output
        workflow.add_edge("conversational_node", "output_guardrails_node")
        workflow.add_edge("clarification_node", "output_guardrails_node")
        workflow.add_edge("memory_manager_node", "output_guardrails_node")

        # Research paths to verification
        workflow.add_edge("documentation_agent_node", "verification_node")
        workflow.add_edge("investigation_agent_node", "verification_node")

        # Verification conditional retry edge
        workflow.add_conditional_edges(
            "verification_node",
            self.check_verification,
            {
                "output_guardrails_node": "output_guardrails_node",
                "documentation_agent_node": "documentation_agent_node",
                "investigation_agent_node": "investigation_agent_node",
            },
        )

        # Output to persistence to END
        workflow.add_edge("output_guardrails_node", "persistence_node")
        workflow.add_edge("persistence_node", END)

        return workflow.compile()
