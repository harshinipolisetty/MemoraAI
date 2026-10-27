"""LLM client integration supporting Groq, Gemini, and deterministic offline synthesis.

Keeps model invocation decoupled from agent logic. Provides fallback synthesis
for testing and offline operations without requiring live API keys.
"""

from typing import Any, Callable, Dict, List, Optional
from asterline.config.settings import get_settings


class LLMClient:
    """Unified generation model client with provider routing and offline testing mode."""

    def __init__(self, mock_synthesizer: Optional[Callable[[str, str], str]] = None):
        self.settings = get_settings()
        self.mock_synthesizer = mock_synthesizer

    def generate(
        self,
        prompt: str,
        system_prompt: str = "You are Asterline Labs Research Assistant. Answer strictly from retrieved evidence.",
    ) -> str:
        """Generate text response using configured provider or mock synthesizer."""
        if self.mock_synthesizer is not None:
            return self.mock_synthesizer(system_prompt, prompt)

        # Check if live API key is configured
        if not self.settings.llm_api_key or not self.settings.llm_api_key.get_secret_value().strip():
            # In offline/test mode without API key, return deterministic evidence synthesis
            return self._offline_fallback_synthesis(system_prompt, prompt)

        api_key = self.settings.llm_api_key.get_secret_value()
        provider = self.settings.llm_provider.lower()

        try:
            if provider == "groq":
                from langchain_groq import ChatGroq
                llm = ChatGroq(
                    model_name=self.settings.llm_model,
                    api_key=api_key,
                    temperature=self.settings.llm_temperature,
                    max_tokens=self.settings.llm_max_tokens,
                )
                from langchain_core.messages import HumanMessage, SystemMessage
                messages = [SystemMessage(content=system_prompt), HumanMessage(content=prompt)]
                response = llm.invoke(messages)
                return str(response.content)

            elif provider in ("gemini", "google"):
                from langchain_google_genai import ChatGoogleGenerativeAI
                llm = ChatGoogleGenerativeAI(
                    model=self.settings.llm_model,
                    google_api_key=api_key,
                    temperature=self.settings.llm_temperature,
                )
                from langchain_core.messages import HumanMessage, SystemMessage
                messages = [SystemMessage(content=system_prompt), HumanMessage(content=prompt)]
                response = llm.invoke(messages)
                return str(response.content)

            else:
                return self._offline_fallback_synthesis(system_prompt, prompt)

        except Exception as e:
            # Re-raise or fallback safely
            raise RuntimeError(f"Error invoking LLM provider '{provider}': {str(e)}") from e

    def _offline_fallback_synthesis(self, system_prompt: str, prompt: str) -> str:
        """Deterministic offline evidence synthesizer for test suites and offline environments."""
        return (
            "Evidence-grounded synthesis (offline mode): Based strictly on the provided documentation chunks, "
            "the documented facts are as stated in the evidence."
        )
