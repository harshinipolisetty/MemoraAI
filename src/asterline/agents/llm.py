"""LLM client integration supporting Groq, Gemini, and deterministic offline synthesis.

Keeps model invocation decoupled from agent logic. Provides fallback synthesis
for testing and offline operations without requiring live API keys.
"""

from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional
from asterline.config.settings import get_settings


@dataclass
class LLMResponse:
    """Structured response from LLM generation."""

    content: str
    is_fallback: bool
    provider: str
    model: str
    error: Optional[str] = None
    fallback_reason: Optional[str] = None


class LLMClient:
    """Unified generation model client with provider routing, structured response, and safe fallback."""

    def __init__(self, mock_synthesizer: Optional[Callable[[str, str], str]] = None):
        self.settings = get_settings()
        self.mock_synthesizer = mock_synthesizer

    def generate(
        self,
        prompt: str,
        system_prompt: str = "You are Asterline Labs Research Assistant. Answer strictly from retrieved evidence.",
    ) -> str:
        """Generate text response using configured provider or mock synthesizer."""
        res = self.generate_with_metadata(prompt=prompt, system_prompt=system_prompt)
        return res.content

    def generate_with_metadata(
        self,
        prompt: str,
        system_prompt: str = "You are Asterline Labs Research Assistant. Answer strictly from retrieved evidence.",
    ) -> LLMResponse:
        """Generate response with explicit fallback tracking and provider error metadata."""
        if self.mock_synthesizer is not None:
            content = self.mock_synthesizer(system_prompt, prompt)
            return LLMResponse(
                content=content,
                is_fallback=False,
                provider="mock",
                model="mock-synthesizer",
            )

        # Check if live API key is configured
        if not self.settings.llm_api_key or not self.settings.llm_api_key.get_secret_value().strip():
            return LLMResponse(
                content=self._offline_fallback_synthesis(system_prompt, prompt),
                is_fallback=True,
                provider=self.settings.llm_provider,
                model=self.settings.llm_model,
                fallback_reason="No LLM API key configured (offline deterministic fallback)",
            )

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
                return LLMResponse(
                    content=str(response.content),
                    is_fallback=False,
                    provider=provider,
                    model=self.settings.llm_model,
                )

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
                return LLMResponse(
                    content=str(response.content),
                    is_fallback=False,
                    provider=provider,
                    model=self.settings.llm_model,
                )

            else:
                return LLMResponse(
                    content=self._offline_fallback_synthesis(system_prompt, prompt),
                    is_fallback=True,
                    provider=provider,
                    model=self.settings.llm_model,
                    fallback_reason=f"Provider '{provider}' not implemented; using fallback",
                )

        except Exception as e:
            err_msg = f"LLM provider '{provider}' failed: {str(e)}"
            fallback_content = (
                f"[Fallback Mode: Provider Unavailable] {self._offline_fallback_synthesis(system_prompt, prompt)}"
            )
            return LLMResponse(
                content=fallback_content,
                is_fallback=True,
                provider=provider,
                model=self.settings.llm_model,
                error=err_msg,
                fallback_reason="provider_exception",
            )

    def _offline_fallback_synthesis(self, system_prompt: str, prompt: str) -> str:
        """Deterministic offline evidence synthesizer for test suites and offline environments."""
        return (
            "Evidence-grounded synthesis (offline mode): Based strictly on the provided documentation chunks, "
            "the documented facts are as stated in the evidence."
        )

