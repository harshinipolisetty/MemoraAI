"""LLM client integration supporting Groq, Gemini, and deterministic offline synthesis.

Keeps model invocation decoupled from agent logic. Provides fallback synthesis
for testing and offline operations without requiring live API keys.
"""

import hashlib
import threading
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Tuple
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


# Module-level thread-safe client cache across LLM invocations
_CLIENT_CACHE: Dict[Tuple, Any] = {}
_CLIENT_LOCK = threading.Lock()


def _clean_llm_text(text: str) -> str:
    """Sanitize LLM output by normalizing Unicode typographic symbols to standard ASCII."""
    if not isinstance(text, str):
        return ""
    replacements = {
        "\u2010": "-", "\u2011": "-", "\u2012": "-", "\u2013": "-", "\u2014": "-",
        "\u2018": "'", "\u2019": "'", "\u201c": '"', "\u201d": '"',
        "\u202f": " ", "\u00a0": " ", "\u2022": "*", "\u2026": "...",
        "\u3010": "[", "\u3011": "]",
    }
    cleaned = text
    for u, a in replacements.items():
        cleaned = cleaned.replace(u, a)
    return cleaned


class LLMClient:
    """Unified generation model client with provider routing, structured response, client reuse, and safe fallback."""

    def __init__(
        self,
        mock_synthesizer: Optional[Callable[[str, str], str]] = None,
        request_timeout: Optional[float] = 30.0,
    ):
        self.settings = get_settings()
        self.mock_synthesizer = mock_synthesizer
        self.request_timeout = request_timeout

    @classmethod
    def clear_client_cache(cls) -> None:
        """Clear cached LLM client instances (e.g. for testing or config reloads)."""
        with _CLIENT_LOCK:
            _CLIENT_CACHE.clear()

    @staticmethod
    def _get_groq_client(
        model_name: str,
        api_key: str,
        temperature: float,
        max_tokens: int,
        request_timeout: Optional[float] = 30.0,
    ) -> Any:
        """Safely retrieve or instantiate a reusable ChatGroq client under lock."""
        import langchain_groq
        chat_groq_cls = langchain_groq.ChatGroq

        # Key includes class ID so monkeypatched mock classes in tests never collide with real clients
        api_key_hash = hashlib.sha256(api_key.encode("utf-8")).hexdigest()
        cache_key = (
            id(chat_groq_cls),
            model_name,
            api_key_hash,
            temperature,
            max_tokens,
            request_timeout,
        )

        with _CLIENT_LOCK:
            if cache_key in _CLIENT_CACHE:
                return _CLIENT_CACHE[cache_key]

            client = chat_groq_cls(
                model_name=model_name,
                api_key=api_key,
                temperature=temperature,
                max_tokens=max_tokens,
                request_timeout=request_timeout,
            )
            _CLIENT_CACHE[cache_key] = client
            return client

    def generate(
        self,
        prompt: str,
        system_prompt: str = "You are Asterline Labs Research Assistant. Answer strictly from retrieved evidence.",
        max_tokens: Optional[int] = None,
    ) -> str:
        """Generate text response using configured provider or mock synthesizer."""
        res = self.generate_with_metadata(prompt=prompt, system_prompt=system_prompt, max_tokens=max_tokens)
        return res.content

    def generate_with_metadata(
        self,
        prompt: str,
        system_prompt: str = "You are Asterline Labs Research Assistant. Answer strictly from retrieved evidence.",
        max_tokens: Optional[int] = None,
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
        effective_max_tokens = max_tokens if max_tokens is not None else self.settings.llm_max_tokens

        try:
            if provider == "groq":
                llm = self._get_groq_client(
                    model_name=self.settings.llm_model,
                    api_key=api_key,
                    temperature=self.settings.llm_temperature,
                    max_tokens=effective_max_tokens,
                    request_timeout=self.request_timeout,
                )
                from langchain_core.messages import HumanMessage, SystemMessage
                messages = [SystemMessage(content=system_prompt), HumanMessage(content=prompt)]
                response = llm.invoke(messages)
                return LLMResponse(
                    content=_clean_llm_text(str(response.content)),
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
            # If primary Groq model hit token rate limits (HTTP 429), attempt secondary live model
            if provider == "groq" and ("429" in str(e) or "rate_limit" in str(e).lower()):
                alt_models = ["openai/gpt-oss-20b", "qwen/qwen3.8-27b", "openai/gpt-oss-120b"]
                for alt in alt_models:
                    if alt != self.settings.llm_model:
                        try:
                            alt_llm = self._get_groq_client(
                                model_name=alt,
                                api_key=api_key,
                                temperature=self.settings.llm_temperature,
                                max_tokens=effective_max_tokens,
                                request_timeout=self.request_timeout,
                            )
                            from langchain_core.messages import HumanMessage, SystemMessage
                            alt_messages = [SystemMessage(content=system_prompt), HumanMessage(content=prompt)]
                            alt_resp = alt_llm.invoke(alt_messages)
                            return LLMResponse(
                                content=_clean_llm_text(str(alt_resp.content)),
                                is_fallback=False,
                                provider=provider,
                                model=alt,
                            )
                        except Exception:
                            continue

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


