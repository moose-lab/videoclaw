"""Thin async wrapper around LiteLLM for LLM calls used by Director/Agents.

Centralises model routing, token tracking, and JSON parsing so that every
component in VideoClaw uses a single, consistent interface for language-model
completions.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from typing import Any

import litellm

from videoclaw.config import get_config

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Token usage tracking
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class TokenUsage:
    """Cumulative token counters for cost reporting."""

    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0

    def record(self, usage: dict[str, int]) -> None:
        """Add counts from a single API response's ``usage`` dict."""
        self.prompt_tokens += usage.get("prompt_tokens", 0)
        self.completion_tokens += usage.get("completion_tokens", 0)
        self.total_tokens += usage.get("total_tokens", 0)


# ---------------------------------------------------------------------------
# LLM client
# ---------------------------------------------------------------------------


class LLMClient:
    """High-level async client for language-model completions.

    Uses `litellm <https://docs.litellm.ai/>`_ under the hood so callers can
    transparently target OpenAI, Anthropic, local Ollama, or any other
    provider supported by LiteLLM — just change the model string.

    Parameters
    ----------
    default_model:
        The model identifier to use when none is passed to individual calls.
        Accepts any LiteLLM-compatible model string (e.g. ``"gpt-4o"``,
        ``"claude-sonnet-4-20250514"``, ``"ollama/llama3"``, ``"openai/moonshot-v1-8k"``).
    """

    # Moonshot (Kimi) model prefix detection
    MOONSHOT_PREFIXES = ("openai/moonshot", "moonshot")
    # Evolink (Kimi K2) model prefix detection
    EVOLINK_PREFIXES = ("openai/kimi-k2", "kimi-k2")

    def __init__(self, default_model: str = "gpt-4o") -> None:
        self._default_model = default_model
        self.usage = TokenUsage()
        self._config = get_config()

    def _is_moonshot_model(self, model: str) -> bool:
        """Check if the model is a Moonshot (Kimi) model."""
        return any(model.startswith(prefix) for prefix in self.MOONSHOT_PREFIXES)

    def _is_evolink_model(self, model: str) -> bool:
        """Check if the model is an Evolink (Kimi K2) model."""
        return any(model.startswith(prefix) for prefix in self.EVOLINK_PREFIXES)

    def _get_model_config(self, model: str) -> dict[str, Any]:
        """Get provider-specific configuration for the model."""
        config: dict[str, Any] = {}

        if self._is_moonshot_model(model):
            # Moonshot uses OpenAI-compatible API
            if self._config.moonshot_api_key:
                config["api_key"] = self._config.moonshot_api_key
            config["api_base"] = self._config.moonshot_api_base

            # Normalize model name for LiteLLM (remove openai/ prefix if present)
            if model.startswith("openai/"):
                config["model"] = model  # LiteLLM handles this

        elif self._is_evolink_model(model):
            # Evolink uses OpenAI-compatible API (Kimi K2)
            if self._config.evolink_api_key:
                config["api_key"] = self._config.evolink_api_key
            config["api_base"] = self._config.evolink_api_base

        return config

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def complete(
        self,
        prompt: str,
        *,
        system: str = "",
        model: str | None = None,
        temperature: float = 0.7,
        response_format: type | None = None,
        max_tokens: int | None = None,
    ) -> str:
        """Return a plain-text completion for *prompt*.

        Parameters
        ----------
        prompt:
            The user message.
        system:
            Optional system message prepended to the conversation.
        model:
            Override the default model for this call.
        temperature:
            Sampling temperature.
        response_format:
            If set, passed through to the API (e.g. ``{"type": "json_object"}``
            for structured output).
        max_tokens:
            Optional upper bound on completion length.
        """
        resolved_model = model or self._default_model
        messages = self._build_messages(prompt, system)

        kwargs: dict[str, Any] = {
            "model": resolved_model,
            "messages": messages,
            "temperature": temperature,
        }
        if response_format is not None:
            kwargs["response_format"] = response_format
        if max_tokens is not None:
            kwargs["max_tokens"] = max_tokens

        # Apply provider-specific config (e.g., Moonshot API base)
        kwargs.update(self._get_model_config(resolved_model))

        logger.debug(
            "[llm] Calling %s (temp=%.2f, tokens=%s)",
            resolved_model,
            temperature,
            max_tokens or "default",
        )

        response = await litellm.acompletion(**kwargs)

        # Track usage.
        if hasattr(response, "usage") and response.usage is not None:
            self.usage.record(
                {
                    "prompt_tokens": response.usage.prompt_tokens,
                    "completion_tokens": response.usage.completion_tokens,
                    "total_tokens": response.usage.total_tokens,
                }
            )

        content: str = response.choices[0].message.content or ""
        logger.debug("[llm] Response length: %d chars", len(content))
        return content

    async def complete_json(
        self,
        prompt: str,
        *,
        system: str = "",
        model: str | None = None,
        temperature: float = 0.3,
    ) -> dict[str, Any]:
        """Return a parsed JSON dict from the model.

        The system prompt is augmented with an explicit instruction to return
        valid JSON, and ``response_format`` is set where the API supports it.
        Lower *temperature* default (0.3) to reduce malformed output.
        """
        json_system = (
            system + "\n\n" if system else ""
        ) + "You MUST respond with valid JSON only. No markdown, no explanation."

        raw = await self.complete(
            prompt,
            system=json_system,
            model=model,
            temperature=temperature,
            response_format={"type": "json_object"},  # type: ignore[arg-type]
        )

        return self._parse_json(raw)

    async def chat(
        self,
        messages: list[dict[str, str]],
        *,
        model: str | None = None,
        temperature: float = 0.7,
        max_tokens: int | None = None,
    ) -> str:
        """Send a pre-built message list and return the assistant's reply.

        This is a convenience wrapper around :meth:`complete` for callers
        (like the Director) that construct their own message history.
        """
        resolved_model = model or self._default_model

        kwargs: dict[str, Any] = {
            "model": resolved_model,
            "messages": messages,
            "temperature": temperature,
        }
        if max_tokens is not None:
            kwargs["max_tokens"] = max_tokens

        # Apply provider-specific config (e.g., Moonshot API base)
        kwargs.update(self._get_model_config(resolved_model))

        logger.debug("[llm] chat call to %s (%d messages)", resolved_model, len(messages))
        response = await litellm.acompletion(**kwargs)

        if hasattr(response, "usage") and response.usage is not None:
            self.usage.record(
                {
                    "prompt_tokens": response.usage.prompt_tokens,
                    "completion_tokens": response.usage.completion_tokens,
                    "total_tokens": response.usage.total_tokens,
                }
            )

        content: str = response.choices[0].message.content or ""
        return content

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _build_messages(
        prompt: str,
        system: str,
    ) -> list[dict[str, str]]:
        messages: list[dict[str, str]] = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})
        return messages

    @staticmethod
    def _parse_json(raw: str) -> dict[str, Any]:
        """Best-effort extraction of a JSON object from *raw*.

        Handles common LLM quirks such as wrapping JSON in markdown fences.
        """
        text = raw.strip()

        # Strip markdown code fences if present.
        if text.startswith("```"):
            # Remove opening fence (possibly ```json)
            first_nl = text.index("\n") if "\n" in text else 3
            text = text[first_nl + 1 :]
            if text.endswith("```"):
                text = text[:-3]
            text = text.strip()

        try:
            parsed = json.loads(text)
        except json.JSONDecodeError as exc:
            logger.error("Failed to parse LLM JSON response: %s", exc)
            logger.debug("Raw response:\n%s", raw)
            raise ValueError(f"LLM returned invalid JSON: {exc}") from exc

        if not isinstance(parsed, dict):
            raise ValueError(
                f"Expected a JSON object (dict), got {type(parsed).__name__}"
            )

        return parsed

    # ------------------------------------------------------------------
    # Dunder
    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        return (
            f"<LLMClient model={self._default_model!r} "
            f"tokens_used={self.usage.total_tokens}>"
        )
