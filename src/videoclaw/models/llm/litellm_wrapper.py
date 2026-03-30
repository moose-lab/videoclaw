"""Thin async wrapper around LiteLLM for LLM calls used by Director/Agents.

Centralises model routing, token tracking, and JSON parsing so that every
component in VideoClaw uses a single, consistent interface for language-model
completions.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
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
        ``"claude-sonnet-4-6"``, ``"kimi-k2-thinking"``, ``"gpt-5.1-chat"``,
        ``"deepseek-chat"``, ``"gemini-3.1-pro"``).
        Models matching Evolink prefixes are auto-routed via Evolink gateway.
    """

    # Moonshot (Kimi v1) model prefix detection
    MOONSHOT_PREFIXES = ("openai/moonshot", "moonshot")

    # Evolink unified gateway — routes through OpenAI-compatible or Anthropic API
    EVOLINK_PREFIXES = (
        # Kimi K2
        "kimi-k2", "openai/kimi-k2",
        "kimi-2.5", "openai/kimi-2.5",
        # Claude (via Evolink — uses Anthropic Messages API /v1/messages)
        "claude-sonnet-4-6", "claude-opus-4-6",
        "claude-sonnet-4-5", "claude-opus-4-5",
        "claude-opus-4-1", "claude-sonnet-4-",
        "claude-haiku-4-5",
        # GPT (via Evolink)
        "gpt-5.1", "gpt-5.2", "gpt-5.4",
        # Gemini (via Evolink)
        "gemini-2.5", "gemini-3",
        # DeepSeek (via Evolink)
        "deepseek-chat", "deepseek-reasoner",
        # Doubao Seed (via Evolink)
        "doubao-seed",
        # MiniMax (via Evolink)
        "minimax-m2.5",
        # Evolink auto routing
        "evolink/auto",
    )

    # Claude models on Evolink use the Anthropic Messages API (/v1/messages),
    # not the OpenAI chat/completions format.
    EVOLINK_CLAUDE_PREFIXES = (
        "claude-sonnet-4-6", "claude-opus-4-6",
        "claude-sonnet-4-5", "claude-opus-4-5",
        "claude-opus-4-1", "claude-sonnet-4-",
        "claude-haiku-4-5",
    )

    def __init__(self, default_model: str = "gpt-4o") -> None:
        self._default_model = default_model
        self.usage = TokenUsage()
        self._config = get_config()

    def _is_moonshot_model(self, model: str) -> bool:
        """Check if the model is a Moonshot (Kimi v1) model."""
        return any(model.startswith(prefix) for prefix in self.MOONSHOT_PREFIXES)

    def _is_evolink_model(self, model: str) -> bool:
        """Check if the model is routable through Evolink gateway."""
        bare = model.removeprefix("openai/").removeprefix("anthropic/")
        return any(bare.startswith(prefix) for prefix in self.EVOLINK_PREFIXES)

    def _is_evolink_claude(self, model: str) -> bool:
        """Check if the model is a Claude model routed via Evolink.

        Claude models on Evolink use the Anthropic Messages API format
        (``POST /v1/messages``), not the OpenAI chat/completions format.
        """
        bare = model.removeprefix("openai/").removeprefix("anthropic/")
        return any(bare.startswith(prefix) for prefix in self.EVOLINK_CLAUDE_PREFIXES)

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
            if self._config.evolink_api_key:
                config["api_key"] = self._config.evolink_api_key

            if self._is_evolink_claude(model):
                # Claude on Evolink uses Anthropic Messages API format.
                # LiteLLM: anthropic/ prefix → calls POST /v1/messages
                # api_base should be without /v1 — LiteLLM appends /v1/messages
                base = self._config.evolink_api_base.rstrip("/")
                if base.endswith("/v1"):
                    base = base[:-3]
                config["api_base"] = base

                bare = model.removeprefix("openai/").removeprefix("anthropic/")
                config["model"] = f"anthropic/{bare}"
            else:
                # Non-Claude Evolink models use OpenAI-compatible API
                config["api_base"] = self._config.evolink_api_base

                if not model.startswith("openai/"):
                    config["model"] = f"openai/{model}"

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
        timeout: float | None = None,
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

        # Thinking models need longer timeouts
        if timeout is not None:
            kwargs["timeout"] = timeout
        elif "thinking" in resolved_model:
            kwargs["timeout"] = 600  # 10 minutes for thinking models

        # Apply provider-specific config (e.g., Moonshot API base)
        kwargs.update(self._get_model_config(resolved_model))

        # Use streaming for thinking models AND Evolink-proxied models
        # to avoid Cloudflare 524 timeout on long generations.
        use_stream = "thinking" in resolved_model or self._is_evolink_model(resolved_model)

        logger.debug(
            "[llm] chat call to %s (%d messages, stream=%s)",
            resolved_model, len(messages), use_stream,
        )

        if use_stream:
            kwargs["stream"] = True
            response = await litellm.acompletion(**kwargs)
            chunks: list[str] = []
            reasoning_chunks: list[str] = []
            async for chunk in response:
                delta = chunk.choices[0].delta
                if delta:
                    if delta.content:
                        chunks.append(delta.content)
                    # Thinking models may put output in reasoning_content
                    rc = getattr(delta, "reasoning_content", None)
                    if rc:
                        reasoning_chunks.append(rc)
            content = "".join(chunks)
            if not content and reasoning_chunks:
                # Fallback: extract JSON from reasoning output
                reasoning_text = "".join(reasoning_chunks)
                logger.warning(
                    "[llm] Streaming content empty, extracting from reasoning (%d chars)",
                    len(reasoning_text),
                )
                # Try to find JSON in reasoning output
                import re
                json_match = re.search(r"```(?:json)?\s*\n([\s\S]*?)\n```", reasoning_text)
                if json_match:
                    content = json_match.group(1).strip()
                elif reasoning_text.strip().startswith("{"):
                    content = reasoning_text.strip()
            logger.debug(
                "[llm] Streamed %d content chunks, %d reasoning chunks",
                len(chunks), len(reasoning_chunks),
            )
        else:
            response = await litellm.acompletion(**kwargs)
            if hasattr(response, "usage") and response.usage is not None:
                self.usage.record(
                    {
                        "prompt_tokens": response.usage.prompt_tokens,
                        "completion_tokens": response.usage.completion_tokens,
                        "total_tokens": response.usage.total_tokens,
                    }
                )
            content = response.choices[0].message.content or ""

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
