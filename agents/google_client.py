"""Google AI Studio client wrapper providing an LM Studio compatible interface."""

from __future__ import annotations

import logging
import os
import re
import time
from dataclasses import dataclass
from typing import Iterable, List, Optional

from youtube_blog_pipeline.agents.lmstudio_client import Message
from youtube_blog_pipeline.config import DEFAULT_CONFIG
try:
    from dotenv import load_dotenv
except ImportError:  # pragma: no cover - optional dependency
    load_dotenv = None
from pathlib import Path

LOGGER = logging.getLogger(__name__)

# Load environment variables from default locations and package-level .env
if load_dotenv:
    load_dotenv()
    load_dotenv(Path(__file__).resolve().parents[1] / ".env")


class GoogleAIError(RuntimeError):
    """Raised when the Google AI Studio client encounters an error."""


@dataclass
class GoogleAIResponse:
    """Container for responses if additional metadata is ever needed."""

    content: str


class GoogleAIClient:
    """Synchronous Google AI Studio client wrapper."""

    def __init__(self, *, model: Optional[str] = None, api_key: Optional[str] = None) -> None:
        self._config = DEFAULT_CONFIG.google
        self._model = model or self._config.model
        self._api_key = api_key or self._discover_api_key()
        self._client = None
        self._types = None

    def _discover_api_key(self) -> str:
        for env_name in self._config.api_key_env_vars:
            value = os.getenv(env_name)
            if value:
                return value
        raise GoogleAIError(
            "Google AI Studio API key not found. Set one of: "
            f"{', '.join(self._config.api_key_env_vars)}."
        )

    def _ensure_client(self) -> None:
        if self._client is not None:
            return
        try:
            from google import genai
            from google.genai import types  # noqa: WPS433
        except Exception as exc:  # noqa: BLE001
            raise GoogleAIError(
                "google-genai package is required for Google AI Studio integration."
            ) from exc
        self._client = genai.Client(api_key=self._api_key)
        self._types = types

    @staticmethod
    def _is_rate_limited(exc: Exception) -> bool:
        message = str(exc)
        lowered = message.lower()
        return (
            "resource_exhausted" in lowered
            or "quota" in lowered
            or "rate limit" in lowered
            or "too many requests" in lowered
        )

    @staticmethod
    def _extract_retry_delay(message: str) -> Optional[float]:
        patterns = [
            r"retryDelay[^0-9]*([0-9]+(?:\.[0-9]+)?)s",
            r"retry in ([0-9]+(?:\.[0-9]+)?)s",
            r"retry in ([0-9]+(?:\.[0-9]+)?) seconds",
        ]
        for pattern in patterns:
            match = re.search(pattern, message, flags=re.IGNORECASE)
            if match:
                try:
                    return float(match.group(1))
                except (ValueError, TypeError):
                    continue
        return None

    def chat_completion(
        self,
        messages: Iterable[Message],
        *,
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
        stop: Optional[List[str]] = None,
    ) -> str:
        self._ensure_client()

        types = self._types
        assert types is not None  # for type checkers

        system_instruction: Optional[str] = None
        contents: List[types.Content] = []

        for message in messages:
            role = message.role.lower()
            if role == "system":
                if system_instruction is None:
                    system_instruction = message.content
                else:
                    contents.append(
                        types.Content(
                            role="user",
                            parts=[types.Part.from_text(text=message.content)],
                        )
                    )
                continue

            converted_role = "user" if role == "user" else "model"
            contents.append(
                types.Content(
                    role=converted_role,
                    parts=[types.Part.from_text(text=message.content)],
                )
            )

        if not contents:
            contents.append(
                types.Content(
                    role="user",
                    parts=[types.Part.from_text(text="Provide response.")],
                )
            )

        if system_instruction is not None and not self._config.supports_system_instruction:
            contents.insert(
                0,
                types.Content(
                    role="user",
                    parts=[types.Part.from_text(text=system_instruction)],
                ),
            )
            system_instruction = None

        config_kwargs = {}
        max_tokens = max_tokens if max_tokens is not None else self._config.max_tokens
        if max_tokens and max_tokens > 0:
            config_kwargs["max_output_tokens"] = max_tokens
        if temperature is None:
            temperature = self._config.temperature
        config_kwargs["temperature"] = temperature
        if top_p is None:
            top_p = self._config.top_p
        config_kwargs["top_p"] = top_p
        stop_sequences = stop or list(self._config.stop_sequences or ())
        if stop_sequences:
            config_kwargs["stop_sequences"] = stop_sequences
        if system_instruction is not None and self._config.supports_system_instruction:
            config_kwargs["system_instruction"] = system_instruction
        config = types.GenerateContentConfig(**config_kwargs)

        attempts = 0
        max_attempts = max(0, self._config.rate_limit_retries)
        last_exc: Optional[Exception] = None

        while True:
            try:
                response = self._client.models.generate_content(  # type: ignore[union-attr]
                    model=self._model,
                    contents=contents,
                    config=config,
                )
                break
            except Exception as exc:  # noqa: BLE001
                attempts += 1
                last_exc = exc
                message = str(exc)
                if attempts > max_attempts or not self._is_rate_limited(exc):
                    LOGGER.error("Google AI Studio request failed: %s", message)
                    raise GoogleAIError(f"Google AI Studio request failed: {message}") from exc

                delay = self._extract_retry_delay(message)
                if delay is None or delay <= 0:
                    delay = self._config.rate_limit_backoff_seconds * attempts
                LOGGER.warning(
                    "Google AI Studio rate limit encountered (attempt %d/%d). Retrying in %.1fs.",
                    attempts,
                    max_attempts,
                    delay,
                )
                time.sleep(delay)

        if last_exc is not None and attempts > 0:
            LOGGER.info("Google AI Studio request succeeded after %d retry(ies).", attempts)

        text = getattr(response, "text", None)
        if text:
            return text.strip()

        try:
            candidates = getattr(response, "candidates", []) or []
            assembled: List[str] = []
            for candidate in candidates:
                content = getattr(candidate, "content", None)
                if not content:
                    continue
                for part in getattr(content, "parts", []) or []:
                    piece = getattr(part, "text", "")
                    if piece:
                        assembled.append(piece)
            if assembled:
                return "\n".join(assembled).strip()
        except Exception as exc:  # noqa: BLE001
            LOGGER.debug("Failed to assemble response from candidates: %s", exc)

        raise GoogleAIError("Google AI Studio response missing text content.")
