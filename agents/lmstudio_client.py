"""Minimal client for LM Studio OpenAI-compatible API."""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional

import requests

from youtube_blog_pipeline.config import DEFAULT_CONFIG, LMStudioConfig

LOGGER = logging.getLogger(__name__)


class LMStudioError(RuntimeError):
    """Raised when the LM Studio server returns an error."""


@dataclass
class Message:
    role: str
    content: str


class LMStudioClient:
    """Synchronous client wrapper."""

    def __init__(self, config: Optional[LMStudioConfig] = None) -> None:
        self._config = config or DEFAULT_CONFIG.lmstudio

    def chat_completion(
        self,
        messages: Iterable[Message],
        *,
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
        stop: Optional[List[str]] = None,
    ) -> str:
        payload = {
            "model": self._config.model,
            "messages": [message.__dict__ for message in messages],
            # If max_tokens is -1, LM Studio treats it as unlimited within context
            "max_tokens": max_tokens if max_tokens is not None else self._config.max_tokens,
            "temperature": temperature if temperature is not None else self._config.temperature,
            "top_p": top_p if top_p is not None else self._config.top_p,
        }
        if stop:
            payload["stop"] = stop

        try:
            LOGGER.debug("LMStudio request: %s", json.dumps({k: v for k, v in payload.items() if k != 'messages'}))
            response = requests.post(
                f"{self._config.base_url}/chat/completions",
                headers=self._config.request_headers,
                data=json.dumps(payload),
                timeout=self._config.timeout,
            )
        except requests.RequestException as exc:  # noqa: B904
            raise LMStudioError(f"Failed to contact LM Studio API: {exc}") from exc

        if response.status_code != 200:
            LOGGER.error("LMStudio non-200: %s %s", response.status_code, response.text[:400])
            raise LMStudioError(f"LM Studio API returned {response.status_code}: {response.text[:400]}")

        data = response.json()
        try:
            content = data["choices"][0]["message"]["content"].strip()
            # Log first part of the model output to terminal
            LOGGER.info("LMStudio content: %s", content[:1000])
            return content
        except (KeyError, IndexError) as exc:  # noqa: B904
            LOGGER.error("LMStudio unexpected response: %s", response.text[:400])
            raise LMStudioError("Unexpected response format from LM Studio API") from exc
