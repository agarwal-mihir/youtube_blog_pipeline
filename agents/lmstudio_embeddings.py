from __future__ import annotations

import json
import logging
from typing import Iterable, List, Optional

import requests

from youtube_blog_pipeline.config import DEFAULT_CONFIG

LOGGER = logging.getLogger(__name__)


class LMStudioEmbeddingsClient:
    def __init__(self, *, model: Optional[str] = None, timeout: Optional[float] = 120.0) -> None:
        self._base = DEFAULT_CONFIG.lmstudio.base_url
        self._headers = DEFAULT_CONFIG.lmstudio.request_headers
        self._model = model or DEFAULT_CONFIG.chaptering.embedding_model
        self._timeout = timeout or 120.0

    def embed(self, texts: Iterable[str]) -> List[List[float]]:
        payload = {"model": self._model, "input": list(texts)}
        try:
            resp = requests.post(
                f"{self._base}/embeddings",
                headers=self._headers,
                data=json.dumps(payload),
                timeout=self._timeout,
            )
        except requests.RequestException as exc:  # noqa: B904
            raise RuntimeError(f"Embeddings request failed: {exc}") from exc
        if resp.status_code != 200:
            LOGGER.error("Embeddings error %s: %s", resp.status_code, resp.text[:200])
            raise RuntimeError(f"Embeddings error {resp.status_code}: {resp.text[:200]}")
        data = resp.json()
        return [item["embedding"] for item in data.get("data", [])]
