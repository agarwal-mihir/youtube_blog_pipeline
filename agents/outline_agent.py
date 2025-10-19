"""Generates structured outlines using LM Studio."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Optional

from youtube_blog_pipeline.agents.lmstudio_client import LMStudioClient, Message
from youtube_blog_pipeline.config import DEFAULT_CONFIG
from youtube_blog_pipeline.processing.text_processing import Chunk


@dataclass
class OutlineTopic:
    title: str
    summary: str
    start: Optional[float] = None
    end: Optional[float] = None
    segment_indices: Optional[List[int]] = None


def _format_chunk_preview(chunks: Iterable[Chunk], limit: int = 3) -> str:
    preview_lines: List[str] = []
    for idx, chunk in enumerate(chunks):
        if idx >= limit:
            break
        if DEFAULT_CONFIG.redact_timestamps:
            preview_lines.append(f"Chunk {idx + 1}: {chunk.text[:400]}")
        else:
            preview_lines.append(
                f"Chunk {idx + 1} [{chunk.start:.0f}s - {chunk.end:.0f}s]: {chunk.text[:400]}"
            )
    return "\n".join(preview_lines)


def generate_outline(
    chunks: List[Chunk],
    client: Optional[LMStudioClient] = None,
    video_title: Optional[str] = None,
    video_description: Optional[str] = None,
    sample_chunk_count: int = 3,
) -> List[OutlineTopic]:
    client = client or LMStudioClient()

    preview = _format_chunk_preview(chunks, limit=sample_chunk_count)
    title_line = f"Video Title: {video_title}" if video_title else "Video title unavailable."
    desc_line = (
        f"Video Description: {video_description[:400]}"
        if video_description
        else "Video description not provided."
    )

    messages = [
        Message(role="system", content=DEFAULT_CONFIG.system_prompt),
        Message(
            role="user",
            content=(
                f"{title_line}\n{desc_line}\n\n"
                f"Transcript sample:\n{preview}\n\n"
                f"{DEFAULT_CONFIG.outline_prompt}\n"
                "Output as JSON array with objects: {title, summary, start?, end?}."
            ),
        ),
    ]

    response = client.chat_completion(messages)

    outline: List[OutlineTopic] = []
    try:
        import json

        data = json.loads(response)
        for entry in data:
            outline.append(
                OutlineTopic(
                    title=entry.get("title", "Untitled"),
                    summary=entry.get("summary", ""),
                    start=entry.get("start"),
                    end=entry.get("end"),
                )
            )
    except Exception:  # noqa: BLE001 preserve raw fallback
        outline.append(OutlineTopic(title="Full Session", summary=response))

    return outline
