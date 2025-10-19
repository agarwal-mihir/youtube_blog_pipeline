"""Review agent refines drafted sections."""

from __future__ import annotations

from typing import List, Optional

from youtube_blog_pipeline.agents.lmstudio_client import LMStudioClient, Message
from youtube_blog_pipeline.config import DEFAULT_CONFIG
from youtube_blog_pipeline.processing.text_processing import gather_text_for_indices
from youtube_blog_pipeline.services.transcript_service import TranscriptSegment


def review_section(
    section_markdown: str,
    related_indices: List[int],
    segments: List[TranscriptSegment],
    client: Optional[LMStudioClient] = None,
) -> str:
    client = client or LMStudioClient()
    excerpt = gather_text_for_indices(segments, related_indices)

    messages = [
        Message(role="system", content=DEFAULT_CONFIG.system_prompt),
        Message(
            role="user",
            content=(
                f"Existing markdown section:\n{section_markdown}\n\n"
                f"Relevant transcript excerpts:\n{excerpt}\n\n"
                f"{DEFAULT_CONFIG.reviewer_prompt}\nReturn only the revised Markdown section."
            ),
        ),
    ]

    return client.chat_completion(messages)
