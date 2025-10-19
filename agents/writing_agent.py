"""Agent responsible for drafting markdown sections."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Optional

from youtube_blog_pipeline.agents.lmstudio_client import LMStudioClient, Message
from youtube_blog_pipeline.config import DEFAULT_CONFIG
from youtube_blog_pipeline.chaptering.cluster import Chapter
from youtube_blog_pipeline.chaptering.paragraphs import StructuredParagraph
from youtube_blog_pipeline.processing.text_processing import Chunk, gather_text_for_indices
from youtube_blog_pipeline.services.transcript_service import TranscriptSegment


@dataclass
class SectionDraft:
    title: str
    content: str
    related_indices: List[int]


def draft_sections(
    outline_topics,
    chunks: List[Chunk],
    segments: List[TranscriptSegment],
    prior_summaries: Optional[List[str]] = None,
    client: Optional[LMStudioClient] = None,
    *,
    detail_level: str = "high",
) -> List[SectionDraft]:
    client = client or LMStudioClient()
    prior_summaries = prior_summaries or []

    drafts: List[SectionDraft] = []
    running_summary: List[str] = list(prior_summaries)
    detail_key = (detail_level or "high").lower()
    detail_instruction = DEFAULT_CONFIG.detail_prompts.get(detail_key, DEFAULT_CONFIG.detail_prompts["high"])
    max_tokens = DEFAULT_CONFIG.detail_max_tokens.get(detail_key)

    for topic in outline_topics:
        override_indices = getattr(topic, "segment_indices", None)
        if override_indices:
            relevant_indices = sorted({idx for idx in override_indices if 0 <= idx < len(segments)})
        else:
            relevant_indices = []
            for chunk in chunks:
                if topic.start is not None and topic.end is not None:
                    if chunk.end < topic.start or chunk.start > topic.end:
                        continue
                relevant_indices.extend(chunk.segment_indices)
            if not relevant_indices:
                relevant_indices = chunks[0].segment_indices if chunks else []

        context_text = gather_text_for_indices(segments, relevant_indices)
        previous_notes = "\n".join(running_summary[-DEFAULT_CONFIG.history_max_entries:])
        prior_block = previous_notes or "None yet."

        prompt = (
            f"Topic: {topic.title}\nSummary cue: {topic.summary}\n"
            f"Previously covered chapters (avoid repeating these details):\n{prior_block}\n\n"
            "Transcript excerpts:\n"
            f"{context_text}\n\n"
            f"{detail_instruction} Ensure the notes begin with the heading '## {topic.title}'. "
            "Typeset mathematics with proper LaTeX ($...$ inline, $$...$$ display) whenever formulas are needed."
        )

        messages = [
            Message(role="system", content=DEFAULT_CONFIG.system_prompt),
            Message(role="user", content=prompt),
        ]

        completion_kwargs = {}
        if max_tokens is not None:
            completion_kwargs["max_tokens"] = max_tokens
        content = client.chat_completion(messages, **completion_kwargs)
        drafts.append(SectionDraft(title=topic.title, content=content, related_indices=relevant_indices))

        running_summary.append(f"{topic.title}: {topic.summary}")

    return drafts


def draft_chapter_sections(
    chapters: List[Chapter],
    paragraphs: List[StructuredParagraph],
    segments: List[TranscriptSegment],
    client: Optional[LMStudioClient] = None,
    *,
    detail_level: str = "high",
) -> List[SectionDraft]:
    client = client or LMStudioClient()

    drafts: List[SectionDraft] = []
    history_summaries: List[str] = []
    detail_key = (detail_level or "high").lower()
    detail_instruction = DEFAULT_CONFIG.detail_prompts.get(detail_key, DEFAULT_CONFIG.detail_prompts["high"])
    max_tokens = DEFAULT_CONFIG.detail_max_tokens.get(detail_key)

    def _summarize_text(text: str, limit: int) -> str:
        cleaned = " ".join(text.split())
        if len(cleaned) <= limit:
            return cleaned
        truncated = cleaned[:limit].rsplit(" ", 1)[0].strip()
        return f"{truncated}..." if truncated else cleaned[:limit]

    for chapter in chapters:
        para_indices = [idx for idx in chapter.paragraph_indices if 0 <= idx < len(paragraphs)]
        if not para_indices:
            continue

        chapter_title = chapter.title or "Chapter"
        paragraph_text = "\n\n".join(paragraphs[idx].text for idx in para_indices)

        segment_indices: List[int] = []
        seen_indices = set()
        for para_idx in para_indices:
            start_idx, end_idx = paragraphs[para_idx].segment_range
            for seg_idx in range(start_idx, end_idx):
                if 0 <= seg_idx < len(segments) and seg_idx not in seen_indices:
                    seen_indices.add(seg_idx)
                    segment_indices.append(seg_idx)

        if not segment_indices:
            segment_indices = list(range(len(segments)))

        previous_notes = "\n".join(history_summaries[-DEFAULT_CONFIG.history_max_entries:])
        if previous_notes:
            prior_block = (
                "Previously covered chapters (avoid repeating these details):\n"
                f"{previous_notes}\n\n"
            )
        else:
            prior_block = "Previously covered chapters: None yet.\n\n"

        time_range = ""
        if chapter.start is not None and chapter.end is not None:
            time_range = f"Time range: {chapter.start:.0f}s – {chapter.end:.0f}s.\n"

        prompt = (
            f"Chapter title: {chapter_title}\n"
            f"{time_range}"
            f"{prior_block}"
            "Transcript paragraphs for this chapter:\n"
            f"{paragraph_text}\n\n"
            f"{detail_instruction} Ensure the chapter begins with the heading '## {chapter_title}'. "
            "Follow the flow of the transcript and avoid restating prior chapters except for brief references. "
            "Typeset mathematics with LaTeX delimiters ($...$ inline, $$...$$ display) whenever formulas appear."
        )

        messages = [
            Message(role="system", content=DEFAULT_CONFIG.system_prompt),
            Message(role="user", content=prompt),
        ]

        completion_kwargs = {}
        if max_tokens is not None:
            completion_kwargs["max_tokens"] = max_tokens
        content = client.chat_completion(messages, **completion_kwargs)
        if f"## {chapter_title}" not in content:
            content = f"## {chapter_title}\n\n{content.strip()}"

        drafts.append(
            SectionDraft(
                title=chapter_title,
                content=content.strip(),
                related_indices=segment_indices,
            )
        )

        summary_source = " ".join(paragraphs[idx].text for idx in para_indices)
        history_summaries.append(
            f"{chapter_title}: {_summarize_text(summary_source, DEFAULT_CONFIG.history_summary_chars)}"
        )

    return drafts
