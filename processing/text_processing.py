"""Utilities for transcript preprocessing, normalization, and chunking."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Iterable, List, Sequence

from youtube_blog_pipeline.config import DEFAULT_CONFIG
from youtube_blog_pipeline.services.transcript_service import TranscriptSegment
from youtube_blog_pipeline.postprocessing.timestamp_cleaner import strip_timestamps

LOGGER = logging.getLogger(__name__)


def normalize_segments(segments: Sequence[TranscriptSegment]) -> List[TranscriptSegment]:
    """Return segments stripped of extra whitespace and merged where appropriate."""

    normalized: List[TranscriptSegment] = []
    cfg = DEFAULT_CONFIG.chunking

    for segment in segments:
        text = strip_timestamps(" ".join(segment.text.split())) if DEFAULT_CONFIG.redact_timestamps else " ".join(segment.text.split())
        if not text:
            continue

        if normalized:
            prev = normalized[-1]
            gap = segment.start - (prev.start + prev.duration)
            if gap < 0 or (0 < gap <= cfg.merge_gap_seconds):
                merged_text = f"{prev.text} {text}".strip()
                normalized[-1] = TranscriptSegment(
                    text=merged_text,
                    start=prev.start,
                    duration=(segment.start + segment.duration) - prev.start,
                )
                continue

        normalized.append(TranscriptSegment(text=text, start=segment.start, duration=segment.duration))

    LOGGER.info("Normalized transcript to %d segments", len(normalized))
    return normalized


@dataclass
class Chunk:
    """Represents a chunk of transcript text for prompting."""

    text: str
    start: float
    end: float
    segment_indices: List[int]


def _estimate_tokens(text: str) -> int:
    return max(1, int(len(text) / 4))


def chunk_transcript(
    segments: Sequence[TranscriptSegment],
    max_tokens: int | None = None,
    overlap_tokens: int | None = None,
) -> List[Chunk]:
    """Chunk transcript into overlapping windows based on token estimates."""

    cfg = DEFAULT_CONFIG.chunking
    limit = max_tokens or cfg.max_tokens
    overlap = overlap_tokens or cfg.overlap_tokens

    chunks: List[Chunk] = []
    buffer: List[str] = []
    indices: List[int] = []
    current_tokens = 0
    start_ts: float | None = None
    last_end: float | None = None

    for idx, segment in enumerate(segments):
        segment_tokens = _estimate_tokens(segment.text)
        if not buffer:
            start_ts = segment.start

        if current_tokens + segment_tokens > limit and buffer:
            end_ts = last_end if last_end is not None else segment.start
            chunks.append(
                Chunk(
                    text=" ".join(buffer),
                    start=start_ts or 0.0,
                    end=end_ts,
                    segment_indices=list(indices),
                )
            )

            prev_indices = chunks[-1].segment_indices
            step_back_tokens = 0
            buffer = []
            indices = []
            current_tokens = 0
            for lookback_idx in reversed(prev_indices):
                prior_segment = segments[lookback_idx]
                prior_tokens = _estimate_tokens(prior_segment.text)
                step_back_tokens += prior_tokens
                buffer.insert(0, prior_segment.text)
                indices.insert(0, lookback_idx)
                current_tokens += prior_tokens
                if step_back_tokens >= overlap:
                    break

            start_ts = segments[indices[0]].start if indices else segment.start

        buffer.append(segment.text)
        indices.append(idx)
        current_tokens += segment_tokens
        last_end = segment.start + segment.duration

    if buffer:
        chunks.append(
            Chunk(
                text=" ".join(buffer),
                start=start_ts or 0.0,
                end=last_end or (start_ts or 0.0),
                segment_indices=list(indices),
            )
        )

    LOGGER.info("Chunked transcript into %d chunks", len(chunks))
    return chunks


def gather_text_for_indices(segments: Sequence[TranscriptSegment], indices: Iterable[int]) -> str:
    text = " ".join(segments[i].text for i in indices)
    return strip_timestamps(text) if DEFAULT_CONFIG.redact_timestamps else text
