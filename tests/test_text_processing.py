"""Tests for transcript text processing utilities."""

from __future__ import annotations

from youtube_blog_pipeline.processing.text_processing import chunk_transcript, normalize_segments
from youtube_blog_pipeline.postprocessing.emoji_cleaner import strip_emoji
from youtube_blog_pipeline.services.transcript_service import TranscriptSegment


def _segments() -> list[TranscriptSegment]:
    return [
        TranscriptSegment(text="Intro", start=0.0, duration=5.0),
        TranscriptSegment(text="Definition", start=10.0, duration=5.0),
        TranscriptSegment(text="Example", start=25.0, duration=5.0),
        TranscriptSegment(text="Proof", start=40.0, duration=5.0),
    ]


def test_normalize_segments_merges_close_entries() -> None:
    normalized = normalize_segments(_segments())
    assert len(normalized) == 4
    assert normalized[0].text == "Intro"


def test_chunk_transcript_creates_overlap() -> None:
    segments = normalize_segments(_segments())
    chunks = chunk_transcript(segments, max_tokens=3, overlap_tokens=2)
    assert len(chunks) >= 2
    assert chunks[0].segment_indices[0] == 0
    if len(chunks) > 1:
        assert set(chunks[0].segment_indices).intersection(chunks[1].segment_indices)


def test_strip_emoji_removes_characters() -> None:
    assert strip_emoji("Hello 😊") == "Hello "
