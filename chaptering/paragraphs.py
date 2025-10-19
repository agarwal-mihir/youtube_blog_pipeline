from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass
from typing import Iterable, List, Sequence, Tuple

from youtube_blog_pipeline.agents.lmstudio_client import LMStudioClient, Message
from youtube_blog_pipeline.agents.lmstudio_embeddings import LMStudioEmbeddingsClient
from youtube_blog_pipeline.services.transcript_service import TranscriptSegment

LOGGER = logging.getLogger(__name__)


PARAGRAPH_PROMPT = (
    "You are a helpful assistant. Improve readability, add punctuation, remove verbal tics, and"
    " structure the text into paragraphs separated by blank lines. Keep wording faithful."
    " Return the result wrapped in <answer>...</answer>."
)


@dataclass
class StructuredParagraph:
    text: str
    start: float
    end: float
    segment_range: Tuple[int, int]
    chunk_index: int


def _chunk_segments(segments: Sequence[TranscriptSegment], max_chars: int) -> List[Tuple[str, int, int]]:
    chunks: List[Tuple[str, int, int]] = []
    buf: List[str] = []
    start_idx = 0
    total = 0
    for idx, seg in enumerate(segments):
        text = seg.text.strip()
        if not text:
            continue
        if not buf:
            start_idx = idx
            total = 0
        addition = (" " if buf else "") + text
        if buf and total + len(addition) > max_chars:
            chunks.append(("".join(buf), start_idx, idx))
            buf = [text]
            start_idx = idx
            total = len(text)
        else:
            buf.append(addition if buf else text)
            total += len(addition)
    if buf:
        chunks.append(("".join(buf), start_idx, len(segments)))
    return chunks


_ANSWER_RE = re.compile(r"<answer>(.*?)</answer>", re.DOTALL | re.IGNORECASE)


def _llm_format_text(text: str, client: LMStudioClient, *, max_chars: int) -> str:
    messages = [
        Message(role="system", content=PARAGRAPH_PROMPT),
        Message(role="user", content=text[:max_chars]),
    ]
    raw = client.chat_completion(messages, temperature=0.0)
    match = _ANSWER_RE.search(raw)
    if match:
        return match.group(1).strip()
    LOGGER.warning("Paragraph formatter did not return <answer> tags; using raw content")
    return raw.strip()


def _split_paragraphs(text: str) -> List[str]:
    paras = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]
    return paras


def format_transcript_to_paragraphs(
    segments: Sequence[TranscriptSegment],
    client: LMStudioClient,
    *,
    max_chars: int,
) -> List[Tuple[str, int, int]]:
    chunked = _chunk_segments(segments, max_chars)
    structured: List[Tuple[str, int, int]] = []
    for chunk_idx, (chunk_text, start_idx, end_idx) in enumerate(chunked):
        formatted = _llm_format_text(chunk_text, client, max_chars=max_chars)
        paragraphs = _split_paragraphs(formatted)
        if not paragraphs:
            continue

        seg_indices = list(range(start_idx, end_idx))
        if not seg_indices:
            seg_indices = [start_idx]
        seg_count = len(seg_indices)

        boundaries = [(i * seg_count) // len(paragraphs) for i in range(len(paragraphs) + 1)]
        cursor = 0
        for idx, para in enumerate(paragraphs):
            start_off = max(cursor, boundaries[idx])
            if start_off >= seg_count:
                start_off = seg_count - 1
            if start_off < 0:
                start_off = 0

            if idx == len(paragraphs) - 1:
                end_off = seg_count
            else:
                end_off = max(boundaries[idx + 1], start_off + 1)
            if end_off > seg_count:
                end_off = seg_count
            if end_off <= start_off:
                end_off = min(seg_count, start_off + 1)

            slice_indices = seg_indices[start_off:end_off]
            if not slice_indices:
                slice_indices = [seg_indices[start_off]]
                end_off = min(seg_count, start_off + 1)

            seg_start = slice_indices[0]
            if idx == len(paragraphs) - 1:
                seg_end = end_idx
            else:
                seg_end = min(end_idx, slice_indices[-1] + 1)
            structured.append((para, seg_start, seg_end, chunk_idx))

            cursor = max(end_off, cursor)
    return structured


def _embed_texts(texts: Sequence[str], embed_client: LMStudioEmbeddingsClient) -> List[List[float]]:
    if not texts:
        return []
    return embed_client.embed(texts)


def _select_segment_texts(segments: Sequence[TranscriptSegment], words: int) -> List[str]:
    out: List[str] = []
    for seg in segments:
        tokens = seg.text.split()
        out.append(" ".join(tokens[:words]))
    return out


def _paragraph_samples(paragraphs: Sequence[str], words: int) -> List[str]:
    out: List[str] = []
    for para in paragraphs:
        tokens = para.split()
        out.append(" ".join(tokens[:words]))
    return out


def align_paragraphs_with_segments(
    structured: Sequence[Tuple[str, int, int, int]],
    segments: Sequence[TranscriptSegment],
    embed_client: LMStudioEmbeddingsClient,
    *,
    sample_words: int,
) -> List[StructuredParagraph]:
    if not structured:
        return []
    paragraph_texts = [item[0] for item in structured]
    para_samples = _paragraph_samples(paragraph_texts, sample_words)
    seg_samples = _select_segment_texts(segments, sample_words)

    if not seg_samples:
        LOGGER.warning("No transcript segments available for alignment")
        return []

    para_vecs = _embed_texts(para_samples, embed_client)
    seg_vecs = _embed_texts(seg_samples, embed_client)

    if not seg_vecs:
        LOGGER.warning("No segment embeddings available for alignment")
        return []

    aligned: List[StructuredParagraph] = []
    last_consumed = 0
    prev_end_time = 0.0
    for (para_text, orig_start, orig_end, chunk_idx), para_vec in zip(structured, para_vecs):
        if orig_end <= orig_start:
            orig_end = orig_start + 1

        margin = 3
        start_bound = max(0, orig_start - margin, last_consumed)
        end_bound = min(len(seg_vecs), orig_end + margin)
        if start_bound >= len(seg_vecs):
            start_bound = len(seg_vecs) - 1
        if end_bound <= start_bound:
            end_bound = min(len(seg_vecs), start_bound + 1)

        best_idx = start_bound
        best_score = float("-inf")
        for idx in range(start_bound, end_bound):
            score = float(sum(a * b for a, b in zip(para_vec, seg_vecs[idx])))
            if score > best_score:
                best_score = score
                best_idx = idx
            if score >= 0.95:
                break

        valid_start = min(max(orig_start, 0), len(segments) - 1)
        valid_end = min(max(orig_end, valid_start + 1), len(segments))
        best_idx = min(max(best_idx, valid_start), valid_end - 1)
        start_idx = min(best_idx, valid_start)
        span_len = max(1, orig_end - orig_start)
        span_end_idx = min(len(segments) - 1, max(valid_end - 1, start_idx + span_len - 1))

        seg = segments[start_idx]
        start_time = max(prev_end_time, seg.start)
        end_time = max(start_time, segments[span_end_idx].start + segments[span_end_idx].duration)

        last_consumed = max(span_end_idx + 1, start_idx + 1)
        prev_end_time = end_time

        aligned.append(
            StructuredParagraph(
                text=para_text,
                start=start_time,
                end=end_time,
                segment_range=(start_idx, span_end_idx + 1),
                chunk_index=chunk_idx,
            )
        )
    return aligned
