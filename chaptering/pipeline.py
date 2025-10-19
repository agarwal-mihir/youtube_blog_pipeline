from __future__ import annotations

from typing import List, Tuple

from youtube_blog_pipeline.agents.lmstudio_client import LMStudioClient
from youtube_blog_pipeline.agents.lmstudio_embeddings import LMStudioEmbeddingsClient
from youtube_blog_pipeline.chaptering.paragraphs import (
    StructuredParagraph,
    align_paragraphs_with_segments,
    format_transcript_to_paragraphs,
)
from youtube_blog_pipeline.chaptering.cluster import (
    Chapter,
    cluster_paragraphs,
    embed_paragraphs,
    title_chapters,
)
from youtube_blog_pipeline.services.transcript_service import TranscriptSegment
from youtube_blog_pipeline.config import DEFAULT_CONFIG


def generate_chapters(
    segments: List[TranscriptSegment],
    client: LMStudioClient,
    embed_client: LMStudioEmbeddingsClient,
) -> Tuple[List[Chapter], List[StructuredParagraph]]:
    cfg = DEFAULT_CONFIG.chaptering
    if not segments:
        return [], []
    T = segments[-1].start + segments[-1].duration

    raw_paragraphs = format_transcript_to_paragraphs(
        segments,
        client,
        max_chars=cfg.paragraph_max_chars,
    )
    paragraphs = align_paragraphs_with_segments(
        raw_paragraphs,
        segments,
        embed_client,
        sample_words=cfg.paragraph_sample_words,
    )
    paragraphs.sort(key=lambda p: (p.chunk_index, p.start))

    para_vecs = embed_paragraphs(paragraphs, embed_client)
    clusters = cluster_paragraphs(para_vecs, paragraphs, cfg.sim_threshold)
    clusters.sort(key=lambda c: c.start)
    titled = title_chapters(clusters, paragraphs, client)

    for ch in titled:
        ch.end = min(ch.end, T)
    return titled, paragraphs
