from __future__ import annotations

from dataclasses import dataclass
from typing import List
import numpy as np

from youtube_blog_pipeline.agents.lmstudio_client import LMStudioClient, Message
from youtube_blog_pipeline.agents.lmstudio_embeddings import LMStudioEmbeddingsClient
from youtube_blog_pipeline.chaptering.paragraphs import StructuredParagraph


@dataclass
class Chapter:
    title: str
    start: float
    end: float
    paragraph_indices: List[int]


def _cos(a: List[float], b: List[float]) -> float:
    va = np.asarray(a, dtype=np.float32)
    vb = np.asarray(b, dtype=np.float32)
    denom = np.linalg.norm(va) * np.linalg.norm(vb)
    if denom == 0:
        return 0.0
    return float(np.dot(va, vb) / denom)


def embed_paragraphs(paragraphs: List[StructuredParagraph], embed_client: LMStudioEmbeddingsClient) -> List[List[float]]:
    texts = [para.text for para in paragraphs]
    return embed_client.embed(texts)


def cluster_paragraphs(
    vectors: List[List[float]],
    paragraphs: List[StructuredParagraph],
    sim_threshold: float,
) -> List[Chapter]:
    chapters: List[Chapter] = []
    if not paragraphs:
        return chapters
    cur = Chapter(
        title="",
        start=paragraphs[0].start,
        end=paragraphs[0].end,
        paragraph_indices=[0],
    )
    for idx in range(1, len(paragraphs)):
        sim = _cos(vectors[idx - 1], vectors[idx])
        if sim >= sim_threshold:
            cur.end = max(cur.end, paragraphs[idx].end)
            cur.paragraph_indices.append(idx)
        else:
            chapters.append(cur)
            cur = Chapter(
                title="",
                start=paragraphs[idx].start,
                end=paragraphs[idx].end,
                paragraph_indices=[idx],
            )
    chapters.append(cur)
    return chapters


def title_chapters(chapters: List[Chapter], paragraphs: List[StructuredParagraph], client: LMStudioClient) -> List[Chapter]:
    titled: List[Chapter] = []
    previous_end = 0.0
    for ch in chapters:
        texts = [paragraphs[i].text for i in ch.paragraph_indices]
        joined = "\n\n".join(texts)[:4000]
        messages = [
            Message(role="system", content="Return a short, descriptive section title. No emojis."),
            Message(
                role="user",
                content=(
                    "Provide a 3-6 word title capturing the main idea of the following transcript paragraphs."
                    "\n---\n" + joined
                ),
            ),
        ]
        response = client.chat_completion(messages).strip()
        lines = [line.strip() for line in response.splitlines() if line.strip()]
        title = lines[0] if lines else "Section"

        start = max(previous_end, ch.start)
        end = max(start, ch.end)
        previous_end = end

        titled.append(
            Chapter(
                title=title or "Section",
                start=start,
                end=end,
                paragraph_indices=ch.paragraph_indices,
            )
        )
    return titled
