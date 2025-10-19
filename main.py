"""CLI entrypoint for the YouTube transcript to markdown pipeline."""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Any, List, Optional, Tuple

from youtube_blog_pipeline.agents.google_client import GoogleAIClient, GoogleAIError
from youtube_blog_pipeline.agents.lmstudio_embeddings import LMStudioEmbeddingsClient
from youtube_blog_pipeline.agents.lmstudio_client import LMStudioClient
from youtube_blog_pipeline.agents.outline_agent import OutlineTopic, generate_outline
from youtube_blog_pipeline.agents.writing_agent import draft_sections, draft_chapter_sections
from youtube_blog_pipeline.assembly.blog_assembler import assemble_markdown, write_markdown
from youtube_blog_pipeline.chaptering.pipeline import generate_chapters
from youtube_blog_pipeline.config import DEFAULT_CONFIG
from youtube_blog_pipeline.postprocessing.emoji_cleaner import strip_emoji
from youtube_blog_pipeline.processing.text_processing import chunk_transcript, normalize_segments
from youtube_blog_pipeline.services.transcript_service import fetch_transcript

try:
    from dotenv import load_dotenv
except ImportError:  # pragma: no cover - optional dependency
    load_dotenv = None

if load_dotenv:
    load_dotenv()
    load_dotenv(Path(__file__).resolve().parent / ".env")

LOGGER = logging.getLogger("youtube_blog_pipeline")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Generate blog post from YouTube transcript")
    parser.add_argument("youtube_url", help="YouTube video URL or ID")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("output/blog_post.md"),
        help="Destination markdown file",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=DEFAULT_CONFIG.chunking.max_tokens,
        help="Approximate max tokens per chunk",
    )
    parser.add_argument(
        "--overlap",
        type=int,
        default=DEFAULT_CONFIG.chunking.overlap_tokens,
        help="Approximate overlap tokens between chunks",
    )
    parser.add_argument(
        "--save-outline",
        type=Path,
        help="Optional path to write generated outline as JSON",
    )
    parser.add_argument(
        "--provider",
        default="lmstudio",
        choices=["lmstudio", "google"],
        help="LLM provider for generation (default: lmstudio)",
    )
    parser.add_argument(
        "--detail-level",
        default="high",
        choices=["low", "high"],
        help="Controls how verbose the generated notes should be (default: high).",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
    )
    return parser


def _trim_summary(text: str, limit: int = 400) -> str:
    snippet = text.strip()
    if not snippet:
        return ""
    if len(snippet) <= limit:
        return snippet
    truncated = snippet[:limit].rsplit(" ", 1)[0].strip()
    if not truncated:
        truncated = snippet[:limit].strip()
    return f"{truncated}..."


def _topics_from_chapters(chapters, paragraphs) -> List[OutlineTopic]:
    topics: List[OutlineTopic] = []
    for idx, chapter in enumerate(chapters):
        text_fragments: List[str] = []
        segment_indices_set = set()
        for p_idx in chapter.paragraph_indices:
            if 0 <= p_idx < len(paragraphs):
                para = paragraphs[p_idx]
                text_fragments.append(para.text)
                segment_indices_set.update(range(para.segment_range[0], para.segment_range[1]))
        combined_text = " ".join(text_fragments)
        summary = _trim_summary(combined_text) or (chapter.title or f"Chapter {idx + 1}")
        topics.append(
            OutlineTopic(
                title=chapter.title or f"Chapter {idx + 1}",
                summary=summary,
                start=chapter.start,
                end=chapter.end,
                segment_indices=sorted(segment_indices_set) if segment_indices_set else None,
            )
        )
    return topics


def run_pipeline(
    youtube_url: str,
    *,
    chunk_size: Optional[int] = None,
    overlap: Optional[int] = None,
    client: Optional[Any] = None,
    output_path: Optional[Path] = None,
    outline_path: Optional[Path] = None,
    provider: str = "lmstudio",
    detail_level: str = "high",
) -> Tuple[str, list[OutlineTopic]]:
    chunk_size = chunk_size or DEFAULT_CONFIG.chunking.max_tokens
    overlap = overlap or DEFAULT_CONFIG.chunking.overlap_tokens

    detail_level = (detail_level or "high").lower()
    if detail_level not in {"low", "high"}:
        detail_level = "high"

    segments = normalize_segments(fetch_transcript(youtube_url))
    chunks = chunk_transcript(segments, max_tokens=chunk_size, overlap_tokens=overlap)

    provider_key = (provider or "lmstudio").lower()
    if provider_key not in {"lmstudio", "google"}:
        raise ValueError(f"Unsupported provider: {provider}")
    if client is None:
        if provider_key == "google":
            client = GoogleAIClient()
        else:
            client = LMStudioClient()
    embed_client = LMStudioEmbeddingsClient()
    chapters: list = []
    structured_paragraphs: list = []
    try:
        chapters, structured_paragraphs = generate_chapters(segments, client, embed_client)
    except Exception as exc:  # noqa: BLE001
        LOGGER.warning("Chapter generation failed; falling back to LLM outline: %s", exc)

    outline: List[OutlineTopic] = []
    sections_markdown: List[str] = []

    if chapters:
        outline = _topics_from_chapters(chapters, structured_paragraphs)
        chapter_drafts = draft_chapter_sections(
            chapters,
            structured_paragraphs,
            segments,
            client=client,
            detail_level=detail_level,
        )
        sections_markdown = [draft.content for draft in chapter_drafts if draft.content.strip()]

    if not sections_markdown:
        outline = generate_outline(chunks, client=client)
        drafts = draft_sections(
            outline,
            chunks,
            segments,
            client=client,
            detail_level=detail_level,
        )
        sections_markdown = [draft.content for draft in drafts if draft.content.strip()]

    title = outline[0].title if outline else "Generated Blog"
    markdown = assemble_markdown(sections_markdown, title=title, video_url=youtube_url)
    cleaned = strip_emoji(markdown)

    if output_path:
        write_markdown(output_path, cleaned)
    if outline_path:
        outline_path.parent.mkdir(parents=True, exist_ok=True)
        outline_path.write_text(
            json.dumps([topic.__dict__ for topic in outline], indent=2),
            encoding="utf-8",
        )

    return cleaned, outline


def main(argv: Optional[list[str]] = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)

    logging.basicConfig(level=getattr(logging, args.log_level))

    LOGGER.info("Running pipeline...")
    try:
        markdown, outline = run_pipeline(
            args.youtube_url,
            chunk_size=args.chunk_size,
            overlap=args.overlap,
            output_path=args.output,
            outline_path=args.save_outline,
            provider=args.provider,
            detail_level=args.detail_level,
        )
    except GoogleAIError as exc:
        LOGGER.error("Google AI Studio error: %s", exc)
        raise
    except Exception as exc:  # noqa: BLE001
        LOGGER.error("Pipeline failed: %s", exc)
        raise

    if args.output:
        LOGGER.info("Markdown written to %s", args.output)
    if args.save_outline:
        LOGGER.info("Outline written to %s", args.save_outline)


if __name__ == "__main__":
    main()
