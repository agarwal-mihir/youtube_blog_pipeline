"""Assemble final markdown output."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Iterable, Optional


def assemble_markdown(
    sections: Iterable[str],
    *,
    title: str,
    video_url: str,
    description: Optional[str] = None,
    include_frontmatter: bool = True,
) -> str:
    date_str = datetime.utcnow().strftime("%Y-%m-%d")
    description = description or "Detailed lecture notes generated from transcript."

    frontmatter = (
        "---\n"
        f"title: \"{title}\"\n"
        f"date: {date_str}\n"
        f"original_video: {video_url}\n"
        f"summary: \"{description}\"\n"
        "toc: true\n"
        "---\n\n"
    ) if include_frontmatter else ""

    body = "\n\n".join(section.strip() for section in sections if section.strip())
    return f"{frontmatter}{body}\n"


def write_markdown(output_path: Path, markdown: str) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(markdown, encoding="utf-8")
