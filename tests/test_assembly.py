from __future__ import annotations

from pathlib import Path
from youtube_blog_pipeline.assembly.blog_assembler import assemble_markdown, write_markdown


def test_assemble_markdown_includes_frontmatter():
    md = assemble_markdown(["## A\nB", "## C\nD"], title="T", video_url="https://youtu.be/x")
    assert md.startswith("---\n")
    assert "title: \"T\"" in md
    assert "original_video: https://youtu.be/x" in md
    assert "## A" in md and "## C" in md


def test_write_markdown(tmp_path: Path):
    out = tmp_path / "out.md"
    content = "---\n---\nBody"
    write_markdown(out, content)
    assert out.exists()
    assert out.read_text(encoding="utf-8").endswith("Body")
