"""Utility to remove emoji characters from markdown files."""

from __future__ import annotations

import argparse
import re
from pathlib import Path


EMOJI_PATTERN = re.compile(
    r"[\U0001F600-\U0001F64F\U0001F300-\U0001F5FF\U0001F680-\U0001F6FF\U0001F1E0-\U0001F1FF"
    r"\U00002700-\U000027BF\U0001F900-\U0001F9FF\U0001FA70-\U0001FAFF\U00002600-\U000026FF"
    r"\U00002B00-\U00002BFF]",
    flags=re.UNICODE,
)


def strip_emoji(text: str) -> str:
    return EMOJI_PATTERN.sub("", text)


def clean_markdown(path: Path) -> None:
    content = path.read_text(encoding="utf-8")
    cleaned = strip_emoji(content)
    path.write_text(cleaned, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Remove emoji from markdown file")
    parser.add_argument("markdown_path", type=Path, help="Path to markdown file to clean")
    args = parser.parse_args()
    clean_markdown(args.markdown_path)


if __name__ == "__main__":
    main()
