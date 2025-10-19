"""Utilities to remove timestamp patterns from text."""

from __future__ import annotations

import re


_BRACKETED = re.compile(r"[\[(](?:\d{1,2}:){1,2}\d{2}(?:\.\d{1,3})?[)\]]")
_COLONED = re.compile(r"\b(?:\d{1,2}:){1,2}\d{2}(?:\.\d{1,3})?\b")
_HMS = re.compile(r"\b\d+h(?:\d+m)?(?:\d+s)?\b|\b\d+m\d+s\b|\b\d+s\b")


def strip_timestamps(text: str) -> str:
    """Remove common timecode formats like 01:23, 1:02:03.5, [00:12], 1h23m45s, 45s.

    This is conservative: it targets bracketed/paren timecodes, coloned groups,
    and h/m/s units, while keeping regular decimals (e.g., 3.14) intact.
    """

    if not text:
        return text

    cleaned = _BRACKETED.sub(" ", text)
    cleaned = _COLONED.sub(" ", cleaned)
    cleaned = _HMS.sub(" ", cleaned)
    # collapse whitespace
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    return cleaned
