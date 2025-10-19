"""Service for retrieving YouTube video transcripts.

Supports both legacy (<=0.6.x) and modern (>=1.x) versions of youtube-transcript-api.
Prefers instance API (fetch/list) when available, with fallbacks to legacy
class-based API. Provides clear, user-friendly error messages.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Iterable, List, Optional

# Import module and exceptions compatibly across versions
import youtube_transcript_api as _yta  # type: ignore

YouTubeTranscriptApi = _yta.YouTubeTranscriptApi  # type: ignore[attr-defined]

try:  # modern versions export errors under _errors
    from youtube_transcript_api._errors import (  # type: ignore
        TranscriptsDisabled as _TranscriptsDisabled,
        NoTranscriptFound as _NoTranscriptFound,
        TooManyRequests as _TooManyRequests,
        VideoUnavailable as _VideoUnavailable,
        RequestBlocked as _RequestBlocked,
        IpBlocked as _IpBlocked,
    )
    TranscriptsDisabled = _TranscriptsDisabled
    NoTranscriptFound = _NoTranscriptFound
    TooManyRequests = _TooManyRequests
    VideoUnavailable = _VideoUnavailable
    RequestBlocked = _RequestBlocked
    IpBlocked = _IpBlocked
except Exception:  # fallback for legacy versions
    TranscriptsDisabled = getattr(_yta, "TranscriptsDisabled", Exception)
    NoTranscriptFound = getattr(_yta, "NoTranscriptFound", Exception)
    TooManyRequests = getattr(_yta, "TooManyRequests", Exception)
    VideoUnavailable = getattr(_yta, "VideoUnavailable", Exception)
    RequestBlocked = getattr(_yta, "RequestBlocked", Exception)
    IpBlocked = getattr(_yta, "IpBlocked", Exception)

# Build a safe tuple of rate-limit related exceptions; avoid catching broad Exception fallbacks
_rate_limit_classes = []
for _cls in (TooManyRequests, RequestBlocked, IpBlocked):
    try:
        if _cls is not Exception:
            _rate_limit_classes.append(_cls)
    except Exception:
        pass
RATE_LIMIT_EXCEPTIONS = tuple(_rate_limit_classes)

LOGGER = logging.getLogger(__name__)


@dataclass
class TranscriptSegment:
    """Represents a single transcript segment."""

    text: str
    start: float
    duration: float


class TranscriptUnavailableError(RuntimeError):
    """Raised when transcript retrieval fails."""


def extract_video_id(url_or_id: str) -> str:
    """Extract the video ID from a YouTube URL or return the input if already an ID."""

    if len(url_or_id) == 11 and "/" not in url_or_id:
        return url_or_id

    # Common URL patterns
    patterns = ("v=", "youtu.be/", "shorts/", "embed/")
    for token in patterns:
        if token in url_or_id:
            remainder = url_or_id.split(token, 1)[1]
            candidate = remainder.split("?", 1)[0].split("&", 1)[0].strip()
            if candidate:
                return candidate

    raise ValueError(f"Unable to extract video ID from: {url_or_id}")


def fetch_transcript(
    url_or_id: str,
    preferred_languages: Optional[Iterable[str]] = None,
) -> List[TranscriptSegment]:
    """Fetch the transcript for a YouTube video with robust fallbacks and errors."""

    video_id = extract_video_id(url_or_id)
    languages = list(preferred_languages or ("en", "en-US", "en-GB"))

    try:
        # Prefer modern instance API if available
        if not hasattr(YouTubeTranscriptApi, "list_transcripts"):
            api = YouTubeTranscriptApi()
            try:
                fetched = api.fetch(video_id, languages=languages)
                transcript = fetched.to_raw_data()
            except NoTranscriptFound:
                # Attempt list + generated fallback
                tlist = api.list(video_id)
                try:
                    t = tlist.find_manually_created_transcript(languages)
                except NoTranscriptFound:
                    t = tlist.find_generated_transcript(languages)
                transcript = t.fetch().to_raw_data()
        else:
            # Legacy class API path
            tlist = YouTubeTranscriptApi.list_transcripts(video_id)
            try:
                t = tlist.find_manually_created_transcript(languages)
            except NoTranscriptFound:
                t = tlist.find_generated_transcript(languages)
            transcript = t.fetch()
    except RATE_LIMIT_EXCEPTIONS as exc:  # type: ignore[misc]
        raise TranscriptUnavailableError("Rate limited by YouTube; please retry in a few minutes.") from exc
    except TranscriptsDisabled as exc:  # type: ignore[misc]
        raise TranscriptUnavailableError("Transcripts disabled for this video") from exc
    except VideoUnavailable as exc:  # type: ignore[misc]
        raise TranscriptUnavailableError("Video unavailable or region-restricted.") from exc
    except NoTranscriptFound as exc:  # type: ignore[misc]
        raise TranscriptUnavailableError(
            f"No transcript found for preferred languages: {', '.join(languages)}"
        ) from exc
    except Exception as exc:  # noqa: BLE001
        raise TranscriptUnavailableError(f"Unexpected transcript error: {exc}") from exc

    segments = [
        TranscriptSegment(text=entry["text"].strip(), start=entry["start"], duration=entry["duration"])
        for entry in transcript
        if entry.get("text")
    ]
    LOGGER.info("Fetched %d transcript segments for video %s", len(segments), video_id)
    return segments
