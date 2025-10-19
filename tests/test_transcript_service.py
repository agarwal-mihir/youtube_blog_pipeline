from __future__ import annotations

import importlib
import sys
import types
import pytest

from youtube_blog_pipeline.services import transcript_service as ts


def test_extract_video_id_variants():
    assert ts.extract_video_id("dQw4w9WgXcQ") == "dQw4w9WgXcQ"
    assert ts.extract_video_id("https://www.youtube.com/watch?v=dQw4w9WgXcQ") == "dQw4w9WgXcQ"
    assert ts.extract_video_id("https://youtu.be/dQw4w9WgXcQ?si=abc") == "dQw4w9WgXcQ"
    assert ts.extract_video_id("https://www.youtube.com/embed/dQw4w9WgXcQ") == "dQw4w9WgXcQ"
    assert ts.extract_video_id("https://www.youtube.com/shorts/dQw4w9WgXcQ?feature=share") == "dQw4w9WgXcQ"


def test_fetch_transcript_success(monkeypatch):
    entries = [
        {"text": "Hello", "start": 0.0, "duration": 1.0},
        {"text": "World", "start": 1.0, "duration": 1.0},
    ]
    fake_api = importlib.import_module("youtube_blog_pipeline.tests.conftest").make_fake_transcript_api(entries)
    monkeypatch.setitem(sys.modules, "youtube_transcript_api", fake_api)

    import importlib as _importlib
    _importlib.reload(ts)

    segs = ts.fetch_transcript("https://youtu.be/aaaaaaaaaaa")
    assert len(segs) == 2
    assert segs[0].text == "Hello"


def test_fetch_transcript_no_transcript_found(monkeypatch):
    # Fake API that always raises NoTranscriptFound on manual and generated
    class FakeTranscriptList:
        def find_manually_created_transcript(self, languages):
            raise ts.NoTranscriptFound()

        def find_generated_transcript(self, languages):
            raise ts.NoTranscriptFound()

    class FakeYouTubeTranscriptApi:
        @staticmethod
        def list_transcripts(video_id):
            return FakeTranscriptList()

    fake_api = types.SimpleNamespace(
        YouTubeTranscriptApi=FakeYouTubeTranscriptApi,
        NoTranscriptFound=ts.NoTranscriptFound,
        TooManyRequests=ts.TooManyRequests,
        VideoUnavailable=ts.VideoUnavailable,
        TranscriptsDisabled=ts.TranscriptsDisabled,
    )
    monkeypatch.setitem(sys.modules, "youtube_transcript_api", fake_api)
    import importlib as _importlib
    _importlib.reload(ts)

    with pytest.raises(ts.TranscriptUnavailableError) as ei:
        ts.fetch_transcript("dQw4w9WgXcQ")
    assert "No transcript found" in str(ei.value)


def test_fetch_transcript_rate_limited(monkeypatch):
    class FakeTranscriptList:
        def find_manually_created_transcript(self, languages):
            raise ts.TooManyRequests()

        def find_generated_transcript(self, languages):
            raise ts.TooManyRequests()

    class FakeYouTubeTranscriptApi:
        @staticmethod
        def list_transcripts(video_id):
            return FakeTranscriptList()

    fake_api = types.SimpleNamespace(
        YouTubeTranscriptApi=FakeYouTubeTranscriptApi,
        NoTranscriptFound=ts.NoTranscriptFound,
        TooManyRequests=ts.TooManyRequests,
        VideoUnavailable=ts.VideoUnavailable,
        TranscriptsDisabled=ts.TranscriptsDisabled,
    )
    monkeypatch.setitem(sys.modules, "youtube_transcript_api", fake_api)
    import importlib as _importlib
    _importlib.reload(ts)

    with pytest.raises(ts.TranscriptUnavailableError) as ei:
        ts.fetch_transcript("dQw4w9WgXcQ")
    assert "Rate limited" in str(ei.value)


def test_fetch_transcript_video_unavailable(monkeypatch):
    class FakeTranscriptList:
        def find_manually_created_transcript(self, languages):
            raise ts.VideoUnavailable()

        def find_generated_transcript(self, languages):
            raise ts.VideoUnavailable()

    class FakeYouTubeTranscriptApi:
        @staticmethod
        def list_transcripts(video_id):
            return FakeTranscriptList()

    fake_api = types.SimpleNamespace(
        YouTubeTranscriptApi=FakeYouTubeTranscriptApi,
        NoTranscriptFound=ts.NoTranscriptFound,
        TooManyRequests=ts.TooManyRequests,
        VideoUnavailable=ts.VideoUnavailable,
        TranscriptsDisabled=ts.TranscriptsDisabled,
    )
    monkeypatch.setitem(sys.modules, "youtube_transcript_api", fake_api)
    import importlib as _importlib
    _importlib.reload(ts)

    with pytest.raises(ts.TranscriptUnavailableError) as ei:
        ts.fetch_transcript("dQw4w9WgXcQ")
    assert "Video unavailable" in str(ei.value)
