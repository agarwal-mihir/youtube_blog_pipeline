from __future__ import annotations

import types
import pytest


@pytest.fixture
def sample_segments():
    return [
        {"text": "Intro", "start": 0.0, "duration": 5.0},
        {"text": "Definition", "start": 10.0, "duration": 5.0},
        {"text": "Example", "start": 25.0, "duration": 5.0},
        {"text": "Proof", "start": 40.0, "duration": 5.0},
    ]


class StubLMStudioClient:
    def __init__(self, responses: list[str] | None = None):
        self._responses = responses or []
        self.calls: list[dict] = []

    def chat_completion(
        self,
        messages,
        *,
        max_tokens=None,
        temperature=None,
        top_p=None,
        stop=None,
    ) -> str:
        self.calls.append({"messages": messages})
        if self._responses:
            return self._responses.pop(0)
        # sensible default: return minimal JSON for outline, or echo request for others
        import json

        user_content = "\n\n".join(m.content for m in messages if getattr(m, "role", "") == "user")
        if "Output as JSON array" in user_content:
            return json.dumps([
                {"title": "Section 1", "summary": "Summary 1", "start": 0, "end": 30},
                {"title": "Section 2", "summary": "Summary 2", "start": 30, "end": 60},
            ])
        return "## Section\nContent"


@pytest.fixture
def stub_lmstudio_client():
    return StubLMStudioClient()


def make_fake_transcript_api(entries):
    class FakeTranscript:
        def fetch(self):
            return entries

    class FakeTranscriptList:
        def find_manually_created_transcript(self, languages):
            raise NoTranscriptFound()

        def find_generated_transcript(self, languages):
            return FakeTranscript()

    class TooManyRequests(Exception):
        pass

    class VideoUnavailable(Exception):
        pass

    class TranscriptsDisabled(Exception):
        pass

    class NoTranscriptFound(Exception):
        pass

    class FakeYouTubeTranscriptApi:
        @staticmethod
        def list_transcripts(video_id):
            return FakeTranscriptList()

    ns = types.SimpleNamespace(
        YouTubeTranscriptApi=FakeYouTubeTranscriptApi,
        TooManyRequests=TooManyRequests,
        VideoUnavailable=VideoUnavailable,
        TranscriptsDisabled=TranscriptsDisabled,
        NoTranscriptFound=NoTranscriptFound,
    )
    return ns
