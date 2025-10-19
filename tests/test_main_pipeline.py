from __future__ import annotations

import importlib
from youtube_blog_pipeline import main as pipeline
from youtube_blog_pipeline.services import transcript_service as ts


class FakeClient:
    def __init__(self):
        self._calls = 0

    def chat_completion(self, messages, **kwargs):
        self._calls += 1
        # Alternate between outline JSON and simple markdown
        import json

        user_text = "\n".join(m.content for m in messages if getattr(m, "role", "") == "user")
        if "Output as JSON array" in user_text:
            return json.dumps([
                {"title": "Intro", "summary": "Overview", "start": 0, "end": 30},
                {"title": "Topic", "summary": "Details", "start": 30, "end": 60},
            ])
        if "Return the result wrapped in <answer>" in user_text:
            return "<answer>Intro paragraph.\n\nNext paragraph.</answer>"
        if "Provide a 3-6 word title" in user_text:
            return "Introduction"
        if "Transcript paragraphs for this chapter" in user_text:
            title_line = next(
                (line.split(":", 1)[1].strip() for line in user_text.splitlines() if line.startswith("Chapter title:")),
                "Chapter",
            )
            return f"## {title_line}\nNotes for {title_line}."
        return "## Section\nContent."


def test_run_pipeline_end_to_end(monkeypatch):
    # Stub transcript fetch to return deterministic segments
    monkeypatch.setattr(
        pipeline,
        "fetch_transcript",
        lambda url: [
            ts.TranscriptSegment(text="Intro", start=0.0, duration=5.0),
            ts.TranscriptSegment(text="Definition", start=10.0, duration=5.0),
            ts.TranscriptSegment(text="Example", start=25.0, duration=5.0),
            ts.TranscriptSegment(text="Proof", start=40.0, duration=5.0),
        ],
    )

    client = FakeClient()
    markdown, outline = pipeline.run_pipeline("https://youtu.be/aaaaaaaaaaa", client=client)
    assert outline
    assert markdown and "##" in markdown
