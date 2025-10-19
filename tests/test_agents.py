from __future__ import annotations

from youtube_blog_pipeline.agents.outline_agent import OutlineTopic, generate_outline
from youtube_blog_pipeline.agents.writing_agent import draft_sections, draft_chapter_sections
from youtube_blog_pipeline.chaptering.cluster import Chapter
from youtube_blog_pipeline.chaptering.paragraphs import StructuredParagraph
from youtube_blog_pipeline.processing.text_processing import Chunk
from youtube_blog_pipeline.services.transcript_service import TranscriptSegment


def _chunks():
    return [
        Chunk(text="Intro part one", start=0.0, end=30.0, segment_indices=[0, 1]),
        Chunk(text="Topic part two", start=30.0, end=60.0, segment_indices=[2, 3]),
    ]


def _segments():
    return [
        TranscriptSegment(text="Intro", start=0.0, duration=5.0),
        TranscriptSegment(text="Definition", start=10.0, duration=5.0),
        TranscriptSegment(text="Example", start=25.0, duration=5.0),
        TranscriptSegment(text="Proof", start=40.0, duration=5.0),
    ]


def test_generate_outline_parses_json(stub_lmstudio_client):
    outline = generate_outline(_chunks(), client=stub_lmstudio_client)
    assert outline
    assert outline[0].title


def test_draft_sections_returns_content(stub_lmstudio_client):
    outline = [OutlineTopic(title="Section 1", summary="Summary 1", start=0, end=30)]
    drafts = draft_sections(outline, _chunks(), _segments(), client=stub_lmstudio_client)
    assert drafts
    assert drafts[0].title == "Section 1"
    assert drafts[0].content


def test_draft_chapter_sections_returns_content(stub_lmstudio_client):
    chapter = Chapter(title="Intro", start=0.0, end=60.0, paragraph_indices=[0, 1])
    paragraphs = [
        StructuredParagraph(text="Intro paragraph.", start=0.0, end=10.0, segment_range=(0, 1), chunk_index=0),
        StructuredParagraph(text="Next paragraph.", start=10.0, end=20.0, segment_range=(1, 2), chunk_index=0),
    ]
    drafts = draft_chapter_sections([chapter], paragraphs, _segments(), client=stub_lmstudio_client)
    assert drafts
    assert drafts[0].title == "Intro"
    assert drafts[0].content.startswith("## Intro")
