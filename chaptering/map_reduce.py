from __future__ import annotations

import json
from dataclasses import dataclass
from typing import List

from youtube_blog_pipeline.agents.lmstudio_client import LMStudioClient, Message
from youtube_blog_pipeline.chaptering.windows import Window


@dataclass
class MiniSummary:
    topic: str
    when: list[float]  # [start, end]
    bullets: list[str]
    keywords: list[str]


def _prompt_for_superchunk(sc: Window) -> str:
    return (
        "You are given a transcript excerpt and its time window in seconds. "
        "Summarize this window in 1-2 sentences, and return JSON with keys: "
        "topic (string), when [start,end] (numbers), bullets (array of 2-4 short points), "
        "keywords (array of up to 5 tokens). Use the provided start and end seconds exactly.\n\n"
        f"start: {sc.start}\nend: {sc.end}\n\ntranscript:\n{sc.text[:4000]}\n"
        "Return only JSON."
    )


def map_summarize(superchunks: List[Window], client: LMStudioClient) -> List[MiniSummary]:
    out: List[MiniSummary] = []
    for sc in superchunks:
        messages = [
            Message(role="system", content="You output strict JSON only."),
            Message(role="user", content=_prompt_for_superchunk(sc)),
        ]
        resp = client.chat_completion(messages)
        try:
            data = json.loads(resp)
            out.append(
                MiniSummary(
                    topic=data.get("topic", ""),
                    when=[float(sc.start), float(sc.end)],  # enforce provided
                    bullets=[b for b in (data.get("bullets") or []) if isinstance(b, str)][:4],
                    keywords=[k for k in (data.get("keywords") or []) if isinstance(k, str)][:5],
                )
            )
        except Exception:
            out.append(
                MiniSummary(topic="", when=[float(sc.start), float(sc.end)], bullets=[], keywords=[])
            )
    return out
