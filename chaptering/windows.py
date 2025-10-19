from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Sequence, Tuple

from youtube_blog_pipeline.services.transcript_service import TranscriptSegment


@dataclass
class Window:
    start: float
    end: float
    text: str
    index_range: Tuple[int, int]  # [start_idx, end_idx) into segments


def build_windows(segments: Sequence[TranscriptSegment], win_len: int, stride: int) -> List[Window]:
    if not segments:
        return []
    t0 = segments[0].start
    tN = segments[-1].start + segments[-1].duration
    windows: List[Window] = []
    cur = t0
    while cur < tN:
        w_start = cur
        w_end = min(cur + win_len, tN)
        # collect segments in [w_start, w_end)
        start_idx = None
        end_idx = None
        buf: List[str] = []
        for i, s in enumerate(segments):
            s_end = s.start + s.duration
            if s_end <= w_start:
                continue
            if s.start >= w_end:
                break
            if start_idx is None:
                start_idx = i
            end_idx = i + 1
            buf.append(s.text)
        windows.append(
            Window(
                start=w_start,
                end=w_end,
                text=" ".join(buf).strip(),
                index_range=(start_idx or 0, end_idx or 0),
            )
        )
        cur += stride
    return windows


def super_chunks(windows: Sequence[Window], size: int) -> List[Window]:
    out: List[Window] = []
    for i in range(0, len(windows), size):
        batch = windows[i : i + size]
        if not batch:
            continue
        start = batch[0].start
        end = batch[-1].end
        text = " ".join(w.text for w in batch if w.text).strip()
        idx0 = batch[0].index_range[0]
        idx1 = batch[-1].index_range[1]
        out.append(Window(start=start, end=end, text=text, index_range=(idx0, idx1)))
    return out
