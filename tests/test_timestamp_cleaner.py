from __future__ import annotations

from youtube_blog_pipeline.postprocessing.timestamp_cleaner import strip_timestamps


def test_strip_coloned_variants():
    assert strip_timestamps("See 01:23 section") == "See section"
    assert strip_timestamps("Time 1:02:03.5 note") == "Time note"


def test_strip_bracketed():
    assert strip_timestamps("[00:12] intro") == "intro"
    assert strip_timestamps("(12:34) mid") == "mid"


def test_strip_hms():
    assert strip_timestamps("after 1h23m45s we continue") == "after we continue"
    assert strip_timestamps("pause 12m34s resume") == "pause resume"
    assert strip_timestamps("wait 45s") == "wait"


def test_preserve_decimals():
    assert strip_timestamps("pi=3.14 approx") == "pi=3.14 approx"
