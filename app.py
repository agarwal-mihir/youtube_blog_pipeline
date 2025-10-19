"""Streamlit interface for generating markdown notes from YouTube videos."""

from __future__ import annotations

import streamlit as st

from youtube_blog_pipeline.agents.google_client import GoogleAIError
from youtube_blog_pipeline.agents.lmstudio_manager import ensure_lmstudio_ready
from youtube_blog_pipeline.main import run_pipeline
from youtube_blog_pipeline.services.transcript_service import extract_video_id
from pathlib import Path

try:
    from dotenv import load_dotenv
except ImportError:  # pragma: no cover - optional dependency
    load_dotenv = None

if load_dotenv:
    load_dotenv()
    load_dotenv(Path(__file__).resolve().parent / ".env")


st.set_page_config(page_title="YouTube Transcript to Notes", layout="wide")
st.title("YouTube Chapters & Notes")

# Load server + embeddings on first load
if "lmstudio_embeddings_ready" not in st.session_state:
    with st.spinner("Starting LM Studio server and loading embedding model..."):
        try:
            ensure_lmstudio_ready(load_llm=False)
            st.caption("LM Studio ready.")
        except Exception as _exc:  # noqa: BLE001
            st.warning(f"LM Studio not fully ready: {_exc}")
            st.info("Check terminal logs for LM Studio manager debug output.")
    st.session_state["lmstudio_embeddings_ready"] = True

with st.form("input_form"):
    youtube_url = st.text_input("YouTube URL", placeholder="https://www.youtube.com/watch?v=...")
    provider_label = st.selectbox(
        "Generation model",
        options=[
            "LM Studio (local Qwen)",
            "Google AI Studio (Gemini)",
        ],
        index=0,
    )
    detail_label = st.selectbox(
        "Note detail level",
        options=[
            "High detail (comprehensive lecture notes)",
            "Low detail (concise key takeaways)",
        ],
        index=0,
    )
    submitted = st.form_submit_button("Generate Chapters & Notes")

if submitted:
    url = youtube_url.strip()
    if not url:
        st.error("Please provide a valid YouTube URL.")
    else:
        try:
            video_id = extract_video_id(url)
        except ValueError as exc:
            st.error(f"Unable to parse YouTube ID: {exc}")
        else:
            notes_path = Path("output") / f"{video_id}_notes.md"
            outline_path = Path("output") / f"{video_id}_outline.json"

            provider_key = (
                "google" if "google" in provider_label.lower() else "lmstudio"
            )
            detail_level = "low" if "low" in detail_label.lower() else "high"

            if provider_key == "lmstudio" and not st.session_state.get("lmstudio_llm_ready"):
                with st.spinner("Loading LM Studio language model..."):
                    try:
                        ensure_lmstudio_ready(load_llm=True, load_embeddings=False)
                    except Exception as exc:  # noqa: BLE001
                        st.error(f"Unable to load LM Studio language model: {exc}")
                        st.stop()
                st.session_state["lmstudio_llm_ready"] = True

            with st.spinner("Running pipeline..."):
                try:
                    markdown, outline = run_pipeline(
                        url,
                        output_path=notes_path,
                        outline_path=outline_path,
                        provider=provider_key,
                        detail_level=detail_level,
                    )
                except GoogleAIError as exc:
                    st.error(f"Google AI Studio error: {exc}")
                except Exception as exc:  # noqa: BLE001
                    st.error(f"Pipeline failed: {exc}")
                else:
                    st.success("Chapters and notes generated.")
                    st.caption(f"Markdown stored at `{notes_path}`")
                    st.caption(f"Outline stored at `{outline_path}`")

                    if outline:
                        st.subheader("Chapters")
                        for idx, topic in enumerate(outline, start=1):
                            timing = ""
                            if topic.start is not None and topic.end is not None:
                                timing = f" ({topic.start:0.0f}s – {topic.end:0.0f}s)"
                            st.markdown(f"**{idx}. {topic.title}{timing}**\n\n{topic.summary}")

                    st.subheader("Lecture Notes (Markdown)")
                    st.download_button(
                        label="Download Markdown",
                        data=markdown.encode("utf-8"),
                        file_name=notes_path.name,
                        mime="text/markdown",
                    )
                    st.markdown(markdown)
