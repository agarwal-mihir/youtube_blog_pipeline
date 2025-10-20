"""Global configuration defaults for the YouTube blog pipeline."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Optional


@dataclass
class LMStudioConfig:
    """Configuration for communicating with the LM Studio local API."""

    base_url: str = "http://127.0.0.1:1234/v1"
    model: str = "qwen/qwen3-4b-2507"
    temperature: float = 0.2
    # Use -1 to let LM Studio generate up to the model/runtime limits
    max_tokens: int = -1
    top_p: float = 0.95
    # Per-request timeout (connect + read) for LM Studio API calls
    timeout: float = 1800.0
    # Documented model max context length (not directly settable via API)
    context_length: int = 262144
    request_headers: Dict[str, str] = field(
        default_factory=lambda: {"Content-Type": "application/json"}
    )


@dataclass
class ChunkingConfig:
    """Configuration for transcript chunking."""

    max_tokens: int = 1800
    overlap_tokens: int = 200
    merge_gap_seconds: float = 1.5


@dataclass
class PipelineConfig:
    """Top-level pipeline configuration values."""

    lmstudio: LMStudioConfig = field(default_factory=LMStudioConfig)
    @dataclass
    class GoogleAIConfig:
        model: str = "models/gemma-3-27b-it"
        temperature: float = 0.2
        top_p: float = 0.95
        max_tokens: int = 4096
        stop_sequences: Optional[tuple[str, ...]] = None
        api_key_env_vars: tuple[str, ...] = ("GOOGLE_API_KEY", "GEMINI_API_KEY")
        supports_system_instruction: bool = False
        rate_limit_retries: int = 3
        rate_limit_backoff_seconds: float = 60.0

    chunking: ChunkingConfig = field(default_factory=ChunkingConfig)
    system_prompt: str = (
        "You are an expert technical writer producing rigorous lecture notes from transcripts. "
        "Every statement must be directly supported by the provided transcript excerpts and follow their chronological order. "
        "Do not add outside knowledge, speculation, predictions, or examples that are not explicitly mentioned. "
        "If the transcript lacks detail, briefly acknowledge the limitation without requesting additional input, then continue. "
        "Never ask the user for more information. "
        "Do not use emojis in any output."
    )
    outline_prompt: str = (
        "Given the transcript context, produce a structured outline of topics with concise "
        "summaries. Provide for each topic: a descriptive title and a 1-2 sentence summary."
    )
    reviewer_prompt: str = (
        "Review the provided section draft alongside transcript excerpts. Add missing details, "
        "clarify proofs, ensure accuracy, and maintain continuity. Output the improved markdown "
        "section starting from the existing heading. Ensure the result is valid Markdown and typeset "
        "any mathematics with correct LaTeX delimiters ($...$ for inline, $$...$$ for display)."
    )
    history_max_entries: int = 4
    history_summary_chars: int = 220
    detail_prompts: Dict[str, str] = field(
        default_factory=lambda: {
            "high": (
                "Write comprehensive Markdown lecture notes for this chapter strictly using the supplied transcript paragraphs. "
                "Retell the material in order and elaborate only when the transcript explicitly supports the detail. "
                "Do not introduce background knowledge, speculation, hypotheticals, or examples that the speaker does not mention. "
                "If the transcript is sparse, produce a faithful, concise explanation and note that detail is limited."
            ),
            "low": (
                "Write a concise Markdown summary for this chapter using brief prose drawn only from the supplied transcript paragraphs. "
                "Limit the output to one or two short paragraphs covering the essential ideas explicitly mentioned. "
                "Do not speculate, add external context, or use bullet or "
                "numbered lists."
            ),
        }
    )
    detail_max_tokens: Dict[str, Optional[int]] = field(
        default_factory=lambda: {
            "high": None,
            "low": 700,
        }
    )

    # When True, scrub timestamps from transcripts and previews
    redact_timestamps: bool = True
    
    # Coverage-guaranteed chaptering settings
    @dataclass
    class ChapteringConfig:
        window_len_seconds: int = 180
        stride_seconds: int = 30
        superchunk_size: int = 5
        sim_threshold: float = 0.58
        gap_threshold_seconds: int = 10
        # Embedding model served via LM Studio embeddings endpoint
        embedding_model: str = "Qwen/Qwen3-Embedding-0.6B-GGUF/Qwen3-Embedding-0.6B-Q8_0.gguf"
        # Desired context length for the embedding model
        embedding_context_length: int = 8192
        map_max_chars: int = 40000
        paragraph_max_chars: int = 3500
        paragraph_sample_words: int = 50

    chaptering: ChapteringConfig = field(default_factory=ChapteringConfig)
    use_coverage_outline: bool = True
    google: GoogleAIConfig = field(default_factory=GoogleAIConfig)


DEFAULT_CONFIG = PipelineConfig()
