<details>
<summary><strong>YouTube Transcript → Blog Pipeline</strong></summary>

Transform YouTube transcripts into polished, chaptered lecture notes using LM Studio (local) or Google AI Studio (Gemini) models. The pipeline normalizes transcripts, chunks them for LLM prompting, drafts sections, and assembles publication-ready Markdown.

</details>

## Features

- **Dual model support**: generate content with local LM Studio models or Google AI Studio.
- **Automatic chaptering**: paragraph alignment, semantic clustering, and title generation with configurable thresholds.
- **Structured drafting loop**: outline creation, section drafting, and emoji/timestamp cleanup.
- **Multiple surfaces**: Streamlit UI (`app.py`) and CLI (`main.py`).
- **Deterministic tests**: extensive pytest coverage for services, processing, and agent utilities.

## Quickstart

```bash
git clone https://github.com/agarwal-mihir/youtube_blog_pipeline.git
cd youtube_blog_pipeline
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -e .
```

## Configuration

The pipeline defaults are defined in `config.py`. Key options:

| Section | Purpose | Highlights |
| --- | --- | --- |
| `lmstudio` | Local OpenAI-compatible settings | Base URL, model key (`qwen/qwen3-4b-2507`), context length, request headers |
| `chunking` | Transcript segmentation | `max_tokens`, `overlap_tokens`, merge gap tolerance |
| `chaptering` | Coverage-first chapter discovery | Embedding model, similarity threshold, window/stride seconds |
| `detail_prompts` | Draft verbosity | High/low detail instructions and token limits |
| `google` | Google AI Studio | Model (`models/gemma-3-27b-it`), API key env vars |

`.env` files are automatically loaded (root `.env` plus package-level `.env`).

### Environment variables

| Name | Usage |
| --- | --- |
| `GOOGLE_API_KEY` or `GEMINI_API_KEY` | Required when `provider=google` |
| `LMSTUDIO_*` (server defaults) | LM Studio must be running and accessible at `DEFAULT_CONFIG.lmstudio.base_url` |

## LM Studio setup

1. Install LM Studio and the specified chat model (`qwen/qwen3-4b-2507`).
2. Start the server and load models (chat + embedding) via `agents/lmstudio_manager.ensure_lmstudio_ready()` or Streamlit’s first-run spinner.
3. Ensure the API is reachable at `http://127.0.0.1:1234/v1`.

Optional helper:

```bash
python scripts/start_lmstudio.py qwen/qwen3-4b-2507 --context-length 262144
```

## CLI usage

```bash
python -m youtube_blog_pipeline.main \
  "https://www.youtube.com/watch?v=dQw4w9WgXcQ" \
  --output output/blog_post.md \
  --save-outline output/outline.json \
  --provider lmstudio \
  --detail-level high
```

Arguments:

- `--provider`: `lmstudio` (default) or `google`.
- `--detail-level`: `high` (comprehensive) or `low` (concise paragraph summaries).
- `--chunk-size`, `--overlap`: override transcript chunking.
- `--log-level`: standard Python logging level.

Outputs:

- Markdown lecture notes (with YAML front matter) written to `--output` path.
- JSON outline (if `--save-outline` is provided).

## Streamlit app

```bash
streamlit run app.py
```

Features:

- Select LM Studio or Google AI Studio providers.
- Choose detail level (high/low).
- Inline feedback when LM Studio bootstrap fails.
- Download markdown results directly from the UI.

## Architecture overview

```
services/transcript_service.py   → fetch + normalize YouTube transcripts
processing/text_processing.py    → segment normalization & chunk creation
chaptering/*                     → paragraph formatting, embeddings, clustering, titling
agents/*                         → outline, drafting, and LM integrations
assembly/blog_assembler.py       → generate final markdown with front matter
postprocessing/*                 → strip emojis and timestamps
app.py / main.py                 → Streamlit UI and CLI entrypoints
```

Tests in `tests/` stub external dependencies (YouTube, LM Studio) to keep execution self-contained.

## Testing

```bash
pytest
```

All suites pass locally (`18 passed`).

## Troubleshooting

- **Missing transcript**: the pipeline raises `TranscriptUnavailableError` with descriptive guidance.
- **LM Studio not ready**: use Streamlit prompts or run `ensure_lmstudio_ready()` manually.
- **Google AI auth errors**: confirm `GOOGLE_API_KEY` or `GEMINI_API_KEY` is exported before running.
- **Long runs/timeouts**: adjust chunk sizes or detail level to reduce prompt size.

## Contributing

1. Fork & clone the repository.
2. Create a feature branch.
3. Add tests for new functionality.
4. Submit a pull request with context and screenshots if UI changes are included.

## License

MIT © Mihir Agarwal.
