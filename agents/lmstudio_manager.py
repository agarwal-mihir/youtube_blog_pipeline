import json
import logging
import os
import shlex
import subprocess

from youtube_blog_pipeline.config import DEFAULT_CONFIG

LOGGER = logging.getLogger(__name__)


def _run(args: list[str]) -> None:
    command = " ".join(shlex.quote(arg) for arg in args)
    LOGGER.info("Running: %s", command)
    exit_code = os.system(command)
    if exit_code != 0:
        raise RuntimeError(f"Command failed with exit code {exit_code}: {command}")


def _loaded_identifiers() -> list[str]:
    try:
        result = subprocess.run(
            ["lms", "ps", "--json"],
            capture_output=True,
            text=True,
            check=True,
        )
    except subprocess.CalledProcessError:
        return []

    try:
        entries = json.loads(result.stdout or "[]")
    except json.JSONDecodeError:
        return []

    identifiers: set[str] = set()
    for entry in entries:
        identifier = entry.get("identifier")
        path = entry.get("path")
        if identifier:
            identifiers.add(str(identifier))
        if path:
            identifiers.add(str(path))
    return list(identifiers)


def _is_loaded(target: str) -> bool:
    if not target:
        return False
    loaded = _loaded_identifiers()
    return any(item == target or item.startswith(f"{target}:") for item in loaded)


def ensure_lmstudio_ready(
    *,
    load_server: bool = True,
    load_llm: bool = True,
    load_embeddings: bool = True,
) -> None:
    if load_server:
        _run(["lms", "server", "start"])

    model = DEFAULT_CONFIG.lmstudio.model
    embedding_model = getattr(DEFAULT_CONFIG.chaptering, "embedding_model", "")

    if load_llm and model:
        if not _is_loaded(model):
            cmd = ["lms", "load", model, "-y"]
            ctx = getattr(DEFAULT_CONFIG.lmstudio, "context_length", None)
            if ctx:
                cmd.extend(["--context-length", str(ctx)])
            _run(cmd)
        else:
            LOGGER.info("Model '%s' already loaded; skipping", model)

    if load_embeddings and embedding_model:
        if not _is_loaded(embedding_model):
            cmd = ["lms", "load", embedding_model, "-y"]
            embed_ctx = getattr(DEFAULT_CONFIG.chaptering, "embedding_context_length", None)
            if embed_ctx:
                cmd.extend(["--context-length", str(embed_ctx)])
            _run(cmd)
        else:
            LOGGER.info("Embedding model '%s' already loaded; skipping", embedding_model)
