"""Model-name resolution and backend session cache.

Public request model names (short aliases) map onto ``qwen3_embed``
registry names; names ending in ``-GGUF`` route to the GGUF backend,
everything else to ONNX. Sessions are cached per resolved name so one
daemon serves every product off a single model instance.

Cache location: ``HULL_EMBEDDING_CACHE_DIR`` wins, else qwen3-embed's own
lookup (``QWEN3_EMBED_CACHE_PATH`` -> ``$XDG_CACHE_HOME`` or the platform
user cache). Inference threads: ``HULL_EMBEDDING_THREADS``.
"""

from __future__ import annotations

import os
import threading

from hull_embedding_daemon.backends._qwen3 import Qwen3BackendBase

# Public alias -> qwen3_embed registry name. Aliases are resolved
# case-insensitively; a name that matches no alias is passed through
# unchanged so operators can point at custom / local qwen3-embed models.
EMBED_MODELS: dict[str, str] = {
    "qwen3-0.6b": "n24q02m/Qwen3-Embedding-0.6B-ONNX",
    "qwen3-0.6b-q4f16": "n24q02m/Qwen3-Embedding-0.6B-ONNX-Q4F16",
    "qwen3-0.6b-gguf": "n24q02m/Qwen3-Embedding-0.6B-GGUF",
}

RERANK_MODELS: dict[str, str] = {
    "qwen3-rerank-0.6b": "n24q02m/Qwen3-Reranker-0.6B-ONNX",
    "qwen3-rerank-0.6b-q4f16": "n24q02m/Qwen3-Reranker-0.6B-ONNX-Q4F16",
    "qwen3-rerank-0.6b-yesno": "n24q02m/Qwen3-Reranker-0.6B-ONNX-YesNo",
    "qwen3-rerank-0.6b-gguf": "n24q02m/Qwen3-Reranker-0.6B-GGUF",
}

CACHE_DIR_ENV = "HULL_EMBEDDING_CACHE_DIR"
THREADS_ENV = "HULL_EMBEDDING_THREADS"

_sessions: dict[tuple[str, str], Qwen3BackendBase] = {}
_sessions_lock = threading.Lock()


def resolve_model(requested: str, kind: str) -> str:
    """Map a request ``model`` name onto the qwen3-embed registry name.

    ``kind`` is ``"embed"`` or ``"rerank"`` and selects the alias table;
    unrecognised names pass through so qwen3_embed can validate them.
    """
    table = EMBED_MODELS if kind == "embed" else RERANK_MODELS
    return table.get(requested.lower(), requested)


def is_gguf_model(model_name: str) -> bool:
    """True when the resolved name selects a GGUF artifact."""
    return model_name.lower().endswith("-gguf")


def _threads() -> int | None:
    raw = os.environ.get(THREADS_ENV)
    if not raw:
        return None
    try:
        value = int(raw)
    except ValueError:
        return None
    return value if value > 0 else None


def get_backend(requested: str, kind: str) -> Qwen3BackendBase:
    """Return the cached backend for a request model name, creating it lazily."""
    model_name = resolve_model(requested, kind)
    key = (kind, model_name.lower())
    with _sessions_lock:
        backend = _sessions.get(key)
        if backend is None:
            if is_gguf_model(model_name):
                from hull_embedding_daemon.backends.gguf import GGUFBackend  # noqa: PLC0415

                backend = GGUFBackend(model_name, cache_dir=os.environ.get(CACHE_DIR_ENV), threads=_threads())
            else:
                from hull_embedding_daemon.backends.onnx import ONNXBackend  # noqa: PLC0415

                backend = ONNXBackend(model_name, cache_dir=os.environ.get(CACHE_DIR_ENV), threads=_threads())
            _sessions[key] = backend
        return backend


def reset_sessions() -> None:
    """Drop every cached backend. Used by tests and before reconfiguring."""
    with _sessions_lock:
        _sessions.clear()
