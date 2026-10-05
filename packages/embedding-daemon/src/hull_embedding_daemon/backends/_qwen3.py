"""Shared wiring for the qwen3-embed backends.

Both the ONNX and the GGUF backends are thin adapters over ``qwen3_embed``:
the library dispatches on the model name to its ONNX / GGUF loaders, so each
backend here is only responsible for lazy session creation, dims validation
and converting the library's numpy output into plain ``list[float]``.

``qwen3_embed`` is imported lazily inside the getters so importing this module
stays free of the heavy ``[embedding]`` stack (the API module calls
``require_embedding_extra()`` before it gets this far).
"""

from __future__ import annotations

import math
from typing import Any

# Qwen3-0.6B native width; MRL truncation accepts anything in [DIM_MIN, DIM_MAX].
DIM_MIN = 32
DIM_MAX = 1024


def check_dims(dims: int, native: int = DIM_MAX) -> int:
    """Validate the requested output dimensionality against MRL bounds."""
    if dims < DIM_MIN or dims > native:
        raise ValueError(
            f"dims must be between {DIM_MIN} and {native} for this model, got {dims}",
        )
    return dims


def as_float_matrix(embeddings: Any) -> list[list[float]]:
    """Convert an iterable of numpy vectors to a plain nested float list."""
    return [[float(v) for v in row] for row in embeddings]


def ranked_indices(scores: list[float], top_n: int | None = None) -> list[tuple[int, float]]:
    """Return ``(index, score)`` pairs sorted by score descending.

    Ties keep the original document order (stable sort on the index). ``top_n``
    caps the list; ``None`` returns every document.
    """
    order = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
    if top_n is not None:
        order = order[:top_n]
    return [(i, scores[i]) for i in order]


def is_l2_normalized(vector: list[float], tol: float = 1e-3) -> bool:
    """Sanity check used by tests and the health of MRL truncation."""
    norm = math.sqrt(sum(v * v for v in vector))
    return abs(norm - 1.0) <= tol


class Qwen3BackendBase:
    """Lazy qwen3-embed session holder.

    ``model_path`` is a qwen3-embed registry name (e.g.
    ``n24q02m/Qwen3-Embedding-0.6B-ONNX``) or an absolute local model
    directory. Nothing is downloaded or loaded until ``embed`` / ``rerank``
    is called; the first call resolves the model into the cache directory.
    """

    #: Extra needed when the model class raises ImportError for a missing
    #: runtime (GGUF needs llama-cpp-python). Subclasses override the hint.
    _INSTALL_HINT: str = 'pip install "hull-core[embedding]"'

    def __init__(self, model_path: str, cache_dir: str | None = None, threads: int | None = None) -> None:
        self._model_path = model_path
        self._cache_dir = cache_dir
        self._threads = threads
        self._embedder: Any | None = None
        self._reranker: Any | None = None

    # -- lazy session getters -------------------------------------------------

    def _get_embedder(self) -> Any:
        if self._embedder is None:
            self._embedder = self._load("embedder")
        return self._embedder

    def _get_reranker(self) -> Any:
        if self._reranker is None:
            self._reranker = self._load("reranker")
        return self._reranker

    def _load(self, kind: str) -> Any:
        from qwen3_embed import TextCrossEncoder, TextEmbedding  # noqa: PLC0415

        cls = TextEmbedding if kind == "embedder" else TextCrossEncoder
        try:
            return cls(
                model_name=self._model_path,
                cache_dir=self._cache_dir,
                threads=self._threads,
            )
        except ImportError as exc:
            msg = str(exc)
            if "pip install" in msg:
                # Rewrite the library's own install hint to the hull extra.
                raise ImportError(msg.replace("qwen3-embed[gguf]", "hull-core[embedding-gguf]")) from exc
            raise ImportError(f"{exc} Install it with: {self._INSTALL_HINT}") from exc

    # -- inference ------------------------------------------------------------

    def embed(self, texts: list[str], dims: int = DIM_MAX) -> list[list[float]]:
        check_dims(dims)
        vectors = self._get_embedder().embed(texts, dim=dims)
        return as_float_matrix(vectors)

    def rerank(self, query: str, docs: list[str], top_n: int | None = None) -> list[tuple[int, float]]:
        scores = [float(s) for s in self._get_reranker().rerank(query, docs)]
        return ranked_indices(scores, top_n)
