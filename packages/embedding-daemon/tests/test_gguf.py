"""Tests for the GGUF backend's qwen3-embed wiring.

Same seam as the ONNX tests: ``qwen3_embed`` classes are patched, so the
llama.cpp runtime is never required and nothing is downloaded.
"""

from __future__ import annotations

from typing import Iterable

import numpy as np
import pytest

import qwen3_embed

from hull_embedding_daemon.backends.gguf import GGUFBackend

MODEL = "n24q02m/Qwen3-Embedding-0.6B-GGUF"


class _FakeTextEmbedding:
    def __init__(self, model_name: str, cache_dir: str | None = None, threads: int | None = None, **kw) -> None:
        self.model_name = model_name
        self.cache_dir = cache_dir

    def embed(self, documents: Iterable[str], **kw) -> Iterable[np.ndarray]:
        return [np.ones(kw.get("dim", 1024), dtype=np.float32) for _ in documents]


class _MissingLlamaCpp:
    """What qwen3-embed's GGUF loader raises when llama-cpp-python is absent."""

    def __init__(self, *a, **kw) -> None:
        raise ImportError("llama-cpp-python is required for GGUF models")


def test_gguf_backend_init() -> None:
    backend = GGUFBackend("path/to/model.gguf")
    assert backend._model_path == "path/to/model.gguf"


def test_embed_loads_gguf_model_lazily(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(qwen3_embed, "TextEmbedding", _FakeTextEmbedding)
    backend = GGUFBackend(MODEL, cache_dir="/tmp/models")
    out = backend.embed(["hello"], dims=64)
    assert len(out) == 1
    embedder = backend._embedder
    assert embedder is not None
    assert embedder.model_name == MODEL
    assert embedder.cache_dir == "/tmp/models"


def test_missing_llama_cpp_gets_gguf_hint(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(qwen3_embed, "TextEmbedding", _MissingLlamaCpp)
    backend = GGUFBackend(MODEL)
    with pytest.raises(ImportError, match=r"hull-core\[embedding-gguf\]"):
        backend.embed(["x"])


def test_missing_llama_cpp_rerank_hint(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(qwen3_embed, "TextCrossEncoder", _MissingLlamaCpp)
    backend = GGUFBackend("n24q02m/Qwen3-Reranker-0.6B-GGUF")
    with pytest.raises(ImportError, match=r"hull-core\[embedding-gguf\]"):
        backend.rerank("q", ["d"])
