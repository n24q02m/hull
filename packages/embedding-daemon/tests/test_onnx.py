"""Tests for the ONNX backend's qwen3-embed wiring.

Deterministic and offline: the ``qwen3_embed.TextEmbedding`` /
``TextCrossEncoder`` classes are patched at the seam, so no model artifact
is ever downloaded. Shape conversion, dims validation and score ordering
run against the real backend code.
"""

from __future__ import annotations

from typing import Iterable

import numpy as np
import pytest

import qwen3_embed

from hull_embedding_daemon.backends._qwen3 import DIM_MAX, DIM_MIN
from hull_embedding_daemon.backends.onnx import ONNXBackend

MODEL = "n24q02m/Qwen3-Embedding-0.6B-ONNX"


class _FakeTextEmbedding:
    """Records construction and returns fixed-width unit vectors."""

    instances: list["_FakeTextEmbedding"] = []

    def __init__(self, model_name: str, cache_dir: str | None = None, threads: int | None = None, **kw) -> None:
        self.model_name = model_name
        self.cache_dir = cache_dir
        self.threads = threads
        self.calls: list[tuple[list[str], dict]] = []
        _FakeTextEmbedding.instances.append(self)

    def embed(self, documents: Iterable[str], **kw) -> Iterable[np.ndarray]:
        docs = list(documents)
        self.calls.append((docs, dict(kw)))
        dim = kw.get("dim", DIM_MAX)
        return [np.full(dim, 1.0 / np.sqrt(dim), dtype=np.float32) for _ in docs]


class _FakeTextCrossEncoder:
    instances: list["_FakeTextCrossEncoder"] = []

    def __init__(self, model_name: str, cache_dir: str | None = None, threads: int | None = None, **kw) -> None:
        self.model_name = model_name
        _FakeTextCrossEncoder.instances.append(self)

    def rerank(self, query: str, documents: Iterable[str], **kw) -> Iterable[float]:
        # Deterministic scores: doc index order is deliberately not sorted.
        return [0.2, 0.9, 0.5]


@pytest.fixture(autouse=True)
def _reset_fakes() -> None:
    _FakeTextEmbedding.instances.clear()
    _FakeTextCrossEncoder.instances.clear()


def test_onnx_backend_init() -> None:
    backend = ONNXBackend("path/to/model.onnx")
    assert backend._model_path == "path/to/model.onnx"


def test_embed_lazy_loads_on_first_use(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(qwen3_embed, "TextEmbedding", _FakeTextEmbedding)
    backend = ONNXBackend(MODEL, cache_dir="/tmp/models", threads=2)
    # Construction must not touch the library.
    assert _FakeTextEmbedding.instances == []

    out = backend.embed(["hello", "world"], dims=256)

    assert len(out) == 2
    assert len(out[0]) == 256
    assert out[0][0] == pytest.approx(1.0 / np.sqrt(256))
    assert len(_FakeTextEmbedding.instances) == 1
    inst = _FakeTextEmbedding.instances[0]
    assert inst.model_name == MODEL
    assert inst.cache_dir == "/tmp/models"
    assert inst.threads == 2
    assert inst.calls == [(["hello", "world"], {"dim": 256})]


def test_embed_reuses_session(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(qwen3_embed, "TextEmbedding", _FakeTextEmbedding)
    backend = ONNXBackend(MODEL)
    backend.embed(["a"])
    backend.embed(["b"])
    assert len(_FakeTextEmbedding.instances) == 1


def test_embed_rejects_out_of_range_dims() -> None:
    backend = ONNXBackend(MODEL)
    with pytest.raises(ValueError, match="dims"):
        backend.embed(["x"], dims=DIM_MAX + 1)
    with pytest.raises(ValueError, match="dims"):
        backend.embed(["x"], dims=DIM_MIN - 1)


def test_embed_returns_plain_float_lists(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(qwen3_embed, "TextEmbedding", _FakeTextEmbedding)
    out = ONNXBackend(MODEL).embed(["x"], dims=64)
    assert type(out) is list
    assert type(out[0]) is list
    assert all(type(v) is float for v in out[0])


def test_rerank_returns_index_score_pairs_sorted(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(qwen3_embed, "TextCrossEncoder", _FakeTextCrossEncoder)
    backend = ONNXBackend("n24q02m/Qwen3-Reranker-0.6B-ONNX")
    ranked = backend.rerank("q", ["d0", "d1", "d2"])
    assert ranked == [(1, 0.9), (2, 0.5), (0, 0.2)]
    # Rerank must not have built an embedding session.
    assert _FakeTextEmbedding.instances == []


def test_rerank_top_n(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(qwen3_embed, "TextCrossEncoder", _FakeTextCrossEncoder)
    backend = ONNXBackend("n24q02m/Qwen3-Reranker-0.6B-ONNX")
    assert backend.rerank("q", ["d0", "d1", "d2"], top_n=2) == [(1, 0.9), (2, 0.5)]


def test_missing_runtime_gets_install_hint(monkeypatch: pytest.MonkeyPatch) -> None:
    class _Missing:
        def __init__(self, *a, **kw) -> None:
            raise ImportError("onnxruntime is required")

    monkeypatch.setattr(qwen3_embed, "TextEmbedding", _Missing)
    backend = ONNXBackend(MODEL)
    with pytest.raises(ImportError, match=r"n24q02m-hull\[embedding\]"):
        backend.embed(["x"])
