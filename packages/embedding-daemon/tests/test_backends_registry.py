"""Tests for the model-name registry and backend session cache."""

from __future__ import annotations

import pytest

from hull_embedding_daemon.backends import registry
from hull_embedding_daemon.backends.gguf import GGUFBackend
from hull_embedding_daemon.backends.onnx import ONNXBackend


@pytest.fixture(autouse=True)
def _fresh_sessions():
    registry.reset_sessions()
    yield
    registry.reset_sessions()


def test_resolve_embed_alias() -> None:
    assert registry.resolve_model("qwen3-0.6b", "embed") == "n24q02m/Qwen3-Embedding-0.6B-ONNX"


def test_resolve_rerank_alias() -> None:
    assert registry.resolve_model("qwen3-rerank-0.6b", "rerank") == "n24q02m/Qwen3-Reranker-0.6B-ONNX"


def test_resolve_is_case_insensitive() -> None:
    assert registry.resolve_model("QWEN3-0.6B", "embed") == "n24q02m/Qwen3-Embedding-0.6B-ONNX"


def test_resolve_passthrough_for_canonical_names() -> None:
    canonical = "n24q02m/Qwen3-Embedding-0.6B-ONNX-Q4F16"
    assert registry.resolve_model(canonical, "embed") == canonical
    # The same string in the other endpoint's table also passes through.
    assert registry.resolve_model("custom/model", "rerank") == "custom/model"


def test_gguf_names_route_to_gguf_backend() -> None:
    backend = registry.get_backend("qwen3-0.6b-gguf", "embed")
    assert isinstance(backend, GGUFBackend)
    assert backend._model_path == "n24q02m/Qwen3-Embedding-0.6B-GGUF"


def test_onnx_names_route_to_onnx_backend() -> None:
    backend = registry.get_backend("qwen3-0.6b", "embed")
    assert isinstance(backend, ONNXBackend)
    assert backend._model_path == "n24q02m/Qwen3-Embedding-0.6B-ONNX"


def test_sessions_are_cached_per_resolved_name() -> None:
    a = registry.get_backend("qwen3-0.6b", "embed")
    b = registry.get_backend("QWEN3-0.6B", "embed")
    c = registry.get_backend("qwen3-0.6b-gguf", "embed")
    assert a is b
    assert a is not c


def test_cache_dir_env_is_forwarded(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("HULL_EMBEDDING_CACHE_DIR", "/tmp/hull-models")
    backend = registry.get_backend("qwen3-0.6b", "embed")
    assert backend._cache_dir == "/tmp/hull-models"


def test_threads_env_is_forwarded(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("HULL_EMBEDDING_THREADS", "4")
    backend = registry.get_backend("qwen3-0.6b", "embed")
    assert backend._threads == 4


def test_invalid_threads_env_falls_back(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("HULL_EMBEDDING_THREADS", "banana")
    backend = registry.get_backend("qwen3-0.6b", "embed")
    assert backend._threads is None
