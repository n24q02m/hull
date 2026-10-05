"""ONNX backend -- CPU or CUDA ExecutionProvider.

Wraps ``qwen3_embed``'s ONNX loaders (Qwen3-Embedding / Qwen3-Reranker).
The ExecutionProvider is auto-detected by onnxruntime: with plain
``onnxruntime`` it runs on CPU; with ``onnxruntime-gpu`` the same module
picks up CUDA automatically. Models download from Hugging Face into the
user cache on first use, never at import.
"""

from __future__ import annotations

from hull_embedding_daemon.backends._qwen3 import Qwen3BackendBase


class ONNXBackend(Qwen3BackendBase):
    """ONNX embedding / rerank sessions over ``qwen3_embed`` models."""

    _INSTALL_HINT = 'pip install "hull-core[embedding]"'
