"""GGUF backend via llama-cpp-python.

Used when ONNX is unavailable or a quantized GGUF is preferred for CPU
inference. Wraps ``qwen3_embed``'s GGUF loaders, which require the extra
``llama-cpp-python`` dependency -- that package ships source-only on PyPI
(needs a C++ toolchain), so it is opt-in via ``n24q02m-hull[embedding-gguf]``
rather than part of the base ``[embedding]`` stack.
"""

from __future__ import annotations

from hull_embedding_daemon.backends._qwen3 import Qwen3BackendBase


class GGUFBackend(Qwen3BackendBase):
    """GGUF embedding / rerank sessions; needs ``llama-cpp-python``."""

    _INSTALL_HINT = 'pip install "n24q02m-hull[embedding-gguf]"'
