"""Import guard for the optional ``[embedding]`` extra of the ``hull-core`` dist.

``hull_embedding_daemon`` ships inside the ``hull-core`` wheel, but its server
stack is only installed by ``pip install "hull-core[embedding]"``. The API
module and the CLI call ``require_embedding_extra()`` before touching that
stack, so a base install fails with the install hint instead of a bare
``ModuleNotFoundError`` for ``fastapi``. The package ``__init__`` does not
call it, so ``__version__`` stays importable without the extra.
"""

from __future__ import annotations

import importlib.util
from collections.abc import Callable
from typing import Any

# Import names of the [embedding] extra in pyproject.toml, in declaration order.
EMBEDDING_EXTRA_MODULES: tuple[str, ...] = (
    "fastapi",
    "uvicorn",
    "qwen3_embed",
    "onnxruntime",
    "numpy",
    "httpx",
    "pydantic",
)

INSTALL_HINT = 'pip install "hull-core[embedding]"'

FindSpec = Callable[[str], Any]


def missing_embedding_modules(find_spec: FindSpec = importlib.util.find_spec) -> list[str]:
    """Return the [embedding] extra modules that cannot be found, in declaration order."""
    return [name for name in EMBEDDING_EXTRA_MODULES if find_spec(name) is None]


def require_embedding_extra(find_spec: FindSpec = importlib.util.find_spec) -> None:
    """Raise ``ImportError`` naming the missing modules when the extra is absent."""
    missing = missing_embedding_modules(find_spec)
    if missing:
        raise ImportError(
            f"hull_embedding_daemon requires the optional 'embedding' extra of hull-core; missing module(s): "
            f"{', '.join(missing)}. Install it with: {INSTALL_HINT}",
            name=missing[0],
        )
