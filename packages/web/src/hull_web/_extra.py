"""Import guard for the optional ``[web]`` extra of the ``hull-core`` dist.

``hull_web`` ships inside the ``hull-core`` wheel, but its dependencies are
only installed by ``pip install "hull-core[web]"``. Importing this module
checks that every dependency of the extra is importable and raises an
``ImportError`` with the install hint otherwise, instead of a bare
``ModuleNotFoundError`` from deep inside a submodule.
"""

from __future__ import annotations

import importlib.util
from collections.abc import Callable
from typing import Any

# Import names of the [web] extra in pyproject.toml, in declaration order.
WEB_EXTRA_MODULES: tuple[str, ...] = (
    "httpx",
    "curl_cffi",
    "crawl4ai",
    "filelock",
    "langgraph",
    "patchright",
    "browserforge",
    "capsolver",
    "pydantic",
    "PIL",
)

INSTALL_HINT = 'pip install "hull-core[web]"'

FindSpec = Callable[[str], Any]


def missing_web_modules(find_spec: FindSpec = importlib.util.find_spec) -> list[str]:
    """Return the [web] extra modules that cannot be found, in declaration order."""
    return [name for name in WEB_EXTRA_MODULES if find_spec(name) is None]


def require_web_extra(find_spec: FindSpec = importlib.util.find_spec) -> None:
    """Raise ``ImportError`` naming the missing modules when the extra is absent."""
    missing = missing_web_modules(find_spec)
    if missing:
        raise ImportError(
            f"hull_web requires the optional 'web' extra of hull-core; missing module(s): "
            f"{', '.join(missing)}. Install it with: {INSTALL_HINT}",
            name=missing[0],
        )


require_web_extra()
