"""Import guard for the optional ``[llm]`` extra of the ``n24q02m-hull`` dist.

``hull_core.llm`` ships inside the ``n24q02m-hull`` wheel, but its runtime
dependency (litellm) is only installed by ``pip install "n24q02m-hull[llm]"``.
Importing this module first raises an ``ImportError`` with the install hint
instead of a bare ``ModuleNotFoundError`` from deep inside a submodule.
"""

from __future__ import annotations

try:
    import litellm  # noqa: F401
except ImportError as e:  # pragma: no cover - exercised only without the extra
    raise ImportError("hull_core.llm requires the [llm] extra: pip install 'n24q02m-hull[llm]'") from e
