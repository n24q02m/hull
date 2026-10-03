"""Import guard for the optional ``[llm]`` extra of the ``hull-core`` dist.

``hull_core.llm`` ships inside the ``hull-core`` wheel, but its runtime
dependency (litellm) is only installed by ``pip install "hull-core[llm]"``.
Importing this module first raises an ``ImportError`` with the install hint
instead of a bare ``ModuleNotFoundError`` from deep inside a submodule.
"""

from __future__ import annotations

try:
    import litellm  # noqa: F401
except ImportError as e:  # pragma: no cover - exercised only without the extra
    raise ImportError("hull_core.llm requires the [llm] extra: pip install 'hull-core[llm]'") from e
