"""The interpreter floor that both ``doctor`` commands check.

``MIN_PYTHON`` mirrors ``requires-python`` in the root ``pyproject.toml``; a
test pins the two together, so widening the supported window cannot leave the
doctor rejecting an interpreter the dist itself accepts.
"""

from __future__ import annotations

import sys

MIN_PYTHON = (3, 11, 5)


def python_check() -> tuple[bool, str]:
    """Return ``(ok, line)`` in the ``[ok]/[fail]`` shape the doctors print."""
    current = tuple(sys.version_info[:3])
    shown = ".".join(str(part) for part in current)
    if current >= MIN_PYTHON:
        return True, f"[ok] python {shown}"
    floor = ".".join(str(part) for part in MIN_PYTHON)
    return False, f"[fail] python {shown} (hull requires >= {floor})"
