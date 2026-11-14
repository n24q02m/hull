"""The doctor's interpreter floor stays pinned to the dist's ``requires-python``."""

from __future__ import annotations

import re
import sys
import tomllib
from pathlib import Path

import pytest

from hull_core._runtime import MIN_PYTHON, python_check

ROOT_PYPROJECT = Path(__file__).resolve().parents[3] / "pyproject.toml"


def test_min_python_matches_requires_python() -> None:
    spec = tomllib.loads(ROOT_PYPROJECT.read_text(encoding="utf-8"))["project"]["requires-python"]
    match = re.fullmatch(r">=(\d+)\.(\d+)(?:\.(\d+))?", spec)
    assert match is not None, f"unexpected requires-python shape: {spec!r}"
    floor = tuple(int(part or 0) for part in match.groups())
    assert floor == MIN_PYTHON


def test_python_check_accepts_the_floor(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(sys, "version_info", (*MIN_PYTHON, "final", 0))
    ok, line = python_check()
    monkeypatch.undo()
    assert ok
    assert line == "[ok] python " + ".".join(str(part) for part in MIN_PYTHON)


def test_python_check_rejects_below_the_floor(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(sys, "version_info", (3, 11, 4, "final", 0))
    ok, line = python_check()
    monkeypatch.undo()
    assert not ok
    assert line == "[fail] python 3.11.4 (hull requires >= 3.11.5)"
