"""``import hull_web`` without the ``[web]`` extra must fail with an install hint."""

from __future__ import annotations

import importlib
import sys

import pytest

from hull_web import _extra

INSTALL_HINT = 'pip install "n24q02m-hull[web]"'


def test_missing_web_modules_empty_when_all_present() -> None:
    assert _extra.missing_web_modules(find_spec=lambda name: object()) == []


def test_missing_web_modules_lists_absent_modules_in_order() -> None:
    absent = {"crawl4ai", "PIL"}
    found = _extra.missing_web_modules(find_spec=lambda name: None if name in absent else object())
    assert found == [m for m in _extra.WEB_EXTRA_MODULES if m in absent]


def test_require_web_extra_raises_with_install_hint() -> None:
    with pytest.raises(ImportError) as exc_info:
        _extra.require_web_extra(find_spec=lambda name: None if name == "patchright" else object())
    message = str(exc_info.value)
    assert INSTALL_HINT in message
    assert "patchright" in message
    assert exc_info.value.name == "patchright"


def test_require_web_extra_passes_when_all_present() -> None:
    _extra.require_web_extra(find_spec=lambda name: object())


def test_extra_modules_cover_every_web_dependency() -> None:
    # Import names of the [web] extra dependencies declared in pyproject.toml.
    assert set(_extra.WEB_EXTRA_MODULES) == {
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
    }


@pytest.mark.parametrize("module", ["crawl4ai", "curl_cffi", "PIL"])
def test_import_hull_web_without_extra_raises_install_hint(monkeypatch: pytest.MonkeyPatch, module: str) -> None:
    # A None entry in sys.modules makes find_spec() report the module as absent.
    monkeypatch.setitem(sys.modules, module, None)
    for name in [n for n in sys.modules if n == "hull_web" or n.startswith("hull_web.")]:
        monkeypatch.delitem(sys.modules, name)

    with pytest.raises(ImportError) as exc_info:
        importlib.import_module("hull_web")

    assert INSTALL_HINT in str(exc_info.value)
    assert module in str(exc_info.value)
