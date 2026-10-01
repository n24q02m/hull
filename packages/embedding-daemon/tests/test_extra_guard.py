"""The ``[embedding]`` extra guard: clear ImportError / exit 1 with an install hint."""

from __future__ import annotations

import importlib
import sys
from unittest.mock import patch

import pytest

from hull_embedding_daemon import _extra
from hull_embedding_daemon.__main__ import main

INSTALL_HINT = 'pip install "hull-core[embedding]"'


def test_missing_embedding_modules_empty_when_all_present() -> None:
    assert _extra.missing_embedding_modules(find_spec=lambda name: object()) == []


def test_missing_embedding_modules_lists_absent_modules_in_order() -> None:
    absent = {"onnxruntime", "fastapi"}
    found = _extra.missing_embedding_modules(find_spec=lambda name: None if name in absent else object())
    assert found == ["fastapi", "onnxruntime"]


def test_require_embedding_extra_raises_with_install_hint() -> None:
    with pytest.raises(ImportError) as exc_info:
        _extra.require_embedding_extra(find_spec=lambda name: None if name in {"fastapi", "numpy"} else object())
    message = str(exc_info.value)
    assert INSTALL_HINT in message
    assert "fastapi, numpy" in message
    assert exc_info.value.name == "fastapi"


def test_require_embedding_extra_passes_when_all_present() -> None:
    _extra.require_embedding_extra(find_spec=lambda name: object())


@pytest.mark.parametrize("module", ["fastapi", "onnxruntime", "numpy"])
def test_import_api_without_extra_raises_install_hint(monkeypatch: pytest.MonkeyPatch, module: str) -> None:
    # A None entry in sys.modules makes find_spec() report the module as absent.
    monkeypatch.setitem(sys.modules, module, None)
    monkeypatch.delitem(sys.modules, "hull_embedding_daemon.api", raising=False)

    with pytest.raises(ImportError) as exc_info:
        importlib.import_module("hull_embedding_daemon.api")

    assert INSTALL_HINT in str(exc_info.value)
    assert module in str(exc_info.value)


def test_package_import_stays_light_without_extra(monkeypatch: pytest.MonkeyPatch) -> None:
    # __version__ must stay readable from a base install (no [embedding] extra).
    monkeypatch.setitem(sys.modules, "fastapi", None)
    monkeypatch.setitem(sys.modules, "onnxruntime", None)
    for name in [n for n in sys.modules if n == "hull_embedding_daemon" or n.startswith("hull_embedding_daemon.")]:
        monkeypatch.delitem(sys.modules, name)

    pkg = importlib.import_module("hull_embedding_daemon")
    assert pkg.__version__


@pytest.mark.parametrize("module", ["fastapi", "uvicorn", "onnxruntime"])
def test_main_returns_1_with_hint_when_extra_missing(monkeypatch: pytest.MonkeyPatch, module: str) -> None:
    import uvicorn

    # Patch the real module object first; sys.modules["uvicorn"] may be nulled below.
    with (
        patch.object(uvicorn, "run") as mock_run,
        patch("sys.stderr") as mock_stderr,
        patch("sys.argv", ["hull-embedding-daemon"]),
    ):
        monkeypatch.setitem(sys.modules, module, None)
        result = main()
    assert result == 1
    mock_run.assert_not_called()
    written = "".join(call.args[0] for call in mock_stderr.write.call_args_list)
    assert INSTALL_HINT in written
    assert module in written
