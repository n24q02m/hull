"""Tests for hull_embedding_daemon.__main__."""

from __future__ import annotations

import sys
from unittest.mock import patch

from hull_embedding_daemon.__main__ import main


def test_main_uvicorn_import_error() -> None:
    """Test that main() returns 1 when uvicorn is not installed."""
    with patch.dict(sys.modules, {"uvicorn": None}):
        with patch("sys.stderr") as mock_stderr:
            with patch("sys.argv", ["hull-embedding-daemon"]):
                result = main()
                assert result == 1
                called_text = "".join(call.args[0] for call in mock_stderr.write.call_args_list)
                assert "uvicorn is required to run hull-embedding-daemon" in called_text


def test_main_success() -> None:
    """Test that main() returns 0 and calls uvicorn.run with correct arguments."""
    with patch("uvicorn.run") as mock_run:
        with patch("sys.argv", ["hull-embedding-daemon", "--host", "0.0.0.0", "--port", "8888", "--log-level", "debug"]):
            result = main()
            assert result == 0
            mock_run.assert_called_once_with(
                "hull_embedding_daemon.api:app",
                host="0.0.0.0",
                port=8888,
                log_level="debug",
            )


def test_main_default_args() -> None:
    """Test that main() uses default arguments when none are provided."""
    with patch("uvicorn.run") as mock_run:
        with patch("sys.argv", ["hull-embedding-daemon"]):
            result = main()
            assert result == 0
            mock_run.assert_called_once_with(
                "hull_embedding_daemon.api:app",
                host="127.0.0.1",
                port=9800,
                log_level="info",
            )


def test_main_module_execution() -> None:
    """Test the if __name__ == "__main__": block."""
    import runpy

    with patch("uvicorn.run"), patch("sys.exit") as mock_exit, patch("sys.argv", ["hull-embedding-daemon"]):
        runpy.run_module("hull_embedding_daemon.__main__", run_name="__main__")
        mock_exit.assert_called_once_with(0)
