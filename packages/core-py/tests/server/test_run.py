"""Blocking server runner: loopback policy for no-auth + lock-then-bind order."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import pytest

from hull_core.config.settings import HullSettings, ServerSettings
from hull_core.server.run import ServerConfigError, _is_loopback, run_server


def _settings(tmp_path: Path, *, host: str = "127.0.0.1", port: int = 8765, auth: str = "no-auth") -> HullSettings:
    return HullSettings(config_dir=tmp_path, server=ServerSettings(auth=auth, host=host, port=port))


class TestIsLoopback:
    @pytest.mark.parametrize("host", ["localhost", "127.0.0.1", "::1", "127.4.5.6"])
    def test_loopback_hosts(self, host: str) -> None:
        assert _is_loopback(host) is True

    @pytest.mark.parametrize("host", ["0.0.0.0", "8.8.8.8", "192.168.1.10", "example.com", ""])
    def test_non_loopback_hosts(self, host: str) -> None:
        assert _is_loopback(host) is False


class TestNoAuthBindRefused:
    def test_refuses_explicit_public_host(self, tmp_path: Path) -> None:
        with pytest.raises(ServerConfigError) as exc:
            run_server(_settings(tmp_path), host="0.0.0.0")
        assert "no-auth" in str(exc.value)

    def test_refuses_public_host_from_settings(self, tmp_path: Path) -> None:
        """The refusal must not depend on the CLI flag -- a config.toml bind is enough."""
        with pytest.raises(ServerConfigError):
            run_server(_settings(tmp_path, host="192.168.1.5"))

    def test_token_mode_may_bind_publicly(self, tmp_path: Path) -> None:
        served: list[dict] = []
        with (
            patch("hull_core.server.run.build_app", return_value=object()),
            patch(
                "hull_core.server.run.uvicorn.run",
                side_effect=lambda app, **kwargs: served.append(kwargs),
            ),
        ):
            run_server(_settings(tmp_path, auth="token"), host="0.0.0.0", lock_root=tmp_path / "locks")
        assert served[0]["host"] == "0.0.0.0"


class TestRunServerLockAndBind:
    def test_lock_is_held_while_serving_and_released_after(self, tmp_path: Path) -> None:
        root = tmp_path / "locks"
        held_during_serve: list[bool] = []

        def serve(app, **kwargs):  # noqa: ANN001, ANN003
            held_during_serve.append(any(root.glob("hull-*.lock")))

        with (
            patch("hull_core.server.run.build_app", return_value=object()),
            patch("hull_core.server.run.uvicorn.run", side_effect=serve),
        ):
            run_server(_settings(tmp_path, port=9101), lock_root=root)

        assert held_during_serve == [True], "the port must be locked before uvicorn binds it"
        assert not list(root.glob("hull-*.lock")), "the lock is released when the server returns"

    def test_binds_explicit_host_and_port(self, tmp_path: Path) -> None:
        seen: dict = {}

        def serve(app, **kwargs):  # noqa: ANN001, ANN003
            seen.update(kwargs)

        with (
            patch("hull_core.server.run.build_app", return_value=object()),
            patch("hull_core.server.run.uvicorn.run", side_effect=serve),
        ):
            run_server(_settings(tmp_path, port=1111), host="localhost", port=2222, lock_root=tmp_path / "locks")

        assert seen == {"host": "localhost", "port": 2222, "log_level": "info"}

    def test_falls_back_to_settings_bind(self, tmp_path: Path) -> None:
        seen: dict = {}

        def serve(app, **kwargs):  # noqa: ANN001, ANN003
            seen.update(kwargs)

        with (
            patch("hull_core.server.run.build_app", return_value=object()),
            patch("hull_core.server.run.uvicorn.run", side_effect=serve),
        ):
            run_server(_settings(tmp_path, host="127.0.0.1", port=3333), lock_root=tmp_path / "locks")

        assert (seen["host"], seen["port"]) == ("127.0.0.1", 3333)

    def test_builds_the_app_from_settings(self, tmp_path: Path) -> None:
        settings = _settings(tmp_path)
        built: list[HullSettings] = []

        def build(cfg):  # noqa: ANN001
            built.append(cfg)
            return object()

        with (
            patch("hull_core.server.run.build_app", side_effect=build),
            patch("hull_core.server.run.uvicorn.run", side_effect=lambda app, **_k: None),
        ):
            run_server(settings, lock_root=tmp_path / "locks")

        assert built == [settings]
