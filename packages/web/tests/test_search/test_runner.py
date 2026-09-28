"""Tests for hull_web.search.runner -- SearXNG cross-process singleton manager.

All subprocess/network calls are mocked. No real SearXNG processes.
"""

from __future__ import annotations

import asyncio
import json
import os
import socket
import subprocess
import sys
import time
from pathlib import Path
from unittest.mock import ANY, AsyncMock, MagicMock, patch

import httpx
import pytest

from hull_web.search import ensure_searxng, shutdown_searxng
from hull_web.search.runner import (
    _SETTINGS_TEMPLATE,
    _cleanup_process,
    _find_available_port,
    _get_docker_lock,
    _get_pip_command,
    _get_process_kwargs,
    _get_settings_path,
    _get_startup_lock,
    _handle_restart_and_start,
    _install_searxng,
    _is_pid_alive,
    _is_process_alive,
    _is_searxng_installed,
    _kill_stale_port_process,
    _port_listener_pids,
    _quick_health_check,
    _read_discovery,
    _remove_discovery,
    _select_start_port,
    _start_docker_searxng,
    _try_reuse_existing,
    _wait_for_service,
    _write_discovery,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
# _is_pid_alive
# ===========================================================================


class TestIsPidAlive:
    def test_current_process_is_alive(self):
        """os.getpid() should always be alive."""
        assert _is_pid_alive(os.getpid()) is True

    def test_pid_zero_is_not_alive(self):
        """PID 0 (kernel/idle) should not be considered alive."""
        assert _is_pid_alive(0) is False

    def test_negative_pid_is_not_alive(self):
        """Negative PIDs are invalid and should not be alive."""
        assert _is_pid_alive(-1) is False
        assert _is_pid_alive(-9999) is False

    @pytest.mark.skipif(sys.platform == "win32", reason="Unix-only zombie check")
    def test_zombie_process_detected(self, tmp_path):
        """A zombie process on Linux should be detected as not alive."""
        # Mock /proc/{pid}/status with zombie state
        pid = 99999
        with (
            patch("os.kill") as mock_kill,
            patch.object(Path, "exists", return_value=True),
            patch.object(Path, "read_text", return_value="State:\tZ (zombie)\n"),
        ):
            mock_kill.return_value = None  # os.kill succeeds (PID in table)
            assert _is_pid_alive(pid) is False

    @pytest.mark.skipif(sys.platform != "win32", reason="Windows-only ctypes check")
    def test_windows_dead_process(self):
        """A non-existent PID on Windows should return False."""
        # PID 4000000 is very unlikely to exist
        assert _is_pid_alive(4000000) is False

    def test_very_large_pid_not_alive(self):
        """An absurdly large PID should not be alive."""
        assert _is_pid_alive(999999999) is False


# ===========================================================================
# Discovery file management
# ===========================================================================


class TestDiscovery:
    def test_read_discovery_no_file(self, tmp_discovery):
        """Returns None when discovery file doesn't exist."""
        assert _read_discovery() is None

    def test_write_and_read_discovery(self, tmp_discovery):
        """Write then read round-trips correctly."""
        _write_discovery(18888, 12345)
        data = _read_discovery()
        assert data is not None
        assert data["port"] == 18888
        assert data["pid"] == 12345
        assert data["owner_pid"] == os.getpid()
        assert "started_at" in data

    def test_read_discovery_invalid_json(self, tmp_discovery):
        """Returns None on malformed JSON."""
        tmp_discovery.parent.mkdir(parents=True, exist_ok=True)
        fd = os.open(tmp_discovery, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
        with os.fdopen(fd, "w") as f:
            f.write("not json")
        assert _read_discovery() is None

    def test_read_discovery_missing_keys(self, tmp_discovery):
        """Returns None when required keys are missing."""
        tmp_discovery.parent.mkdir(parents=True, exist_ok=True)
        fd = os.open(tmp_discovery, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
        with os.fdopen(fd, "w") as f:
            f.write(json.dumps({"port": 8080}))  # Missing pid
        assert _read_discovery() is None

    @pytest.mark.skipif(sys.platform == "win32", reason="POSIX permissions not supported on Windows")
    def test_read_discovery_insecure_permissions(self, tmp_discovery):
        """Returns None when discovery file has insecure permissions."""
        _write_discovery(18888, 12345)
        os.chmod(tmp_discovery, 0o644)
        assert _read_discovery() is None

    @pytest.mark.skipif(sys.platform == "win32", reason="POSIX permissions not supported on Windows")
    def test_write_discovery_secure_permissions(self, tmp_discovery):
        """Discovery file is created with 0o600 permissions."""
        _write_discovery(18888, 12345)
        assert (os.stat(tmp_discovery).st_mode & 0o777) == 0o600

    def test_remove_discovery(self, tmp_discovery):
        """Removes the discovery file if it exists."""
        _write_discovery(18888, 12345)
        assert tmp_discovery.exists()
        _remove_discovery()
        assert not tmp_discovery.exists()

    def test_remove_discovery_nonexistent(self, tmp_discovery):
        """Does not raise when file doesn't exist."""
        _remove_discovery()  # Should not raise


# ===========================================================================
# _quick_health_check
# ===========================================================================


class TestQuickHealthCheck:
    async def test_healthy_instance(self):
        """Returns True when /healthz returns 200."""
        mock_response = MagicMock()
        mock_response.status_code = 200

        with patch("hull_web.search.runner.safe_httpx_client") as mock_client_cls:
            mock_client = AsyncMock()
            mock_client.get = AsyncMock(return_value=mock_response)
            mock_client_cls.return_value.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client_cls.return_value.__aexit__ = AsyncMock(return_value=None)

            result = await _quick_health_check("http://127.0.0.1:18888")
            assert result is True

    async def test_unhealthy_instance(self):
        """Returns False when all retries fail."""
        with patch("hull_web.search.runner.safe_httpx_client") as mock_client_cls:
            mock_client = AsyncMock()
            mock_client.get = AsyncMock(side_effect=httpx.ConnectError("refused"))
            mock_client_cls.return_value.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client_cls.return_value.__aexit__ = AsyncMock(return_value=None)

            result = await _quick_health_check("http://127.0.0.1:18888", retries=1)
            assert result is False

    async def test_retries_on_failure_then_succeeds(self):
        """Retries and returns True on eventual success."""
        mock_fail = MagicMock()
        mock_fail.status_code = 500
        mock_ok = MagicMock()
        mock_ok.status_code = 200

        with patch("hull_web.search.runner.safe_httpx_client") as mock_client_cls:
            mock_client = AsyncMock()
            mock_client.get = AsyncMock(side_effect=[mock_fail, mock_ok])
            mock_client_cls.return_value.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client_cls.return_value.__aexit__ = AsyncMock(return_value=None)

            result = await _quick_health_check("http://127.0.0.1:18888", retries=2)
            assert result is True


# ===========================================================================
# _try_reuse_existing
# ===========================================================================


class TestTryReuseExisting:
    async def test_returns_url_if_instance_running(self, tmp_discovery):
        """Returns URL if discovery file points to a healthy instance."""
        _write_discovery(18888, os.getpid())

        with patch("hull_web.search.runner._quick_health_check", new_callable=AsyncMock, return_value=True):
            url = await _try_reuse_existing()
            assert url == "http://127.0.0.1:18888"

    async def test_returns_none_if_not_running(self, tmp_discovery):
        """Returns None if health check fails."""
        _write_discovery(18888, os.getpid())

        with patch("hull_web.search.runner._quick_health_check", new_callable=AsyncMock, return_value=False):
            url = await _try_reuse_existing()
            assert url is None

    async def test_returns_none_if_no_discovery(self, tmp_discovery):
        """Returns None when discovery file doesn't exist."""
        url = await _try_reuse_existing()
        assert url is None

    async def test_returns_none_if_pid_dead(self, tmp_discovery):
        """Returns None and cleans up if PID in discovery is dead."""
        _write_discovery(18888, 999999999)

        with patch("hull_web.search.runner._is_pid_alive", return_value=False):
            url = await _try_reuse_existing()
            assert url is None
            assert not tmp_discovery.exists()

    async def test_returns_none_if_missing_port(self, tmp_discovery):
        """Returns None if discovery data is missing port."""
        tmp_discovery.parent.mkdir(parents=True, exist_ok=True)
        tmp_discovery.write_text(json.dumps({"pid": 1234}))  # Missing port
        url = await _try_reuse_existing()
        assert url is None


# ===========================================================================
# _find_available_port
# ===========================================================================


class TestFindAvailablePort:
    def test_returns_valid_port(self):
        """Returns a valid non-privileged port."""
        port = _find_available_port(18888, max_tries=50)
        assert 1024 <= port <= 65535

    def test_raises_if_no_port_available(self):
        """Raises RuntimeError if all ports in range are in use."""
        with patch("socket.socket") as mock_socket_cls:
            mock_socket = MagicMock()
            mock_socket.__enter__ = MagicMock(return_value=mock_socket)
            mock_socket.__exit__ = MagicMock(return_value=False)
            mock_socket.bind = MagicMock(side_effect=OSError("Address in use"))
            mock_socket_cls.return_value = mock_socket

            with pytest.raises(RuntimeError, match="No available port found"):
                _find_available_port(18888, max_tries=5)

    def test_skips_used_ports(self):
        """Finds a free port even if some are in use."""
        # Bind a port to make it unavailable, then check the function works
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.bind(("127.0.0.1", 0))
            _ = s.getsockname()[1]

        # The function should still find a port (just not the used one)
        port = _find_available_port(18888, max_tries=50)
        assert isinstance(port, int)
        assert port >= 18888

    def test_find_available_port_invalid_range(self):
        """Covers the case where start_port is outside [1024, 65535]."""
        # Low port
        port = _find_available_port(80, max_tries=5)
        assert 49152 <= port <= 65535

        # High port
        port = _find_available_port(70000, max_tries=5)
        assert 49152 <= port <= 65535

    def test_find_available_port_retries_on_bind_error(self):
        """Covers OSError during s.bind()."""
        with patch("socket.socket") as mock_socket_cls:
            # First mock socket fails on bind, second succeeds
            mock_s1 = MagicMock()
            mock_s1.__enter__.return_value = mock_s1
            mock_s1.bind.side_effect = OSError("First port taken")

            mock_s2 = MagicMock()
            mock_s2.__enter__.return_value = mock_s2
            # mock_s2.bind succeeds (default)

            mock_socket_cls.side_effect = [mock_s1, mock_s2]

            port = _find_available_port(18888, max_tries=5)
            assert isinstance(port, int)
            assert mock_socket_cls.call_count == 2

    def test_find_available_port_retries_on_socket_creation_error(self):
        """Covers OSError during socket.socket() instantiation."""
        with patch("socket.socket") as mock_socket_cls:
            # First call to socket.socket() fails
            mock_s2 = MagicMock()
            mock_s2.__enter__.return_value = mock_s2

            mock_socket_cls.side_effect = [OSError("Socket creation failed"), mock_s2]

            port = _find_available_port(18888, max_tries=5)
            assert isinstance(port, int)
            assert mock_socket_cls.call_count == 2


# ===========================================================================
# _wait_for_service
# ===========================================================================


class TestWaitForService:
    async def test_returns_true_when_healthy(self):
        """Returns True immediately when service is healthy."""
        mock_response = MagicMock()
        mock_response.status_code = 200

        with patch("hull_web.search.runner.safe_httpx_client") as mock_client_cls:
            mock_client = AsyncMock()
            mock_client.get = AsyncMock(return_value=mock_response)
            mock_client_cls.return_value.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client_cls.return_value.__aexit__ = AsyncMock(return_value=None)

            result = await _wait_for_service("http://127.0.0.1:18888", timeout=2.0)
            assert result is True

    async def test_returns_false_on_timeout(self):
        """Returns False when service never becomes healthy."""
        with patch("hull_web.search.runner.safe_httpx_client") as mock_client_cls:
            mock_client = AsyncMock()
            mock_client.get = AsyncMock(side_effect=httpx.ConnectError("refused"))
            mock_client_cls.return_value.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client_cls.return_value.__aexit__ = AsyncMock(return_value=None)

            result = await _wait_for_service("http://127.0.0.1:18888", timeout=0.5)
            assert result is False


# ===========================================================================
# _is_searxng_installed / _install_searxng
# ===========================================================================


class TestSearxngInstallation:
    def test_is_searxng_installed_true(self):
        """Returns True when searx.webapp is importable."""
        with patch("importlib.util.find_spec", return_value=MagicMock()):
            assert _is_searxng_installed() is True

    def test_is_searxng_installed_false(self):
        """Returns False when searx.webapp is not found."""
        with patch("importlib.util.find_spec", return_value=None):
            assert _is_searxng_installed() is False

    def test_is_searxng_installed_module_not_found(self):
        """Returns False when find_spec raises ModuleNotFoundError."""
        with patch("importlib.util.find_spec", side_effect=ModuleNotFoundError):
            assert _is_searxng_installed() is False

    def test_install_searxng_success(self):
        """Returns True when pip install succeeds."""
        mock_result = MagicMock()
        mock_result.returncode = 0
        mock_result.stderr = ""

        with (
            patch("subprocess.run", return_value=mock_result),
            patch("hull_web.search.runner._get_pip_command", return_value=["pip", "install"]),
        ):
            assert _install_searxng() is True

    def test_install_searxng_failure(self):
        """Returns False when pip install fails."""
        mock_ok = MagicMock()
        mock_ok.returncode = 0
        mock_ok.stderr = ""

        mock_fail = MagicMock()
        mock_fail.returncode = 1
        mock_fail.stderr = "error: could not build"

        with (
            patch("subprocess.run", side_effect=[mock_ok, mock_fail]),
            patch("hull_web.search.runner._get_pip_command", return_value=["pip", "install"]),
        ):
            assert _install_searxng() is False

    def test_install_searxng_timeout(self):
        """Returns False when pip install times out."""
        with (
            patch("subprocess.run", side_effect=subprocess.TimeoutExpired(cmd="pip", timeout=120)),
            patch("hull_web.search.runner._get_pip_command", return_value=["pip", "install"]),
        ):
            assert _install_searxng() is False

    def test_install_searxng_deps_failure(self):
        """Returns False when build deps installation fails."""
        mock_fail = MagicMock()
        mock_fail.returncode = 1
        mock_fail.stderr = "dependency error"

        with (
            patch("subprocess.run", return_value=mock_fail),
            patch("hull_web.search.runner._get_pip_command", return_value=["pip", "install"]),
        ):
            assert _install_searxng() is False


# ===========================================================================
# _get_pip_command
# ===========================================================================


class TestGetPipCommand:
    def test_prefers_uv(self):
        """Uses uv pip when uv is available."""
        with patch("shutil.which", side_effect=lambda x: "/usr/bin/uv" if x == "uv" else None):
            cmd = _get_pip_command()
            assert cmd[0] == "/usr/bin/uv"
            assert "pip" in cmd
            assert "--python" in cmd

    def test_uses_pip(self):
        """Uses pip when uv is not available."""
        with patch("shutil.which", side_effect=lambda x: "/usr/bin/pip" if x == "pip" else None):
            cmd = _get_pip_command()
            assert cmd == ["/usr/bin/pip", "install"]

    def test_fallback_python_m_pip(self):
        """Falls back to python -m pip."""
        with patch("shutil.which", return_value=None):
            cmd = _get_pip_command()
            assert cmd == [sys.executable, "-m", "pip", "install"]


# ===========================================================================
# _get_settings_path
# ===========================================================================


class TestConfigDir:
    def test_config_dir_moved_to_hull(self):
        """Config dir is ~/.hull/searxng (de-web-core naming)."""
        import hull_web.search.runner as mod

        assert mod._CONFIG_DIR == Path.home() / ".hull" / "searxng"
        assert mod._DISCOVERY_FILE == mod._CONFIG_DIR / "searxng_instance.json"


class TestGetSettingsPath:
    def test_creates_settings_file(self, tmp_config_dir):
        """Creates a per-process settings file with correct port and secret."""
        path = _get_settings_path(18888)
        assert path.exists()
        content = path.read_text()
        assert "port: 18888" in content
        assert "hull SearXNG" in content
        assert "web-core" not in content
        # Secret should be a hex string (not the template placeholder)
        assert "{secret_key}" not in content
        assert "{port}" not in content

    def test_per_process_filename(self, tmp_config_dir):
        """Settings file is securely named with a prefix and correct suffix."""
        path = _get_settings_path(18888)
        assert path.name.startswith("searxng_settings_")
        assert path.name.endswith(".yml")

    def test_http2_disabled_on_windows(self, tmp_config_dir):
        """HTTP/2 is disabled on Windows to avoid deadlocks."""
        with patch("sys.platform", "win32"):
            path = _get_settings_path(18888)
            content = path.read_text()
            assert "enable_http2: false" in content

    def test_http2_enabled_on_linux(self, tmp_config_dir):
        """HTTP/2 is enabled on non-Windows platforms."""
        with patch("sys.platform", "linux"):
            path = _get_settings_path(18888)
            content = path.read_text()
            assert "enable_http2: true" in content


# ===========================================================================
# _get_process_kwargs
# ===========================================================================


class TestGetProcessKwargs:
    def test_unix_uses_start_new_session(self):
        """On Unix, uses start_new_session=True for process group management."""
        with patch("sys.platform", "linux"), patch("os.getuid", return_value=1000, create=True):
            kwargs = _get_process_kwargs()
            assert kwargs.get("start_new_session") is True
            assert "preexec_fn" not in kwargs

    def test_unix_root_drops_privileges(self):
        """On Unix as root, drops privileges to 'nobody'."""
        mock_pw = MagicMock()
        mock_pw.pw_uid = 65534
        mock_pw.pw_gid = 65534
        mock_pwd = MagicMock()
        mock_pwd.getpwnam.return_value = mock_pw
        with (
            patch("sys.platform", "linux"),
            patch("os.getuid", return_value=0, create=True),
            patch.dict("sys.modules", {"pwd": mock_pwd}),
        ):
            kwargs = _get_process_kwargs()
            assert kwargs["user"] == 65534
            assert kwargs["group"] == 65534

    @pytest.mark.skipif(sys.platform != "win32", reason="CREATE_NEW_PROCESS_GROUP only exists on Windows")
    def test_windows_uses_creation_flags(self):
        """On Windows, uses CREATE_NEW_PROCESS_GROUP."""
        with patch("sys.platform", "win32"):
            kwargs = _get_process_kwargs()
            assert "creationflags" in kwargs
            assert kwargs["creationflags"] == subprocess.CREATE_NEW_PROCESS_GROUP


# ===========================================================================
# _is_process_alive
# ===========================================================================


class TestGetSecureEnv:
    def test_filters_environment(self):
        """Only whitelisted environment variables are kept."""
        from hull_web.search.runner import _get_secure_env

        settings_path = Path("/tmp/settings.yml")
        with patch.dict("os.environ", {"PATH": "/bin", "MY_VAR": "discarded_value", "PYTHONPATH": "src"}):
            env = _get_secure_env(settings_path)
            assert env["PATH"] == "/bin"
            assert env["PYTHONPATH"] == "src"
            assert "MY_VAR" not in env
            assert env["SEARXNG_SETTINGS_PATH"] == str(settings_path)


class TestIsProcessAlive:
    def test_returns_false_when_no_process(self):
        """Returns False when _searxng_process is None."""
        assert _is_process_alive() is False

    def test_returns_true_when_alive(self):
        """Returns True when process poll() returns None (alive)."""
        import hull_web.search.runner as mod

        mock_proc = MagicMock()
        mock_proc.poll.return_value = None
        mod._searxng_process = mock_proc
        assert _is_process_alive() is True

    def test_returns_false_when_dead(self):
        """Returns False when process poll() returns exit code."""
        import hull_web.search.runner as mod

        mock_proc = MagicMock()
        mock_proc.poll.return_value = 1
        mod._searxng_process = mock_proc
        assert _is_process_alive() is False


# ===========================================================================
# _kill_stale_port_process (ownership-aware) / _port_listener_pids
# ===========================================================================


class TestKillStalePortProcess:
    async def test_invalid_port_is_manageable(self):
        """Invalid ports report nothing to free."""
        assert await _kill_stale_port_process(0) is True
        assert await _kill_stale_port_process(-1) is True
        assert await _kill_stale_port_process(70000) is True
        assert await _kill_stale_port_process("abc") is True  # type: ignore[arg-type]

    async def test_no_listener_returns_true(self, monkeypatch):
        """A port with no listener needs no freeing."""
        import hull_web.search.runner as mod

        monkeypatch.setattr(mod, "_port_listener_pids", AsyncMock(return_value=set()))
        mock_kill = AsyncMock()
        monkeypatch.setattr(mod, "_sigterm_then_kill", mock_kill)

        assert await _kill_stale_port_process(18888) is True
        mock_kill.assert_not_called()

    async def test_foreign_listener_not_killed_returns_false(self, monkeypatch):
        """A port held by a process we did not spawn must never be signalled."""
        import hull_web.search.runner as mod

        monkeypatch.setattr(mod, "_port_listener_pids", AsyncMock(return_value={99999}))
        mock_kill = AsyncMock()
        monkeypatch.setattr(mod, "_sigterm_then_kill", mock_kill)

        assert await _kill_stale_port_process(18888) is False
        mock_kill.assert_not_called()

    async def test_owned_listener_from_discovery_is_killed(self, monkeypatch):
        """A stale listener recorded in our own discovery state is killed."""
        import hull_web.search.runner as mod

        monkeypatch.setattr(mod, "_read_discovery", lambda: {"pid": 99999, "port": 18888, "owner_pid": 4242})
        monkeypatch.setattr(mod, "_port_listener_pids", AsyncMock(return_value={99999}))
        mock_kill = AsyncMock()
        monkeypatch.setattr(mod, "_sigterm_then_kill", mock_kill)

        assert await _kill_stale_port_process(18888) is True
        mock_kill.assert_awaited_once_with(99999, "stale port 18888")

    async def test_owned_listener_from_own_subprocess_is_killed(self, monkeypatch):
        import hull_web.search.runner as mod

        mock_proc = MagicMock()
        mock_proc.pid = 99999
        mock_proc.poll.return_value = None
        mod._searxng_process = mock_proc
        try:
            monkeypatch.setattr(mod, "_port_listener_pids", AsyncMock(return_value={99999}))
            mock_kill = AsyncMock()
            monkeypatch.setattr(mod, "_sigterm_then_kill", mock_kill)

            assert await _kill_stale_port_process(18888) is True
            mock_kill.assert_awaited_once_with(99999, "stale port 18888")
        finally:
            mod._searxng_process = None

    async def test_own_pid_is_skipped_not_signalled(self, monkeypatch):
        """A listener that is this very process is left alone (cannot kill self)."""
        import hull_web.search.runner as mod

        monkeypatch.setattr(mod, "_read_discovery", lambda: {"pid": os.getpid(), "port": 18888})
        monkeypatch.setattr(mod, "_port_listener_pids", AsyncMock(return_value={os.getpid()}))
        mock_kill = AsyncMock()
        monkeypatch.setattr(mod, "_sigterm_then_kill", mock_kill)

        assert await _kill_stale_port_process(18888) is True
        mock_kill.assert_not_called()

    async def test_mixed_owned_and_foreign_fails_closed(self, monkeypatch):
        """Any foreign PID on the port blocks killing the owned one too."""
        import hull_web.search.runner as mod

        monkeypatch.setattr(mod, "_read_discovery", lambda: {"pid": 99999, "port": 18888})
        monkeypatch.setattr(mod, "_port_listener_pids", AsyncMock(return_value={99999, 88888}))
        mock_kill = AsyncMock()
        monkeypatch.setattr(mod, "_sigterm_then_kill", mock_kill)

        assert await _kill_stale_port_process(18888) is False
        mock_kill.assert_not_called()


class TestPortListenerPids:
    @pytest.mark.skipif(sys.platform != "win32", reason="Windows-only test")
    async def test_windows_netstat(self):
        """On Windows, parses netstat LISTENING lines into PIDs."""
        mock_result = MagicMock()
        mock_result.stdout = "  TCP    127.0.0.1:18888    0.0.0.0:0    LISTENING    99999\n"

        with patch("subprocess.run", return_value=mock_result):
            assert await _port_listener_pids(18888) == {99999}

    @pytest.mark.skipif(sys.platform == "win32", reason="Unix-only test")
    async def test_unix_lsof(self):
        """On Unix, uses lsof to list listener PIDs."""
        mock_result = MagicMock()
        mock_result.returncode = 0
        mock_result.stdout = "99999\n"

        with patch("subprocess.run", return_value=mock_result):
            assert await _port_listener_pids(18888) == {99999}

    @pytest.mark.skipif(sys.platform == "win32", reason="Unix-only test")
    async def test_unix_lsof_not_found_falls_back_to_fuser_query(self):
        """On Unix without lsof, queries fuser (no -k: never signals anything)."""
        mock_fuser_result = MagicMock()
        mock_fuser_result.returncode = 0
        mock_fuser_result.stdout = "  99999\n"

        def side_effect(args, **kwargs):
            if args[0] == "lsof":
                raise FileNotFoundError("lsof not found")
            return mock_fuser_result

        with patch("subprocess.run", side_effect=side_effect) as mock_run:
            assert await _port_listener_pids(18888) == {99999}
            assert mock_run.call_count == 2
            mock_run.assert_any_call(
                ["lsof", "-ti", ":18888"],
                shell=False,
                stdin=subprocess.DEVNULL,
                capture_output=True,
                text=True,
                timeout=5,
            )
            mock_run.assert_any_call(
                ["fuser", "18888/tcp"],
                shell=False,
                stdin=subprocess.DEVNULL,
                capture_output=True,
                text=True,
                timeout=5,
            )

    @pytest.mark.skipif(sys.platform == "win32", reason="Unix-only test")
    async def test_unix_lsof_and_fuser_not_found(self):
        """On Unix, handles cases where both lsof and fuser are missing."""
        import hull_web.search.runner as mod

        with (
            patch("subprocess.run", side_effect=FileNotFoundError("not found")),
            patch("hull_web.search.runner.logger") as mock_logger,
        ):
            assert await _port_listener_pids(18888) == set()
            mock_logger.debug.assert_any_call("Could not query processes on port %d using fuser: %r", 18888, ANY)


# ===========================================================================
# _select_start_port
# ===========================================================================


class TestSelectStartPort:
    async def test_free_port_kept(self, monkeypatch):
        import hull_web.search.runner as mod

        monkeypatch.setattr(mod, "_find_available_port", lambda start_port, max_tries=50: 18888)
        mock_kill = AsyncMock(return_value=True)
        monkeypatch.setattr(mod, "_kill_stale_port_process", mock_kill)

        assert await mod._select_start_port(18888) == 18888
        mock_kill.assert_awaited_once_with(18888)

    async def test_foreign_owner_falls_back_to_alternate_port(self, monkeypatch):
        """Foreign port owner must never be killed; pick an alternate port instead."""

        def fake_find(start_port, max_tries=50):
            return 18888 if start_port == 18888 else 20001

        import hull_web.search.runner as mod

        monkeypatch.setattr(mod, "_find_available_port", fake_find)
        mock_kill = AsyncMock(return_value=False)
        monkeypatch.setattr(mod, "_kill_stale_port_process", mock_kill)

        assert await mod._select_start_port(18888) == 20001
        mock_kill.assert_awaited_once_with(18888)


# ===========================================================================
# _cleanup_process / shutdown_searxng
# ===========================================================================


class TestCleanupProcess:
    def test_cleanup_as_owner(self, tmp_discovery, tmp_path):
        """Owner kills process, removes discovery file, and deletes settings path."""
        import hull_web.search.runner as mod

        mock_proc = MagicMock()
        mock_proc.poll.return_value = None
        mock_proc.pid = 12345
        mock_proc.stderr = None

        mod._searxng_process = mock_proc
        mod._searxng_port = 18888
        mod._is_owner = True

        # Mock settings path
        dummy_settings = tmp_path / "searxng_settings_test.yml"
        dummy_settings.write_text("test")
        mod._searxng_settings_path = dummy_settings

        _write_discovery(18888, 12345)
        assert tmp_discovery.exists()

        _cleanup_process()

        assert mod._searxng_process is None
        assert mod._searxng_port is None
        assert mod._is_owner is False
        assert not tmp_discovery.exists()
        assert not dummy_settings.exists()

    def test_cleanup_as_non_owner(self, tmp_discovery):
        """Non-owner clears local refs but does not kill or remove discovery."""
        import hull_web.search.runner as mod

        mock_proc = MagicMock()
        mock_proc.poll.return_value = None
        mock_proc.pid = 12345

        mod._searxng_process = mock_proc
        mod._searxng_port = 18888
        mod._is_owner = False

        _write_discovery(18888, 12345)

        _cleanup_process()

        assert mod._searxng_process is None
        assert mod._searxng_port is None
        # Discovery file should still exist (not our responsibility)
        assert tmp_discovery.exists()

    def test_cleanup_no_process(self):
        """Does not raise when no process exists."""
        _cleanup_process()  # Should not raise

    def test_shutdown_searxng_calls_cleanup(self, tmp_discovery):
        """shutdown_searxng delegates to _cleanup_process."""
        import hull_web.search.runner as mod

        mock_proc = MagicMock()
        mock_proc.poll.return_value = None
        mock_proc.pid = 12345
        mock_proc.stderr = None

        mod._searxng_process = mock_proc
        mod._searxng_port = 18888
        mod._is_owner = True

        shutdown_searxng()

        assert mod._searxng_process is None

    def test_cleanup_removes_settings_file(self, tmp_config_dir):
        """Cleanup removes the dynamically generated settings file."""
        import hull_web.search.runner as mod

        settings_file = tmp_config_dir / "searxng_settings_test.yml"
        settings_file.write_text("test")
        mod._searxng_settings_path = settings_file
        assert settings_file.exists()

        _cleanup_process()

        assert not settings_file.exists()


# ===========================================================================
# ensure_searxng
# ===========================================================================


class TestEnsureSearxng:
    async def test_returns_env_var_url(self, monkeypatch):
        """Returns SEARXNG_URL env var if set, no auto-start."""
        monkeypatch.setenv("SEARXNG_URL", "http://external:8080")
        url = await ensure_searxng()
        assert url == "http://external:8080"

    async def test_returns_explicit_url(self, monkeypatch):
        """Returns explicit url parameter, no auto-start."""
        monkeypatch.delenv("SEARXNG_URL", raising=False)
        url = await ensure_searxng(url="http://my-searxng:9999")
        assert url == "http://my-searxng:9999"

    async def test_explicit_url_overrides_env(self, monkeypatch):
        """Explicit url parameter takes priority over env var."""
        monkeypatch.setenv("SEARXNG_URL", "http://env:8080")
        url = await ensure_searxng(url="http://explicit:9999")
        assert url == "http://explicit:9999"

    async def test_strips_trailing_slash(self, monkeypatch):
        """Strips trailing slash from URL."""
        monkeypatch.setenv("SEARXNG_URL", "http://external:8080/")
        url = await ensure_searxng()
        assert url == "http://external:8080"

    async def test_returns_url_from_discovery(self, tmp_discovery, monkeypatch):
        """Returns URL from discovery file if valid instance is running."""
        monkeypatch.delenv("SEARXNG_URL", raising=False)
        _write_discovery(18888, os.getpid())

        with patch("hull_web.search.runner._quick_health_check", new_callable=AsyncMock, return_value=True):
            url = await ensure_searxng()
            assert url == "http://127.0.0.1:18888"

    async def test_auto_start_disabled_raises(self, tmp_discovery, monkeypatch):
        """Raises RuntimeError when no instance found and auto_start=False."""
        monkeypatch.delenv("SEARXNG_URL", raising=False)

        with pytest.raises(RuntimeError, match="auto_start is disabled"):
            await ensure_searxng(auto_start=False)

    async def test_concurrent_calls_use_lock(self, monkeypatch):
        """Concurrent calls are serialized by the asyncio lock."""
        monkeypatch.setenv("SEARXNG_URL", "http://external:8080")

        # Launch multiple concurrent calls
        results = await asyncio.gather(
            ensure_searxng(),
            ensure_searxng(),
            ensure_searxng(),
        )
        assert all(r == "http://external:8080" for r in results)

    async def test_fast_path_reuses_alive_process(self, monkeypatch):
        """Fast path returns URL when our own process is alive and healthy."""
        import hull_web.search.runner as mod

        monkeypatch.delenv("SEARXNG_URL", raising=False)

        mock_proc = MagicMock()
        mock_proc.poll.return_value = None  # alive
        mock_proc.pid = 12345

        mod._searxng_process = mock_proc
        mod._searxng_port = 18888

        with patch("hull_web.search.runner._quick_health_check", new_callable=AsyncMock, return_value=True):
            url = await ensure_searxng()
            assert url == "http://127.0.0.1:18888"

    async def test_restart_on_crash(self, tmp_discovery, monkeypatch):
        """Restarts SearXNG when the process has crashed."""
        import hull_web.search.runner as mod

        monkeypatch.delenv("SEARXNG_URL", raising=False)

        # Simulate a crashed process
        mock_proc = MagicMock()
        mock_proc.poll.return_value = 1  # exited
        mock_proc.pid = 12345
        mock_proc.stderr = None
        mod._searxng_process = mock_proc
        mod._searxng_port = 18888

        with (
            patch("hull_web.search.runner._try_reuse_existing", new_callable=AsyncMock, return_value=None),
            patch("hull_web.search.runner._start_docker_searxng", new_callable=AsyncMock, return_value=None),
            patch("hull_web.search.runner._is_searxng_installed", return_value=True),
            patch(
                "hull_web.search.runner._start_docker_searxng",
                new_callable=AsyncMock,
                return_value=None,
            ),
            patch(
                "hull_web.search.runner._start_searxng_subprocess",
                new_callable=AsyncMock,
                return_value="http://127.0.0.1:18889",
            ),
        ):
            url = await ensure_searxng()
            assert url == "http://127.0.0.1:18889"

    async def test_installs_and_starts(self, tmp_discovery, monkeypatch):
        """Installs SearXNG and starts when not installed (explicit opt-in)."""
        monkeypatch.delenv("SEARXNG_URL", raising=False)
        monkeypatch.setenv("HULL_SEARXNG_AUTO_INSTALL", "1")

        with (
            patch("hull_web.search.runner._try_reuse_existing", new_callable=AsyncMock, return_value=None),
            patch("hull_web.search.runner._start_docker_searxng", new_callable=AsyncMock, return_value=None),
            patch("hull_web.search.runner._is_searxng_installed", return_value=False),
            patch("hull_web.search.runner._install_searxng", return_value=True),
            patch(
                "hull_web.search.runner._start_searxng_subprocess",
                new_callable=AsyncMock,
                return_value="http://127.0.0.1:18888",
            ),
        ):
            url = await ensure_searxng()
            assert url == "http://127.0.0.1:18888"

    async def test_install_failure_raises(self, tmp_discovery, monkeypatch):
        """Raises RuntimeError when SearXNG installation fails."""
        monkeypatch.delenv("SEARXNG_URL", raising=False)
        monkeypatch.setenv("HULL_SEARXNG_AUTO_INSTALL", "1")

        with (
            patch("hull_web.search.runner._try_reuse_existing", new_callable=AsyncMock, return_value=None),
            patch("hull_web.search.runner._start_docker_searxng", new_callable=AsyncMock, return_value=None),
            patch("hull_web.search.runner._is_searxng_installed", return_value=False),
            patch("hull_web.search.runner._install_searxng", return_value=False),
            pytest.raises(RuntimeError, match="installation failed"),
        ):
            await ensure_searxng()

    async def test_restart_limit_reached(self, tmp_discovery, monkeypatch):
        """Raises RuntimeError when restart limit is reached."""
        import hull_web.search.runner as mod

        monkeypatch.delenv("SEARXNG_URL", raising=False)

        mod._restart_count = 3
        mod._last_restart_time = time.time()  # Recent, so counter won't reset

        with (
            patch("hull_web.search.runner._try_reuse_existing", new_callable=AsyncMock, return_value=None),
            pytest.raises(RuntimeError, match="restart limit reached"),
        ):
            await ensure_searxng()

    async def test_start_failure_raises(self, tmp_discovery, monkeypatch):
        """Raises RuntimeError when subprocess start fails."""
        monkeypatch.delenv("SEARXNG_URL", raising=False)

        with (
            patch("hull_web.search.runner._try_reuse_existing", new_callable=AsyncMock, return_value=None),
            patch("hull_web.search.runner._start_docker_searxng", new_callable=AsyncMock, return_value=None),
            patch("hull_web.search.runner._is_searxng_installed", return_value=True),
            patch("hull_web.search.runner._start_searxng_subprocess", new_callable=AsyncMock, return_value=None),
            pytest.raises(RuntimeError, match="start failed"),
        ):
            await ensure_searxng()


# ===========================================================================
# _get_startup_lock
# ===========================================================================


class TestGetStartupLock:
    def test_returns_asyncio_lock(self):
        """Returns an asyncio.Lock instance."""
        lock = _get_startup_lock()
        assert isinstance(lock, asyncio.Lock)

    def test_returns_same_lock(self):
        """Returns the same lock on subsequent calls."""
        lock1 = _get_startup_lock()
        lock2 = _get_startup_lock()
        assert lock1 is lock2


# ===========================================================================
# Settings template
# ===========================================================================


# ===========================================================================
# _start_docker_searxng
# ===========================================================================


class TestStartDockerSearxng:
    async def test_docker_binary_not_found(self):
        """Returns None if docker binary is not in PATH."""
        with patch("shutil.which", return_value=None):
            url = await _start_docker_searxng(8888)
            assert url is None

    async def test_docker_daemon_not_running(self):
        """Returns None if docker info fails (daemon not running)."""
        mock_res = MagicMock()
        mock_res.returncode = 1
        with (
            patch("shutil.which", return_value="/usr/bin/docker"),
            patch("subprocess.run", return_value=mock_res),
        ):
            url = await _start_docker_searxng(8888)
            assert url is None

    async def test_reuse_healthy_container(self, tmp_config_dir):
        """Reuses an existing container if it is healthy."""
        import hull_web.search.runner as mod

        mock_res_info = MagicMock()
        mock_res_info.returncode = 0

        mock_res_ps = MagicMock()
        mock_res_ps.stdout = "container_id_123"

        mock_lock = MagicMock()

        with (
            patch("shutil.which", return_value="/usr/bin/docker"),
            patch("subprocess.run", side_effect=[mock_res_info, mock_res_ps]),
            patch("hull_web.search.runner._get_docker_lock", return_value=mock_lock),
            patch("hull_web.search.runner._quick_health_check", new_callable=AsyncMock, return_value=True),
        ):
            url = await _start_docker_searxng(8888)
            assert url == "http://127.0.0.1:41592"
            assert mod._searxng_docker_container == "searxng-hull-41592"
            assert mod._is_owner is False

    async def test_spawn_new_container_success(self, tmp_config_dir):
        """Spawns a new container when none exists and it becomes healthy."""
        import hull_web.search.runner as mod

        mock_res_info = MagicMock()
        mock_res_info.returncode = 0

        mock_res_ps = MagicMock()
        mock_res_ps.stdout = ""  # No container running

        mock_res_rm = MagicMock()

        mock_lock = MagicMock()

        mock_popen = MagicMock()
        mock_popen.wait = AsyncMock(return_value=0)
        mock_popen.returncode = 0

        with (
            patch("shutil.which", return_value="/usr/bin/docker"),
            patch("subprocess.run", side_effect=[mock_res_info, mock_res_ps, mock_res_rm]),
            patch("hull_web.search.runner._get_docker_lock", return_value=mock_lock),
            patch("hull_web.search.runner._write_secure_text") as mock_write,
            patch("subprocess.Popen", return_value=mock_popen) as mock_popen_cls,
            patch("hull_web.search.runner._wait_for_service", new_callable=AsyncMock, return_value=True),
            patch("hull_web.search.runner._write_discovery", return_value=None),
        ):
            url = await _start_docker_searxng(8888)
            assert url == "http://127.0.0.1:41592"
            assert mod._searxng_docker_container == "searxng-hull-41592"
            assert mod._is_owner is True
            mock_write.assert_called_once()

        # Image must be a pinned release tag, never the floating 'latest'.
        image = next(arg for arg in mock_popen_cls.call_args[0][0] if arg.startswith("searxng/searxng:"))
        assert image == "searxng/searxng:2026.4.7-08ef7a63d"

    async def test_respawn_unhealthy_container(self, tmp_config_dir):
        """Respawns if container exists but is unhealthy."""
        import hull_web.search.runner as mod

        mock_res_info = MagicMock()
        mock_res_info.returncode = 0

        mock_res_ps = MagicMock()
        mock_res_ps.stdout = "container_id_123"

        mock_res_rm = MagicMock()

        mock_lock = MagicMock()

        mock_popen = MagicMock()
        mock_popen.wait = AsyncMock(return_value=0)
        mock_popen.returncode = 0

        with (
            patch("shutil.which", return_value="/usr/bin/docker"),
            patch("subprocess.run", side_effect=[mock_res_info, mock_res_ps, mock_res_rm]),
            patch("hull_web.search.runner._get_docker_lock", return_value=mock_lock),
            patch("hull_web.search.runner._quick_health_check", new_callable=AsyncMock, return_value=False),
            patch("hull_web.search.runner._write_secure_text"),
            patch("subprocess.Popen", return_value=mock_popen),
            patch("hull_web.search.runner._wait_for_service", new_callable=AsyncMock, return_value=True),
            patch("hull_web.search.runner._write_discovery", return_value=None),
        ):
            url = await _start_docker_searxng(8888)
            assert url == "http://127.0.0.1:41592"
            assert mod._is_owner is True

    async def test_docker_run_failure(self, tmp_config_dir):
        """Returns None if docker run fails."""
        mock_res_info = MagicMock()
        mock_res_info.returncode = 0
        mock_res_ps = MagicMock()
        mock_res_ps.stdout = ""
        mock_res_rm = MagicMock()

        mock_lock = MagicMock()

        mock_popen = MagicMock()
        mock_popen.wait = AsyncMock(return_value=1)
        mock_popen.returncode = 1

        with (
            patch("shutil.which", return_value="/usr/bin/docker"),
            patch("subprocess.run", side_effect=[mock_res_info, mock_res_ps, mock_res_rm]),
            patch("hull_web.search.runner._get_docker_lock", return_value=mock_lock),
            patch("hull_web.search.runner._write_secure_text"),
            patch("subprocess.Popen", return_value=mock_popen),
        ):
            url = await _start_docker_searxng(8888)
            assert url is None

    async def test_service_timeout_cleanup(self, tmp_config_dir):
        """Cleans up and returns None if service never becomes healthy."""
        mock_res_info = MagicMock()
        mock_res_info.returncode = 0
        mock_res_ps = MagicMock()
        mock_res_ps.stdout = ""
        mock_res_rm_init = MagicMock()
        mock_res_rm_cleanup = MagicMock()

        mock_lock = MagicMock()

        mock_popen = MagicMock()
        mock_popen.wait = AsyncMock(return_value=0)
        mock_popen.returncode = 0

        with (
            patch("shutil.which", return_value="/usr/bin/docker"),
            patch(
                "subprocess.run", side_effect=[mock_res_info, mock_res_ps, mock_res_rm_init, mock_res_rm_cleanup]
            ) as mock_run,
            patch("hull_web.search.runner._get_docker_lock", return_value=mock_lock),
            patch("hull_web.search.runner._write_secure_text"),
            patch("subprocess.Popen", return_value=mock_popen),
            patch("hull_web.search.runner._wait_for_service", new_callable=AsyncMock, return_value=False),
        ):
            url = await _start_docker_searxng(8888)
            assert url is None
            # Verify cleanup rm was called
            assert mock_run.call_count == 4

    async def test_general_exception_handling(self):
        """Returns None and logs error on unexpected exceptions."""
        with (
            patch("shutil.which", return_value="/usr/bin/docker"),
            patch("asyncio.to_thread", side_effect=Exception("unexpected")),
        ):
            url = await _start_docker_searxng(8888)
            assert url is None


class TestSettingsTemplate:
    def test_template_has_placeholders(self):
        """Template contains the expected format placeholders."""
        assert "{port}" in _SETTINGS_TEMPLATE
        assert "{secret_key}" in _SETTINGS_TEMPLATE
        assert "{enable_http2}" in _SETTINGS_TEMPLATE

    def test_template_renders_cleanly(self):
        """Template renders without errors."""
        rendered = _SETTINGS_TEMPLATE.format(
            port=18888,
            secret_key="test-key-123",
            enable_http2="true",
        )
        assert "port: 18888" in rendered
        assert 'secret_key: "test-key-123"' in rendered
        assert "enable_http2: true" in rendered


# ===========================================================================
# Module exports
# ===========================================================================


class TestModuleExports:
    def test_ensure_searxng_exported(self):
        """ensure_searxng is available from hull_web.search."""
        from hull_web.search import ensure_searxng as fn

        assert callable(fn)

    def test_shutdown_searxng_exported(self):
        """shutdown_searxng is available from hull_web.search."""
        from hull_web.search import shutdown_searxng as fn

        assert callable(fn)

    def test_all_contains_runner_exports(self):
        """__all__ includes runner exports."""
        from hull_web.search import __all__ as all_exports

        assert "ensure_searxng" in all_exports
        assert "shutdown_searxng" in all_exports


# ===========================================================================
# _get_docker_lock
# ===========================================================================


class TestGetDockerLock:
    def test_returns_filelock(self, tmp_config_dir):
        """Returns a filelock.FileLock instance with correct path and timeout."""
        import filelock

        lock = _get_docker_lock()
        assert isinstance(lock, filelock.FileLock)
        assert Path(lock.lock_file).name == "docker_startup.lock"
        assert lock.timeout == 60.0

    def test_returns_same_lock(self, tmp_config_dir):
        """Returns the same lock on subsequent calls."""
        lock1 = _get_docker_lock()
        lock2 = _get_docker_lock()
        assert lock1 is lock2

    def test_creates_config_dir(self, tmp_path, monkeypatch):
        """Creates the config directory if it does not exist."""
        config_dir = tmp_path / "new-config-dir"
        monkeypatch.setattr("hull_web.search.runner._CONFIG_DIR", config_dir)
        monkeypatch.setattr("hull_web.search.runner._docker_lock", None)

        assert not config_dir.exists()
        _get_docker_lock()
        assert config_dir.exists()

    def test_get_config_dir_permissions(self, tmp_path, monkeypatch):
        """_get_config_dir sets correct permissions (0o700) on non-Windows."""
        import os
        import sys

        from hull_web.search.runner import _get_config_dir

        config_dir = tmp_path / "perm-test-dir"
        monkeypatch.setattr("hull_web.search.runner._CONFIG_DIR", config_dir)

        _get_config_dir()
        assert config_dir.exists()

        if sys.platform != "win32":
            mode = os.stat(config_dir).st_mode
            assert (mode & 0o777) == 0o700


# ===========================================================================
# _handle_restart_and_start
# ===========================================================================


class TestHandleRestartAndStart:
    @pytest.fixture(autouse=True)
    def _reset_state(self):
        import hull_web.search.runner as mod

        mod._searxng_process = None
        mod._searxng_port = None
        mod._restart_count = 0
        mod._last_restart_time = 0.0
        mod._searxng_docker_container = None
        yield

    async def test_crash_detection_logging(self):
        """Verifies crashed process is logged and cleared."""
        import hull_web.search.runner as mod

        mock_proc = MagicMock()
        mock_proc.poll.return_value = -1
        mock_proc.stderr.read.return_value = b"some error"
        mod._searxng_process = mock_proc

        with (
            patch("hull_web.search.runner.logger") as mock_logger,
            patch("hull_web.search.runner._start_docker_searxng", new_callable=AsyncMock, return_value="http://ok"),
            patch("asyncio.to_thread", side_effect=lambda f, *args: f(*args)),
        ):
            await _handle_restart_and_start(start_port=8888)
            assert mod._searxng_process is None
            # Check warning call - first argument to warning
            warning_call = [call for call in mock_logger.warning.call_args_list if "crashed" in call.args[0]]
            assert len(warning_call) > 0

    async def test_crash_detection_stderr_does_not_forge_log_line(self):
        """stderr cua subprocess la du lieu ngoai -- CRLF khong duoc tao dong log gia."""
        import logging

        import hull_web.search.runner as mod

        crlf = b"\x0d\x0a"
        mock_proc = MagicMock()
        mock_proc.poll.return_value = -1
        mock_proc.stderr.read.return_value = b"boom" + crlf + b"2026-01-01 CRITICAL root: forged entry"
        mod._searxng_process = mock_proc

        records: list[logging.LogRecord] = []

        class _Capture(logging.Handler):
            def emit(self, record: logging.LogRecord) -> None:
                records.append(record)

        handler = _Capture()
        mod.logger.addHandler(handler)
        old_level = mod.logger.level
        mod.logger.setLevel(logging.DEBUG)
        try:
            with (
                patch(
                    "hull_web.search.runner._start_docker_searxng",
                    new_callable=AsyncMock,
                    return_value="http://ok",
                ),
                patch("asyncio.to_thread", side_effect=lambda f, *args: f(*args)),
            ):
                await _handle_restart_and_start(start_port=8888)
        finally:
            mod.logger.removeHandler(handler)
            mod.logger.setLevel(old_level)

        crashed = [r.getMessage() for r in records if "crashed" in r.getMessage()]
        assert crashed, "expected a crash warning"
        assert all("\r" not in m and "\n" not in m for m in crashed)
        # Escape chu khong nuot: noi dung stderr van doc duoc.
        assert any("forged entry" in m for m in crashed)

    async def test_crash_detection_stderr_read_failure(self):
        """Verifies crash detection handles stderr read failure gracefully."""
        import hull_web.search.runner as mod

        mock_proc = MagicMock()
        mock_proc.poll.return_value = -1
        mock_proc.stderr.read.side_effect = Exception("read error")
        mod._searxng_process = mock_proc

        with (
            patch("hull_web.search.runner.logger") as mock_logger,
            patch("hull_web.search.runner._start_docker_searxng", new_callable=AsyncMock, return_value="http://ok"),
            patch("asyncio.to_thread", side_effect=lambda f, *args: f(*args)),
        ):
            await _handle_restart_and_start(start_port=8888)
            assert mod._searxng_process is None
            warning_call = [call for call in mock_logger.warning.call_args_list if "crashed" in call.args[0]]
            assert len(warning_call) > 0
            # stderr should be empty in the log message
            assert "stderr: " in warning_call[0].args[0]

    async def test_restart_counter_reset(self):
        """Resets restart count if enough time passed."""
        import hull_web.search.runner as mod

        mod._restart_count = 5
        mod._last_restart_time = time.time() - 301

        with patch("hull_web.search.runner._start_docker_searxng", new_callable=AsyncMock, return_value="http://ok"):
            await _handle_restart_and_start(start_port=8888)
            assert mod._restart_count == 0

    async def test_docker_fallback_success(self):
        """Returns Docker URL if successful."""
        import hull_web.search.runner as mod

        with patch(
            "hull_web.search.runner._start_docker_searxng", new_callable=AsyncMock, return_value="http://docker-url"
        ):
            url = await _handle_restart_and_start(start_port=8888)
            assert url == "http://docker-url"
            assert mod._restart_count == 0

    async def test_subprocess_success_after_docker_fail(self):
        """Falls back to subprocess if Docker fails."""
        import hull_web.search.runner as mod

        with (
            patch("hull_web.search.runner._start_docker_searxng", new_callable=AsyncMock, return_value=None),
            patch("hull_web.search.runner._is_searxng_installed", return_value=True),
            patch(
                "hull_web.search.runner._start_searxng_subprocess",
                new_callable=AsyncMock,
                return_value="http://sub-url",
            ),
        ):
            url = await _handle_restart_and_start(start_port=8888)
            assert url == "http://sub-url"
            assert mod._restart_count == 0

    async def test_restart_limit_reached(self):
        """Raises RuntimeError if limit reached."""
        import hull_web.search.runner as mod

        mod._restart_count = 3
        mod._last_restart_time = time.time()

        with pytest.raises(RuntimeError, match="restart limit reached"):
            await _handle_restart_and_start(start_port=8888)

    async def test_install_failure(self, monkeypatch):
        """Raises RuntimeError if installation fails."""
        monkeypatch.setenv("HULL_SEARXNG_AUTO_INSTALL", "1")
        with (
            patch("hull_web.search.runner._start_docker_searxng", new_callable=AsyncMock, return_value=None),
            patch("hull_web.search.runner._is_searxng_installed", return_value=False),
            patch("hull_web.search.runner._install_searxng", return_value=False),
            pytest.raises(RuntimeError, match="installation failed"),
        ):
            await _handle_restart_and_start(start_port=8888)

    async def test_install_without_opt_in_raises_and_does_not_pip_install(self, monkeypatch):
        """Runtime pip-install is opt-in: default OFF points the host at SEARXNG_URL."""
        import hull_web.search.runner as mod

        monkeypatch.delenv("HULL_SEARXNG_AUTO_INSTALL", raising=False)
        mock_install = MagicMock(return_value=True)

        with (
            patch("hull_web.search.runner._start_docker_searxng", new_callable=AsyncMock, return_value=None),
            patch("hull_web.search.runner._is_searxng_installed", return_value=False),
            patch("hull_web.search.runner._install_searxng", mock_install),
            patch("hull_web.search.runner._start_searxng_subprocess", new_callable=AsyncMock, return_value=None),
            pytest.raises(RuntimeError, match="SEARXNG_URL"),
        ):
            await _handle_restart_and_start(start_port=8888)
        mock_install.assert_not_called()

    async def test_install_opt_in_env_allows_install(self, monkeypatch):
        """HULL_SEARXNG_AUTO_INSTALL=1 restores the legacy runtime-install path."""
        monkeypatch.setenv("HULL_SEARXNG_AUTO_INSTALL", "1")

        with (
            patch("hull_web.search.runner._start_docker_searxng", new_callable=AsyncMock, return_value=None),
            patch("hull_web.search.runner._is_searxng_installed", return_value=False),
            patch("hull_web.search.runner._install_searxng", return_value=True),
            patch(
                "hull_web.search.runner._start_searxng_subprocess",
                new_callable=AsyncMock,
                return_value="http://sub-url",
            ),
        ):
            assert await _handle_restart_and_start(start_port=8888) == "http://sub-url"

    async def test_cooldown_applied(self):
        """Applies cooldown if restart count > 0."""
        import hull_web.search.runner as mod

        mod._restart_count = 1
        mod._last_restart_time = time.time()

        with (
            patch("hull_web.search.runner._start_docker_searxng", new_callable=AsyncMock, return_value=None),
            patch("hull_web.search.runner._is_searxng_installed", return_value=True),
            patch("asyncio.sleep", new_callable=AsyncMock) as mock_sleep,
            patch("hull_web.search.runner._start_searxng_subprocess", new_callable=AsyncMock, return_value="http://ok"),
        ):
            await _handle_restart_and_start(start_port=8888)
            mock_sleep.assert_called_once_with(2.0)  # _RESTART_COOLDOWN * 1

    async def test_subprocess_failure_raises(self):
        """Raises RuntimeError if subprocess also fails."""
        with (
            patch("hull_web.search.runner._start_docker_searxng", new_callable=AsyncMock, return_value=None),
            patch("hull_web.search.runner._is_searxng_installed", return_value=True),
            patch("hull_web.search.runner._start_searxng_subprocess", new_callable=AsyncMock, return_value=None),
            pytest.raises(RuntimeError, match="start failed after all attempts"),
        ):
            await _handle_restart_and_start(start_port=8888)

    async def test_crash_detection_no_stderr(self):
        """Verifies crash detection handles missing stderr."""
        import hull_web.search.runner as mod

        mock_proc = MagicMock()
        mock_proc.poll.return_value = -1
        mock_proc.stderr = None
        mod._searxng_process = mock_proc

        with (
            patch("hull_web.search.runner.logger") as mock_logger,
            patch("hull_web.search.runner._start_docker_searxng", new_callable=AsyncMock, return_value="http://ok"),
        ):
            await _handle_restart_and_start(start_port=8888)
            assert mod._searxng_process is None
            warning_call = [call for call in mock_logger.warning.call_args_list if "crashed" in call.args[0]]
            assert len(warning_call) > 0

    async def test_install_required_and_success(self, monkeypatch):
        """Verifies it proceeds if installation is required and succeeds."""
        monkeypatch.setenv("HULL_SEARXNG_AUTO_INSTALL", "1")

        with (
            patch("hull_web.search.runner._start_docker_searxng", new_callable=AsyncMock, return_value=None),
            patch("hull_web.search.runner._is_searxng_installed", return_value=False),
            patch("hull_web.search.runner._install_searxng", return_value=True),
            patch(
                "hull_web.search.runner._start_searxng_subprocess",
                new_callable=AsyncMock,
                return_value="http://sub-url",
            ),
        ):
            url = await _handle_restart_and_start(start_port=8888)
            assert url == "http://sub-url"
