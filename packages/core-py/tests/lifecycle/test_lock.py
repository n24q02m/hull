"""Tests for cross-process lifecycle lock."""

from __future__ import annotations

import os
from datetime import datetime, timedelta, timezone
import subprocess
import sys
import textwrap
import time
import uuid
from pathlib import Path

import pytest

from hull_core.lifecycle.lock import (
    DEFAULT_LOCK_TTL_HOURS,
    LifecycleLock,
    _lock_dir,
    _locks_dir,
    _parse_lock_text,
    refresh_lock_timestamp,
    sweep_stale_locks,
)


@pytest.fixture
def lock_root(tmp_path: Path) -> Path:
    """Per-test lock directory so concurrent tests don't collide."""
    root = tmp_path / "locks"
    root.mkdir(parents=True, exist_ok=True)
    return root


@pytest.fixture
def unique_name() -> str:
    """Unique server name per test (belt-and-braces alongside lock_root)."""
    return f"test-srv-{uuid.uuid4().hex[:8]}"


class TestAcquireAndRelease:
    def test_acquires_and_releases(self, lock_root: Path, unique_name: str) -> None:
        lock = LifecycleLock(name=unique_name, port=9000, root=lock_root)
        lock_file = lock_root / f"{unique_name}-9000.lock"

        with lock:
            assert lock_file.exists(), "lock file must exist while held"

        # Scaffold unlinks on __exit__, so the file should be gone after release.
        assert not lock_file.exists(), "lock file must be removed after release"

    def test_acquires_multiple_times_sequentially(self, lock_root: Path, unique_name: str) -> None:
        """After release, a fresh LifecycleLock with same (name, port) can re-acquire."""
        lock1 = LifecycleLock(name=unique_name, port=9000, root=lock_root)
        with lock1:
            pass

        lock2 = LifecycleLock(name=unique_name, port=9000, root=lock_root)
        with lock2:
            pass  # should not raise

    def test_lock_stores_pid_and_port(self, lock_root: Path, unique_name: str) -> None:
        """While held, the lock file contains the current PID and port."""
        lock = LifecycleLock(name=unique_name, port=9000, root=lock_root)
        lock_file = lock_root / f"{unique_name}-9000.lock"

        with lock:
            content = lock_file.read_text(encoding="utf-8")
            lines = [line.strip() for line in content.splitlines() if line.strip()]
            assert lines[0] == str(os.getpid()), f"expected lock file to start with pid {os.getpid()}, got {lines!r}"
            assert lines[1] == "9000", f"expected lock file to contain port 9000, got {lines!r}"

    def test_path_property_exposes_lock_file_location(self, lock_root: Path, unique_name: str) -> None:
        lock = LifecycleLock(name=unique_name, port=9000, root=lock_root)
        assert lock.path == lock_root / f"{unique_name}-9000.lock"


class TestDifferentLocksDoNotConflict:
    def test_different_ports_do_not_conflict(self, lock_root: Path, unique_name: str) -> None:
        lock_a = LifecycleLock(name=unique_name, port=9000, root=lock_root)
        lock_b = LifecycleLock(name=unique_name, port=9001, root=lock_root)

        with lock_a, lock_b:
            assert (lock_root / f"{unique_name}-9000.lock").exists()
            assert (lock_root / f"{unique_name}-9001.lock").exists()

    def test_different_names_do_not_conflict(self, lock_root: Path) -> None:
        name_a = f"srv-a-{uuid.uuid4().hex[:6]}"
        name_b = f"srv-b-{uuid.uuid4().hex[:6]}"

        lock_a = LifecycleLock(name=name_a, port=9000, root=lock_root)
        lock_b = LifecycleLock(name=name_b, port=9000, root=lock_root)

        with lock_a, lock_b:
            assert (lock_root / f"{name_a}-9000.lock").exists()
            assert (lock_root / f"{name_b}-9000.lock").exists()


def _helper_script(lock_root: Path, name: str, port: int, ready_file: Path) -> str:
    """Python source for a subprocess that holds a lock until stdin closes."""
    # Inject the src path of the installed package by using the sys.path of the
    # parent process - simplest is to rely on the parent's environment, since
    # `uv run pytest` already has hull_core importable.
    return textwrap.dedent(
        f"""
        import sys
        from pathlib import Path
        from hull_core.lifecycle.lock import LifecycleLock

        lock = LifecycleLock(name={name!r}, port={port}, root=Path({str(lock_root)!r}))
        with lock:
            Path({str(ready_file)!r}).write_text("ready", encoding="utf-8")
            # Block on stdin so the parent test controls when we release the lock.
            sys.stdin.read()
        """
    ).strip()


class TestContention:
    """Verify that a second attempt to acquire the same lock fails."""

    def test_contention_raises_across_processes(self, tmp_path: Path, lock_root: Path, unique_name: str) -> None:
        ready_file = tmp_path / "ready.txt"
        script = _helper_script(lock_root, unique_name, 9000, ready_file)

        # Launch subprocess that acquires the lock and waits on stdin.
        proc = subprocess.Popen(
            [sys.executable, "-c", script],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )

        try:
            # Wait for helper to signal it has acquired the lock.
            deadline = time.monotonic() + 10.0
            while time.monotonic() < deadline:
                if ready_file.exists():
                    break
                if proc.poll() is not None:
                    stdout, stderr = proc.communicate(timeout=1.0)
                    pytest.fail(
                        "helper process exited before acquiring lock: "
                        f"rc={proc.returncode} stdout={stdout!r} stderr={stderr!r}"
                    )
                time.sleep(0.05)
            else:
                pytest.fail("helper did not acquire lock within 10s")

            # Now try to acquire in this process - must raise.
            contender = LifecycleLock(name=unique_name, port=9000, root=lock_root)
            with pytest.raises(RuntimeError, match="another process holds"):
                with contender:
                    pass
        finally:
            # Release the helper: closing stdin makes sys.stdin.read() return.
            if proc.stdin is not None:
                try:
                    proc.stdin.close()
                except Exception:
                    pass
            try:
                proc.wait(timeout=10.0)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait(timeout=5.0)

    def test_lock_available_after_holder_releases(self, tmp_path: Path, lock_root: Path, unique_name: str) -> None:
        ready_file = tmp_path / "ready.txt"
        script = _helper_script(lock_root, unique_name, 9000, ready_file)

        proc = subprocess.Popen(
            [sys.executable, "-c", script],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )

        try:
            deadline = time.monotonic() + 10.0
            while time.monotonic() < deadline:
                if ready_file.exists():
                    break
                if proc.poll() is not None:
                    stdout, stderr = proc.communicate(timeout=1.0)
                    pytest.fail(
                        "helper process exited before acquiring lock: "
                        f"rc={proc.returncode} stdout={stdout!r} stderr={stderr!r}"
                    )
                time.sleep(0.05)
            else:
                pytest.fail("helper did not acquire lock within 10s")
        finally:
            if proc.stdin is not None:
                try:
                    proc.stdin.close()
                except Exception:
                    pass
            try:
                proc.wait(timeout=10.0)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait(timeout=5.0)

        # Helper has exited, lock should be free again.
        reacquired = LifecycleLock(name=unique_name, port=9000, root=lock_root)
        with reacquired:
            pass  # must not raise


def _write_lock(path: Path, *, pid: int, port: int, token: str = "tok", age_hours: float = 0.0) -> None:
    """Write a lock payload padded to the on-disk width LifecycleLock uses."""
    spawned = datetime.now(timezone.utc) - timedelta(hours=age_hours)
    path.write_text(f"{pid}\n{port}\n{token}\n{spawned.isoformat()}\n".ljust(512, " "), encoding="utf-8")


class TestParseLockText:
    def test_parses_four_line_payload(self) -> None:
        md = _parse_lock_text("4242\n9100\ntok-abc\n2026-09-30T10:00:00+00:00\n")
        assert md is not None
        assert (md.pid, md.port, md.token) == (4242, 9100, "tok-abc")
        assert md.spawned_at.year == 2026

    def test_legacy_longer_payload_keeps_first_four_lines(self) -> None:
        md = _parse_lock_text("7\n80\ntok\n2026-09-30T10:00:00+00:00\nlegacy\ntrailing\n")
        assert md is not None
        assert (md.pid, md.port) == (7, 80)

    def test_padded_record_parses(self) -> None:
        padded = ("9\n80\ntok\n2026-09-30T10:00:00+00:00\n").ljust(512, " ")
        md = _parse_lock_text(padded)
        assert md is not None and md.pid == 9

    @pytest.mark.parametrize(
        "raw",
        [
            pytest.param("", id="empty"),
            pytest.param("1\n2\n3\n", id="three-lines"),
            pytest.param("dead-pid\n80\ntok\n2026-09-30T10:00:00+00:00\n", id="pid-not-a-number"),
            pytest.param("7\nnot-a-port\ntok\n2026-09-30T10:00:00+00:00\n", id="port-not-a-number"),
        ],
    )
    def test_malformed_payloads_are_rejected(self, raw: str) -> None:
        assert _parse_lock_text(raw) is None


class TestLockDirResolution:
    def test_explicit_root_wins(self, tmp_path: Path) -> None:
        assert _locks_dir(tmp_path) == tmp_path

    def test_default_root_is_under_home(self) -> None:
        assert _locks_dir() == Path.home() / ".config" / "mcp" / "locks"

    def test_lock_dir_delegates_to_the_no_arg_variant(self, tmp_path: Path, monkeypatch) -> None:  # noqa: ANN001
        monkeypatch.setattr("hull_core.lifecycle.lock._locks_dir", lambda root=None: tmp_path)
        assert _lock_dir() == tmp_path


class TestRefreshLockTimestamp:
    def test_updates_spawned_at_and_preserves_identity(self, tmp_path: Path) -> None:
        lock_file = tmp_path / "hull-9000.lock"
        _write_lock(lock_file, pid=4242, port=9000, token="tok-abc", age_hours=3)
        before_size = lock_file.stat().st_size

        refresh_lock_timestamp(lock_file)

        md = _parse_lock_text(lock_file.read_text(encoding="utf-8"))
        assert md is not None
        assert (md.pid, md.port, md.token) == (4242, 9000, "tok-abc")
        assert md.spawned_at > datetime.now(timezone.utc) - timedelta(minutes=1)
        assert lock_file.stat().st_size == before_size, "padding must keep the on-disk width stable"

    def test_missing_file_is_a_no_op(self, tmp_path: Path) -> None:
        refresh_lock_timestamp(tmp_path / "absent.lock")  # must not raise

    def test_malformed_file_is_left_alone(self, tmp_path: Path) -> None:
        lock_file = tmp_path / "hull-9000.lock"
        lock_file.write_text("garbage\n", encoding="utf-8")

        refresh_lock_timestamp(lock_file)

        assert lock_file.read_text(encoding="utf-8") == "garbage\n"

    def test_unwritable_lock_does_not_raise(self, tmp_path: Path, monkeypatch) -> None:  # noqa: ANN001
        lock_file = tmp_path / "hull-9000.lock"
        _write_lock(lock_file, pid=1, port=9000)

        def boom(*_args, **_kwargs):
            raise OSError("file is locked")

        monkeypatch.setattr("builtins.open", boom)
        refresh_lock_timestamp(lock_file)  # best-effort: the server must stay alive


class TestSweepStaleLocks:
    def test_missing_dir_returns_zero(self, tmp_path: Path) -> None:
        assert sweep_stale_locks("hull", root=tmp_path / "absent") == 0

    def test_expired_lock_removed_fresh_lock_kept(self, tmp_path: Path) -> None:
        stale = tmp_path / "hull-9000.lock"
        fresh = tmp_path / "hull-9001.lock"
        _write_lock(stale, pid=1, port=9000, age_hours=DEFAULT_LOCK_TTL_HOURS + 1)
        _write_lock(fresh, pid=2, port=9001)

        assert sweep_stale_locks("hull", root=tmp_path) == 1
        assert not stale.exists()
        assert fresh.exists(), "a lock inside the TTL belongs to a live server"

    def test_other_servers_are_untouched(self, tmp_path: Path) -> None:
        other = tmp_path / "wet-9000.lock"
        _write_lock(other, pid=1, port=9000, age_hours=DEFAULT_LOCK_TTL_HOURS + 1)

        assert sweep_stale_locks("hull", root=tmp_path) == 0
        assert other.exists()

    def test_malformed_lock_is_reclaimed(self, tmp_path: Path) -> None:
        corrupt = tmp_path / "hull-9000.lock"
        corrupt.write_text("dead-pid\n", encoding="utf-8")

        assert sweep_stale_locks("hull", root=tmp_path) == 1
        assert not corrupt.exists()

    def test_unreadable_lock_is_reclaimed(self, tmp_path: Path, monkeypatch) -> None:  # noqa: ANN001
        broken = tmp_path / "hull-9000.lock"
        _write_lock(broken, pid=1, port=9000)
        original = Path.read_text

        def deny(self, *args, **kwargs):  # noqa: ANN001, ANN002, ANN003
            if self.name.endswith(".lock"):
                raise OSError("denied")
            return original(self, *args, **kwargs)

        monkeypatch.setattr(Path, "read_text", deny)
        assert sweep_stale_locks("hull", root=tmp_path) == 1
        assert not broken.exists()

    @pytest.mark.parametrize("state", ["expired", "malformed", "unreadable"])
    def test_unremovable_lock_is_reported_as_kept(self, tmp_path: Path, monkeypatch, state: str) -> None:  # noqa: ANN001
        stale = tmp_path / "hull-9000.lock"
        if state == "malformed":
            stale.write_text("dead-pid\n", encoding="utf-8")
        else:
            _write_lock(stale, pid=1, port=9000, age_hours=DEFAULT_LOCK_TTL_HOURS + 1)
            if state == "unreadable":
                original = Path.read_text

                def deny(self, *args, **kwargs):  # noqa: ANN001, ANN002, ANN003
                    if self.name.endswith(".lock"):
                        raise OSError("denied")
                    return original(self, *args, **kwargs)

                monkeypatch.setattr(Path, "read_text", deny)

        def deny_unlink(self, **_kwargs):  # noqa: ANN001, ANN003
            raise OSError("in use")

        monkeypatch.setattr(Path, "unlink", deny_unlink)
        assert sweep_stale_locks("hull", root=tmp_path) == 0

    def test_default_root_used_when_root_omitted(self, tmp_path: Path, monkeypatch) -> None:  # noqa: ANN001
        stale = tmp_path / "hull-9000.lock"
        _write_lock(stale, pid=1, port=9000, age_hours=DEFAULT_LOCK_TTL_HOURS + 1)
        monkeypatch.setattr("hull_core.lifecycle.lock._lock_dir", lambda: tmp_path)

        assert sweep_stale_locks("hull") == 1
        assert not stale.exists()

    def test_ttl_argument_is_honoured(self, tmp_path: Path) -> None:
        lock = tmp_path / "hull-9000.lock"
        _write_lock(lock, pid=1, port=9000, age_hours=2)

        assert sweep_stale_locks("hull", ttl_hours=1, root=tmp_path) == 1
        assert sweep_stale_locks("hull", ttl_hours=DEFAULT_LOCK_TTL_HOURS, root=tmp_path) == 0


def test_bad_timestamp_is_rejected() -> None:
    assert _parse_lock_text("1\n2\ntok\nnot-a-date\n") is None


class TestAcquireFailurePaths:
    def test_open_failure_raises_runtime_error(self, lock_root: Path, unique_name: str, monkeypatch) -> None:  # noqa: ANN001
        def boom(*_args, **_kwargs):
            raise PermissionError("denied")

        monkeypatch.setattr("os.open", boom)
        with pytest.raises(RuntimeError, match="Failed to open lock file"):
            with LifecycleLock(name=unique_name, port=9000, root=lock_root):
                pass

    def test_release_survives_a_failed_unlink(self, lock_root: Path, unique_name: str, monkeypatch) -> None:  # noqa: ANN001
        lock = LifecycleLock(name=unique_name, port=9000, root=lock_root)
        lock.__enter__()

        def boom(self, **_kwargs):  # noqa: ANN001, ANN003
            raise OSError("in use")

        monkeypatch.setattr(Path, "unlink", boom)
        lock.__exit__(None, None, None)  # must not raise
        assert lock._fh is None, "the handle is closed even when the file cannot be unlinked"
