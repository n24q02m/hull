"""Concurrency: leases and cooldowns are shared, never reused across tasks."""

from __future__ import annotations

import threading
from pathlib import Path

import pytest

from helpers import Clock, ProviderFailure
from hull_core.credentials.models import PoolExhaustedError
from hull_core.credentials.store import CredentialStore

BASE_URL = "https://openrouter.ai/api/v1"


def seed(store: CredentialStore, count: int = 2) -> list:
    return [
        store.add_credential("openrouter", f"sk-live-concurrent-{index:04d}", base_url=BASE_URL)
        for index in range(count)
    ]


def test_two_threads_never_lease_the_same_credential(store: CredentialStore) -> None:
    seed(store, 2)
    pool = store.pool("openrouter", lease_seconds=30)
    barrier = threading.Barrier(2)
    leased: list[int] = []
    failures: list[BaseException] = []
    lock = threading.Lock()

    def worker() -> None:
        try:
            credential = pool.acquire()
            with lock:
                leased.append(credential.id)
            barrier.wait(timeout=10)
        except BaseException as exc:  # surface thread failures in the test result
            with lock:
                failures.append(exc)

    threads = [threading.Thread(target=worker) for _ in range(2)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=15)
    assert failures == []
    assert len(leased) == 2
    assert len(set(leased)) == 2


def test_a_task_never_reuses_the_credential_that_just_got_a_429(store: CredentialStore) -> None:
    """The core concurrency contract, orchestrated deterministically.

    Task A leases credential 1 and reports a 429; task B, already running,
    must hold credential 2 and must not be handed credential 1 afterwards.
    """
    first, second = seed(store)
    pool = store.pool("openrouter", lease_seconds=30)
    a_has_first = threading.Event()
    b_acquired = threading.Event()
    a_failed = threading.Event()
    b_done = threading.Event()
    results: dict[str, int] = {}

    def task_a() -> None:
        credential = pool.acquire()
        results["a_first"] = credential.id
        a_has_first.set()
        assert b_acquired.wait(timeout=10)
        pool.report_failure(credential, 429, retry_after=60)
        a_failed.set()
        assert b_done.wait(timeout=10)
        results["a_after_429"] = pool.acquire().id

    def task_b() -> None:
        assert a_has_first.wait(timeout=10)
        credential = pool.acquire()
        results["b"] = credential.id
        b_acquired.set()
        assert a_failed.wait(timeout=10)
        # A's 429 cooled credential 1 only; B's credential is untouched.
        assert store.health("openrouter")[0].usable is False
        pool.report_success(credential)
        b_done.set()

    threads = [threading.Thread(target=task_a), threading.Thread(target=task_b)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=20)
    assert results["a_first"] == first.id
    assert results["b"] == second.id
    # After the 429, the next lease is the other credential — never the cooled one.
    assert results["a_after_429"] == second.id


def test_cooldown_written_by_one_connection_is_seen_by_another(tmp_path: Path, clock: Clock) -> None:
    path = tmp_path / "shared.db"
    writer = CredentialStore(path, clock=clock)
    reader = CredentialStore(path, clock=clock)
    try:
        first = writer.add_credential("openrouter", "sk-live-shared-0001", base_url=BASE_URL)
        second = writer.add_credential("openrouter", "sk-live-shared-0002", base_url=BASE_URL)
        leased = writer.pool("openrouter").acquire()
        assert leased.id == first.id
        writer.pool("openrouter").report_failure(leased, 429, retry_after=60)
        assert reader.pool("openrouter").acquire().id == second.id
        with pytest.raises(PoolExhaustedError, match="cooling_down=1"):
            reader.pool("openrouter", lease_seconds=1).acquire(exclude=[second.id])
    finally:
        writer.close()
        reader.close()


def test_concurrent_failures_never_leak_a_key_into_the_error(store: CredentialStore) -> None:
    seed(store, 2)
    pool = store.pool("openrouter", max_5xx_retries=0)
    errors: list[str] = []
    lock = threading.Lock()

    def worker() -> None:
        try:
            pool.run(lambda credential: (_ for _ in ()).throw(ProviderFailure(401, f"bad key {credential.api_key}")))
        except Exception as exc:  # PoolExhaustedError or the losing thread's own error
            with lock:
                errors.append(f"{exc!s}{exc!r}")

    threads = [threading.Thread(target=worker) for _ in range(2)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=15)
    assert errors
    for text in errors:
        assert "sk-live-concurrent-" not in text
