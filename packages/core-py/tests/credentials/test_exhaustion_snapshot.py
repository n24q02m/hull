"""A refused lease explains itself with the counts read in the same transaction.

``PoolExhaustedError`` carries the pool counts so a caller can tell "every key is
leased by another in-flight call, wait" from "every key is disabled or cooling,
give up". Those counts used to come from a second read after the refused lease
committed: a holder that released its lease in between made the error report
``leased=0`` for a pool that was busy, and a caller waiting on leases gave up.
Without exclusions a refusal means every entry is disabled, cooling down or
leased, so the counts must add up to the total.
"""

from __future__ import annotations

import asyncio
import threading
from collections.abc import Callable

import pytest

from helpers import Clock
from hull_core.credentials.models import Credential, PoolExhaustedError
from hull_core.credentials.pool import KeyPool
from hull_core.credentials.store import CredentialStore

BASE_URL = "https://openrouter.ai/api/v1"
ROUNDS = 2000
WORKERS = 2


def _acquire(pool: KeyPool, _runner: asyncio.Runner) -> None:
    pool.report_success(pool.acquire())


def _run(pool: KeyPool, _runner: asyncio.Runner) -> None:
    pool.run(lambda _credential: None)


def _arun(pool: KeyPool, runner: asyncio.Runner) -> None:
    async def call(_credential: Credential) -> None:
        return None

    runner.run(pool.arun(call))


# Each worker keeps one event loop for all its rounds: a fresh ``asyncio.run``
# per round builds a proactor self-pipe through the localhost socketpair
# fallback on Windows thousands of times, and one of those accepts can hang.
ENTRY_POINTS: dict[str, Callable[[KeyPool, asyncio.Runner], None]] = {
    "acquire": _acquire,
    "run": _run,
    "arun": _arun,
}


def _unexplained(error: PoolExhaustedError) -> bool:
    return error.disabled + error.cooling_down + error.leased != error.total


@pytest.mark.parametrize("entry", sorted(ENTRY_POINTS))
def test_a_refusal_under_contention_counts_the_lease_that_caused_it(store: CredentialStore, entry: str) -> None:
    store.add_credential("openrouter", "sk-live-snapshot-0001", base_url=BASE_URL)
    pool = KeyPool(store, "openrouter")
    use = ENTRY_POINTS[entry]
    refusals: list[PoolExhaustedError] = []
    errors: list[str] = []
    guard = threading.Lock()
    start = threading.Barrier(WORKERS)

    def worker() -> None:
        start.wait(timeout=10)
        with asyncio.Runner() as runner:
            for _ in range(ROUNDS):
                try:
                    use(pool, runner)
                except PoolExhaustedError as exc:
                    with guard:
                        refusals.append(exc)
                except BaseException as exc:  # surface every thread failure in the test result
                    with guard:
                        errors.append(repr(exc))
                    return

    threads = [threading.Thread(target=worker) for _ in range(WORKERS)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=120)

    assert errors == []
    assert [str(exc) for exc in refusals if _unexplained(exc)] == []


def test_lease_first_or_diagnose_leases_a_usable_entry(store: CredentialStore, clock: Clock) -> None:
    added = store.add_credential("openrouter", "sk-live-snapshot-0002", base_url=BASE_URL)

    credential, detail = store.lease_first_or_diagnose("openrouter", None, now=clock.value, lease_seconds=60)

    assert credential is not None and credential.id == added.id
    assert detail is None
    assert store.diagnose("openrouter")["leased"] == 1


def test_lease_first_or_diagnose_returns_the_counts_behind_a_refusal(store: CredentialStore, clock: Clock) -> None:
    leased = store.add_credential("openrouter", "sk-live-snapshot-0003", base_url=BASE_URL)
    cooling = store.add_credential("openrouter", "sk-live-snapshot-0004", base_url=BASE_URL)
    disabled = store.add_credential("openrouter", "sk-live-snapshot-0005", base_url=BASE_URL)
    held = store.lease_first("openrouter", None, now=clock.value, lease_seconds=60)
    assert held is not None and held.id == leased.id
    store.record_failure(cooling.id, 429, now=clock.value, cooldown_until=clock.value + 30)
    store.disable(disabled.id)

    credential, detail = store.lease_first_or_diagnose("openrouter", None, now=clock.value, lease_seconds=60)

    assert credential is None
    assert detail == {"total": 3, "disabled": 1, "cooling_down": 1, "leased": 1, "retry_after": 30.0}


def test_the_public_exhausted_error_reads_live_counts(store: CredentialStore) -> None:
    """Outside a refused lease there is no snapshot, so the public builder reads now."""
    disabled = store.add_credential("openrouter", "sk-live-snapshot-0007", base_url=BASE_URL)
    store.disable(disabled.id)

    error = KeyPool(store, "openrouter").exhausted_error(last_error="upstream said no")

    assert (error.total, error.disabled, error.cooling_down, error.leased) == (1, 1, 0, 0)
    assert error.last_error == "upstream said no"


def test_an_exhausted_run_reports_the_refusal_not_a_later_read(store: CredentialStore, clock: Clock) -> None:
    """The final error of ``run`` reuses the refused lease's counts.

    Releasing the lease after the refusal (what a concurrent holder does) must
    not turn a busy pool into an unexplained exhaustion.
    """
    store.add_credential("openrouter", "sk-live-snapshot-0006", base_url=BASE_URL)
    pool = KeyPool(store, "openrouter")
    held = pool.acquire()
    real = store.lease_first_or_diagnose

    def refuse_then_release(*args, **kwargs):  # noqa: ANN002, ANN003, ANN202
        result = real(*args, **kwargs)
        if result[0] is None:
            pool.release(held)
        return result

    store.lease_first_or_diagnose = refuse_then_release  # type: ignore[method-assign]
    with pytest.raises(PoolExhaustedError) as caught:
        pool.run(lambda _credential: None)

    assert (caught.value.total, caught.value.leased) == (1, 1)
    assert store.diagnose("openrouter")["leased"] == 0
