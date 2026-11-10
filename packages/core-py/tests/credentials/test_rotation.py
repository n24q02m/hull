"""Rotation policy: 401/403, 429, 5xx, cooldown expiry, exhaustion, redaction."""

from __future__ import annotations

from datetime import datetime, timezone
from email.utils import format_datetime

import pytest

from helpers import Clock, ProviderFailure
from hull_core.credentials.models import PoolExhaustedError
from hull_core.credentials.pool import (
    KeyPool,
    classify_status,
    cooldown_seconds,
    parse_rate_limit_reset,
    parse_retry_after,
)
from hull_core.credentials.store import CredentialStore

BASE_URL = "https://openrouter.ai/api/v1"


def seed(store: CredentialStore, count: int = 2) -> list:
    return [
        store.add_credential("openrouter", f"sk-live-rotation-{index:04d}", base_url=BASE_URL, label=f"c{index}")
        for index in range(count)
    ]


@pytest.mark.parametrize("status", [401, 403])
def test_auth_failures_disable_and_rotate(store: CredentialStore, status: int) -> None:
    first, second = seed(store)
    seen: list[int] = []

    def call(credential):
        seen.append(credential.id)
        if credential.id == first.id:
            raise ProviderFailure(status, "invalid api key")
        return "ok"

    assert store.pool("openrouter").run(call) == "ok"
    assert seen == [first.id, second.id]
    health = {item.id: item for item in store.health("openrouter")}
    assert health[first.id].disabled is True
    assert health[first.id].last_status == status
    assert health[second.id].disabled is False and health[second.id].last_status == 200


def test_429_honours_retry_after_seconds_and_rotates(store: CredentialStore, clock: Clock) -> None:
    first, second = seed(store)

    def call(credential):
        if credential.id == first.id:
            raise ProviderFailure(429, "rate limited", retry_after="120")
        return "ok"

    assert store.pool("openrouter").run(call) == "ok"
    health = {item.id: item for item in store.health("openrouter")}
    assert health[first.id].last_status == 429
    assert health[first.id].cooldown_until == pytest.approx(clock.value + 120)
    assert health[first.id].usable is False
    assert health[second.id].last_status == 200


def test_429_honours_retry_after_http_date(store: CredentialStore, clock: Clock) -> None:
    first, second = seed(store)
    header = format_datetime(datetime.fromtimestamp(clock.value + 90, tz=timezone.utc), usegmt=True)

    def call(credential):
        if credential.id == first.id:
            raise ProviderFailure(429, "rate limited", headers={"Retry-After": header})
        return "ok"

    assert store.pool("openrouter").run(call) == "ok"
    assert store.health("openrouter")[0].cooldown_until == pytest.approx(clock.value + 90, abs=1)


def test_429_honours_rate_limit_reset_epoch(store: CredentialStore, clock: Clock) -> None:
    first, second = seed(store)

    def call(credential):
        if credential.id == first.id:
            raise ProviderFailure(429, "rate limited", rate_limit_reset=clock.value + 75)
        return "ok"

    assert store.pool("openrouter").run(call) == "ok"
    assert store.health("openrouter")[0].cooldown_until == pytest.approx(clock.value + 75)


def test_cooldown_expires_and_the_credential_returns(store: CredentialStore, clock: Clock) -> None:
    first, second = seed(store)

    def rate_limited(credential):
        if credential.id == first.id:
            raise ProviderFailure(429, "rate limited", retry_after=30)
        return "ok"

    pool = store.pool("openrouter")
    assert pool.run(rate_limited) == "ok"
    with pytest.raises(PoolExhaustedError):
        # second is leased by nobody but first is cooling; a fresh call that
        # only tries the cooling credential must fail rather than spin.
        pool.acquire(exclude=[second.id])
    clock.advance(31)
    assert pool.acquire().id == first.id


def test_5xx_retries_the_same_credential_once_then_moves_on(store: CredentialStore) -> None:
    first, second = seed(store)
    seen: list[int] = []

    def call(credential):
        seen.append(credential.id)
        if credential.id == first.id:
            raise ProviderFailure(500, "upstream exploded")
        return "ok"

    assert store.pool("openrouter").run(call) == "ok"
    assert seen == [first.id, first.id, second.id]
    health = {item.id: item for item in store.health("openrouter")}
    assert health[first.id].failure_count == 2
    assert health[first.id].last_status == 500
    assert health[first.id].disabled is False


def test_attempt_budget_is_bounded_not_infinite(store: CredentialStore) -> None:
    seed(store, 3)
    seen: list[int] = []

    def always_500(credential):
        seen.append(credential.id)
        raise ProviderFailure(503, "unavailable")

    with pytest.raises(PoolExhaustedError) as excinfo:
        store.pool("openrouter", max_5xx_retries=2).run(always_500)
    # 3 credentials * (1 try + 2 retries), then the pool gives up.
    assert len(seen) == 9
    assert "no usable 'openrouter' credential" in str(excinfo.value)


def test_exhausted_error_carries_last_error_with_every_key_redacted(store: CredentialStore) -> None:
    first, second = seed(store)

    def call(credential):
        raise ProviderFailure(401, f"invalid api key {credential.api_key} (account leak)")

    with pytest.raises(PoolExhaustedError) as excinfo:
        store.pool("openrouter").run(call)
    error = excinfo.value
    rendered = f"{error!s}\n{error!r}\n{error.last_error}"
    assert first.api_key not in rendered
    assert second.api_key not in rendered
    assert "invalid api key [redacted]" in str(error)
    assert (error.total, error.disabled, error.cooling_down) == (2, 2, 0)


def test_empty_pool_reports_total_zero(store: CredentialStore) -> None:
    with pytest.raises(PoolExhaustedError, match="total=0") as excinfo:
        store.pool("openrouter").run(lambda credential: "never")
    assert excinfo.value.last_error is None


def test_non_credential_failure_is_re_raised_and_releases_the_lease(store: CredentialStore) -> None:
    first, _ = seed(store)

    def bad_request(credential):
        raise ProviderFailure(400, "malformed request")

    with pytest.raises(ProviderFailure):
        store.pool("openrouter").run(bad_request)
    # No credential was blamed: first is immediately acquirable and healthy.
    assert store.pool("openrouter").acquire().id == first.id
    assert store.health("openrouter")[0].failure_count == 0


def test_non_http_exception_is_re_raised_and_releases_the_lease(store: CredentialStore) -> None:
    first, _ = seed(store)
    with pytest.raises(ValueError, match="bad payload"):
        store.pool("openrouter").run(lambda credential: (_ for _ in ()).throw(ValueError("bad payload")))
    assert store.pool("openrouter").acquire().id == first.id


async def test_arun_rotates_like_run(store: CredentialStore) -> None:
    first, second = seed(store)

    async def call(credential):
        if credential.id == first.id:
            raise ProviderFailure(401, "invalid api key")
        return "async-ok"

    assert await store.pool("openrouter").arun(call) == "async-ok"
    assert store.health("openrouter")[0].disabled is True


def test_acquire_leases_each_credential_once(store: CredentialStore) -> None:
    first, second = seed(store)
    pool = store.pool("openrouter", lease_seconds=30)
    acquired = {pool.acquire().id, pool.acquire().id}
    assert acquired == {first.id, second.id}
    with pytest.raises(PoolExhaustedError, match="leased=2"):
        pool.acquire()
    pool.report_success(store.get_credential(first.id))
    assert pool.acquire().id == first.id


def test_pool_repr_and_health_never_hold_keys(store: CredentialStore) -> None:
    first, _ = seed(store)
    pool = store.pool("openrouter")
    assert first.api_key not in repr(pool)
    assert first.api_key not in repr(pool.health())
    assert first.api_key not in repr(pool.credentials())


@pytest.mark.parametrize(
    ("status", "action"),
    [
        (None, "fatal"),
        (400, "fatal"),
        (401, "disable"),
        (403, "disable"),
        (429, "cooldown"),
        (500, "retry"),
        (503, "retry"),
        (599, "retry"),
        (600, "fatal"),
    ],
)
def test_classify_status(status: int | None, action: str) -> None:
    assert classify_status(status) == action


def test_parse_retry_after_forms(clock: Clock) -> None:
    assert parse_retry_after(None) is None
    assert parse_retry_after("") is None
    assert parse_retry_after("12") == 12
    assert parse_retry_after(12) == 12
    assert parse_retry_after("-5") == 0
    assert parse_retry_after("not-a-date") is None
    header = format_datetime(datetime.fromtimestamp(clock.value + 45, tz=timezone.utc), usegmt=True)
    assert parse_retry_after(header, now=clock.value) == pytest.approx(45, abs=1)
    past = format_datetime(datetime.fromtimestamp(clock.value - 45, tz=timezone.utc), usegmt=True)
    assert parse_retry_after(past, now=clock.value) == 0


def test_parse_rate_limit_reset_forms(clock: Clock) -> None:
    assert parse_rate_limit_reset(None) is None
    assert parse_rate_limit_reset(30) == 30
    assert parse_rate_limit_reset("30") == 30
    assert parse_rate_limit_reset(clock.value + 60, now=clock.value) == pytest.approx(60)
    assert parse_rate_limit_reset(0) == 0
    assert parse_rate_limit_reset("garbage") is None


def test_cooldown_precedence_and_clamps(clock: Clock) -> None:
    assert cooldown_seconds(retry_after=20, rate_limit_reset=90, now=clock.value) == 20
    assert cooldown_seconds(rate_limit_reset=90, now=clock.value) == 90
    assert cooldown_seconds(now=clock.value) == 60
    assert cooldown_seconds(retry_after=0, now=clock.value) == 1
    assert cooldown_seconds(retry_after=99_999, now=clock.value) == 3600


def test_key_pool_validates_configuration(store: CredentialStore) -> None:
    with pytest.raises(ValueError, match="provider"):
        KeyPool(store, "  ")
    with pytest.raises(ValueError, match="max_5xx_retries"):
        KeyPool(store, "openrouter", max_5xx_retries=-1)
    with pytest.raises(ValueError, match="lease_seconds"):
        KeyPool(store, "openrouter", lease_seconds=0)
