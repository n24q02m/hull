"""Credential rotation over one provider pool.

Failure policy (identical in ``packages/core-ts/src/storage/key-pool.ts``):

===========  ===========================================================
status       action
===========  ===========================================================
401 / 403    disable the credential and move to the next one
429          cool the credential down (``Retry-After`` /
             ``rate_limit_reset`` honoured, clamped 1s..1h) and move on
5xx          retry the same credential once, then move to the next one
anything     release the lease and re-raise (request problem, not a
else         credential problem)
===========  ===========================================================

The pool is bounded: at most ``credentials * (1 + max_5xx_retries)`` calls
per :meth:`KeyPool.run`, and every wait for a cooling-down pool raises
:class:`~hull_core.credentials.models.PoolExhaustedError` carrying the last
error text. The last error is redacted against every stored key of the
provider, so a provider that echoes the key back cannot leak it through hull.
"""

from __future__ import annotations

import time
from datetime import timezone
from email.utils import parsedate_to_datetime
from typing import Any, Callable, Collection, TypeVar

from hull_core.credentials.models import (
    Credential,
    CredentialHealth,
    PoolExhaustedError,
)
from hull_core.credentials.store import CredentialStore

T = TypeVar("T")

DEFAULT_COOLDOWN_SECONDS = 60.0
MIN_COOLDOWN_SECONDS = 1.0
MAX_COOLDOWN_SECONDS = 3600.0
DISABLE_STATUSES = frozenset({401, 403})
COOLDOWN_STATUSES = frozenset({429})


def _now(now: float | None) -> float:
    return time.time() if now is None else now


def _is_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def parse_retry_after(value: Any, *, now: float | None = None) -> float | None:
    """``Retry-After`` as delay seconds (numeric form or HTTP-date)."""
    if value is None:
        return None
    if _is_number(value):
        return max(0.0, float(value))
    text = str(value).strip()
    if not text:
        return None
    try:
        return max(0.0, float(text))
    except ValueError:
        pass
    try:
        moment = parsedate_to_datetime(text)
    except (TypeError, ValueError):
        return None
    if moment.tzinfo is None:
        moment = moment.replace(tzinfo=timezone.utc)
    return max(0.0, moment.timestamp() - _now(now))


def parse_rate_limit_reset(value: Any, *, now: float | None = None) -> float | None:
    """``rate_limit_reset``/``x-ratelimit-reset`` as delay seconds.

    Providers use two shapes: an epoch second (``1760000000``) and a delta
    already in seconds (``42``). Values above ``1e9`` are treated as epochs.
    """
    if value is None:
        return None
    if _is_number(value):
        number = float(value)
    else:
        text = str(value).strip()
        if not text:
            return None
        try:
            number = float(text)
        except ValueError:
            return parse_retry_after(text, now=now)
    if number <= 0:
        return 0.0
    return max(0.0, number - _now(now)) if number > 1_000_000_000 else number


def cooldown_seconds(
    *,
    retry_after: Any = None,
    rate_limit_reset: Any = None,
    now: float | None = None,
    default: float = DEFAULT_COOLDOWN_SECONDS,
    minimum: float = MIN_COOLDOWN_SECONDS,
    maximum: float = MAX_COOLDOWN_SECONDS,
) -> float:
    """Cooldown for a 429: provider hint first, then the default, always clamped."""
    for candidate in (
        parse_retry_after(retry_after, now=now),
        parse_rate_limit_reset(rate_limit_reset, now=now),
    ):
        if candidate is not None:
            return min(maximum, max(minimum, candidate))
    return min(maximum, max(minimum, default))


def classify_status(status: int | None) -> str:
    """Map an HTTP status to ``disable`` / ``cooldown`` / ``retry`` / ``fatal``."""
    if status is None:
        return "fatal"
    if status in DISABLE_STATUSES:
        return "disable"
    if status in COOLDOWN_STATUSES:
        return "cooldown"
    if 500 <= status < 600:
        return "retry"
    return "fatal"


def _header(headers: Any, name: str) -> Any:
    """Case-insensitive header read for httpx.Headers and plain dicts alike."""
    if headers is None:
        return None
    wanted = name.lower()
    try:
        value = headers.get(name)
        if value is not None:
            return value
        for key, item in headers.items():
            if str(key).lower() == wanted:
                return item
    except Exception:
        return None
    return None


class KeyPool:
    """Ordered, health-aware rotation over one provider's credentials."""

    def __init__(
        self,
        store: CredentialStore,
        provider: str,
        *,
        account_id: str | None = None,
        clock: Callable[[], float] | None = None,
        max_5xx_retries: int = 1,
        lease_seconds: float = 60.0,
    ) -> None:
        if not isinstance(provider, str) or not provider.strip():
            raise ValueError("provider must be a non-empty string")
        if max_5xx_retries < 0:
            raise ValueError("max_5xx_retries must be >= 0")
        if lease_seconds <= 0:
            raise ValueError("lease_seconds must be > 0")
        self.store = store
        self.provider = provider.strip()
        self.account_id = account_id
        self.max_5xx_retries = int(max_5xx_retries)
        self.lease_seconds = float(lease_seconds)
        self._clock = clock if clock is not None else store.now

    # -- read API ---------------------------------------------------------

    def credentials(self) -> list[Credential]:
        """Every entry in this pool (disabled/cooling included), in priority order."""
        return self.store.pool_credentials(self.provider, self.account_id)

    def health(self) -> list[CredentialHealth]:
        """Masked state of the pool — safe to log or return to an operator."""
        return self.store.pool_health(self.provider, self.account_id)

    def __repr__(self) -> str:
        total = self.store.diagnose(self.provider, self.account_id)["total"]
        return f"KeyPool(provider={self.provider!r}, account_id={self.account_id!r}, credentials={total})"

    # -- rotation ---------------------------------------------------------

    def acquire(self, *, exclude: Collection[int] = ()) -> Credential:
        """Lease the next usable credential or raise ``PoolExhaustedError``."""
        credential = self.store.lease_first(
            self.provider,
            self.account_id,
            now=self._clock(),
            lease_seconds=self.lease_seconds,
            exclude=exclude,
        )
        if credential is None:
            raise self.exhausted_error()
        return credential

    def report_success(self, credential: Credential) -> None:
        """Record a 2xx: clears failures, cooldown and the lease."""
        self.store.record_success(credential.id)

    def report_failure(
        self,
        credential: Credential,
        status: int | None,
        *,
        retry_after: Any = None,
        rate_limit_reset: Any = None,
    ) -> str:
        """Apply the failure policy; returns ``disable``/``cooldown``/``retry``/``fatal``."""
        action = classify_status(status)
        now = self._clock()
        if action == "cooldown":
            until = now + cooldown_seconds(retry_after=retry_after, rate_limit_reset=rate_limit_reset, now=now)
            self.store.record_failure(credential.id, status, now=now, cooldown_until=until)
        elif action in ("disable", "retry"):
            self.store.record_failure(credential.id, status, now=now, disable=action == "disable")
        else:
            # Not an HTTP answer from the provider: do not blame the credential.
            self.store.release_lease(credential.id)
        return action

    def enable(self, credential_id: int) -> None:
        """Re-enable a credential disabled by 401/403 after the operator fixes it."""
        self.store.enable(credential_id)

    def redact(self, text: str | None) -> str:
        """Replace every stored key of this provider in ``text`` with ``[redacted]``."""
        if not text:
            return ""
        for credential in self.store.credentials(self.provider):
            if credential.api_key and credential.api_key in text:
                text = text.replace(credential.api_key, "[redacted]")
        return text

    @staticmethod
    def status_of(error: BaseException) -> int | None:
        """HTTP status carried by a provider error (``ProviderError.status``)."""
        status = getattr(error, "status", None)
        return status if isinstance(status, int) and not isinstance(status, bool) else None

    def exhausted_error(self, *, last_error: str | None = None) -> PoolExhaustedError:
        """Build the pool's final error from live counts; never holds a key."""
        detail = self.store.diagnose(self.provider, self.account_id)
        message = f"no usable {self.provider!r} credential"
        if self.account_id is not None:
            message += f" for account {self.account_id!r}"
        message += (
            f" (total={detail['total']}, disabled={detail['disabled']}, "
            f"cooling_down={detail['cooling_down']}, leased={detail['leased']}"
        )
        if detail["retry_after"] is not None:
            message += f", retry in {float(detail['retry_after']):.0f}s"
        message += ")"
        redacted = self.redact(last_error)
        if redacted:
            message += f"; last error: {redacted[:500]}"
        return PoolExhaustedError(
            message,
            provider=self.provider,
            account_id=self.account_id,
            total=int(detail["total"]),
            disabled=int(detail["disabled"]),
            cooling_down=int(detail["cooling_down"]),
            leased=int(detail["leased"]),
            retry_after=detail["retry_after"],
            last_error=redacted or None,
        )

    # -- one-call surface -------------------------------------------------

    def run(
        self, call: Callable[[Credential], T], *, status_of: Callable[[BaseException], int | None] | None = None
    ) -> T:
        """Run ``call(credential)`` with rotation; returns its result or raises."""
        rotation = _Rotation(self, status_of or self.status_of)
        while (credential := rotation.next_credential()) is not None:
            try:
                result = call(credential)
            except Exception as exc:
                if rotation.failed(credential, exc) == "fatal":
                    raise
                continue
            self.report_success(credential)
            return result
        raise rotation.exhausted()

    async def arun(
        self, call: Callable[[Credential], Any], *, status_of: Callable[[BaseException], int | None] | None = None
    ) -> Any:
        """Async sibling of :meth:`run`; ``call`` is awaited."""
        rotation = _Rotation(self, status_of or self.status_of)
        while (credential := rotation.next_credential()) is not None:
            try:
                result = await call(credential)
            except Exception as exc:
                if rotation.failed(credential, exc) == "fatal":
                    raise
                continue
            self.report_success(credential)
            return result
        raise rotation.exhausted()


class _Rotation:
    """Attempt budget + retry bookkeeping for one :meth:`KeyPool.run` call."""

    def __init__(self, pool: KeyPool, status_of: Callable[[BaseException], int | None]) -> None:
        self.pool = pool
        self.status_of = status_of
        total = int(pool.store.diagnose(pool.provider, pool.account_id)["total"])
        self.limit = max(1, total * (1 + pool.max_5xx_retries))
        self.attempts = 0
        self.retried: dict[int, int] = {}
        self.skip: set[int] = set()
        self.last_error: str | None = None

    def next_credential(self) -> Credential | None:
        if self.attempts >= self.limit:
            return None
        try:
            credential = self.pool.acquire(exclude=self.skip)
        except PoolExhaustedError:
            return None
        self.attempts += 1
        return credential

    def failed(self, credential: Credential, error: BaseException) -> str:
        retry_after = getattr(error, "retry_after", None)
        rate_limit_reset = getattr(error, "rate_limit_reset", None)
        headers = getattr(error, "headers", None)
        if headers is not None:
            if retry_after is None:
                retry_after = _header(headers, "retry-after")
            if rate_limit_reset is None:
                rate_limit_reset = _header(headers, "x-ratelimit-reset")
        action = self.pool.report_failure(
            credential,
            self.status_of(error),
            retry_after=retry_after,
            rate_limit_reset=rate_limit_reset,
        )
        self.last_error = self.pool.redact(str(error))
        if action == "retry":
            self.retried[credential.id] = self.retried.get(credential.id, 0) + 1
            if self.retried[credential.id] > self.pool.max_5xx_retries:
                self.skip.add(credential.id)
        elif action in ("disable", "cooldown"):
            self.skip.add(credential.id)
        return action

    def exhausted(self) -> PoolExhaustedError:
        return self.pool.exhausted_error(last_error=self.last_error)
