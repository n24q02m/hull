"""SQLite-backed provider key pool and account registry (``~/.hull/hull.db``).

The tables are additive and migration-safe: ``CREATE TABLE/INDEX IF NOT
EXISTS`` only, no ``ALTER``, so a database created by an older hull (only the
``kv`` table) keeps working and an older hull keeps ignoring the new tables.

Scope rules
-----------
* A pool is one provider identity (``openrouter``, ``openai``, a host slug).
  ``account_id IS NULL`` is the provider's default pool; a named account has
  its own pool and rotation never hops across accounts.
* Entries are ordered by ``priority`` then insertion order.
* Rotation state is shared through the database, not process memory, so two
  tasks (or two processes) never reuse a credential that is cooling down or
  already leased by the other one.

Secrets are sealed by :mod:`hull_core.credentials.crypto`; this module only
ever stores the envelope, a display hint and a fingerprint.
"""

from __future__ import annotations

import json
import sqlite3
import threading
import time
from contextlib import contextmanager
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, Generator, Iterable, Mapping

from hull_core.credentials.crypto import SecretCipher, key_fingerprint, key_hint
from hull_core.credentials.models import (
    Account,
    AccountError,
    Credential,
    CredentialExistsError,
    CredentialHealth,
    CredentialNotFoundError,
    MigrationResult,
    provider_from_base_url,
)

if TYPE_CHECKING:
    from hull_core.config.models import ModelCell
    from hull_core.credentials.pool import KeyPool
    from hull_core.credentials.registry import AccountRegistry

POOL_SCHEMA = """
CREATE TABLE IF NOT EXISTS provider_accounts (
    account_id TEXT PRIMARY KEY,
    provider   TEXT NOT NULL,
    name       TEXT NOT NULL,
    email      TEXT,
    workspace  TEXT,
    labels     TEXT NOT NULL DEFAULT '{}',
    created_at TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ','now')),
    updated_at TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ','now'))
);

CREATE UNIQUE INDEX IF NOT EXISTS provider_accounts_name
    ON provider_accounts(provider, name);

CREATE TABLE IF NOT EXISTS provider_credentials (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    provider        TEXT NOT NULL,
    account_id      TEXT,
    label           TEXT,
    base_url        TEXT NOT NULL,
    model           TEXT,
    secret          TEXT NOT NULL,
    key_hint        TEXT NOT NULL,
    key_fingerprint TEXT NOT NULL,
    priority        INTEGER NOT NULL DEFAULT 0,
    disabled        INTEGER NOT NULL DEFAULT 0,
    last_status     INTEGER,
    failure_count   INTEGER NOT NULL DEFAULT 0,
    cooldown_until  REAL NOT NULL DEFAULT 0,
    lease_until     REAL NOT NULL DEFAULT 0,
    created_at      TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ','now')),
    updated_at      TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ','now'))
);

CREATE UNIQUE INDEX IF NOT EXISTS provider_credentials_identity
    ON provider_credentials(provider, ifnull(account_id, ''), key_fingerprint, base_url);

CREATE INDEX IF NOT EXISTS provider_credentials_pool
    ON provider_credentials(provider, account_id, priority, id);
"""

_COLUMNS = (
    "id, provider, account_id, label, base_url, model, secret, key_hint, "
    "priority, disabled, last_status, failure_count, cooldown_until, lease_until"
)
_TOUCH = "updated_at = strftime('%Y-%m-%dT%H:%M:%fZ','now')"


def default_pool_db_path() -> Path:
    """The instance database every hull surface already uses: ``~/.hull/hull.db``."""
    from hull_core.config.settings import default_config_dir

    return default_config_dir() / "hull.db"


def _text(value: str, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field} must be a non-empty string")
    return value.strip()


def _optional_text(value: str | None) -> str | None:
    if value is None:
        return None
    value = value.strip()
    return value or None


class CredentialStore:
    """Ordered provider credential pool plus named accounts, on local SQLite."""

    def __init__(
        self,
        path: Path | str | None = None,
        *,
        cipher: SecretCipher | None = None,
        clock: Callable[[], float] = time.time,
    ) -> None:
        self.path = Path(path) if path is not None else default_pool_db_path()
        self.path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
        self._clock = clock
        self._cipher = cipher if cipher is not None else SecretCipher(self.path.parent / ".credential-key")
        self._lock = threading.RLock()
        self._conn = sqlite3.connect(str(self.path), check_same_thread=False, isolation_level=None)
        self._conn.row_factory = sqlite3.Row
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute("PRAGMA busy_timeout=5000")
        with self._lock:
            self._conn.executescript(POOL_SCHEMA)

    # -- plumbing ---------------------------------------------------------

    def now(self) -> float:
        return float(self._clock())

    @contextmanager
    def _immediate(self) -> Generator[None, None, None]:
        """One write transaction, serialized against other connections."""
        self._conn.execute("BEGIN IMMEDIATE")
        try:
            yield
        except BaseException:
            self._conn.execute("ROLLBACK")
            raise
        else:
            self._conn.execute("COMMIT")

    def _credential(self, row: sqlite3.Row) -> Credential:
        return Credential(
            id=int(row["id"]),
            provider=str(row["provider"]),
            account_id=row["account_id"],
            label=row["label"],
            base_url=str(row["base_url"]),
            model=row["model"],
            api_key=self._cipher.decrypt(str(row["secret"])),
            priority=int(row["priority"]),
            key_hint=str(row["key_hint"]),
        )

    def _health(self, row: sqlite3.Row, now: float) -> CredentialHealth:
        cooldown_until = float(row["cooldown_until"])
        lease_until = float(row["lease_until"])
        remaining = max(0.0, cooldown_until - now)
        return CredentialHealth(
            id=int(row["id"]),
            provider=str(row["provider"]),
            account_id=row["account_id"],
            label=row["label"],
            base_url=str(row["base_url"]),
            model=row["model"],
            key_hint=str(row["key_hint"]),
            priority=int(row["priority"]),
            disabled=bool(row["disabled"]),
            last_status=None if row["last_status"] is None else int(row["last_status"]),
            failure_count=int(row["failure_count"]),
            cooldown_until=cooldown_until,
            lease_until=lease_until,
            cooldown_remaining=remaining,
            usable=not bool(row["disabled"]) and cooldown_until <= now and lease_until <= now,
        )

    def _row(self, credential_id: int) -> sqlite3.Row:
        row = self._conn.execute(
            f"SELECT {_COLUMNS} FROM provider_credentials WHERE id = ?", (credential_id,)
        ).fetchone()
        if row is None:
            raise CredentialNotFoundError(f"no provider credential with id {credential_id}")
        return row

    def _require_account(self, account_id: str, provider: str) -> sqlite3.Row:
        row = self._conn.execute(
            "SELECT account_id, provider FROM provider_accounts WHERE account_id = ?", (account_id,)
        ).fetchone()
        if row is None:
            raise AccountError(f"unknown account {account_id!r}; add it with the account registry first")
        if str(row["provider"]) != provider:
            raise AccountError(f"account {account_id!r} belongs to provider {row['provider']!r}, not {provider!r}")
        return row

    # -- writes -----------------------------------------------------------

    def add_credential(
        self,
        provider: str,
        api_key: str,
        *,
        base_url: str,
        model: str | None = None,
        account_id: str | None = None,
        label: str | None = None,
        priority: int | None = None,
    ) -> Credential:
        """Add one ordered pool entry; duplicate (provider, account, key, base_url) is refused."""
        provider = _text(provider, "provider")
        api_key = _text(api_key, "api_key")
        base_url = _text(base_url, "base_url").rstrip("/")
        account_id = _optional_text(account_id)
        label = _optional_text(label)
        model = _optional_text(model)
        secret = self._cipher.encrypt(api_key)
        fingerprint = key_fingerprint(api_key)
        hint = key_hint(api_key)
        with self._lock, self._immediate():
            if account_id is not None:
                self._require_account(account_id, provider)
            if priority is None:
                row = self._conn.execute(
                    "SELECT COALESCE(MAX(priority), -1) + 1 AS next FROM provider_credentials "
                    "WHERE provider = ? AND account_id IS ?",
                    (provider, account_id),
                ).fetchone()
                priority = int(row["next"])
            try:
                cur = self._conn.execute(
                    "INSERT INTO provider_credentials "
                    "(provider, account_id, label, base_url, model, secret, key_hint, key_fingerprint, priority) "
                    "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                    (provider, account_id, label, base_url, model, secret, hint, fingerprint, int(priority)),
                )
            except sqlite3.IntegrityError:
                raise CredentialExistsError(
                    f"provider {provider!r} already has this credential"
                    + (f" for account {account_id!r}" if account_id else "")
                ) from None
            row = self._row(int(cur.lastrowid or 0))
        return self._credential(row)

    def remove_credential(self, credential_id: int) -> None:
        with self._lock, self._immediate():
            cur = self._conn.execute("DELETE FROM provider_credentials WHERE id = ?", (credential_id,))
            if cur.rowcount == 0:
                raise CredentialNotFoundError(f"no provider credential with id {credential_id}")

    def set_priority(self, credential_id: int, priority: int) -> None:
        with self._lock, self._immediate():
            cur = self._conn.execute(
                f"UPDATE provider_credentials SET priority = ?, {_TOUCH} WHERE id = ?",
                (int(priority), credential_id),
            )
            if cur.rowcount == 0:
                raise CredentialNotFoundError(f"no provider credential with id {credential_id}")

    def enable(self, credential_id: int) -> None:
        """Clear disabled/cooldown/failure state after an operator fixes the key."""
        with self._lock, self._immediate():
            cur = self._conn.execute(
                "UPDATE provider_credentials SET disabled = 0, last_status = NULL, failure_count = 0, "
                f"cooldown_until = 0, lease_until = 0, {_TOUCH} WHERE id = ?",
                (credential_id,),
            )
            if cur.rowcount == 0:
                raise CredentialNotFoundError(f"no provider credential with id {credential_id}")

    def disable(self, credential_id: int) -> None:
        with self._lock, self._immediate():
            cur = self._conn.execute(
                f"UPDATE provider_credentials SET disabled = 1, lease_until = 0, {_TOUCH} WHERE id = ?",
                (credential_id,),
            )
            if cur.rowcount == 0:
                raise CredentialNotFoundError(f"no provider credential with id {credential_id}")

    def reset_health(self, provider: str | None = None) -> int:
        """Clear rotation state for one provider (or every provider); returns rows touched."""
        sql = (
            "UPDATE provider_credentials SET disabled = 0, last_status = NULL, failure_count = 0, "
            f"cooldown_until = 0, lease_until = 0, {_TOUCH}"
        )
        params: tuple[Any, ...] = ()
        if provider is not None:
            sql += " WHERE provider = ?"
            params = (provider,)
        with self._lock, self._immediate():
            return int(self._conn.execute(sql, params).rowcount)

    # -- reads ------------------------------------------------------------

    def credentials(self, provider: str | None = None) -> list[Credential]:
        """Every decrypted entry (all accounts), ordered by provider/priority/id."""
        sql = f"SELECT {_COLUMNS} FROM provider_credentials"
        params: tuple[Any, ...] = ()
        if provider is not None:
            sql += " WHERE provider = ?"
            params = (provider,)
        sql += " ORDER BY provider, priority, id"
        rows = self._conn.execute(sql, params).fetchall()
        return [self._credential(row) for row in rows]

    def pool_credentials(self, provider: str, account_id: str | None = None) -> list[Credential]:
        """The rotation pool for one provider/account scope, in priority order."""
        rows = self._conn.execute(
            f"SELECT {_COLUMNS} FROM provider_credentials WHERE provider = ? AND account_id IS ? ORDER BY priority, id",
            (_text(provider, "provider"), _optional_text(account_id)),
        ).fetchall()
        return [self._credential(row) for row in rows]

    def health(self, provider: str | None = None) -> list[CredentialHealth]:
        """Masked state of every entry, including disabled and cooling ones."""
        sql = f"SELECT {_COLUMNS} FROM provider_credentials"
        params: tuple[Any, ...] = ()
        if provider is not None:
            sql += " WHERE provider = ?"
            params = (provider,)
        sql += " ORDER BY provider, priority, id"
        now = self.now()
        return [self._health(row, now) for row in self._conn.execute(sql, params).fetchall()]

    def pool_health(self, provider: str, account_id: str | None = None) -> list[CredentialHealth]:
        now = self.now()
        rows = self._conn.execute(
            f"SELECT {_COLUMNS} FROM provider_credentials WHERE provider = ? AND account_id IS ? ORDER BY priority, id",
            (_text(provider, "provider"), _optional_text(account_id)),
        ).fetchall()
        return [self._health(row, now) for row in rows]

    def get_credential(self, credential_id: int) -> Credential:
        return self._credential(self._row(credential_id))

    def diagnose(self, provider: str, account_id: str | None = None) -> dict[str, Any]:
        """Pool counts + earliest cooldown, for an accurate exhaustion message."""
        now = self.now()
        rows = self._conn.execute(
            "SELECT disabled, cooldown_until, lease_until FROM provider_credentials "
            "WHERE provider = ? AND account_id IS ?",
            (_text(provider, "provider"), _optional_text(account_id)),
        ).fetchall()
        disabled = sum(1 for row in rows if int(row["disabled"]))
        cooling = [
            float(row["cooldown_until"])
            for row in rows
            if not int(row["disabled"]) and float(row["cooldown_until"]) > now
        ]
        leased = sum(
            1
            for row in rows
            if not int(row["disabled"]) and float(row["cooldown_until"]) <= now and float(row["lease_until"]) > now
        )
        return {
            "total": len(rows),
            "disabled": disabled,
            "cooling_down": len(cooling),
            "leased": leased,
            "retry_after": (min(cooling) - now) if cooling else None,
        }

    # -- rotation state (called by KeyPool) -------------------------------

    def lease_first(
        self,
        provider: str,
        account_id: str | None,
        *,
        now: float,
        lease_seconds: float,
        exclude: Iterable[int] = (),
    ) -> Credential | None:
        """Atomically lease the highest-priority usable entry, or return ``None``."""
        excluded = sorted({int(value) for value in exclude})
        sql = (
            f"SELECT {_COLUMNS} FROM provider_credentials "
            "WHERE provider = ? AND account_id IS ? AND disabled = 0 "
            "AND cooldown_until <= ? AND lease_until <= ?"
        )
        params: list[Any] = [_text(provider, "provider"), _optional_text(account_id), now, now]
        if excluded:
            sql += f" AND id NOT IN ({','.join('?' for _ in excluded)})"
            params.extend(excluded)
        sql += " ORDER BY priority, id LIMIT 1"
        with self._lock, self._immediate():
            row = self._conn.execute(sql, params).fetchone()
            if row is None:
                return None
            self._conn.execute(
                f"UPDATE provider_credentials SET lease_until = ?, {_TOUCH} WHERE id = ?",
                (now + lease_seconds, int(row["id"])),
            )
        return self._credential(row)

    def release_lease(self, credential_id: int) -> None:
        """Drop a lease without touching health (a non-HTTP failure)."""
        with self._lock, self._immediate():
            cur = self._conn.execute(
                f"UPDATE provider_credentials SET lease_until = 0, {_TOUCH} WHERE id = ?",
                (credential_id,),
            )
            if cur.rowcount == 0:
                raise CredentialNotFoundError(f"no provider credential with id {credential_id}")

    def record_success(self, credential_id: int) -> None:
        with self._lock, self._immediate():
            cur = self._conn.execute(
                "UPDATE provider_credentials SET last_status = 200, failure_count = 0, cooldown_until = 0, "
                f"lease_until = 0, {_TOUCH} WHERE id = ?",
                (credential_id,),
            )
            if cur.rowcount == 0:
                raise CredentialNotFoundError(f"no provider credential with id {credential_id}")

    def record_failure(
        self,
        credential_id: int,
        status: int | None,
        *,
        now: float | None = None,
        cooldown_until: float | None = None,
        disable: bool = False,
    ) -> None:
        now = self.now() if now is None else now
        with self._lock, self._immediate():
            cur = self._conn.execute(
                "UPDATE provider_credentials SET last_status = ?, failure_count = failure_count + 1, "
                "cooldown_until = COALESCE(?, cooldown_until), "
                "disabled = CASE WHEN ? THEN 1 ELSE disabled END, "
                f"lease_until = 0, {_TOUCH} WHERE id = ?",
                (status, cooldown_until, 1 if disable else 0, credential_id),
            )
            if cur.rowcount == 0:
                raise CredentialNotFoundError(f"no provider credential with id {credential_id}")

    # -- accounts ---------------------------------------------------------

    def accounts(self) -> AccountRegistry:
        from hull_core.credentials.registry import AccountRegistry

        return AccountRegistry(self)

    def account_rows(self) -> list[sqlite3.Row]:
        return self._conn.execute(
            "SELECT account_id, provider, name, email, workspace, labels FROM provider_accounts ORDER BY provider, name"
        ).fetchall()

    def account_row(self, account_id: str) -> sqlite3.Row | None:
        return self._conn.execute(
            "SELECT account_id, provider, name, email, workspace, labels FROM provider_accounts WHERE account_id = ?",
            (account_id,),
        ).fetchone()

    def account_row_by_name(self, name: str) -> list[sqlite3.Row]:
        rows = self._conn.execute(
            "SELECT account_id, provider, name, email, workspace, labels FROM provider_accounts WHERE name = ?",
            (name,),
        ).fetchall()
        return list(rows)

    def insert_account(self, account: Account) -> None:
        payload = json.dumps(account.labels, sort_keys=True)
        with self._lock, self._immediate():
            try:
                self._conn.execute(
                    "INSERT INTO provider_accounts (account_id, provider, name, email, workspace, labels) "
                    "VALUES (?, ?, ?, ?, ?, ?)",
                    (
                        account.account_id,
                        account.provider,
                        account.name,
                        account.email,
                        account.workspace,
                        payload,
                    ),
                )
            except sqlite3.IntegrityError:
                raise AccountError(
                    f"account {account.account_id!r} or the name {account.name!r} for provider "
                    f"{account.provider!r} already exists"
                ) from None

    def update_account_row(
        self,
        account_id: str,
        *,
        name: str,
        email: str | None,
        workspace: str | None,
        labels_json: str,
    ) -> None:
        with self._lock, self._immediate():
            cur = self._conn.execute(
                f"UPDATE provider_accounts SET name = ?, email = ?, workspace = ?, labels = ?, {_TOUCH} "
                "WHERE account_id = ?",
                (name, email, workspace, labels_json, account_id),
            )
            if cur.rowcount == 0:
                raise AccountError(f"unknown account {account_id!r}")

    def delete_account_row(self, account_id: str, *, drop_credentials: bool) -> None:
        with self._lock, self._immediate():
            count = int(
                self._conn.execute(
                    "SELECT COUNT(*) AS n FROM provider_credentials WHERE account_id = ?", (account_id,)
                ).fetchone()["n"]
            )
            if count and not drop_credentials:
                raise AccountError(
                    f"account {account_id!r} still has {count} credential(s); pass drop_credentials=True "
                    "to delete them with the account"
                )
            self._conn.execute("DELETE FROM provider_credentials WHERE account_id = ?", (account_id,))
            cur = self._conn.execute("DELETE FROM provider_accounts WHERE account_id = ?", (account_id,))
            if cur.rowcount == 0:
                raise AccountError(f"unknown account {account_id!r}")

    # -- pools ------------------------------------------------------------

    def pool(
        self,
        provider: str,
        *,
        account_id: str | None = None,
        clock: Callable[[], float] | None = None,
        max_5xx_retries: int = 1,
        lease_seconds: float = 60.0,
    ) -> KeyPool:
        from hull_core.credentials.pool import KeyPool

        return KeyPool(
            self,
            provider,
            account_id=account_id,
            clock=clock,
            max_5xx_retries=max_5xx_retries,
            lease_seconds=lease_seconds,
        )

    def pool_for_cell(self, cell: ModelCell, **policy: Any) -> KeyPool:
        """Resolve ``[models.<task>]``'s ``provider``/``account`` keys to its pool."""
        provider = str(getattr(cell, "provider", "") or "") or provider_from_base_url(str(cell.base_url))
        account = str(getattr(cell, "account", "") or "") or None
        return self.pool(provider, account_id=account, **policy)

    def migrate_model_cells(
        self,
        cells: Mapping[str, ModelCell] | Iterable[ModelCell],
        *,
        provider: str | None = None,
        account_id: str | None = None,
    ) -> MigrationResult:
        """Seed the pool from the pre-pool single-key config, idempotently.

        Every distinct ``(base_url, api_key)`` becomes one entry in insertion
        order; re-running skips what is already stored, so the call is safe on
        every boot. A cell's ``account`` (or an explicit ``account_id``) scopes
        the entry to that account and must already exist in the registry.
        """
        items = list(cells.values()) if isinstance(cells, Mapping) else list(cells)
        if not items:
            return MigrationResult(provider=provider or "default", added=0, skipped=0)
        if provider is None:
            # Keyless cells do not decide the provider: a config with one key
            # plus three empty task tables is still one provider's migration.
            keyed = [cell for cell in items if str(getattr(cell, "api_key", "") or "")]
            names = {
                str(getattr(cell, "provider", "") or "") or provider_from_base_url(str(cell.base_url))
                for cell in (keyed or items)
            }
            if len(names) != 1:
                raise ValueError(f"model cells span providers {sorted(names)}; pass provider=... explicitly")
            provider = names.pop()
        added = 0
        skipped = 0
        for cell in items:
            api_key = str(getattr(cell, "api_key", "") or "")
            if not api_key:
                continue
            target = account_id if account_id is not None else (str(getattr(cell, "account", "") or "") or None)
            try:
                self.add_credential(
                    provider,
                    api_key,
                    base_url=str(cell.base_url),
                    model=str(cell.model) if getattr(cell, "model", None) else None,
                    account_id=target,
                )
            except CredentialExistsError:
                skipped += 1
            else:
                added += 1
        return MigrationResult(provider=provider, added=added, skipped=skipped)

    def migrate_model_config(
        self,
        models_config: Mapping[str, Any] | None,
        *,
        provider: str | None = None,
        account_id: str | None = None,
        env: Mapping[str, str] | None = None,
    ) -> MigrationResult:
        """Convenience over :meth:`migrate_model_cells` for a raw ``[models]`` table."""
        from hull_core.config.models import resolve_model_cells

        return self.migrate_model_cells(
            resolve_model_cells(models_config, env=env), provider=provider, account_id=account_id
        )

    def close(self) -> None:
        with self._lock:
            self._conn.close()
