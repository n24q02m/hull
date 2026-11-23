"""Tier-3 per-plugin named account registry backed by :class:`PerPluginStore`.

Allows a single user (in stdio, token, or multi-user mode, and within any
Tier-2 profile) to register and switch between multiple named accounts
(e.g. ``work`` and ``personal`` Notion workspaces, Telegram bots, or email
accounts) under leaf ``accounts``.
"""

from __future__ import annotations

from typing import Any

from hull_core.storage.backends import CredentialBackend
from hull_core.storage.data_dir import validate_profile
from hull_core.storage.per_plugin_store import PerPluginStore


def validate_account_alias(alias: str) -> str:
    """Validate an account alias (same safe character rules as profile names)."""
    try:
        return validate_profile(alias)
    except ValueError as exc:
        raise ValueError("Invalid account alias") from exc


class NamedAccountRegistry:
    """Encrypted multi-account store for a plugin (optional ``sub`` and ``profile``)."""

    def __init__(
        self,
        plugin: str,
        sub: str | None = None,
        *,
        backend: CredentialBackend | None = None,
        profile: str | None = None,
    ) -> None:
        self._store = PerPluginStore(
            plugin,
            sub,
            backend=backend,
            sub_key="accounts",
            profile=profile,
        )

    @property
    def store(self) -> PerPluginStore:
        return self._store

    def _read_state(self) -> tuple[str | None, dict[str, dict[str, Any]]]:
        raw = self._store.load()
        if not raw:
            return None, {}
        raw_accounts = raw.get("accounts")
        accounts: dict[str, dict[str, Any]] = {}
        if isinstance(raw_accounts, dict):
            for key, val in raw_accounts.items():
                if isinstance(key, str) and isinstance(val, dict):
                    accounts[key] = dict(val)
        active = raw.get("active")
        active_alias = active if isinstance(active, str) and active in accounts else None
        if active_alias is None and accounts:
            active_alias = next(iter(accounts))
        return active_alias, accounts

    def list_accounts(self) -> list[str]:
        """Return sorted account aliases stored in this registry."""
        _, accounts = self._read_state()
        return sorted(accounts)

    @property
    def active_account(self) -> str | None:
        """Currently active account alias, or ``None`` when empty."""
        active, _ = self._read_state()
        return active

    def get_account(self, alias: str | None = None) -> dict[str, Any] | None:
        """Return credentials for ``alias`` (or the active account when ``alias`` is ``None``)."""
        active, accounts = self._read_state()
        target = validate_account_alias(alias) if alias is not None else active
        if target is None:
            return None
        return accounts.get(target)

    def put_account(
        self,
        alias: str,
        credentials: dict[str, Any],
        *,
        set_active: bool = True,
    ) -> None:
        """Insert or update ``alias`` with ``credentials``."""
        checked = validate_account_alias(alias)
        if not isinstance(credentials, dict):
            raise ValueError("Account credentials must be a dict")
        active, accounts = self._read_state()
        accounts[checked] = dict(credentials)
        new_active = checked if (set_active or active is None) else active
        self._store.save({"active": new_active, "accounts": accounts})

    def set_active(self, alias: str) -> None:
        """Mark an existing account ``alias`` as active."""
        checked = validate_account_alias(alias)
        _, accounts = self._read_state()
        if checked not in accounts:
            raise KeyError(f"Unknown account alias: {checked!r}")
        self._store.save({"active": checked, "accounts": accounts})

    def remove_account(self, alias: str) -> bool:
        """Remove ``alias``; return ``True`` if it existed."""
        checked = validate_account_alias(alias)
        active, accounts = self._read_state()
        if checked not in accounts:
            return False
        del accounts[checked]
        if not accounts:
            self._store.clear()
            return True
        new_active = active if (active is not None and active in accounts) else next(iter(accounts))
        self._store.save({"active": new_active, "accounts": accounts})
        return True
