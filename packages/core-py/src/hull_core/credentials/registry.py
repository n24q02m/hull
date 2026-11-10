"""Named provider accounts for multi-account pools.

A provider such as OpenRouter can be used through several identities (one per
email, workspace or billing owner). Each account owns its own credential pool:
rotation never hops between accounts, so a 429 on one account's key cannot
silently spend another account's quota.

``account_id`` is the stable identifier config points at; ``name`` is the human
label and is unique per provider. ``email``, ``workspace`` and free-form
``labels`` are optional metadata. Credentials can only be attached to an
existing account, and only when its provider matches.
"""

from __future__ import annotations

import json
import re
from typing import TYPE_CHECKING, Any, Mapping

from hull_core.credentials.models import Account, AccountError, CredentialHealth
from hull_core.credentials.store import CredentialStore

if TYPE_CHECKING:
    from hull_core.credentials.pool import KeyPool

_SLUG = re.compile(r"[^a-z0-9]+")
_UNSET: Any = object()


def _text(value: str, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise AccountError(f"{field} must be a non-empty string")
    return value.strip()


def _optional(value: str | None) -> str | None:
    if value is None:
        return None
    value = value.strip()
    return value or None


def _labels(value: Mapping[str, Any] | None) -> dict[str, str]:
    if value is None:
        return {}
    if not isinstance(value, Mapping):
        raise AccountError("labels must be a mapping of string keys to string values")
    return {str(key): str(item) for key, item in value.items()}


def _slug(name: str) -> str:
    slug = _SLUG.sub("-", name.strip().lower()).strip("-")
    if not slug:
        raise AccountError(f"cannot derive an account id from name {name!r}; pass account_id explicitly")
    return slug


def _row_to_account(row: Any) -> Account:
    try:
        raw = json.loads(str(row["labels"] or "{}"))
    except json.JSONDecodeError:
        raw = {}
    labels = {str(key): str(item) for key, item in raw.items()} if isinstance(raw, dict) else {}
    return Account(
        account_id=str(row["account_id"]),
        provider=str(row["provider"]),
        name=str(row["name"]),
        email=row["email"],
        workspace=row["workspace"],
        labels=labels,
    )


class AccountRegistry:
    """CRUD over ``provider_accounts`` plus per-account pool resolution."""

    def __init__(self, store: CredentialStore) -> None:
        self.store = store

    def add(
        self,
        *,
        provider: str,
        name: str,
        account_id: str | None = None,
        email: str | None = None,
        workspace: str | None = None,
        labels: Mapping[str, Any] | None = None,
    ) -> Account:
        """Register one named account; the default id is ``<provider>:<name slug>``."""
        provider = _text(provider, "provider")
        name = _text(name, "name")
        account_id = _text(account_id, "account_id") if account_id is not None else f"{provider}:{_slug(name)}"
        account = Account(
            account_id=account_id,
            provider=provider,
            name=name,
            email=_optional(email),
            workspace=_optional(workspace),
            labels=_labels(labels),
        )
        self.store.insert_account(account)
        return account

    def maybe_get(self, ref: str) -> Account | None:
        """Resolve an account id first, then a unique name; ``None`` when absent."""
        ref = _text(ref, "account reference")
        row = self.store.account_row(ref)
        if row is not None:
            return _row_to_account(row)
        matches = self.store.account_row_by_name(ref)
        if not matches:
            return None
        if len(matches) > 1:
            providers = sorted({str(item["provider"]) for item in matches})
            raise AccountError(f"account name {ref!r} is ambiguous across providers {providers}; pass the account id")
        return _row_to_account(matches[0])

    def get(self, ref: str) -> Account:
        account = self.maybe_get(ref)
        if account is None:
            raise AccountError(f"unknown account {ref!r}; add it with add() first")
        return account

    def list_accounts(self, provider: str | None = None) -> list[Account]:
        accounts = [_row_to_account(row) for row in self.store.account_rows()]
        if provider is not None:
            accounts = [account for account in accounts if account.provider == provider]
        return accounts

    def update(
        self,
        ref: str,
        *,
        name: str | None = None,
        email: str | None = _UNSET,
        workspace: str | None = _UNSET,
        labels: Mapping[str, Any] | None = _UNSET,
    ) -> Account:
        """Partially update metadata; omitted fields keep their value.

        ``labels=None`` clears the labels, ``email=None``/``workspace=None``
        clear those fields. The provider and account id are immutable.
        """
        current = self.get(ref)
        new_email = current.email if email is _UNSET else _optional(email)
        new_workspace = current.workspace if workspace is _UNSET else _optional(workspace)
        new_labels = current.labels if labels is _UNSET else _labels(labels)
        self.store.update_account_row(
            current.account_id,
            name=_text(name, "name") if name is not None else current.name,
            email=new_email,
            workspace=new_workspace,
            labels_json=json.dumps(new_labels, sort_keys=True),
        )
        return self.get(current.account_id)

    def remove(self, ref: str, *, drop_credentials: bool = False) -> None:
        """Delete an account; credentials block the delete unless dropped explicitly."""
        current = self.get(ref)
        self.store.delete_account_row(current.account_id, drop_credentials=drop_credentials)

    def credentials(self, ref: str) -> list[CredentialHealth]:
        """Masked pool state for one account (safe to log)."""
        account = self.get(ref)
        return self.store.pool_health(account.provider, account.account_id)

    def pool(self, ref: str, **policy: Any) -> KeyPool:
        """The KeyPool scoped to one account (provider + account id)."""
        account = self.get(ref)
        return self.store.pool(account.provider, account_id=account.account_id, **policy)
