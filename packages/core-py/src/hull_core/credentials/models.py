"""Provider credential pool and multi-account registry data model.

A *pool* is the ordered list of credentials that belong to one provider
(``openrouter``, ``openai``, a self-hosted base URL's slug, ...). Each entry
keeps the same cell material the products already use — ``base_url`` +
``api_key`` + optional ``model`` — plus the health state rotation needs:
last status, consecutive failure count, cooldown deadline, lease deadline and
a manual disable flag.

A *pool entry* may be attached to a named account (email/workspace-style
multi-account providers). ``account_id`` is what config points at; the human
name, email, workspace and free-form labels live in the account registry.

The key itself never appears in a repr, a health view or an error message; the
pool exposes a hint (last four characters) and a keyed HMAC fingerprint instead.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from urllib.parse import urlparse

# Host -> provider identity used when config does not name one explicitly.
KNOWN_PROVIDER_HOSTS: dict[str, str] = {
    "openrouter.ai": "openrouter",
    "api.openai.com": "openai",
    "api.anthropic.com": "anthropic",
    "generativelanguage.googleapis.com": "gemini",
    "api.cohere.com": "cohere",
    "api.mistral.ai": "mistral",
    "api.together.xyz": "together",
    "api.groq.com": "groq",
    "api.deepseek.com": "deepseek",
    "api.x.ai": "xai",
    "api.voyageai.com": "voyage",
}


def provider_from_base_url(base_url: str) -> str:
    """Derive the provider identity a ``base_url`` belongs to.

    Known hosts map to their short name; anything else becomes the lowercased
    host with dots replaced by hyphens (``api.example.com`` ->
    ``api-example-com``). An unparseable URL falls back to ``"default"``.
    """
    try:
        host = (urlparse(base_url).hostname or "").lower()
    except ValueError:
        host = ""
    if not host:
        return "default"
    for known, provider in KNOWN_PROVIDER_HOSTS.items():
        if host == known or host.endswith(f".{known}"):
            return provider
    return host.replace(".", "-")


@dataclass(frozen=True)
class Credential:
    """One decrypted pool entry, ready to be used for a provider call.

    ``api_key`` is excluded from ``repr``; log :attr:`masked` (or a
    :class:`CredentialHealth` row) instead.
    """

    id: int
    provider: str
    account_id: str | None
    label: str | None
    base_url: str
    model: str | None
    api_key: str = field(repr=False)
    priority: int = 0
    key_hint: str = field(default="…", repr=False)

    @property
    def masked(self) -> str:
        """Safe display form: ``provider/account-or-label:…abcd``."""
        owner = self.account_id or self.label or str(self.id)
        return f"{self.provider}/{owner}:{self.key_hint}"


@dataclass(frozen=True)
class CredentialHealth:
    """Read-only view of one pool entry's state — no key material at all."""

    id: int
    provider: str
    account_id: str | None
    label: str | None
    base_url: str
    model: str | None
    key_hint: str
    priority: int
    disabled: bool
    last_status: int | None
    failure_count: int
    cooldown_until: float
    lease_until: float
    cooldown_remaining: float
    usable: bool


@dataclass(frozen=True)
class Account:
    """A named provider identity (one email/workspace per row)."""

    account_id: str
    provider: str
    name: str
    email: str | None = None
    workspace: str | None = None
    labels: dict[str, str] = field(default_factory=dict)


@dataclass(frozen=True)
class MigrationResult:
    """Outcome of seeding a pool from the pre-pool single-key config."""

    provider: str
    added: int
    skipped: int


class CredentialPoolError(RuntimeError):
    """Base class for credential pool/registry failures (never holds a key)."""


class PoolExhaustedError(CredentialPoolError):
    """No credential in the pool can serve the call right now."""

    def __init__(
        self,
        message: str,
        *,
        provider: str,
        account_id: str | None = None,
        total: int = 0,
        disabled: int = 0,
        cooling_down: int = 0,
        leased: int = 0,
        retry_after: float | None = None,
        last_error: str | None = None,
    ) -> None:
        super().__init__(message)
        self.provider = provider
        self.account_id = account_id
        self.total = total
        self.disabled = disabled
        self.cooling_down = cooling_down
        self.leased = leased
        self.retry_after = retry_after
        self.last_error = last_error


class CredentialExistsError(CredentialPoolError):
    """The same provider/account/key/base_url entry is already stored."""


class CredentialNotFoundError(CredentialPoolError, KeyError):
    """No pool entry has that id."""


class AccountError(CredentialPoolError, ValueError):
    """The account registry rejected the request."""
