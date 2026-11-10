"""Provider key pools and named accounts for hull.

Products used to resolve exactly one ``api_key`` per provider, so a 401 or a
429 took the whole feature down. This package adds the missing layer:

* :class:`~hull_core.credentials.store.CredentialStore` — ordered credentials
  per provider (and per named account) in the existing ``~/.hull/hull.db``,
  with the key sealed by :mod:`hull_core.credentials.crypto`.
* :class:`~hull_core.credentials.pool.KeyPool` — 401/403 disable, 429 cooldown
  (``Retry-After``/``rate_limit_reset``), 5xx retry-then-next, bounded
  attempts and a redacted exhaustion error.
* :class:`~hull_core.credentials.registry.AccountRegistry` — email/workspace
  accounts, each with its own pool.

``base_url + api_key + model`` cells are unchanged; the pool sits under the
provider identity and a cell's optional ``provider``/``account`` keys select
which pool it reads.
"""

from hull_core.credentials.crypto import (
    CredentialCryptoError,
    SecretCipher,
    decrypt_secret,
    encrypt_secret,
    key_hint,
    load_or_create_key,
)
from hull_core.credentials.models import (
    Account,
    AccountError,
    Credential,
    CredentialExistsError,
    CredentialHealth,
    CredentialNotFoundError,
    CredentialPoolError,
    MigrationResult,
    PoolExhaustedError,
    provider_from_base_url,
)
from hull_core.credentials.pool import (
    DEFAULT_COOLDOWN_SECONDS,
    KeyPool,
    classify_status,
    cooldown_seconds,
    parse_rate_limit_reset,
    parse_retry_after,
)
from hull_core.credentials.registry import AccountRegistry
from hull_core.credentials.store import POOL_SCHEMA, CredentialStore, default_pool_db_path

__all__ = [
    "DEFAULT_COOLDOWN_SECONDS",
    "POOL_SCHEMA",
    "Account",
    "AccountError",
    "AccountRegistry",
    "Credential",
    "CredentialCryptoError",
    "CredentialExistsError",
    "CredentialHealth",
    "CredentialNotFoundError",
    "CredentialPoolError",
    "CredentialStore",
    "KeyPool",
    "MigrationResult",
    "PoolExhaustedError",
    "SecretCipher",
    "classify_status",
    "cooldown_seconds",
    "decrypt_secret",
    "default_pool_db_path",
    "encrypt_secret",
    "key_hint",
    "load_or_create_key",
    "parse_rate_limit_reset",
    "parse_retry_after",
    "provider_from_base_url",
]
