"""users.toml — the multi-user table (auth mode 3).

Schema (spec §4)::

    [users.<uid>]
    token_hash = "scrypt$..."        # mint with `hull token hash`
    enabled = true
    namespace = "<uid-or-shared>"    # data isolation root
    allowed_roots = ["/data/alice"]  # filesystem roots tools may touch
    [users.<uid>.limits]
    rpm = 60                         # optional per-minute request cap

Admin is host-side only: edit the file, restart the server. There is no admin
API and no self-service.
"""

from __future__ import annotations

import tomllib
from dataclasses import dataclass, field
from pathlib import Path

from hull_core.auth.tokens import is_valid_encoding, verify_token


class UsersError(ValueError):
    """Raised when users.toml is missing, malformed, or semantically invalid."""


@dataclass(frozen=True)
class UserLimits:
    rpm: int | None = None


@dataclass(frozen=True)
class User:
    uid: str
    token_hash: str
    enabled: bool
    namespace: str
    allowed_roots: tuple[str, ...] = ()
    limits: UserLimits = field(default_factory=UserLimits)


def load_users(path: Path) -> dict[str, User]:
    """Parse and validate users.toml into a ``uid → User`` mapping."""
    if not path.is_file():
        raise UsersError(f"users file not found: {path}")
    try:
        raw = tomllib.loads(path.read_text(encoding="utf-8"))
    except tomllib.TOMLDecodeError as exc:
        raise UsersError(f"users file is not valid TOML: {path}: {exc}") from exc

    table = raw.get("users")
    if not isinstance(table, dict) or not table:
        raise UsersError("users file must contain a non-empty [users] table")

    users: dict[str, User] = {}
    for uid, entry in table.items():
        if not isinstance(entry, dict):
            raise UsersError(f"user {uid!r}: expected a table")
        token_hash = entry.get("token_hash")
        if not isinstance(token_hash, str) or not is_valid_encoding(token_hash):
            raise UsersError(
                f"user {uid!r}: token_hash missing or not a scrypt encoding "
                "(mint one with `hull token hash`)"
            )
        namespace = entry.get("namespace")
        if not isinstance(namespace, str) or not namespace:
            raise UsersError(f"user {uid!r}: namespace must be a non-empty string")
        enabled = entry.get("enabled", True)
        if not isinstance(enabled, bool):
            raise UsersError(f"user {uid!r}: enabled must be a boolean")
        roots = entry.get("allowed_roots", [])
        if not isinstance(roots, list) or not all(isinstance(r, str) for r in roots):
            raise UsersError(f"user {uid!r}: allowed_roots must be a list of strings")
        limits_raw = entry.get("limits", {})
        if not isinstance(limits_raw, dict):
            raise UsersError(f"user {uid!r}: limits must be a table")
        rpm = limits_raw.get("rpm")
        if rpm is not None and (not isinstance(rpm, int) or rpm < 1):
            raise UsersError(f"user {uid!r}: limits.rpm must be a positive integer")
        users[uid] = User(
            uid=uid,
            token_hash=token_hash,
            enabled=enabled,
            namespace=namespace,
            allowed_roots=tuple(roots),
            limits=UserLimits(rpm=rpm),
        )
    return users


def find_user_by_token(users: dict[str, User], token: str) -> User | None:
    """Return the user whose token_hash verifies ``token``, else None."""
    for user in users.values():
        if verify_token(token, user.token_hash):
            return user
    return None


def ensure_path_allowed(user: User, path: Path) -> None:
    """Enforce the user's allowed_roots (spec §4 Q6).

    An empty allowed_roots list grants nothing. Roots are matched lexically
    after resolving both sides, so symlinked escapes require the resolved
    target to sit under a root too.
    """
    resolved = path.resolve()
    for root in user.allowed_roots:
        root_resolved = Path(root).resolve()
        if resolved == root_resolved or root_resolved in resolved.parents:
            return
    raise PermissionError(
        f"path {resolved} is outside allowed roots for user {user.uid!r}"
    )
