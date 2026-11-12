"""Line-preserving edits to a dotted key in a host-owned ``config.toml``.

``config set`` must not reformat the file: the template is heavily commented
and users rely on those comments staying put, and a full TOML round-trip would
drop them. So the setter rewrites exactly one ``key = value`` line, or inserts
one under its section header.

Shared by every hull consumer CLI (``hull``, ``wet``, ``mnemo``, ``crg``, ...)
so the three products cannot drift on quoting, secret masking or error text.
"""

from __future__ import annotations

import json
import re
import tomllib
from pathlib import Path

from hull_core.config.settings import ConfigError

_HEADER_RE = re.compile(r"^\[\s*([^\]]+?)\s*\]$")
_KEY_RE = re.compile(r"^([A-Za-z0-9_\-]+)\s*=")

# Substrings that mean "never echo this value back" in a CLI transcript.
_SECRET_HINTS = ("token", "key", "secret", "password", "passphrase", "salt")


def toml_scalar(raw: str) -> str:
    """Render ``raw`` as a TOML value, keeping numbers/bools typed."""
    try:
        parsed = tomllib.loads(f"v = {raw}")
        value = parsed["v"]
    except tomllib.TOMLDecodeError:
        value = raw
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, str):
        return json.dumps(value, ensure_ascii=False)
    return str(value)


def looks_secret(key: str) -> bool:
    """True when a CLI transcript must not echo ``key``'s value.

    A ``*_hash`` key is a one-way digest, not the credential itself: ``config
    show`` already prints it, and copying it into a ``users.toml`` is the
    documented workflow, so masking it only makes the value harder to verify.
    """
    lowered = key.lower()
    if lowered.endswith("_hash"):
        return False
    return any(hint in lowered for hint in _SECRET_HINTS)


def get_value(path: Path, key: str) -> object:
    """Return the value at dotted ``key``; raise ``ConfigError`` when absent."""
    if not path.is_file():
        raise ConfigError(f"no config at {path} (run `config init`)")
    data = tomllib.loads(path.read_text(encoding="utf-8"))
    current: object = data
    for part in key.split("."):
        if not isinstance(current, dict) or part not in current:
            raise ConfigError(f"key not found: {key} (in {path})")
        current = current[part]
    return current


def format_value(value: object) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, str):
        return json.dumps(value, ensure_ascii=False)
    return str(value)


def set_value(path: Path, key: str, raw_value: str) -> None:
    """Write ``key = raw_value`` into ``path`` in place; keep every comment."""
    if not path.is_file():
        raise ConfigError(f"no config at {path} (run `config init`)")
    parts = key.split(".")
    if len(parts) < 2:
        raise ConfigError(f"key must be dotted (<section>.<key>, e.g. server.port): got {key!r}")
    section, name = ".".join(parts[:-1]), parts[-1]
    value_text = toml_scalar(raw_value)

    lines = path.read_text(encoding="utf-8").splitlines(keepends=True)
    section_idx: int | None = None
    insert_at: int | None = None
    replaced = False
    for i, line in enumerate(lines):
        stripped = line.strip()
        header = _HEADER_RE.match(stripped)
        if header:
            if section_idx is not None:
                insert_at = i  # the next section starts: insert before it
                break
            if header.group(1) == section:
                section_idx = i
            continue
        if section_idx is not None:
            key_match = _KEY_RE.match(stripped)
            if key_match and key_match.group(1) == name:
                lines[i] = f"{name} = {value_text}\n"
                replaced = True
                break
    if section_idx is None:
        raise ConfigError(
            f"section [{section}] not found in {path}; edit the file directly "
            f"to add it (path: `config path`, template: `config show`)"
        )
    if not replaced:
        lines.insert(
            insert_at if insert_at is not None else len(lines),
            f"{name} = {value_text}\n",
        )
    path.write_text("".join(lines), encoding="utf-8")
