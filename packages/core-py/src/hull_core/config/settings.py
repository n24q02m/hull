"""Hull instance configuration: ``~/.hull/config.toml``.

Everything the host owns lives here: server bind/mode and the per-task model
cells (spec §3/§4 — N cells, each ``base_url + api_key + model``, HTTP-plain
OpenAI-spec, OpenRouter pre-wired as default). Keys are host-only: end users
never see or submit them.
"""

from __future__ import annotations

import tomllib
from dataclasses import dataclass, field
from pathlib import Path

AUTH_MODES = ("no-auth", "token", "multi")


class ConfigError(ValueError):
    """Raised when config.toml is malformed or semantically invalid."""


def default_config_dir() -> Path:
    return Path.home() / ".hull"


@dataclass(frozen=True)
class ServerSettings:
    host: str = "127.0.0.1"
    port: int = 8000
    auth: str = "no-auth"
    token_hash: str | None = None
    users_file: Path | None = None
    rpm: int | None = None


@dataclass(frozen=True)
class HullSettings:
    config_dir: Path
    server: ServerSettings
    models: dict = field(default_factory=dict)  # task → raw [models.<task>] table


def load_settings(config_dir: Path | None = None) -> HullSettings:
    """Load config.toml (defaults apply for absent keys/file)."""
    config_dir = config_dir if config_dir is not None else default_config_dir()
    config_path = config_dir / "config.toml"
    raw: dict = {}
    if config_path.is_file():
        try:
            raw = tomllib.loads(config_path.read_text(encoding="utf-8"))
        except tomllib.TOMLDecodeError as exc:
            raise ConfigError(f"config file is not valid TOML: {config_path}: {exc}") from exc

    server_raw = raw.get("server", {})
    if not isinstance(server_raw, dict):
        raise ConfigError("[server] must be a table")
    auth = server_raw.get("auth", "no-auth")
    if auth not in AUTH_MODES:
        raise ConfigError(f"[server] auth must be one of {AUTH_MODES}, got {auth!r}")
    token_hash = server_raw.get("token_hash")
    if auth == "token" and not (isinstance(token_hash, str) and token_hash):
        raise ConfigError('[server] auth = "token" requires a non-empty token_hash')
    users_file = server_raw.get("users_file")
    users_path = Path(users_file).expanduser() if isinstance(users_file, str) and users_file else None
    if auth == "multi" and users_path is None:
        users_path = config_dir / "users.toml"
    rpm = server_raw.get("rpm")
    if rpm is not None and (not isinstance(rpm, int) or rpm < 1):
        raise ConfigError("[server] rpm must be a positive integer")

    server = ServerSettings(
        host=str(server_raw.get("host", "127.0.0.1")),
        port=int(server_raw.get("port", 8000)),
        auth=auth,
        token_hash=token_hash,
        users_file=users_path,
        rpm=rpm,
    )
    models_raw = raw.get("models", {})
    if not isinstance(models_raw, dict):
        raise ConfigError("[models] must be a table of per-task tables")
    return HullSettings(config_dir=config_dir, server=server, models=models_raw)


CONFIG_TEMPLATE = """\
# hull instance config (host-owned). Docs: README.md "Auth" + "Runtime model".
[server]
host = "127.0.0.1"
port = 8000
# no-auth  -> localhost-only, one shared namespace (default)
# token    -> one shared token; set token_hash below (`hull token hash ...`)
# multi    -> users.toml; token -> user -> namespace
auth = "no-auth"
# token_hash = "scrypt$..."
# users_file = "~/.hull/users.toml"
# rpm = 120   # optional shared-mode rate limit

# Per-task provider cells (spec: each task gets its own base_url + api_key +
# model; OpenRouter default pre-wired). api_key may be left empty here and
# supplied via env instead: HULL_EMBED_API_KEY / HULL_RERANK_API_KEY /
# HULL_CHAT_API_KEY / HULL_JEV_SCORE_API_KEY (host-only, never end-user).
[models.embed]
base_url = "https://openrouter.ai/api/v1"
api_key = ""
model = "voyage-4-lite"

[models.rerank]
base_url = "https://openrouter.ai/api/v1"
api_key = ""
model = "voyage-2.5-lite"

[models.chat]
base_url = "https://openrouter.ai/api/v1"
api_key = ""
model = "z-ai/glm-5.3-flash"

[models.jev_score]
base_url = "https://openrouter.ai/api/v1"
api_key = ""
model = "z-ai/glm-5.3-flash"
"""


def write_default_config(config_dir: Path | None = None, *, force: bool = False) -> Path:
    """Write the template config; refuse to clobber unless ``force``."""
    config_dir = config_dir if config_dir is not None else default_config_dir()
    config_path = config_dir / "config.toml"
    if config_path.exists() and not force:
        raise FileExistsError(f"config already exists: {config_path} (use --force to overwrite)")
    config_dir.mkdir(parents=True, exist_ok=True)
    config_path.write_text(CONFIG_TEMPLATE, encoding="utf-8")
    return config_path
