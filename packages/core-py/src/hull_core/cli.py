"""Host-side CLI: control plane + consumer.

Subcommands: ``server start``, ``status``, ``doctor``, ``token hash``,
``config init|path|show|get|set`` and ``db path``, plus ``--version``. With the server
down, consumers are told to start it first — there is no hidden local-core mode
(spec §3).
"""

from __future__ import annotations

import argparse
import getpass
import os
import sys
import tomllib
from pathlib import Path

from hull_core import __version__
from hull_core._runtime import python_check
from hull_core.auth.tokens import hash_token
from hull_core.auth.users import UsersError, load_users
from hull_core.config import tomledit
from hull_core.config.settings import (
    CONFIG_TEMPLATE,
    ConfigError,
    default_config_dir,
    load_settings,
    write_default_config,
)
from hull_core.server.run import ServerConfigError, run_server

# Mirrors the refusal in ``consumer_cli``: a token on argv lands in shell
# history, so both CLIs read it from stdin or a hidden prompt instead.
_TOKEN_ON_ARGV_REFUSAL = (
    "hull: refusing to read a token from the command line (it would land in shell history); "
    "pipe it on stdin or omit it to be prompted"
)


def _cmd_server_start(args: argparse.Namespace) -> int:
    try:
        run_server(host=args.host, port=args.port)
    except ServerConfigError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    except KeyboardInterrupt:
        pass
    return 0


def _cmd_token_hash(args: argparse.Namespace) -> int:
    if args.token is not None:
        print(_TOKEN_ON_ARGV_REFUSAL, file=sys.stderr)
        return 2
    if sys.stdin.isatty():
        token = getpass.getpass("Token: ")
    else:
        token = sys.stdin.read().strip()
    if not token:
        print("hull: empty token", file=sys.stderr)
        return 1
    print(hash_token(token))
    return 0


def _cmd_config_init(args: argparse.Namespace) -> int:
    try:
        path = write_default_config(default_config_dir(), force=args.force)
    except FileExistsError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    print(f"wrote {path}")
    return 0


def _cmd_config_path(_args: argparse.Namespace) -> int:
    print(default_config_dir() / "config.toml")
    return 0


def _cmd_config_show(_args: argparse.Namespace) -> int:
    path = default_config_dir() / "config.toml"
    if path.is_file():
        print(path.read_text(encoding="utf-8"), end="")
    else:
        print(CONFIG_TEMPLATE, end="")
    return 0


def _config_path(args: argparse.Namespace) -> Path:
    return Path(args.config) if getattr(args, "config", None) else default_config_dir() / "config.toml"


def _cmd_config_get(args: argparse.Namespace) -> int:
    try:
        value = tomledit.get_value(_config_path(args), args.key)
    except (ConfigError, tomllib.TOMLDecodeError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    if tomledit.looks_secret(args.key):
        print("*** (value not echoed)")
    else:
        print(tomledit.format_value(value))
    return 0


def _cmd_config_set(args: argparse.Namespace) -> int:
    try:
        tomledit.set_value(_config_path(args), args.key, args.value)
    except (ConfigError, tomllib.TOMLDecodeError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    shown = "*** (value not echoed)" if tomledit.looks_secret(args.key) else tomledit.toml_scalar(args.value)
    print(f"set {args.key} = {shown} in {_config_path(args)}")
    return 0


def _cmd_db_path(_args: argparse.Namespace) -> int:
    print(load_settings(default_config_dir()).config_dir / "hull.db")
    return 0


def _cmd_status(_args: argparse.Namespace) -> int:
    """Config state, bind and storage in one screen; never the token hash."""
    config_dir = default_config_dir()
    path = config_dir / "config.toml"
    try:
        settings = load_settings(config_dir)
    except ConfigError as exc:
        print(f"hull: {exc}", file=sys.stderr)
        return 1
    if not path.is_file():
        print("hull: not configured")
        print(f"  path: {path} (not created yet)")
        return 0
    print("hull: configured")
    print(f"  path: {path}")
    print(f"  auth: {settings.server.auth}")
    print(f"  bind: {settings.server.host}:{settings.server.port}")
    if settings.server.users_file is not None:
        print(f"  users: {settings.server.users_file}")
    print(f"  db: {config_dir / 'hull.db'}")
    return 0


def _cmd_doctor(_args: argparse.Namespace) -> int:
    """Same ``[ok]/[warn]/[fail]`` shape the consumer CLIs print."""
    ok, line = python_check()
    print(line)

    config_dir = default_config_dir()
    path = config_dir / "config.toml"
    if not config_dir.exists():
        # Never mkdir-probe: doctor must not create a dir for an instance that
        # was never configured.
        print(f"[warn] config dir does not exist yet: {config_dir}")
    elif os.access(config_dir, os.W_OK):
        print(f"[ok] config dir writable: {config_dir}")
    else:
        print(f"[fail] config dir not writable: {config_dir}")
        ok = False

    try:
        settings = load_settings(config_dir)
    except ConfigError as exc:
        print(f"[fail] config: {exc}")
        return 1
    if path.is_file():
        print(f"[ok] config: valid (auth={settings.server.auth})")
    else:
        print("[warn] config: not configured (run `hull config init`)")

    users_file = settings.server.users_file
    if settings.server.auth == "multi" and users_file is not None:
        try:
            users = load_users(users_file)
        except UsersError as exc:
            print(f"[fail] users file: {exc}")
            ok = False
        else:
            print(f"[ok] users file: {users_file} ({len(users)} users)")
    return 0 if ok else 1


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="hull",
        description="hull shared core control plane: config, tokens, server and storage",
    )
    parser.add_argument("--version", "-V", action="version", version=f"hull {__version__}")

    sub = parser.add_subparsers(dest="command", required=True)

    server = sub.add_parser("server", help="server lifecycle (start)")
    server_sub = server.add_subparsers(dest="server_command", required=True)
    start = server_sub.add_parser("start", help="start the HTTP MCP server, blocking (--host/--port)")
    start.add_argument("--host", default=None, help="bind host (default from config)")
    start.add_argument("--port", type=int, default=None, help="bind port (default from config)")
    start.set_defaults(func=_cmd_server_start)

    status_p = sub.add_parser("status", help="print the config state, auth mode, bind and DB path")
    status_p.set_defaults(func=_cmd_status)

    doctor_p = sub.add_parser("doctor", help="check python, config dir, config and users file")
    doctor_p.set_defaults(func=_cmd_doctor)

    token = sub.add_parser("token", help="token utilities (hash)")
    token_sub = token.add_subparsers(dest="token_command", required=True)
    token_hash_p = token_sub.add_parser("hash", help="print a token_hash read from stdin or a hidden prompt")
    # Declared only so an argv token reaches our refusal instead of argparse's
    # "unrecognized arguments"; the safe input paths are stdin and the prompt.
    token_hash_p.add_argument("token", nargs="?", default=None, help=argparse.SUPPRESS)
    token_hash_p.set_defaults(func=_cmd_token_hash)

    config = sub.add_parser("config", help="instance config utilities (init, path, show)")
    config_sub = config.add_subparsers(dest="config_command", required=True)
    init_p = config_sub.add_parser("init", help="write ~/.hull/config.toml from the template (--force)")
    init_p.add_argument("--force", action="store_true", help="overwrite an existing config")
    init_p.set_defaults(func=_cmd_config_init)
    path_p = config_sub.add_parser("path", help="print the config file path")
    path_p.set_defaults(func=_cmd_config_path)
    show_p = config_sub.add_parser("show", help="print the effective config (template if absent)")
    show_p.add_argument("--config", default=None, help=argparse.SUPPRESS)
    show_p.set_defaults(func=_cmd_config_show)
    get_p = config_sub.add_parser("get", help="print one value (dotted key, e.g. server.port)")
    get_p.add_argument("key")
    get_p.add_argument("--config", default=None, help=argparse.SUPPRESS)
    get_p.set_defaults(func=_cmd_config_get)
    set_p = config_sub.add_parser("set", help="set one value in place, keeping comments")
    set_p.add_argument("key")
    set_p.add_argument("value")
    set_p.add_argument("--config", default=None, help=argparse.SUPPRESS)
    set_p.set_defaults(func=_cmd_config_set)

    db = sub.add_parser("db", help="storage utilities (path)")
    db_sub = db.add_subparsers(dest="db_command", required=True)
    db_path_p = db_sub.add_parser("path", help="print the SQLite database path")
    db_path_p.set_defaults(func=_cmd_db_path)

    # argparse names a required subcommand group by its ``dest`` when no
    # metavar is set, so ``hull server`` reported "the following arguments are
    # required: server_command" -- an identifier the user never typed. The
    # metavar is also what the usage line shows, so deriving both from
    # ``choices`` keeps them in step as commands are added.
    for group in (sub, server_sub, token_sub, config_sub, db_sub):
        group.metavar = "{" + ",".join(group.choices) + "}"
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return int(args.func(args))
