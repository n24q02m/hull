"""Host-side CLI: control plane + consumer.

Subcommands: ``server start``, ``token hash``, ``config init|path|show``,
``db path``. With the server down, consumers are told to start it first —
there is no hidden local-core mode (spec §3).
"""

from __future__ import annotations

import argparse
import sys

from hull_core.auth.tokens import hash_token
from hull_core.config.settings import (
    CONFIG_TEMPLATE,
    default_config_dir,
    load_settings,
    write_default_config,
)
from hull_core.server.run import ServerConfigError, run_server


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
    print(hash_token(args.token))
    return 0


def _cmd_config_init(args: argparse.Namespace) -> int:
    try:
        path = write_default_config(force=args.force)
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


def _cmd_db_path(_args: argparse.Namespace) -> int:
    settings = load_settings()
    print(settings.config_dir / "hull.db")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="hull", description="hull shared core control plane")
    sub = parser.add_subparsers(dest="command", required=True)

    server = sub.add_parser("server", help="server lifecycle")
    server_sub = server.add_subparsers(dest="server_command", required=True)
    start = server_sub.add_parser("start", help="start the HTTP MCP server (blocking)")
    start.add_argument("--host", default=None, help="bind host (default from config)")
    start.add_argument("--port", type=int, default=None, help="bind port (default from config)")
    start.set_defaults(func=_cmd_server_start)

    token = sub.add_parser("token", help="token utilities")
    token_sub = token.add_subparsers(dest="token_command", required=True)
    token_hash_p = token_sub.add_parser("hash", help="mint a scrypt token_hash from a plaintext token")
    token_hash_p.add_argument("token", help="plaintext token (keep it out of shells that log history)")
    token_hash_p.set_defaults(func=_cmd_token_hash)

    config = sub.add_parser("config", help="instance config utilities")
    config_sub = config.add_subparsers(dest="config_command", required=True)
    init_p = config_sub.add_parser("init", help="write ~/.hull/config.toml from the template")
    init_p.add_argument("--force", action="store_true", help="overwrite an existing config")
    init_p.set_defaults(func=_cmd_config_init)
    path_p = config_sub.add_parser("path", help="print the config file path")
    path_p.set_defaults(func=_cmd_config_path)
    show_p = config_sub.add_parser("show", help="print the effective config (template if absent)")
    show_p.set_defaults(func=_cmd_config_show)

    db = sub.add_parser("db", help="storage utilities")
    db_sub = db.add_subparsers(dest="db_command", required=True)
    db_path_p = db_sub.add_parser("path", help="print the SQLite database path")
    db_path_p.set_defaults(func=_cmd_db_path)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return int(args.func(args))
