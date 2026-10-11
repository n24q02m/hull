"""Console-script CLI builder for MCP servers that consume hull (telegram, ...).

``build_cli`` wraps a server's ``serve(argv) -> int | None`` entry point with
subcommand dispatch while leaving the two shapes ``serve`` already handles
untouched:

- Bare invocation (empty argv) starts the server: ``serve([])``.
- Any flag-prefixed argv (``--http`` etc.) is passed to ``serve`` unchanged,
  except ``-h``/``--help`` (always) and ``--version``/``-V`` (when ``version``
  is set), which are answered here; starting a stdio server on them would
  block on stdin instead of printing.

A leading positional argv[0] names a subcommand: a built-in below or a
server-specific one from ``extra`` (which wins on a name clash). Built-ins
know nothing about relays, OAuth or run modes; they only touch the hull
per-plugin credential store:

``config status`` / ``status``
    Whether a credential blob exists (and decrypts) plus where it lives.
``config delete [--yes]``
    Remove the stored config blob from every location.
``logout [--yes]``
    Remove the stored config and session blobs from every location.
``doctor``
    Python version, data directory, credential backend, credential state.
``token hash``
    Print the ``scrypt$...`` hash of a bearer token read from stdin (piped)
    or a hidden prompt; a token on argv is refused (shell history).

Output convention: informational results go to stdout, failures and prompts
to stderr with a non-zero return code. No built-in ever prints a credential
value, key material or token; only names, paths and states.
"""

from __future__ import annotations

import argparse
import getpass
import os
import sys
from collections.abc import Callable
from typing import cast

from hull_core.auth.tokens import hash_token
from hull_core.storage.backends import LocalFsBackend, backend_from_env
from hull_core.storage.data_dir import data_dir, legacy_data_dirs
from hull_core.storage.per_plugin_store import MachineKeyError, PerPluginStore

ExtraHandler = Callable[[argparse.Namespace], int]
ExtraConfigure = Callable[[argparse.ArgumentParser], None]
ExtraSpec = ExtraHandler | tuple[ExtraConfigure, ExtraHandler]

_SESSION_LEAF = "session"


def _credential_state(store: PerPluginStore) -> str:
    """``configured`` / ``corrupt`` / ``not configured`` without exposing values."""
    if not store.has_stored_blob():
        return "not configured"
    return "configured" if store.load() is not None else "corrupt"


def _print_status(server_name: str, store: PerPluginStore) -> int:
    state = _credential_state(store)
    if state == "corrupt":
        print(f"{server_name}: config is corrupt (undecryptable) -- re-run setup to restore", file=sys.stderr)
        return 1
    print(f"{server_name}: {state}")
    locations = store.blob_locations()
    if locations:
        primary = store.cred_path
        for path in locations:
            note = "" if path == primary else " (legacy dir, migrated on next save)"
            print(f"  path: {path}{note}")
    elif store.cred_path is not None:
        print(f"  path: {store.cred_path} (not created yet)")
    else:
        print(f"  backend: {type(store.backend).__name__}")
    return 0


def _confirm(prompt: str) -> bool:
    print(f"{prompt} [y/N] ", end="", file=sys.stderr, flush=True)
    return sys.stdin.readline().strip().lower() in ("y", "yes")


def _confirmed(server_name: str, prompt: str, *, yes: bool) -> bool:
    if yes:
        return True
    if not sys.stdin.isatty():
        print(f"{server_name}: refusing to delete credentials without --yes in non-interactive mode", file=sys.stderr)
        return False
    if not _confirm(prompt):
        print(f"{server_name}: aborted", file=sys.stderr)
        return False
    return True


def _config_handler(server_name: str, plugin_name: str) -> ExtraHandler:
    def handler(ns: argparse.Namespace) -> int:
        store = PerPluginStore(plugin_name)
        if ns.config_action == "status":
            return _print_status(server_name, store)
        if not _confirmed(server_name, f"Delete stored config for {server_name}?", yes=ns.yes):
            return 1
        store.clear()
        print(f"{server_name}: config deleted")
        return 0

    return handler


def _status_handler(server_name: str, plugin_name: str) -> ExtraHandler:
    def handler(_ns: argparse.Namespace) -> int:
        return _print_status(server_name, PerPluginStore(plugin_name))

    return handler


def _logout_handler(server_name: str, plugin_name: str) -> ExtraHandler:
    def handler(ns: argparse.Namespace) -> int:
        if not _confirmed(server_name, f"Remove stored credentials for {server_name}?", yes=ns.yes):
            return 1
        backend = backend_from_env()
        PerPluginStore(plugin_name, backend=backend).clear()
        PerPluginStore(plugin_name, backend=backend, sub_key=_SESSION_LEAF).clear()
        print(f"{server_name}: stored credentials removed")
        return 0

    return handler


def _doctor_handler(server_name: str, plugin_name: str) -> ExtraHandler:
    def handler(_ns: argparse.Namespace) -> int:
        ok = True
        major, minor = sys.version_info[:2]
        if (major, minor) >= (3, 13):
            print(f"[ok] python {major}.{minor}")
        else:
            print(f"[fail] python {major}.{minor} (hull requires >= 3.13)")
            ok = False

        try:
            backend = backend_from_env()
        except ValueError as err:
            print(f"[fail] credential backend: {err}")
            return 1
        print(f"[ok] credential backend: {type(backend).__name__}")

        if isinstance(backend, LocalFsBackend):
            # Never mkdir-probe: doctor must not create a dir for a server
            # that was never configured.
            directory = data_dir(plugin_name)
            if not directory.exists():
                print(f"[warn] data dir does not exist yet: {directory}")
            elif os.access(directory, os.W_OK):
                print(f"[ok] data dir writable: {directory}")
            else:
                print(f"[fail] data dir not writable: {directory}")
                ok = False
            for candidate in legacy_data_dirs(plugin_name):
                if candidate.dir.exists():
                    print(f"[ok] legacy data dir (read-only fallback): {candidate.dir}")

        try:
            state = _credential_state(PerPluginStore(plugin_name, backend=backend))
        except MachineKeyError as err:
            print(f"[fail] machine key: {err}")
            return 1
        if state == "configured":
            print("[ok] config: configured")
        elif state == "corrupt":
            print("[fail] config: corrupt (re-run setup to restore)")
            ok = False
        else:
            print("[warn] config: not configured")
        return 0 if ok else 1

    return handler


def _token_handler(server_name: str) -> ExtraHandler:
    def handler(ns: argparse.Namespace) -> int:
        if ns.token_value is not None:
            print(
                f"{server_name}: refusing to read a token from the command line (it would land in shell "
                "history); pipe it on stdin or omit it to be prompted",
                file=sys.stderr,
            )
            return 2
        if sys.stdin.isatty():
            token = getpass.getpass("Token: ")
        else:
            token = sys.stdin.read().strip()
        if not token:
            print(f"{server_name}: empty token", file=sys.stderr)
            return 1
        print(hash_token(token))
        return 0

    return handler


def _configure_builtin(name: str, parser: argparse.ArgumentParser) -> None:
    if name == "config":
        parser.add_argument("config_action", choices=["status", "delete"])
        parser.add_argument("--yes", action="store_true", default=False)
    elif name == "logout":
        parser.add_argument("--yes", action="store_true", default=False)
    elif name == "token":
        parser.add_argument("token_action", choices=["hash"])
        parser.add_argument("token_value", nargs="?", default=None, help=argparse.SUPPRESS)


def _usage(server_name: str, handlers: dict[str, ExtraHandler]) -> str:
    return (
        f"usage: {server_name} [-h] [--version] [<subcommand> ...]\n"
        "Any other flags/args are passed through to the MCP server (e.g. --http).\n"
        f"subcommands: {', '.join(sorted(handlers))}"
    )


def build_cli(
    server_name: str,
    *,
    plugin_name: str | None = None,
    serve: Callable[[list[str]], int | None],
    extra: dict[str, ExtraSpec] | None = None,
    version: str | None = None,
) -> Callable[[list[str] | None], int]:
    """Build the console-script entry point for one consumer MCP server.

    ``server_name`` is the console/display name (``"better-telegram-mcp"``);
    ``plugin_name`` is the credential storage slug (``"telegram"``) and
    defaults to ``server_name`` minus a leading ``better-`` and a trailing
    ``-mcp`` (the inverse of the ``~/.better-<plugin>-mcp`` legacy dir name).

    Each ``extra`` value is a bare handler (argument-less subcommand) or a
    ``(configure, handler)`` tuple whose ``configure`` adds arguments to that
    subcommand's parser. An ``extra`` name replaces the built-in of the same
    name, including its argument wiring.
    """
    plugin_name = plugin_name or server_name.removeprefix("better-").removesuffix("-mcp")
    handlers: dict[str, ExtraHandler] = {
        "config": _config_handler(server_name, plugin_name),
        "status": _status_handler(server_name, plugin_name),
        "logout": _logout_handler(server_name, plugin_name),
        "doctor": _doctor_handler(server_name, plugin_name),
        "token": _token_handler(server_name),
    }
    configurers: dict[str, ExtraConfigure] = {}
    overridden: set[str] = set()
    for name, spec in (extra or {}).items():
        # The cast states what the API contract guarantees (a tuple spec is
        # always (configure, handler)); isinstance alone leaves ty unable to
        # exclude the bare-handler arm of the union.
        if isinstance(spec, tuple):
            configure_fn, handler_fn = cast("tuple[ExtraConfigure, ExtraHandler]", spec)
            configurers[name] = configure_fn
        else:
            handler_fn = cast("ExtraHandler", spec)
        handlers[name] = handler_fn
        overridden.add(name)

    def run(argv: list[str] | None = None) -> int:
        if argv is None:
            argv = sys.argv[1:]

        if not argv:
            rc = serve([])
            return 0 if rc is None else rc

        if argv[0] in ("-h", "--help"):
            print(_usage(server_name, handlers))
            return 0

        if version is not None and argv[0] in ("--version", "-V"):
            print(f"{server_name} {version}")
            return 0

        if argv[0].startswith("-"):
            rc = serve(argv)
            return 0 if rc is None else rc

        handler = handlers.get(argv[0])
        if handler is None:
            print(f"{server_name}: unknown subcommand {argv[0]!r}", file=sys.stderr)
            print(_usage(server_name, handlers), file=sys.stderr)
            return 2

        parser = argparse.ArgumentParser(prog=server_name)
        subparsers = parser.add_subparsers(dest="subcommand")
        for name in handlers:
            sub = subparsers.add_parser(name)
            if name in configurers:
                configurers[name](sub)
            elif name not in overridden:
                _configure_builtin(name, sub)
        try:
            ns = parser.parse_args(argv)
        except SystemExit as exc:
            # argparse exits 2 on bad args and 0 on `<sub> -h`; surface that as
            # the return code so the entry point always returns an int.
            return exc.code if isinstance(exc.code, int) else 2
        try:
            return handler(ns)
        except MachineKeyError as err:
            # A foreign .secret is operator-actionable: one clear line, no traceback.
            print(f"{server_name}: {err}", file=sys.stderr)
            return 1
        except ValueError as err:
            # Built-ins raise ValueError only for operator config (bad
            # MCP_STORAGE_BACKEND, invalid plugin name). A server's own handler
            # keeps its traceback: swallowing it would hide real bugs.
            if argv[0] in overridden:
                raise
            print(f"{server_name}: {err}", file=sys.stderr)
            return 1

    return run
