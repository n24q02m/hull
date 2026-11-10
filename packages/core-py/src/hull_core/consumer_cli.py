"""Console-script CLI builder for MCP servers that consume hull (telegram, ...).

``build_cli`` wraps a server's ``serve(argv) -> int | None`` entry point with
subcommand dispatch while leaving the two shapes ``serve`` already handles
untouched:

- Bare invocation (empty argv) starts the server: ``serve([])``.
- Any flag-prefixed argv (``--http`` etc.) is passed to ``serve`` unchanged,
  except ``-h``/``--help`` (always) and ``--version``/``-V`` (when ``version``
  is set), which are answered here; starting a stdio server on them would
  block on stdin instead of printing.

``-h``/``--help`` prints one argparse-rendered page: the pass-through rule,
then every subcommand with the flags it accepts (derived from the subcommand's
own parser, including a server's ``extra`` commands), so a reader can find
``auth --bot-token`` without guessing that ``<subcommand> -h`` exists.

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


# One-line description per built-in; the flags are appended from the parser.
_BUILTIN_HELP = {
    "config": "print the credential state, or delete the stored config",
    "status": "print the credential state and where it is stored",
    "logout": "remove the stored config and session credentials",
    "doctor": "check python, data dir, backend and credentials",
    "token": "mint a token_hash from stdin or a hidden prompt",
}


def _argument_label(action: argparse.Action) -> str | None:
    """The action as it would appear in a usage line, or None when not shown."""
    if action.dest in ("help", "subcommand"):
        # argparse's own -h and the subcommand dispatcher: never user surface.
        return None
    if action.option_strings:
        label = ", ".join(action.option_strings)
        if action.nargs != 0:
            label += f" {action.metavar or action.dest.upper()}"
        return label
    # Positionals fall back the way argparse does: a declared metavar, else the
    # choice set, else the upper-cased dest.
    if action.metavar:
        return str(action.metavar)
    if action.choices:
        return "{" + ",".join(str(choice) for choice in action.choices) + "}"
    return action.dest.upper()


def _argument_summary(sub: argparse.ArgumentParser, *, positionals: bool) -> str:
    """The arguments a subcommand accepts, e.g. ``--bot-token BOT_TOKEN``.

    ``_actions`` is argparse's only iteration hook; every action comes from this
    module's ``add_argument`` calls or a consumer's ``configure`` callback, so
    the top-level help cannot claim an argument the subcommand does not accept.
    """
    labels = []
    for action in sub._actions:
        label = _argument_label(action)
        if label is not None and (positionals or bool(action.option_strings)):
            labels.append(label)
    return ", ".join(labels)


def _subcommand_table(parsers: list[tuple[str, argparse.ArgumentParser]], *, overridden: set[str]) -> str:
    """Epilog listing every subcommand with its arguments, in one aligned block.

    An ``extra`` that replaces a built-in is described by its own arguments:
    keeping the built-in text would describe a command the server replaced.
    """
    rows: list[tuple[str, str]] = []
    for name, sub in sorted(parsers, key=lambda item: item[0]):
        description = None if name in overridden else _BUILTIN_HELP.get(name)
        if description:
            # A built-in reads as "description (flags)"; its positionals are
            # internal wiring (``config status|delete``), not a user surface.
            flags = _argument_summary(sub, positionals=False)
            rows.append((name, f"{description} ({flags})" if flags else description))
        else:
            # An extra has no description here, so its argument list stands in.
            rows.append((name, _argument_summary(sub, positionals=True) or "(no arguments)"))
    width = max(len(name) for name, _ in rows)
    lines = [f"  {name.ljust(width)}  {text}" for name, text in rows]
    return "subcommands (each also accepts -h):\n" + "\n".join(lines)


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

    parser = argparse.ArgumentParser(
        prog=server_name,
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description=(
            "Control commands; a bare invocation starts the MCP server.\n\n"
            "Any other flag or argument (e.g. --http) is passed through to the MCP server."
        ),
    )
    if version is not None:
        parser.add_argument("--version", "-V", action="version", version=f"{server_name} {version}")
    subparsers = parser.add_subparsers(dest="subcommand")
    sub_parsers: list[tuple[str, argparse.ArgumentParser]] = []
    for name in handlers:
        sub = subparsers.add_parser(name)
        if name in configurers:
            configurers[name](sub)
        elif name not in overridden:
            _configure_builtin(name, sub)
        sub_parsers.append((name, sub))
    parser.epilog = _subcommand_table(sub_parsers, overridden=overridden)

    def run(argv: list[str] | None = None) -> int:
        if argv is None:
            argv = sys.argv[1:]

        if not argv:
            rc = serve([])
            return 0 if rc is None else rc

        if argv[0] in ("-h", "--help"):
            print(parser.format_help(), end="")
            return 0

        if version is not None and argv[0] in ("--version", "-V"):
            print(f"{server_name} {version}")
            return 0

        if argv[0].startswith("-"):
            rc = serve(argv)
            return 0 if rc is None else rc

        handler = handlers.get(argv[0])
        if handler is None:
            expected = ", ".join(sorted(handlers))
            print(f"{server_name}: unknown subcommand {argv[0]!r} (expected one of: {expected})", file=sys.stderr)
            print(parser.format_usage(), file=sys.stderr, end="")
            return 2

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
