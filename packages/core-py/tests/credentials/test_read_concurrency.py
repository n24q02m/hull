"""Reads share the store's one SQLite connection with writes, so they take its lock too.

``CredentialStore`` keeps a single ``check_same_thread=False`` connection. Two
threads running a statement on that connection at the same time trip
``sqlite3.InterfaceError: bad parameter or other API misuse`` (the shared
statement cache), and a read that slips into a writer's open ``BEGIN
IMMEDIATE`` sees uncommitted state. Each public read is hammered by two reader
threads while a third thread keeps writing; any exception fails the test.
"""

from __future__ import annotations

import ast
import inspect
import textwrap
import threading
from collections.abc import Callable
from typing import Any

import pytest

from hull_core.credentials.models import Account
from hull_core.credentials.store import CredentialStore

BASE_URL = "https://openrouter.ai/api/v1"
ITERATIONS = 3000
READERS = 2


def _seed(store: CredentialStore) -> int:
    store.insert_account(Account(account_id="acct-1", provider="openrouter", name="team"))
    store.add_credential("openrouter", "sk-live-read-0001", base_url=BASE_URL)
    store.add_credential("openrouter", "sk-live-read-0002", base_url=BASE_URL, account_id="acct-1")
    return store.credentials("openrouter")[0].id


READS: dict[str, Callable[[CredentialStore, int], Any]] = {
    "credentials": lambda store, _cid: store.credentials("openrouter"),
    "pool_credentials": lambda store, _cid: store.pool_credentials("openrouter"),
    "health": lambda store, _cid: store.health("openrouter"),
    "pool_health": lambda store, _cid: store.pool_health("openrouter"),
    "get_credential": lambda store, cid: store.get_credential(cid),
    "diagnose": lambda store, _cid: store.diagnose("openrouter"),
    "account_rows": lambda store, _cid: store.account_rows(),
    "account_row": lambda store, _cid: store.account_row("acct-1"),
    "account_row_by_name": lambda store, _cid: store.account_row_by_name("team"),
}


def _hammer(store: CredentialStore, read: Callable[[CredentialStore, int], Any], credential_id: int) -> list[str]:
    errors: list[str] = []
    guard = threading.Lock()
    stop = threading.Event()
    start = threading.Barrier(READERS + 1)

    def record(exc: BaseException) -> None:
        with guard:
            errors.append(repr(exc))
        stop.set()

    def reader() -> None:
        start.wait(timeout=10)
        for _ in range(ITERATIONS):
            if stop.is_set():
                return
            try:
                read(store, credential_id)
            except BaseException as exc:  # surface every thread failure in the test result
                record(exc)
                return

    def writer() -> None:
        start.wait(timeout=10)
        priority = 0
        while not stop.is_set():
            try:
                priority += 1
                store.set_priority(credential_id, priority)
                store.record_success(credential_id)
            except BaseException as exc:
                record(exc)
                return

    readers = [threading.Thread(target=reader) for _ in range(READERS)]
    writer_thread = threading.Thread(target=writer)
    for thread in (*readers, writer_thread):
        thread.start()
    for thread in readers:
        thread.join(timeout=120)
    stop.set()
    writer_thread.join(timeout=30)
    return errors


@pytest.mark.parametrize("name", sorted(READS))
def test_read_is_safe_against_concurrent_reads_and_writes(store: CredentialStore, name: str) -> None:
    credential_id = _seed(store)
    assert _hammer(store, READS[name], credential_id) == []


def test_every_public_read_is_covered() -> None:
    """A new read method must join ``READS`` so the race test exercises it."""
    writes = {
        "add_credential",
        "remove_credential",
        "set_priority",
        "enable",
        "disable",
        "reset_health",
        "lease_first",
        "release_lease",
        "record_success",
        "record_failure",
        "insert_account",
        "update_account_row",
        "delete_account_row",
        "migrate_model_cells",
        "migrate_model_config",
        "close",
    }
    not_sql = {"now", "accounts", "pool", "pool_for_cell"}
    public = {name for name, _ in inspect.getmembers(CredentialStore, inspect.isfunction) if not name.startswith("_")}
    assert public - writes - not_sql == set(READS)


def _locked(node: ast.With) -> bool:
    for item in node.items:
        expr = item.context_expr
        if isinstance(expr, ast.Call):
            expr = expr.func
        if isinstance(expr, ast.Attribute) and expr.attr in {"_lock", "_immediate"}:
            return True
    return False


def test_every_connection_access_holds_the_lock() -> None:
    """Static guard: ``self._conn`` is only touched inside ``with self._lock`` / ``self._immediate()``.

    ``_immediate`` itself takes the lock, so its own body counts as locked.
    """
    tree = ast.parse(textwrap.dedent(inspect.getsource(CredentialStore)))
    unlocked: list[str] = []

    def visit(node: ast.AST, locked: bool, where: str) -> None:
        if isinstance(node, ast.FunctionDef):
            where = node.name
            locked = node.name == "_immediate"
        if isinstance(node, ast.With) and _locked(node):
            locked = True
        if (
            isinstance(node, ast.Attribute)
            and node.attr == "_conn"
            and isinstance(node.value, ast.Name)
            and node.value.id == "self"
            and not locked
        ):
            unlocked.append(f"{where}:{node.lineno}")
        for child in ast.iter_child_nodes(node):
            visit(child, locked, where)

    visit(tree, False, "<class>")
    assert unlocked == []
