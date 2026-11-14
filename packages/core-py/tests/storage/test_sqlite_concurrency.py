"""``HullDatabase`` reads share its one SQLite connection, so they take its lock too.

The database keeps a single ``check_same_thread=False`` connection. Two threads
running a statement on it at once trip ``sqlite3.InterfaceError: bad parameter
or other API misuse``; reads used to skip the lock that only writes took. Each
read is hammered by reader threads next to a writer; any exception fails.
"""

from __future__ import annotations

import ast
import inspect
import textwrap
import threading
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest

from hull_core.storage.sqlite import HullDatabase

ITERATIONS = 3000
READERS = 3

READS: dict[str, Callable[[HullDatabase], Any]] = {
    "journal_mode": lambda db: db.journal_mode,
    "kv_get": lambda db: db.kv_get("ns", "k1"),
    "kv_list": lambda db: db.kv_list("ns"),
}


@pytest.fixture
def db(tmp_path: Path):  # noqa: ANN201
    database = HullDatabase(tmp_path / "hull.db")
    for index in range(20):
        database.kv_put("ns", f"k{index}", "v" * 100)
    yield database
    database.close()


def _hammer(db: HullDatabase, read: Callable[[HullDatabase], Any]) -> list[str]:
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
                read(db)
            except BaseException as exc:  # surface every thread failure in the test result
                record(exc)
                return

    def writer() -> None:
        start.wait(timeout=10)
        value = 0
        while not stop.is_set():
            try:
                value += 1
                db.kv_put("ns", "k1", str(value))
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
def test_read_is_safe_against_concurrent_reads_and_writes(db: HullDatabase, name: str) -> None:
    assert _hammer(db, READS[name]) == []


def test_every_public_read_is_covered() -> None:
    """A new read must join ``READS`` so the race test exercises it."""
    writes = {"kv_put", "close"}
    public = {name for name, _ in inspect.getmembers(HullDatabase) if not name.startswith("_")} - {"path"}
    assert public - writes == set(READS)


def test_every_connection_access_holds_the_lock() -> None:
    """Static guard: ``self._conn`` is only touched inside ``with self._lock``."""
    tree = ast.parse(textwrap.dedent(inspect.getsource(HullDatabase)))
    unlocked: list[str] = []

    def holds_lock(node: ast.With) -> bool:
        return any(
            isinstance(item.context_expr, ast.Attribute) and item.context_expr.attr == "_lock" for item in node.items
        )

    def visit(node: ast.AST, locked: bool, where: str) -> None:
        if isinstance(node, ast.FunctionDef):
            where = node.name
            locked = False
        if isinstance(node, ast.With) and holds_lock(node):
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
