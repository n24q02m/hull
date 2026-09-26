"""Local SQLite (WAL) storage under ``~/.hull/`` (spec §3).

One database file, WAL journal, namespace-scoped tables: every row carries the
caller's namespace so mode-3 users never see each other's data. Backup/sync is
rclone OUTSIDE this repo — there is no sync code here by design.
"""

from __future__ import annotations

import sqlite3
import threading
from pathlib import Path

_SCHEMA = """
CREATE TABLE IF NOT EXISTS kv (
    namespace TEXT NOT NULL,
    key TEXT NOT NULL,
    value TEXT NOT NULL,
    updated_at TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ','now')),
    PRIMARY KEY (namespace, key)
);
"""


class HullDatabase:
    """Small WAL-mode SQLite facade; thread-safe via a write lock."""

    def __init__(self, path: Path | None = None) -> None:
        if path is None:
            from hull_core.config.settings import default_config_dir

            path = default_config_dir() / "hull.db"
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        self.path = path
        self._lock = threading.Lock()
        self._conn = sqlite3.connect(str(path), check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute("PRAGMA busy_timeout=5000")
        with self._lock:
            self._conn.executescript(_SCHEMA)
            self._conn.commit()

    @property
    def journal_mode(self) -> str:
        return str(self._conn.execute("PRAGMA journal_mode").fetchone()[0])

    def kv_put(self, namespace: str, key: str, value: str) -> None:
        with self._lock:
            self._conn.execute(
                "INSERT INTO kv (namespace, key, value) VALUES (?, ?, ?) "
                "ON CONFLICT(namespace, key) DO UPDATE SET value = excluded.value, "
                "updated_at = strftime('%Y-%m-%dT%H:%M:%fZ','now')",
                (namespace, key, value),
            )
            self._conn.commit()

    def kv_get(self, namespace: str, key: str) -> str | None:
        row = self._conn.execute(
            "SELECT value FROM kv WHERE namespace = ? AND key = ?", (namespace, key)
        ).fetchone()
        return None if row is None else str(row["value"])

    def kv_list(self, namespace: str) -> dict[str, str]:
        rows = self._conn.execute(
            "SELECT key, value FROM kv WHERE namespace = ? ORDER BY key", (namespace,)
        ).fetchall()
        return {str(r["key"]): str(r["value"]) for r in rows}

    def close(self) -> None:
        with self._lock:
            self._conn.close()
