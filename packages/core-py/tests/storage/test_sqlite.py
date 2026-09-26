"""WAL SQLite storage + namespace isolation."""

from __future__ import annotations

from hull_core.storage.sqlite import HullDatabase


def test_wal_mode_enabled(tmp_path) -> None:
    db = HullDatabase(tmp_path / "hull.db")
    try:
        assert db.journal_mode == "wal"
    finally:
        db.close()


def test_kv_namespaced_isolation(tmp_path) -> None:
    db = HullDatabase(tmp_path / "hull.db")
    try:
        db.kv_put("alice", "topic", "alpha")
        db.kv_put("bob", "topic", "beta")
        assert db.kv_get("alice", "topic") == "alpha"
        assert db.kv_get("bob", "topic") == "beta"
        assert db.kv_list("alice") == {"topic": "alpha"}
        # cross-namespace read of an absent key is None, never the other user's value
        assert db.kv_get("carol", "topic") is None
        db.kv_put("alice", "topic", "alpha2")
        assert db.kv_get("alice", "topic") == "alpha2"
    finally:
        db.close()


def test_creates_parent_dirs(tmp_path) -> None:
    db = HullDatabase(tmp_path / "deep" / "nest" / "hull.db")
    try:
        assert db.path.is_file()
    finally:
        db.close()
