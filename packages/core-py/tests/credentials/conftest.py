"""Shared fixtures for provider key pool tests."""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import pytest

from helpers import Clock
from hull_core.credentials.store import CredentialStore


@pytest.fixture
def clock() -> Clock:
    return Clock()


@pytest.fixture
def store(tmp_path: Path, clock: Clock) -> Iterator[CredentialStore]:
    instance = CredentialStore(tmp_path / "hull.db", clock=clock)
    yield instance
    instance.close()


@pytest.fixture
def db_path(tmp_path: Path) -> Path:
    return tmp_path / "hull.db"
