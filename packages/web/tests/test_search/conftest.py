"""Shared test fixtures for hull_web.search."""

from __future__ import annotations

import pytest


@pytest.fixture(autouse=True)
def _clean_runner_state():
    """Reset module-level state in hull_web.search.runner between tests."""
    import hull_web.search.runner as mod

    def _reset():
        mod._searxng_process = None
        mod._searxng_port = None
        mod._searxng_docker_container = None
        mod._searxng_settings_path = None
        mod._restart_count = 0
        mod._last_restart_time = 0.0
        mod._is_owner = False
        mod._startup_lock = None
        mod._docker_lock = None

    _reset()
    yield
    _reset()


@pytest.fixture(autouse=True)
def _clean_client_state():
    """Reset module-level state in hull_web.search.client between tests."""
    import hull_web.search.client as mod

    def _reset():
        mod._shared_clients.clear()

    _reset()
    yield
    _reset()


@pytest.fixture
def tmp_config_dir(tmp_path, monkeypatch):
    """Use a temporary config directory for SearchRunner."""
    import hull_web.search.runner as mod

    config_dir = tmp_path / ".hull" / "searxng"
    config_dir.mkdir(parents=True, exist_ok=True)
    monkeypatch.setattr(mod, "_CONFIG_DIR", config_dir)
    monkeypatch.setattr(mod, "_DISCOVERY_FILE", config_dir / "searxng_instance.json")
    return config_dir


@pytest.fixture
def tmp_discovery(tmp_path, monkeypatch):
    """Use a temporary discovery file for SearchRunner."""
    import hull_web.search.runner as mod

    discovery = tmp_path / "searxng_instance.json"
    monkeypatch.setattr(mod, "_DISCOVERY_FILE", discovery)
    return discovery
