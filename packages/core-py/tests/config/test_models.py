"""Per-task model cells: OpenRouter defaults, per-task env override."""

from __future__ import annotations

import pytest

from hull_core.config.models import (
    DEFAULT_BASE_URL,
    DEFAULT_MODELS,
    TASKS,
    api_key_env,
    model_cell_from_config,
    resolve_model_cells,
)


def test_default_cells_pre_wired_openrouter() -> None:
    cells = resolve_model_cells({})
    assert set(cells) == set(TASKS)
    for task, cell in cells.items():
        assert cell.base_url == DEFAULT_BASE_URL
        assert cell.model == DEFAULT_MODELS[task]
        assert cell.api_key == ""
        assert cell.configured is False


def test_config_table_overrides(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("HULL_EMBED_API_KEY", raising=False)
    cell = model_cell_from_config(
        "embed",
        {"base_url": "https://api.example.ai/v1/", "api_key": "k1", "model": "my-embed"},
    )
    assert cell.base_url == "https://api.example.ai/v1"  # trailing slash stripped
    assert cell.model == "my-embed"
    assert cell.api_key == "k1"
    assert cell.configured is True


def test_env_key_wins_over_file(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("HULL_CHAT_API_KEY", "env-key")
    cell = model_cell_from_config("chat", {"api_key": "file-key"})
    assert cell.api_key == "env-key"


def test_api_key_env_names() -> None:
    assert api_key_env("embed") == "HULL_EMBED_API_KEY"
    assert api_key_env("jev_score") == "HULL_JEV_SCORE_API_KEY"


def test_unknown_task_rejected() -> None:
    with pytest.raises(ValueError, match="unknown model task"):
        model_cell_from_config("poetry", None)
