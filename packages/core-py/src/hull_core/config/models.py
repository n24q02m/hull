"""Per-task model cells: embed / rerank / chat / jev_score (spec §3, §4).

Each task resolves to one cell of ``base_url + api_key + model``. Config file
values come from ``[models.<task>]``; the api_key may be overridden per task
via ``HULL_<TASK>_API_KEY`` (env wins — hosts inject keys at start; keys are
host-only material and never end-user supplied).
"""

from __future__ import annotations

import os
from dataclasses import dataclass

TASKS = ("embed", "rerank", "chat", "jev_score")

DEFAULT_BASE_URL = "https://openrouter.ai/api/v1"
DEFAULT_MODELS = {
    "embed": "voyage-4-lite",
    "rerank": "voyage-2.5-lite",
    "chat": "z-ai/glm-5.3-flash",
    "jev_score": "z-ai/glm-5.3-flash",
}


def api_key_env(task: str) -> str:
    return f"HULL_{task.upper()}_API_KEY"


@dataclass(frozen=True)
class ModelCell:
    task: str
    base_url: str
    api_key: str
    model: str

    @property
    def configured(self) -> bool:
        """True when the host has supplied a key (directly or via env)."""
        return bool(self.api_key)


def model_cell_from_config(task: str, table: dict | None, *, env: dict[str, str] | None = None) -> ModelCell:
    """Build one task's cell from its ``[models.<task>]`` table + env override."""
    if task not in TASKS:
        raise ValueError(f"unknown model task {task!r}; expected one of {TASKS}")
    table = table or {}
    # Read through a fresh local: `env` keeps its declared `dict[str, str] | None`
    # type while os.environ is an `_Environ[str]` mapping.
    env_map = os.environ if env is None else env
    base_url = str(table.get("base_url", DEFAULT_BASE_URL)).rstrip("/")
    model = str(table.get("model", DEFAULT_MODELS[task]))
    api_key = str(env_map.get(api_key_env(task), table.get("api_key", "")) or "")
    return ModelCell(task=task, base_url=base_url, api_key=api_key, model=model)


def resolve_model_cells(models_config: dict | None, *, env: dict[str, str] | None = None) -> dict[str, ModelCell]:
    """Resolve all four task cells; unknown [models.*] tables are ignored."""
    models_config = models_config or {}
    return {task: model_cell_from_config(task, models_config.get(task), env=env) for task in TASKS}
