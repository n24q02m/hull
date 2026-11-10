"""Import guards must track the extras declared in pyproject.toml.

An optional extra that ships an import guard lists the import names of its
dependencies there. This test derives those names from the root pyproject.toml
so adding or dropping a dependency without updating the guard (or vice versa)
fails CI. Only ``[web]`` carries a guard: the remaining extras (``identity``,
``invisible``, ``mteb``, ``llm``, ``dspy``) are optional seams whose import
error surfaces at use time.
"""

from __future__ import annotations

import tomllib
from pathlib import Path

import pytest
from packaging.requirements import Requirement

from hull_web import _extra as web_extra

PYPROJECT = Path(__file__).resolve().parents[1] / "pyproject.toml"

# Distributions whose import name is not the normalized distribution name.
IMPORT_NAME_OVERRIDES = {"pillow": "PIL"}


def _import_name(requirement: str) -> str:
    dist = Requirement(requirement).name.lower()
    return IMPORT_NAME_OVERRIDES.get(dist, dist.replace("-", "_"))


def _extra_import_names(extra: str) -> list[str]:
    data = tomllib.loads(PYPROJECT.read_text(encoding="utf-8"))
    return [_import_name(req) for req in data["project"]["optional-dependencies"][extra]]


@pytest.mark.parametrize(
    ("extra", "guard_modules"),
    [("web", web_extra.WEB_EXTRA_MODULES)],
)
def test_guard_modules_match_pyproject_extra(extra: str, guard_modules: tuple[str, ...]) -> None:
    assert list(guard_modules) == _extra_import_names(extra)
