"""Agent Skills provider helpers built on FastMCP 4.2.2 (SEP-2640)."""

from __future__ import annotations

from pathlib import Path
from typing import Final

from fastmcp import FastMCP
from fastmcp.skills.provider import SkillsDirectoryProvider

SKILLS_EXTENSION_ID: Final[str] = "io.modelcontextprotocol/skills"


def attach_skills_provider(
    mcp: FastMCP,
    skills_dir: str | Path,
    *,
    reload: bool = False,
) -> SkillsDirectoryProvider:
    """Validate ``skills_dir`` and attach a :class:`SkillsDirectoryProvider` to ``mcp``."""
    resolved = Path(skills_dir).expanduser().resolve()
    if not resolved.exists():
        raise FileNotFoundError(f"Skills directory does not exist: {resolved}")
    if not resolved.is_dir():
        raise NotADirectoryError(f"Skills path is not a directory: {resolved}")

    provider = SkillsDirectoryProvider(resolved, reload=reload)
    mcp.add_provider(provider)
    return provider
