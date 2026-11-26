"""Tests for Agent Skills provider attachment (FastMCP 4.2.2 SEP-2640)."""

from __future__ import annotations

from pathlib import Path

import pytest
from fastmcp import Client, FastMCP

from hull_core.server import SKILLS_EXTENSION_ID, attach_skills_provider


def _write_skill(root: Path, name: str, description: str = "Test skill description") -> Path:
    skill_dir = root / name
    skill_dir.mkdir(parents=True, exist_ok=True)
    (skill_dir / "SKILL.md").write_text(
        f"---\nname: {name}\ndescription: {description}\n---\n# {name}\n\nUse this skill.\n",
        encoding="utf-8",
    )
    refs_dir = skill_dir / "references"
    refs_dir.mkdir(exist_ok=True)
    (refs_dir / "guide.md").write_text("# Guide\n\nDetails.\n", encoding="utf-8")
    return skill_dir


def test_skills_extension_id() -> None:
    assert SKILLS_EXTENSION_ID == "io.modelcontextprotocol/skills"


def test_attach_skills_provider_missing_dir(tmp_path: Path) -> None:
    mcp = FastMCP("skills-missing")
    with pytest.raises(FileNotFoundError, match="does not exist"):
        attach_skills_provider(mcp, tmp_path / "nope")


def test_attach_skills_provider_not_a_directory(tmp_path: Path) -> None:
    file_path = tmp_path / "not-a-dir.txt"
    file_path.write_text("hi", encoding="utf-8")
    mcp = FastMCP("skills-file")
    with pytest.raises(NotADirectoryError, match="not a directory"):
        attach_skills_provider(mcp, file_path)


def test_attach_skills_provider_invalid_skill_fails(tmp_path: Path) -> None:
    bad_dir = tmp_path / "bad-name"
    bad_dir.mkdir()
    (bad_dir / "SKILL.md").write_text("---\nname: other-name\ndescription: mismatch\n---\n", encoding="utf-8")
    mcp = FastMCP("skills-invalid")
    with pytest.raises(ValueError, match="not valid Agent Skills"):
        attach_skills_provider(mcp, tmp_path)


async def test_attach_skills_provider_lists_and_reads_resources(tmp_path: Path) -> None:
    _write_skill(tmp_path, "hello-skill", "Greets callers")
    mcp = FastMCP("skills-server")
    provider = attach_skills_provider(mcp, tmp_path, reload=True)
    assert provider in mcp.providers

    async with Client(mcp) as client:
        resources = await client.list_resources()
        uris = {str(r.uri) for r in resources}
        assert "skill://hello-skill/SKILL.md" in uris
        assert "skill://hello-skill/references/guide.md" in uris

        content = await client.read_resource("skill://hello-skill/SKILL.md")
        assert len(content) == 1
        assert "Greets callers" in content[0].text
