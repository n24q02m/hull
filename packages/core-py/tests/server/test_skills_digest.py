"""SHA-256 integrity guard for skills directories (``skills.digest.json``)."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import sys
from pathlib import Path
from typing import Any

import pytest
from fastmcp import Client, FastMCP
from fastmcp.skills.provider import SkillsDirectoryProvider

from hull_core.server import (
    SKILLS_DIGEST_FILENAME,
    SKILLS_DIGEST_VERSION,
    SkillIntegrityError,
    attach_skills_provider,
    compute_skill_digests,
    verify_skill_digests,
    write_skill_digests,
)
from hull_core.server import skills as skills_module

# Shared with the core-ts suite: the same committed manifest must verify on both sides.
FIXTURE = Path(__file__).resolve().parents[3] / "core-ts" / "tests" / "fixtures" / "skills-digest"


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _write(path: Path, data: str | bytes) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data.encode("utf-8") if isinstance(data, str) else data)
    return path


def _skill_md(name: str, description: str = "Test skill description") -> str:
    return f"---\nname: {name}\ndescription: {description}\n---\n# {name}\n\nUse this skill.\n"


def _tree(root: Path) -> Path:
    """Two skills (one nested), a hidden file inside a skill, and files outside any skill."""
    _write(root / "hello-skill" / "SKILL.md", _skill_md("hello-skill", "Greets callers"))
    _write(root / "hello-skill" / "references" / "guide.md", "# Guide\n\nDetails.\n")
    _write(root / "hello-skill" / ".hidden", "hidden but served\n")
    _write(root / "ops" / "deploy" / "SKILL.md", _skill_md("deploy"))
    _write(root / "README.md", "not part of any skill\n")
    _write(root / ".cache" / "junk" / "SKILL.md", "hidden folders are never scanned\n")
    return root


def _symlink_or_skip(link: Path, target: Path) -> None:
    try:
        link.symlink_to(target, target_is_directory=target.is_dir())
    except OSError as exc:  # pragma: no cover - platform without symlink rights
        pytest.skip(f"symlinks unavailable: {exc}")


def _problems(exc: pytest.ExceptionInfo[SkillIntegrityError]) -> list[str]:
    return list(exc.value.problems)


# --- manifest generation ------------------------------------------------------------------


def test_constants() -> None:
    assert SKILLS_DIGEST_FILENAME == "skills.digest.json"
    assert SKILLS_DIGEST_VERSION == 1


def test_integrity_error_is_a_value_error() -> None:
    # FastMCP keeps the last good skills when a reload raises ValueError.
    assert issubclass(SkillIntegrityError, ValueError)


def test_compute_hashes_every_file_inside_skill_folders(tmp_path: Path) -> None:
    root = _tree(tmp_path)
    digests = compute_skill_digests(root)
    assert digests == {
        "hello-skill/.hidden": _sha(b"hidden but served\n"),
        "hello-skill/SKILL.md": _sha(_skill_md("hello-skill", "Greets callers").encode()),
        "hello-skill/references/guide.md": _sha(b"# Guide\n\nDetails.\n"),
        "ops/deploy/SKILL.md": _sha(_skill_md("deploy").encode()),
    }
    assert list(digests) == sorted(digests)


def test_compute_includes_a_skill_nested_in_another_skill(tmp_path: Path) -> None:
    _write(tmp_path / "outer" / "SKILL.md", _skill_md("outer"))
    _write(tmp_path / "outer" / "inner" / "SKILL.md", _skill_md("inner"))
    assert set(compute_skill_digests(tmp_path)) == {"outer/SKILL.md", "outer/inner/SKILL.md"}


def test_compute_ignores_a_skill_md_at_the_top_level(tmp_path: Path) -> None:
    _write(tmp_path / "SKILL.md", _skill_md("top"))
    assert compute_skill_digests(tmp_path) == {}


def test_compute_rejects_a_missing_directory(tmp_path: Path) -> None:
    with pytest.raises(SkillIntegrityError, match="does not exist"):
        compute_skill_digests(tmp_path / "nope")


def test_compute_rejects_a_file_path(tmp_path: Path) -> None:
    file_path = _write(tmp_path / "file.txt", "x")
    with pytest.raises(SkillIntegrityError, match="not a directory"):
        compute_skill_digests(file_path)


def test_compute_rejects_a_symlink_inside_a_skill(tmp_path: Path) -> None:
    root = _tree(tmp_path / "skills")
    _symlink_or_skip(root / "hello-skill" / "link.md", root / "README.md")
    with pytest.raises(SkillIntegrityError) as exc:
        compute_skill_digests(root)
    assert _problems(exc) == ["hello-skill/link.md: symlinks are not allowed"]


def test_compute_rejects_a_symlinked_skill_folder(tmp_path: Path) -> None:
    root = _tree(tmp_path / "skills")
    outside = tmp_path / "outside" / "linked"
    _write(outside / "SKILL.md", _skill_md("linked"))
    _symlink_or_skip(root / "linked", outside)
    with pytest.raises(SkillIntegrityError) as exc:
        compute_skill_digests(root)
    assert _problems(exc) == ["linked: symlinks are not allowed"]


def test_compute_rejects_a_fifo_inside_a_skill(tmp_path: Path) -> None:
    if sys.platform == "win32":
        pytest.skip("needs a POSIX FIFO")
    root = _tree(tmp_path)
    os.mkfifo(root / "hello-skill" / "pipe")
    with pytest.raises(SkillIntegrityError) as exc:
        compute_skill_digests(root)
    assert _problems(exc) == ["hello-skill/pipe: not a regular file"]


class _SpecialEntry:
    """Stands in for a device or socket entry, which Windows cannot create in a temp dir."""

    def __init__(self, directory: str, name: str) -> None:
        self.name = name
        self.path = os.path.join(directory, name)

    def is_symlink(self) -> bool:
        return False

    def is_dir(self, *, follow_symlinks: bool = True) -> bool:
        return False

    def is_file(self, *, follow_symlinks: bool = True) -> bool:
        return False


def test_compute_rejects_any_non_regular_entry(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    root = _tree(tmp_path)
    skill_dir = str(root / "hello-skill")
    real_scandir = os.scandir

    class _Scan:
        def __init__(self, directory: str) -> None:
            self._inner = real_scandir(directory)
            self._extra = [_SpecialEntry(directory, "socket")] if directory == skill_dir else []

        def __enter__(self) -> list[object]:
            return [*self._inner.__enter__(), *self._extra]

        def __exit__(self, *exc: object) -> None:
            self._inner.__exit__(*exc)

    monkeypatch.setattr(skills_module.os, "scandir", _Scan)
    with pytest.raises(SkillIntegrityError) as exc:
        compute_skill_digests(root)
    assert _problems(exc) == ["hello-skill/socket: not a regular file"]


def test_verify_reports_an_unreadable_directory(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    root = _tree(tmp_path)
    write_skill_digests(root)

    def denied(_path: object) -> object:
        raise PermissionError("denied")

    monkeypatch.setattr(skills_module.os, "scandir", denied)
    with pytest.raises(SkillIntegrityError, match="cannot be read: denied"):
        verify_skill_digests(root)


def test_compute_reports_an_unreadable_directory(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    root = _tree(tmp_path)

    def denied(_path: object) -> object:
        raise PermissionError("denied")

    monkeypatch.setattr(skills_module.os, "scandir", denied)
    with pytest.raises(SkillIntegrityError, match="cannot be read: denied"):
        compute_skill_digests(root)


def test_write_produces_the_canonical_manifest(tmp_path: Path) -> None:
    root = _tree(tmp_path)
    path = write_skill_digests(root)
    assert path == root.resolve() / SKILLS_DIGEST_FILENAME
    raw = path.read_bytes()
    assert raw.endswith(b"}\n")
    assert b"\r" not in raw
    payload = json.loads(raw)
    assert list(payload) == ["version", "files"]
    assert payload["version"] == 1
    assert payload["files"] == compute_skill_digests(root)
    # The manifest sits outside every skill folder, so writing it changes nothing.
    assert compute_skill_digests(root) == payload["files"]


def test_write_to_a_custom_path(tmp_path: Path) -> None:
    root = _tree(tmp_path / "skills")
    out = tmp_path / "build" / "digests.json"
    out.parent.mkdir()
    assert write_skill_digests(root, out) == out
    assert not (root / SKILLS_DIGEST_FILENAME).exists()
    assert json.loads(out.read_text("utf-8"))["files"] == compute_skill_digests(root)


# --- verification ---------------------------------------------------------------------------


def test_verify_accepts_a_valid_manifest(tmp_path: Path) -> None:
    root = _tree(tmp_path)
    write_skill_digests(root)
    expected = compute_skill_digests(root)
    assert verify_skill_digests(root) == expected
    assert verify_skill_digests(root, root / SKILLS_DIGEST_FILENAME) == expected
    assert verify_skill_digests(root, str(root / SKILLS_DIGEST_FILENAME)) == expected
    assert verify_skill_digests(root, expected) == expected


def test_verify_rejects_one_tampered_byte(tmp_path: Path) -> None:
    root = _tree(tmp_path)
    write_skill_digests(root)
    guide = root / "hello-skill" / "references" / "guide.md"
    data = bytearray(guide.read_bytes())
    data[0] ^= 0x01
    guide.write_bytes(bytes(data))
    with pytest.raises(SkillIntegrityError) as exc:
        verify_skill_digests(root)
    assert _problems(exc) == ["hello-skill/references/guide.md: digest mismatch"]
    assert "1 skill integrity problem(s)" in str(exc.value)


def test_verify_rejects_a_skill_md_without_an_entry(tmp_path: Path) -> None:
    root = _tree(tmp_path)
    write_skill_digests(root)
    _write(root / "new-skill" / "SKILL.md", _skill_md("new-skill"))
    with pytest.raises(SkillIntegrityError) as exc:
        verify_skill_digests(root)
    assert _problems(exc) == ["new-skill/SKILL.md: no digest entry"]


def test_verify_rejects_an_entry_whose_file_is_missing(tmp_path: Path) -> None:
    root = _tree(tmp_path)
    write_skill_digests(root)
    (root / "hello-skill" / "references" / "guide.md").unlink()
    with pytest.raises(SkillIntegrityError) as exc:
        verify_skill_digests(root)
    assert _problems(exc) == ["hello-skill/references/guide.md: listed in the manifest but missing"]


def test_verify_rejects_an_entry_that_is_a_directory(tmp_path: Path) -> None:
    root = _tree(tmp_path)
    digests = compute_skill_digests(root) | {"hello-skill/references": "0" * 64}
    with pytest.raises(SkillIntegrityError) as exc:
        verify_skill_digests(root, digests)
    assert _problems(exc) == ["hello-skill/references: listed in the manifest but not a regular file"]


def test_verify_checks_a_listed_file_outside_any_skill(tmp_path: Path) -> None:
    root = _tree(tmp_path)
    digests = compute_skill_digests(root) | {"README.md": _sha(b"not part of any skill\n")}
    assert verify_skill_digests(root, digests) == digests
    with pytest.raises(SkillIntegrityError, match="README.md: digest mismatch"):
        verify_skill_digests(root, digests | {"README.md": "0" * 64})


def test_verify_strict_rejects_an_unlisted_supporting_file(tmp_path: Path) -> None:
    root = _tree(tmp_path)
    write_skill_digests(root)
    _write(root / "hello-skill" / "scripts" / "run.sh", "echo injected\n")
    with pytest.raises(SkillIntegrityError) as exc:
        verify_skill_digests(root)
    assert _problems(exc) == ["hello-skill/scripts/run.sh: no digest entry"]


def test_verify_non_strict_allows_unlisted_supporting_files_only(tmp_path: Path) -> None:
    root = _tree(tmp_path)
    write_skill_digests(root)
    _write(root / "hello-skill" / "scripts" / "run.sh", "echo extra\n")
    verify_skill_digests(root, strict=False)
    # Listed files are still verified and every SKILL.md still needs an entry.
    _write(root / "other" / "SKILL.md", _skill_md("other"))
    (root / "ops" / "deploy" / "SKILL.md").write_text(_skill_md("deploy", "changed"), encoding="utf-8")
    with pytest.raises(SkillIntegrityError) as exc:
        verify_skill_digests(root, strict=False)
    assert _problems(exc) == ["ops/deploy/SKILL.md: digest mismatch", "other/SKILL.md: no digest entry"]


def test_verify_reports_every_problem_at_once(tmp_path: Path) -> None:
    root = _tree(tmp_path)
    write_skill_digests(root)
    (root / "hello-skill" / ".hidden").write_text("changed\n", encoding="utf-8")
    (root / "hello-skill" / "references" / "guide.md").unlink()
    _write(root / "added" / "SKILL.md", _skill_md("added"))
    with pytest.raises(SkillIntegrityError) as exc:
        verify_skill_digests(root)
    assert _problems(exc) == [
        "hello-skill/.hidden: digest mismatch",
        "hello-skill/references/guide.md: listed in the manifest but missing",
        "added/SKILL.md: no digest entry",
    ]
    assert str(exc.value).startswith(f"3 skill integrity problem(s) in {root.resolve()}:\n- ")


def test_verify_rejects_a_missing_directory(tmp_path: Path) -> None:
    with pytest.raises(SkillIntegrityError, match="Skills directory does not exist"):
        verify_skill_digests(tmp_path / "nope", {})


def test_verify_rejects_a_missing_manifest(tmp_path: Path) -> None:
    root = _tree(tmp_path)
    with pytest.raises(SkillIntegrityError, match="Skills digest manifest not found"):
        verify_skill_digests(root)
    with pytest.raises(SkillIntegrityError, match="Skills digest manifest not found"):
        verify_skill_digests(root, tmp_path / "elsewhere.json")


def test_verify_rejects_a_symlink_in_the_tree(tmp_path: Path) -> None:
    root = _tree(tmp_path / "skills")
    write_skill_digests(root)
    _symlink_or_skip(root / "alias", root / "hello-skill")
    with pytest.raises(SkillIntegrityError) as exc:
        verify_skill_digests(root)
    assert _problems(exc) == ["alias: symlinks are not allowed"]


_HEX = "a" * 64


@pytest.mark.parametrize(
    ("content", "reason"),
    [
        (b"{not json", "not valid JSON"),
        (b"\xff\xfe", "not valid JSON"),
        (b"[]", "must be a JSON object"),
        (json.dumps({"version": 2, "files": {}}).encode(), "version must be 1"),
        (json.dumps({"version": True, "files": {}}).encode(), "version must be 1"),
        (json.dumps({"version": "1", "files": {}}).encode(), "version must be 1"),
        (json.dumps({"files": {}}).encode(), "version must be 1"),
        (json.dumps({"version": 1}).encode(), "files must be an object"),
        (json.dumps({"version": 1, "files": []}).encode(), "files must be an object"),
        (json.dumps({"version": 1, "files": {}, "extra": 1}).encode(), "unexpected key 'extra'"),
        (json.dumps({"version": 1, "files": {"a/SKILL.md": "A" * 64}}).encode(), "64 lowercase hex"),
        (json.dumps({"version": 1, "files": {"a/SKILL.md": "ab"}}).encode(), "64 lowercase hex"),
        (json.dumps({"version": 1, "files": {"a/SKILL.md": 1}}).encode(), "64 lowercase hex"),
        (json.dumps({"version": 1, "files": {"../x": _HEX}}).encode(), "invalid path '../x'"),
        (json.dumps({"version": 1, "files": {"/abs": _HEX}}).encode(), "invalid path '/abs'"),
        (json.dumps({"version": 1, "files": {"a\\b": _HEX}}).encode(), "invalid path"),
        (json.dumps({"version": 1, "files": {"C:x": _HEX}}).encode(), "invalid path 'C:x'"),
        (json.dumps({"version": 1, "files": {"a//b": _HEX}}).encode(), "invalid path 'a//b'"),
        (json.dumps({"version": 1, "files": {"a/./b": _HEX}}).encode(), "invalid path 'a/./b'"),
        (json.dumps({"version": 1, "files": {"": _HEX}}).encode(), "invalid path ''"),
        (json.dumps({"version": 1, "files": {"a\u0000b": _HEX}}).encode(), "invalid path"),
    ],
)
def test_verify_rejects_an_invalid_manifest(tmp_path: Path, content: bytes, reason: str) -> None:
    root = _tree(tmp_path)
    (root / SKILLS_DIGEST_FILENAME).write_bytes(content)
    with pytest.raises(SkillIntegrityError, match="Invalid skills digest manifest") as exc:
        verify_skill_digests(root)
    assert reason in str(exc.value)


def test_verify_accepts_version_written_as_a_float(tmp_path: Path) -> None:
    # JSON has one number type; core-ts cannot tell 1.0 from 1, so neither side does.
    root = _tree(tmp_path)
    payload = {"version": 1.0, "files": compute_skill_digests(root)}
    (root / SKILLS_DIGEST_FILENAME).write_text(json.dumps(payload), encoding="utf-8")
    verify_skill_digests(root)


def test_verify_rejects_an_unreadable_manifest(tmp_path: Path) -> None:
    root = _tree(tmp_path)
    (root / SKILLS_DIGEST_FILENAME).mkdir()
    with pytest.raises(SkillIntegrityError, match="cannot be read"):
        verify_skill_digests(root)


@pytest.mark.parametrize(
    ("mapping", "reason"),
    [
        ({"a/SKILL.md": "nothex"}, "64 lowercase hex"),
        ({"../a": _HEX}, "invalid path '../a'"),
        ({1: _HEX}, "invalid path 1"),
    ],
)
def test_verify_rejects_an_invalid_mapping(tmp_path: Path, mapping: dict[object, object], reason: str) -> None:
    root = _tree(tmp_path)
    with pytest.raises(SkillIntegrityError, match="Invalid skills digest mapping") as exc:
        verify_skill_digests(root, mapping)  # ty: ignore[invalid-argument-type]
    assert reason in str(exc.value)


# --- attach_skills_provider -----------------------------------------------------------------


async def _uris(mcp: FastMCP) -> set[str]:
    async with Client(mcp) as client:
        return {str(r.uri) for r in await client.list_resources()}


async def test_attach_without_a_manifest_keeps_the_old_behaviour(tmp_path: Path) -> None:
    root = _tree(tmp_path)
    mcp = FastMCP("no-manifest")
    provider = attach_skills_provider(mcp, root)
    assert type(provider) is SkillsDirectoryProvider
    assert "skill://hello-skill/SKILL.md" in await _uris(mcp)


async def test_attach_verifies_a_manifest_found_in_the_directory(tmp_path: Path) -> None:
    root = _tree(tmp_path)
    write_skill_digests(root)
    mcp = FastMCP("auto-manifest")
    provider = attach_skills_provider(mcp, root)
    assert provider in mcp.providers
    uris = await _uris(mcp)
    assert {"skill://hello-skill/SKILL.md", "skill://ops/deploy/SKILL.md"} <= uris


def test_attach_mounts_nothing_when_a_skill_was_tampered_with(tmp_path: Path) -> None:
    root = _tree(tmp_path)
    write_skill_digests(root)
    (root / "hello-skill" / "SKILL.md").write_text(_skill_md("hello-skill", "Evil"), encoding="utf-8")
    mcp = FastMCP("tampered")
    before = list(mcp.providers)
    with pytest.raises(SkillIntegrityError, match="hello-skill/SKILL.md: digest mismatch"):
        attach_skills_provider(mcp, root)
    assert list(mcp.providers) == before


def test_attach_mounts_nothing_when_a_skill_has_no_entry(tmp_path: Path) -> None:
    root = _tree(tmp_path)
    write_skill_digests(root)
    _write(root / "rogue" / "SKILL.md", _skill_md("rogue"))
    mcp = FastMCP("unlisted")
    before = list(mcp.providers)
    with pytest.raises(SkillIntegrityError, match="rogue/SKILL.md: no digest entry"):
        attach_skills_provider(mcp, root)
    assert list(mcp.providers) == before


def test_attach_rejects_an_invalid_manifest_found_in_the_directory(tmp_path: Path) -> None:
    root = _tree(tmp_path)
    (root / SKILLS_DIGEST_FILENAME).write_text("{}", encoding="utf-8")
    with pytest.raises(SkillIntegrityError, match="Invalid skills digest manifest"):
        attach_skills_provider(FastMCP("invalid"), root)


def test_attach_treats_a_dangling_manifest_symlink_as_present(tmp_path: Path) -> None:
    root = _tree(tmp_path / "skills")
    _symlink_or_skip(root / SKILLS_DIGEST_FILENAME, tmp_path / "gone.json")
    with pytest.raises(SkillIntegrityError, match="Skills digest manifest not found"):
        attach_skills_provider(FastMCP("dangling"), root)


def test_attach_keeps_the_guard_on_when_the_manifest_cannot_be_looked_up(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = _tree(tmp_path).resolve()
    write_skill_digests(root)
    (root / "hello-skill" / "SKILL.md").write_text(_skill_md("hello-skill", "Evil"), encoding="utf-8")
    manifest = root / SKILLS_DIGEST_FILENAME
    real_lstat = os.lstat

    def lstat(path: str | os.PathLike[str], **kwargs: Any) -> os.stat_result:
        if os.fspath(path) == str(manifest):
            raise PermissionError("denied")
        return real_lstat(path, **kwargs)

    monkeypatch.setattr(skills_module.os, "lstat", lstat)
    mcp = FastMCP("lookup-denied")
    before = list(mcp.providers)
    with pytest.raises(SkillIntegrityError, match="hello-skill/SKILL.md: digest mismatch"):
        attach_skills_provider(mcp, root)
    assert list(mcp.providers) == before


def test_attach_require_digests_without_a_manifest_fails(tmp_path: Path) -> None:
    root = _tree(tmp_path)
    with pytest.raises(SkillIntegrityError, match="Skills digest manifest not found"):
        attach_skills_provider(FastMCP("required"), root, require_digests=True)


def test_attach_with_an_explicit_option_rejects_a_missing_directory(tmp_path: Path) -> None:
    with pytest.raises(SkillIntegrityError, match="Skills directory does not exist"):
        attach_skills_provider(FastMCP("gone"), tmp_path / "nope", require_digests=True)
    with pytest.raises(SkillIntegrityError, match="Skills directory does not exist"):
        attach_skills_provider(FastMCP("gone"), tmp_path / "nope", digests={})
    file_path = _write(tmp_path / "file.txt", "x")
    with pytest.raises(SkillIntegrityError, match="not a directory"):
        attach_skills_provider(FastMCP("file"), file_path, digests={})


async def test_attach_with_an_in_memory_mapping(tmp_path: Path) -> None:
    root = _tree(tmp_path)
    digests = compute_skill_digests(root)
    mcp = FastMCP("mapping")
    attach_skills_provider(mcp, root, digests=digests)
    assert "skill://ops/deploy/SKILL.md" in await _uris(mcp)
    bad = digests | {"ops/deploy/SKILL.md": "0" * 64}
    with pytest.raises(SkillIntegrityError, match="ops/deploy/SKILL.md: digest mismatch"):
        attach_skills_provider(FastMCP("mapping-bad"), root, digests=bad)


async def test_attach_with_a_manifest_path_outside_the_directory(tmp_path: Path) -> None:
    root = _tree(tmp_path / "skills")
    manifest = write_skill_digests(root, tmp_path / "pinned.json")
    mcp = FastMCP("external")
    attach_skills_provider(mcp, root, digests=manifest)
    assert "skill://hello-skill/SKILL.md" in await _uris(mcp)


async def test_attach_non_strict_allows_an_unlisted_supporting_file(tmp_path: Path) -> None:
    root = _tree(tmp_path)
    write_skill_digests(root)
    _write(root / "hello-skill" / "notes.md", "extra\n")
    with pytest.raises(SkillIntegrityError, match="hello-skill/notes.md: no digest entry"):
        attach_skills_provider(FastMCP("strict"), root)
    mcp = FastMCP("lenient")
    attach_skills_provider(mcp, root, strict_digests=False)
    assert "skill://hello-skill/notes.md" in await _uris(mcp)


def test_attach_checks_the_bytes_the_provider_loaded(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A file swapped between the disk check and the provider's read is still caught."""
    root = _tree(tmp_path)
    write_skill_digests(root)
    (root / "hello-skill" / "SKILL.md").write_text(_skill_md("hello-skill", "Swapped"), encoding="utf-8")
    monkeypatch.setattr(skills_module, "_verify_tree", lambda *_args, **_kwargs: None)
    mcp = FastMCP("race")
    before = list(mcp.providers)
    with pytest.raises(SkillIntegrityError, match="hello-skill/SKILL.md: digest mismatch"):
        attach_skills_provider(mcp, root)
    assert list(mcp.providers) == before


def test_attach_checks_coverage_of_the_bytes_the_provider_loaded(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = _tree(tmp_path)
    write_skill_digests(root)
    _write(root / "hello-skill" / "late.md", "added after the disk check\n")
    _write(root / "late-skill" / "SKILL.md", _skill_md("late-skill"))
    monkeypatch.setattr(skills_module, "_verify_tree", lambda *_args, **_kwargs: None)
    with pytest.raises(SkillIntegrityError) as exc:
        attach_skills_provider(FastMCP("race-strict"), root)
    assert _problems(exc) == ["hello-skill/late.md: no digest entry", "late-skill/SKILL.md: no digest entry"]
    with pytest.raises(SkillIntegrityError) as exc:
        attach_skills_provider(FastMCP("race-lenient"), root, strict_digests=False)
    assert _problems(exc) == ["late-skill/SKILL.md: no digest entry"]


def test_attach_checks_the_provider_loaded_every_listed_file(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    root = _tree(tmp_path)
    write_skill_digests(root)
    (root / "hello-skill" / "references" / "guide.md").unlink()
    monkeypatch.setattr(skills_module, "_verify_tree", lambda *_args, **_kwargs: None)
    with pytest.raises(SkillIntegrityError) as exc:
        attach_skills_provider(FastMCP("race-missing"), root)
    assert _problems(exc) == ["hello-skill/references/guide.md: listed in the manifest but missing"]


async def test_reload_keeps_serving_the_last_verified_skills(tmp_path: Path) -> None:
    root = _tree(tmp_path)
    write_skill_digests(root)
    mcp = FastMCP("reload")
    attach_skills_provider(mcp, root, reload=True)
    skill_md = root / "hello-skill" / "SKILL.md"
    original = skill_md.read_text("utf-8")
    skill_md.write_text(_skill_md("hello-skill", "Tampered after start"), encoding="utf-8")
    async with Client(mcp) as client:
        content = await client.read_resource("skill://hello-skill/SKILL.md")
    assert content[0].text == original
    assert "Tampered" not in content[0].text


async def test_reload_verifies_the_directory_again_on_every_scan(tmp_path: Path) -> None:
    root = _tree(tmp_path)
    write_skill_digests(root)
    mcp = FastMCP("reload-disk")
    attach_skills_provider(mcp, root, reload=True)
    # Removing a whole skill leaves nothing the loaded-bytes check could see, so only the
    # disk check on re-scan catches it; the failing scan keeps the verified skills.
    shutil.rmtree(root / "ops")
    assert {"skill://hello-skill/SKILL.md", "skill://ops/deploy/SKILL.md"} <= await _uris(mcp)


# --- cross-language fixture -----------------------------------------------------------------


def test_fixture_manifest_verifies() -> None:
    committed = json.loads((FIXTURE / SKILLS_DIGEST_FILENAME).read_text("utf-8"))["files"]
    assert verify_skill_digests(FIXTURE) == committed


def test_fixture_manifest_matches_what_this_side_generates(tmp_path: Path) -> None:
    out = write_skill_digests(FIXTURE, tmp_path / "regenerated.json")
    assert out.read_bytes() == (FIXTURE / SKILLS_DIGEST_FILENAME).read_bytes()


async def test_fixture_mounts_with_the_committed_manifest() -> None:
    mcp = FastMCP("fixture")
    attach_skills_provider(mcp, FIXTURE)
    uris = await _uris(mcp)
    assert {"skill://hello-skill/SKILL.md", "skill://ops/deploy/SKILL.md"} <= uris
