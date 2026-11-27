"""Agent Skills provider helpers built on FastMCP 4.2.2 (SEP-2640).

Optional integrity guard: a ``skills.digest.json`` manifest pins the SHA-256 of every
file inside each skill folder. When the guard is on, a skill whose bytes differ from the
manifest, or a ``SKILL.md`` with no manifest entry, is never mounted. The manifest format
and the file walk are shared with ``@n24q02m/hull`` (core-ts), so one manifest verifies
in both languages.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import stat
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, Final

from fastmcp import FastMCP
from fastmcp.skills.provider import SkillsDirectoryProvider
from fastmcp.skills.skill import Skill

SKILLS_EXTENSION_ID: Final[str] = "io.modelcontextprotocol/skills"

SKILLS_DIGEST_FILENAME: Final[str] = "skills.digest.json"
"""Default manifest name, looked up at the root of the skills directory."""

SKILLS_DIGEST_VERSION: Final[int] = 1

_SKILL_FILE: Final[str] = "SKILL.md"
_HEX_DIGEST = re.compile(r"[0-9a-f]{64}")
_DRIVE_PREFIX = re.compile(r"[A-Za-z]:")
_MANIFEST_KEYS: Final[tuple[str, ...]] = ("version", "files")

DigestSource = str | os.PathLike[str] | Mapping[str, str]


class SkillIntegrityError(ValueError):
    """A skills directory failed its digest check; nothing from it is mounted.

    ``problems`` lists one ``"<relative path>: <reason>"`` line per failing file and is
    empty when the failure is about the directory or the manifest itself. It subclasses
    ``ValueError`` so that a ``reload=True`` re-scan which fails the check keeps serving
    the last verified skills instead of crashing the request.
    """

    def __init__(self, message: str, problems: Sequence[str] = ()) -> None:
        super().__init__(message)
        self.problems: tuple[str, ...] = tuple(problems)


def compute_skill_digests(skills_dir: str | os.PathLike[str]) -> dict[str, str]:
    """Return ``{relative POSIX path: SHA-256 hex}`` for every file inside each skill folder.

    A skill folder is a non-hidden folder below ``skills_dir`` that holds a ``SKILL.md``;
    every regular file under it counts, hidden ones included. Files outside skill folders
    are ignored. Any symlink in the walk, or a special file inside a skill, raises
    :class:`SkillIntegrityError`.
    """
    root = _existing_dir(skills_dir)
    try:
        files = _walk_or_raise(root)
        return {rel: _sha256(root, rel) for rel in files}
    except OSError as exc:
        raise _unreadable(root, exc) from exc


def write_skill_digests(skills_dir: str | os.PathLike[str], output: str | os.PathLike[str] | None = None) -> Path:
    """Write the canonical manifest for ``skills_dir`` and return its path.

    The default location is ``<skills_dir>/skills.digest.json``, which sits outside every
    skill folder and so never lists itself. The output is byte-for-byte what core-ts writes.
    """
    digests = compute_skill_digests(skills_dir)
    target = Path(skills_dir).expanduser().resolve() / SKILLS_DIGEST_FILENAME if output is None else Path(output)
    target.write_bytes(_serialize(digests))
    return target


def verify_skill_digests(
    skills_dir: str | os.PathLike[str], digests: DigestSource | None = None, *, strict: bool = True
) -> dict[str, str]:
    """Check ``skills_dir`` against a manifest and return the verified digests.

    ``digests`` is a manifest path, an in-memory ``{path: hex}`` mapping, or ``None`` for
    ``<skills_dir>/skills.digest.json`` (which must then exist). Every entry must be a
    regular file with a matching digest and every ``SKILL.md`` must have an entry. With
    ``strict`` (the default) every file inside a skill folder needs an entry too; with
    ``strict=False`` unlisted supporting files are allowed. All problems are reported in
    one :class:`SkillIntegrityError`.
    """
    root = _existing_dir(skills_dir)
    expected = _expected_digests(root, digests)
    _verify_tree(root, expected, strict=strict)
    return expected


def attach_skills_provider(
    mcp: FastMCP,
    skills_dir: str | Path,
    *,
    reload: bool = False,
    digests: DigestSource | None = None,
    require_digests: bool = False,
    strict_digests: bool = True,
) -> SkillsDirectoryProvider:
    """Validate ``skills_dir`` and attach a :class:`SkillsDirectoryProvider` to ``mcp``.

    The integrity guard is on when ``digests`` is given, when ``require_digests`` is set,
    or when ``<skills_dir>/skills.digest.json`` exists. It then verifies the directory
    (strict unless ``strict_digests=False``) and the bytes the provider loaded before
    anything is mounted, and raises :class:`SkillIntegrityError` on any problem. With
    ``reload=True`` every re-scan is checked against the same digests; one that fails keeps
    the last verified skills serving. With no manifest and no option the directory is
    mounted unchecked, as before.
    """
    resolved = Path(skills_dir).expanduser().resolve()
    guarded = digests is not None or require_digests or _manifest_present(resolved / SKILLS_DIGEST_FILENAME)
    if not guarded:
        if not resolved.exists():
            raise FileNotFoundError(f"Skills directory does not exist: {resolved}")
        if not resolved.is_dir():
            raise NotADirectoryError(f"Skills path is not a directory: {resolved}")
        provider = SkillsDirectoryProvider(resolved, reload=reload)
    else:
        root = _existing_dir(resolved)
        expected = _expected_digests(root, digests)
        provider = _VerifiedSkillsDirectoryProvider(root, expected, strict=strict_digests, reload=reload)
    mcp.add_provider(provider)
    return provider


class _VerifiedSkillsDirectoryProvider(SkillsDirectoryProvider):
    """Verifies the directory on every scan, then the bytes that scan loaded.

    ``SkillsDirectoryProvider`` reads every skill through ``_scan`` both at construction and
    on each ``reload=True`` request; the tests pin that hook. Checking the loaded bytes as
    well means a file swapped between the disk check and the read is never served.
    """

    def __init__(self, root: Path, digests: Mapping[str, str], *, strict: bool, reload: bool) -> None:
        # The base constructor runs the first scan, so the expectations must exist first.
        self._digests = dict(digests)
        self._strict = strict
        self._digest_root = root
        super().__init__(root, reload=reload)

    def _scan(self) -> dict[str, Skill]:
        _verify_tree(self._digest_root, self._digests, strict=self._strict)
        skills = super()._scan()
        problems = _check_loaded(skills, self._digests, strict=self._strict)
        if problems:
            raise _integrity_error(self._digest_root, problems)
        return skills


def _manifest_present(path: Path) -> bool:
    """Whether the guard must turn on for ``path``; only a definite "not there" says no.

    A dangling symlink counts as present (and then fails to load), and so does a path that
    cannot be looked up, so an unreadable manifest never mounts the skills unchecked.
    """
    try:
        os.lstat(path)
    except (FileNotFoundError, NotADirectoryError):
        return False
    except OSError:
        return True
    return True


def _existing_dir(skills_dir: str | os.PathLike[str]) -> Path:
    root = Path(skills_dir).expanduser().resolve()
    if not root.exists():
        raise SkillIntegrityError(f"Skills directory does not exist: {root}")
    if not root.is_dir():
        raise SkillIntegrityError(f"Skills path is not a directory: {root}")
    return root


def _expected_digests(root: Path, digests: DigestSource | None) -> dict[str, str]:
    if digests is None:
        return _load_manifest(root / SKILLS_DIGEST_FILENAME)
    if isinstance(digests, Mapping):
        return _validated(digests, "Invalid skills digest mapping")
    return _load_manifest(Path(digests).expanduser().resolve())


def _load_manifest(path: Path) -> dict[str, str]:
    try:
        raw = path.read_bytes()
    except FileNotFoundError as exc:
        raise SkillIntegrityError(f"Skills digest manifest not found: {path}") from exc
    except OSError as exc:
        raise SkillIntegrityError(f"Skills digest manifest {path} cannot be read: {exc}") from exc
    label = f"Invalid skills digest manifest {path}"
    try:
        payload = json.loads(raw.decode("utf-8"))
    except ValueError as exc:  # JSONDecodeError and UnicodeDecodeError
        raise SkillIntegrityError(f"{label}: not valid JSON ({exc})") from exc
    if not isinstance(payload, dict):
        raise SkillIntegrityError(f"{label}: must be a JSON object")
    for key in payload:
        if key not in _MANIFEST_KEYS:
            raise SkillIntegrityError(f"{label}: unexpected key {key!r}")
    version = payload.get("version")
    # JSON has a single number type, so 1.0 is accepted as it is in core-ts; true is not.
    if isinstance(version, bool) or not isinstance(version, int | float) or version != SKILLS_DIGEST_VERSION:
        raise SkillIntegrityError(f"{label}: version must be {SKILLS_DIGEST_VERSION}")
    files = payload.get("files")
    if not isinstance(files, dict):
        raise SkillIntegrityError(f"{label}: files must be an object")
    return _validated(files, label)


def _validated(entries: Mapping[Any, Any], label: str) -> dict[str, str]:
    result: dict[str, str] = {}
    for key, value in entries.items():
        if not _is_relative_posix(key):
            raise SkillIntegrityError(f"{label}: invalid path {key!r}")
        if not isinstance(value, str) or _HEX_DIGEST.fullmatch(value) is None:
            raise SkillIntegrityError(f"{label}: {key}: digest must be 64 lowercase hex characters")
        result[key] = value
    return dict(sorted(result.items()))


def _is_relative_posix(key: object) -> bool:
    if not isinstance(key, str) or not key or key.startswith("/") or "\\" in key or "\x00" in key:
        return False
    if _DRIVE_PREFIX.match(key):
        return False
    return all(part not in ("", ".", "..") for part in key.split("/"))


def _serialize(digests: Mapping[str, str]) -> bytes:
    payload = {"version": SKILLS_DIGEST_VERSION, "files": dict(sorted(digests.items()))}
    return (json.dumps(payload, indent=2, ensure_ascii=False) + "\n").encode("utf-8")


def _sha256(root: Path, rel: str) -> str:
    return hashlib.sha256(root.joinpath(*rel.split("/")).read_bytes()).hexdigest()


def _walk(root: Path) -> tuple[list[str], list[str]]:
    """Return (files inside skill folders, problems), both as relative POSIX paths."""
    files: list[str] = []
    problems: list[str] = []

    def visit(directory: str, prefix: str, in_skill: bool) -> None:
        with os.scandir(directory) as scan:
            entries = sorted(scan, key=lambda entry: entry.name)
        for entry in entries:
            rel = prefix + entry.name
            if not in_skill and entry.name.startswith("."):
                continue  # hidden folders such as .git are never scanned for skills
            if entry.is_symlink():
                problems.append(f"{rel}: symlinks are not allowed")
            elif entry.is_dir(follow_symlinks=False):
                visit(entry.path, f"{rel}/", in_skill or _holds_skill_file(entry.path))
            elif in_skill:
                if entry.is_file(follow_symlinks=False):
                    files.append(rel)
                else:
                    problems.append(f"{rel}: not a regular file")

    visit(str(root), "", False)
    return sorted(files), problems


def _holds_skill_file(directory: str) -> bool:
    try:
        mode = os.lstat(os.path.join(directory, _SKILL_FILE)).st_mode
    except FileNotFoundError:
        return False
    return not stat.S_ISDIR(mode)


def _walk_or_raise(root: Path) -> list[str]:
    files, problems = _walk(root)
    if problems:
        raise _integrity_error(root, problems)
    return files


def _needs_entry(rel: str, *, strict: bool) -> bool:
    return strict or rel.rsplit("/", 1)[-1] == _SKILL_FILE


def _verify_tree(root: Path, expected: Mapping[str, str], *, strict: bool) -> None:
    problems: list[str] = []
    try:
        for rel, digest in expected.items():
            try:
                mode = os.lstat(root.joinpath(*rel.split("/"))).st_mode
            except (FileNotFoundError, NotADirectoryError):
                problems.append(f"{rel}: listed in the manifest but missing")
                continue
            if not stat.S_ISREG(mode):
                problems.append(f"{rel}: listed in the manifest but not a regular file")
            elif _sha256(root, rel) != digest:
                problems.append(f"{rel}: digest mismatch")
        files, walk_problems = _walk(root)
    except OSError as exc:
        raise _unreadable(root, exc) from exc
    problems.extend(walk_problems)
    problems.extend(
        f"{rel}: no digest entry" for rel in files if rel not in expected and _needs_entry(rel, strict=strict)
    )
    if problems:
        raise _integrity_error(root, problems)


def _check_loaded(skills: Mapping[str, Skill], expected: Mapping[str, str], *, strict: bool) -> list[str]:
    problems: list[str] = []
    for path, skill in sorted(skills.items()):
        loaded = skill.files
        for name in sorted(loaded):
            rel = f"{path}/{name}"
            digest = expected.get(rel)
            if digest is None:
                if _needs_entry(rel, strict=strict):
                    problems.append(f"{rel}: no digest entry")
            elif hashlib.sha256(loaded[name]).hexdigest() != digest:
                problems.append(f"{rel}: digest mismatch")
        prefix = f"{path}/"
        problems.extend(
            f"{rel}: listed in the manifest but missing"
            for rel in expected
            if rel.startswith(prefix) and rel[len(prefix) :] not in loaded
        )
    return problems


def _integrity_error(root: Path, problems: Sequence[str]) -> SkillIntegrityError:
    lines = "\n".join(f"- {problem}" for problem in problems)
    return SkillIntegrityError(f"{len(problems)} skill integrity problem(s) in {root}:\n{lines}", problems)


def _unreadable(root: Path, exc: OSError) -> SkillIntegrityError:
    return SkillIntegrityError(f"Skills directory {root} cannot be read: {exc}")
