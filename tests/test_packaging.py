"""Packaging contract: hull ships as ONE dist, ``n24q02m-hull``, with extras.

Spec v4 addendum 2026-09-30: a single wheel carries the three import packages
(``hull_core``, ``hull_web``, ``hull_embedding_daemon``); the scraping/browser
stack is only pulled in by ``n24q02m-hull[web]`` and the embedding server stack
only by ``n24q02m-hull[embedding]``.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import zipfile
from email.parser import Parser
from pathlib import Path

import pytest
from packaging.requirements import Requirement
from packaging.utils import canonicalize_name

ROOT = Path(__file__).resolve().parents[1]

BASE_DEPS = {"fastmcp", "starlette", "uvicorn", "httpx", "httpcore", "pydantic"}
WEB_DEPS = {
    "httpx",
    "curl-cffi",
    "crawl4ai",
    "filelock",
    "langgraph",
    "patchright",
    "browserforge",
    "capsolver",
    "pydantic",
    "pillow",
}
EMBEDDING_DEPS = {"fastapi", "uvicorn", "qwen3-embed", "onnxruntime", "numpy", "httpx", "pydantic"}
WEB_ONLY = WEB_DEPS - BASE_DEPS
EMBEDDING_ONLY = EMBEDDING_DEPS - BASE_DEPS


def _uv() -> str:
    uv = os.environ.get("UV") or shutil.which("uv")
    if not uv:
        pytest.fail("uv executable not found (neither $UV nor PATH); packaging tests need uv to build the wheel")
    return uv


@pytest.fixture(scope="module")
def wheel(tmp_path_factory: pytest.TempPathFactory) -> Path:
    out = tmp_path_factory.mktemp("dist")
    proc = subprocess.run(
        [_uv(), "build", "--wheel", "--out-dir", str(out), str(ROOT)],
        capture_output=True,
        text=True,
        encoding="utf-8",
        check=False,
    )
    assert proc.returncode == 0, f"uv build failed:\n{proc.stdout}\n{proc.stderr}"
    wheels = sorted(out.glob("*.whl"))
    assert len(wheels) == 1, f"expected exactly one wheel, got {[w.name for w in wheels]}"
    return wheels[0]


@pytest.fixture(scope="module")
def names(wheel: Path) -> list[str]:
    with zipfile.ZipFile(wheel) as zf:
        return zf.namelist()


@pytest.fixture(scope="module")
def metadata(wheel: Path, names: list[str]):
    meta_path = next(n for n in names if n.endswith(".dist-info/METADATA"))
    with zipfile.ZipFile(wheel) as zf:
        return Parser().parsestr(zf.read(meta_path).decode("utf-8"))


@pytest.fixture(scope="module")
def requirements(metadata) -> list[Requirement]:
    return [Requirement(line) for line in metadata.get_all("Requires-Dist") or []]


def _in_extra(req: Requirement, extra: str) -> bool:
    return req.marker is not None and req.marker.evaluate({"extra": extra})


def _unconditional(req: Requirement) -> bool:
    return req.marker is None


def test_wheel_filename_is_n24q02m_hull(wheel: Path) -> None:
    # dist renamed hull-core -> n24q02m-hull (D6); wheel tag follows the dist name
    assert wheel.name.startswith("n24q02m_hull-")


@pytest.mark.parametrize("package", ["hull_core", "hull_web", "hull_embedding_daemon"])
def test_wheel_contains_import_package_at_top_level(names: list[str], package: str) -> None:
    assert f"{package}/__init__.py" in names


def test_wheel_does_not_leak_source_layout(names: list[str]) -> None:
    assert not [n for n in names if n.startswith(("packages/", "src/", "tests/"))]


def test_web_package_ships_py_typed(names: list[str]) -> None:
    assert "hull_web/py.typed" in names


def test_metadata_name_is_n24q02m_hull(metadata) -> None:
    assert metadata["Name"] == "n24q02m-hull"


def test_metadata_declares_web_and_embedding_extras(metadata) -> None:
    extras = set(metadata.get_all("Provides-Extra") or [])
    assert {"web", "embedding"} <= extras


def test_base_dependencies_are_core_only(requirements: list[Requirement]) -> None:
    base = {canonicalize_name(r.name) for r in requirements if _unconditional(r)}
    assert base == BASE_DEPS


def test_web_extra_carries_every_web_dependency(requirements: list[Requirement]) -> None:
    web = {canonicalize_name(r.name) for r in requirements if _in_extra(r, "web")}
    assert web == WEB_DEPS


def test_embedding_extra_carries_every_embedding_dependency(requirements: list[Requirement]) -> None:
    embedding = {canonicalize_name(r.name) for r in requirements if _in_extra(r, "embedding")}
    assert embedding == EMBEDDING_DEPS


@pytest.mark.parametrize("dep", sorted(WEB_ONLY))
def test_web_only_dependency_is_gated_by_web_extra(requirements: list[Requirement], dep: str) -> None:
    matching = [r for r in requirements if canonicalize_name(r.name) == dep]
    assert matching, f"{dep} missing from Requires-Dist"
    for req in matching:
        assert req.marker is not None, f"{dep} is an unconditional dependency"
        assert req.marker.evaluate({"extra": "web"}), f"{dep} not installed by [web]: {req}"
        assert not req.marker.evaluate({"extra": ""}), f"{dep} leaks into the base install: {req}"
        assert not req.marker.evaluate({"extra": "embedding"}), f"{dep} leaks into [embedding]: {req}"


@pytest.mark.parametrize("dep", sorted(EMBEDDING_ONLY))
def test_embedding_only_dependency_is_gated_by_embedding_extra(requirements: list[Requirement], dep: str) -> None:
    matching = [r for r in requirements if canonicalize_name(r.name) == dep]
    assert matching, f"{dep} missing from Requires-Dist"
    for req in matching:
        assert req.marker is not None, f"{dep} is an unconditional dependency"
        assert req.marker.evaluate({"extra": "embedding"}), f"{dep} not installed by [embedding]: {req}"
        assert not req.marker.evaluate({"extra": ""}), f"{dep} leaks into the base install: {req}"
        assert not req.marker.evaluate({"extra": "web"}), f"{dep} leaks into [web]: {req}"


def test_console_scripts(wheel: Path, names: list[str]) -> None:
    ep_path = next(n for n in names if n.endswith(".dist-info/entry_points.txt"))
    with zipfile.ZipFile(wheel) as zf:
        text = zf.read(ep_path).decode("utf-8")
    assert "hull = hull_core.cli:main" in text
    assert "hull-embedding-daemon = hull_embedding_daemon.__main__:main" in text
