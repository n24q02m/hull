"""The CI matrix must be what the repository declares, not a list somebody edits.

``.github/workflows/ci.yml`` resolves both matrices at run time through
``scripts/ci_versions.py``. These tests are the local half of that contract:

  * the derivation still produces the sets the manifests imply, and
  * the workflow has no literal ``python-version`` / ``node-version`` left that
    could silently disagree with it.

Both halves are cheap: no network, no subprocess, and no YAML dependency -- the
repository does not carry PyYAML, so the workflow is read as text and only the
few expressions that matter are matched.

Freshness rule for ``scripts/ci_versions.py``: its two version tables and its
``LAST_VERIFIED`` date are the only hand-maintained input. Extend them (and the
date) when CPython ships a minor or a Node.js line enters/leaves LTS; a table
older than six months is a defect.
"""

from __future__ import annotations

import importlib.util
import json
import re
import sys
from pathlib import Path
from types import ModuleType

import pytest

ROOT = Path(__file__).resolve().parents[1]
WORKFLOW = ROOT / ".github" / "workflows" / "ci.yml"

# The only python-version/node-version values ci.yml may contain: the resolved
# matrix itself, the row handed to setup-*, and the newest row for tooling.
ALLOWED_PYTHON_VALUES = {
    "${{ fromJSON(needs.versions.outputs.python) }}",
    "${{ matrix.python-version }}",
    "${{ needs.versions.outputs.python-newest }}",
}
ALLOWED_NODE_VALUES = {
    "${{ fromJSON(needs.versions.outputs.node) }}",
    "${{ matrix.node-version }}",
}

_JOB = re.compile(r"^ {2}(?P<id>[A-Za-z0-9_-]+):\s*$")


@pytest.fixture(scope="module")
def ci_versions() -> ModuleType:
    """``scripts/ci_versions.py`` as a module (``scripts/`` is not a package)."""
    path = ROOT / "scripts" / "ci_versions.py"
    spec = importlib.util.spec_from_file_location("hull_ci_versions", path)
    assert spec is not None and spec.loader is not None, f"cannot load {path}"
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _workflow_text() -> str:
    return WORKFLOW.read_text(encoding="utf-8")


def _jobs(text: str) -> dict[str, str]:
    """Split the workflow into ``job id -> block`` for the top-level jobs map."""
    blocks: dict[str, list[str]] = {}
    current: str | None = None
    in_jobs = False
    for line in text.splitlines():
        if not in_jobs:
            in_jobs = line.rstrip() == "jobs:"
            continue
        if line and not line.startswith((" ", "\t")):
            break  # a top-level key after `jobs:` ends the job map
        found = _JOB.match(line)
        if found is not None:
            current = found.group("id")
            blocks[current] = [line]
        elif current is not None:
            blocks[current].append(line)
    return {job: "\n".join(lines) for job, lines in blocks.items()}


def _values(text: str, key: str) -> list[str]:
    """Every value assigned to ``key:`` on a non-comment line."""
    pattern = re.compile(rf"^\s*(?:-\s*)?{re.escape(key)}:\s*(?P<value>.+?)\s*$")
    values: list[str] = []
    for line in text.splitlines():
        if not line.strip() or line.lstrip().startswith("#"):
            continue
        found = pattern.match(line)
        if found is not None:
            values.append(found.group("value").strip().strip("\"'"))
    return values


# ---------------------------------------------------------------------------
# The derivation
# ---------------------------------------------------------------------------


def _minor(row: str) -> tuple[int, int]:
    major, minor = row.split(".")
    return int(major), int(minor)


def test_python_matrix_is_the_declared_window(ci_versions: ModuleType) -> None:
    floor = ci_versions.python_floor(ROOT)
    ceiling = ci_versions.python_ceiling(ROOT)
    rows = ci_versions.python_matrix(ROOT)
    minors = [_minor(row) for row in rows]

    assert 1 <= len(rows) <= ci_versions.MAX_ROWS
    # Row one is the declared floor and no minor inside the window is skipped:
    # CPython never skips a minor, so a gap means the table lost an entry.
    assert minors == [(floor[0], floor[1] + index) for index in range(len(minors))]
    if ceiling is None:
        assert minors[-1] == ci_versions.PYTHON_RELEASED_MINORS[-1]
    else:
        # The declared ceiling is a promise to consumers: the minor it excludes
        # is not a row, and the newest row sits directly below it.
        assert all(minor < ceiling for minor in minors)
        assert (minors[-1][0], minors[-1][1] + 1) == ceiling


def test_the_declared_ceiling_keeps_3_15_out(ci_versions: ModuleType) -> None:
    """hull's window is >=3.13,<3.15 until the cp315 wheels exist.

    The ceiling exists because a 3.15 install has to build pyyaml (dspy's
    dependency) and several other compiled packages from source; see the
    requires-python comment in pyproject.toml for the reproduction command.
    """
    rows = ci_versions.python_matrix(ROOT)

    assert "3.15" not in rows
    assert rows[-1] == "3.14"


def test_node_matrix_covers_the_engines_floor(ci_versions: ModuleType) -> None:
    floor_major, _ = ci_versions.node_floor(ROOT)
    rows = ci_versions.node_matrix(ROOT)
    lines = [int(row) for row in rows]

    assert 1 <= len(rows) <= ci_versions.MAX_ROWS
    assert lines == sorted(set(lines)), "rows must be unique and ascending"
    assert all(line >= floor_major for line in lines), "a row below engines.node would fail the install"
    assert lines[-1] == ci_versions.NODE_LTS_LINES[-1], "the newest LTS line is always tested"
    if floor_major in ci_versions.NODE_LTS_LINES:
        assert floor_major in lines, "the floor's own line must be exercised when it is an LTS line"


def test_node_matrix_follows_a_moved_engines_floor(ci_versions: ModuleType, tmp_path: Path) -> None:
    source = json.loads((ROOT / "packages" / "core-ts" / "package.json").read_text(encoding="utf-8"))
    source["engines"]["node"] = ">=26.0.0"
    package_dir = tmp_path / "packages" / "core-ts"
    package_dir.mkdir(parents=True)
    (package_dir / "package.json").write_text(json.dumps(source), encoding="utf-8")

    assert ci_versions.node_matrix(tmp_path) == [str(ci_versions.NODE_LTS_LINES[-1])]


def test_version_tables_are_consistent(ci_versions: ModuleType) -> None:
    python_minors = ci_versions.PYTHON_RELEASED_MINORS
    node_lines = ci_versions.NODE_LTS_LINES

    assert len(python_minors) == len(set(python_minors))
    assert list(python_minors) == sorted(python_minors)
    assert len(node_lines) == len(set(node_lines))
    assert list(node_lines) == sorted(node_lines)
    assert all(line % 2 == 0 for line in node_lines), "odd Node majors never enter LTS"
    assert re.fullmatch(r"\d{4}-\d{2}-\d{2}", ci_versions.LAST_VERIFIED)


DECLARED_WINDOW = 'requires-python = ">=3.13,<3.15"'


def _pyproject_with(tmp_path: Path, window: str) -> Path:
    """A copy of the real pyproject.toml carrying a different Python window."""
    source = (ROOT / "pyproject.toml").read_text(encoding="utf-8")
    mutated = source.replace(DECLARED_WINDOW, window, 1)
    assert mutated != source, f"pyproject.toml no longer declares {DECLARED_WINDOW}; update this test"
    (tmp_path / "pyproject.toml").write_text(mutated, encoding="utf-8")
    return tmp_path


def test_python_matrix_follows_a_moved_floor(ci_versions: ModuleType, tmp_path: Path) -> None:
    root = _pyproject_with(tmp_path, 'requires-python = ">=3.14,<3.15"')

    assert ci_versions.python_matrix(root) == ci_versions.python_matrix(ROOT)[1:]


def test_python_matrix_follows_a_moved_ceiling(ci_versions: ModuleType, tmp_path: Path) -> None:
    root = _pyproject_with(tmp_path, 'requires-python = ">=3.13,<3.14"')

    assert ci_versions.python_matrix(root) == ["3.13"]


def test_an_inclusive_ceiling_excludes_the_next_minor(ci_versions: ModuleType, tmp_path: Path) -> None:
    root = _pyproject_with(tmp_path, 'requires-python = ">=3.13,<=3.13"')

    assert ci_versions.python_ceiling(root) == (3, 14)
    assert ci_versions.python_matrix(root) == ["3.13"]


def test_a_floor_below_the_table_fails_closed(ci_versions: ModuleType, tmp_path: Path) -> None:
    root = _pyproject_with(tmp_path, 'requires-python = ">=3.9,<3.15"')

    with pytest.raises(ci_versions.MatrixError, match="extend the table back"):
        ci_versions.python_matrix(root)


def test_an_empty_window_fails_closed(ci_versions: ModuleType, tmp_path: Path) -> None:
    root = _pyproject_with(tmp_path, 'requires-python = ">=3.15,<3.15"')

    with pytest.raises(ci_versions.MatrixError, match="excludes everything"):
        ci_versions.python_matrix(root)


def test_a_ceiling_past_the_table_fails_closed(ci_versions: ModuleType, tmp_path: Path) -> None:
    root = _pyproject_with(tmp_path, 'requires-python = ">=3.13,<3.18"')

    with pytest.raises(ci_versions.MatrixError, match="extend the table"):
        ci_versions.python_matrix(root)


def test_a_span_wider_than_the_cap_keeps_the_ends(ci_versions: ModuleType) -> None:
    rows = [str(line) for line in range(10, 18)]

    assert ci_versions._cap(rows) == ["10", "14", "15", "16", "17"]


def test_github_output_carries_every_ci_key(ci_versions: ModuleType) -> None:
    rendered = ci_versions.github_output(ci_versions.matrix(ROOT))

    for key in ci_versions.GITHUB_KEYS:
        assert re.search(rf"^{re.escape(key)}(?:=|<<)", rendered, re.MULTILINE), rendered
    assert json.dumps(ci_versions.python_matrix(ROOT)) in rendered
    assert json.dumps(ci_versions.node_matrix(ROOT)) in rendered


# ---------------------------------------------------------------------------
# The workflow cannot override the derivation
# ---------------------------------------------------------------------------


def test_workflow_has_no_literal_python_or_node_version() -> None:
    text = _workflow_text()
    python_values = _values(text, "python-version")
    node_values = _values(text, "node-version")

    assert python_values, "ci.yml declares no python-version at all"
    assert node_values, "ci.yml declares no node-version at all"
    stray_python = sorted(set(python_values) - ALLOWED_PYTHON_VALUES)
    stray_node = sorted(set(node_values) - ALLOWED_NODE_VALUES)
    assert not stray_python, f"ci.yml pins python-version {stray_python}; derive it from scripts/ci_versions.py instead"
    assert not stray_node, f"ci.yml pins node-version {stray_node}; derive it from scripts/ci_versions.py instead"


def test_workflow_resolves_both_matrices_from_the_script() -> None:
    text = _workflow_text()

    assert re.search(r"python3\s+scripts/ci_versions\.py\s+--format\s+github", text), (
        "the versions job must run the derivation on the runner's stock python3"
    )
    assert '"$GITHUB_OUTPUT"' in text
    for key, step_output in (
        ("python", "steps.resolve.outputs.python"),
        ("python-newest", "steps.resolve.outputs.python-newest"),
        ("node", "steps.resolve.outputs.node"),
        ("node-newest", "steps.resolve.outputs.node-newest"),
    ):
        assert f"{key}: ${{{{ {step_output} }}}}" in text, f"versions job does not expose {key}"


def test_matrix_jobs_read_the_resolved_outputs() -> None:
    jobs = _jobs(_workflow_text())
    assert "versions" in jobs

    python_job = jobs["lint-and-test"]
    assert "needs: versions" in python_job, "a matrix sourced from needs.* without needs: resolves empty"
    assert "${{ fromJSON(needs.versions.outputs.python) }}" in python_job
    assert "${{ matrix.python-version }}" in python_job

    typescript_job = jobs["lint-and-test-ts"]
    assert "needs: versions" in typescript_job
    assert "${{ fromJSON(needs.versions.outputs.node) }}" in typescript_job
    assert "${{ matrix.node-version }}" in typescript_job


def test_one_job_owns_the_required_context() -> None:
    """The ruleset requires the single context "Lint & Test".

    A matrix that publishes one check per row under the SAME name lets branch
    protection resolve that name to whichever row reported last, so a red row
    can be masked by a green one. Exactly one aggregating job may therefore
    publish the required name, and the row jobs must carry it only as a prefix.
    """
    jobs = _jobs(_workflow_text())
    owners = [job for job, block in jobs.items() if re.search(r"^    name: Lint & Test$", block, re.MULTILINE)]

    assert owners == ["lint-and-test-gate"], f"exactly one job must publish 'Lint & Test'; found {owners}"
    gate = jobs["lint-and-test-gate"]
    assert "lint-and-test" in gate.split("runs-on:")[0], "the gate must need the Python matrix job"
    assert "RESULT" in gate and "exit 1" in gate, "the gate must fail when a row did"
