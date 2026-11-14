"""The CI matrix must be what the repository declares, not a list somebody edits.

``.github/workflows/ci.yml`` resolves both matrices at run time through
``scripts/ci_versions.py``. These tests are the local half of that contract:

  * the derivation still produces the sets the manifests imply, and
  * the workflow has no literal ``python-version`` / ``node-version`` left that
    could silently disagree with it.

Both halves are cheap: no network, no subprocess, and no YAML dependency -- the
repository does not carry PyYAML, so the workflow is read as text and only the
few expressions that matter are matched.

Freshness rule for ``scripts/ci_versions.py``: its two version tables (with
their start and end-of-life dates) and its ``LAST_VERIFIED`` date are the only
hand-maintained input. Extend them (and the date) when CPython ships a minor or
a Node.js line is scheduled; a table older than six months is a defect. Every
date-dependent test below pins ``today`` so the suite does not change verdict
as the calendar moves.
"""

from __future__ import annotations

import datetime as dt
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

# The day these assertions describe. Pinned, so the suite keeps its verdict as
# the calendar moves; the date-moving tests below pin their own days.
DAY = dt.date(2026, 11, 14)


def _minor(row: str) -> tuple[int, int]:
    major, minor = row.split(".")
    return int(major), int(minor)


def test_upstream_python_window_on_the_verified_day(ci_versions: ModuleType) -> None:
    """3.10 left support on 2026-10-01; 3.15 entered it on 2026-10-09."""
    assert ci_versions.supported_python_minors(DAY) == [(3, 11), (3, 12), (3, 13), (3, 14), (3, 15)]


def test_upstream_node_window_on_the_verified_day(ci_versions: ModuleType) -> None:
    """Node 20 left support on 2026-04-30; Node 26 entered LTS on 2026-10-28."""
    assert ci_versions.supported_node_lines(DAY) == [22, 24, 26]


def test_python_matrix_is_the_declared_window(ci_versions: ModuleType) -> None:
    floor = ci_versions.python_floor(ROOT)
    ceiling = ci_versions.python_ceiling(ROOT)
    rows = ci_versions.python_matrix(ROOT, DAY)
    minors = [_minor(row) for row in rows]
    supported = ci_versions.supported_python_minors(DAY)

    assert 1 <= len(rows) <= ci_versions.MAX_ROWS
    # Row one is the declared floor, or the oldest supported minor when the
    # floor sits on an end-of-life line; no minor inside the window is skipped.
    first = max(floor, supported[0])
    assert minors == [(first[0], first[1] + index) for index in range(len(minors))]
    if ceiling is None:
        assert minors[-1] == supported[-1]
    else:
        assert all(minor < ceiling for minor in minors)


def test_hull_covers_every_supported_python(ci_versions: ModuleType) -> None:
    """Q34: a library runs CI on every CPython minor upstream still supports.

    hull declares ``>=3.11.5`` with no ceiling, so on the verified day the rows
    are exactly the upstream window. A 3.15 install builds pyyaml (pure-Python
    fallback) and zstandard from source until their cp315 wheels exist; the
    3.15 row is the proof that it still works.
    """
    assert ci_versions.python_ceiling(ROOT) is None
    assert ci_versions.python_matrix(ROOT, DAY) == ["3.11", "3.12", "3.13", "3.14", "3.15"]


def test_hull_covers_every_supported_node(ci_versions: ModuleType) -> None:
    assert ci_versions.node_matrix(ROOT, DAY) == ["22", "24", "26"]


def test_an_end_of_life_minor_drops_out_on_its_date(ci_versions: ModuleType) -> None:
    before = ci_versions.python_matrix(ROOT, dt.date(2027, 10, 31))
    after = ci_versions.python_matrix(ROOT, dt.date(2027, 11, 1))

    assert before[0] == "3.11"
    assert after == ["3.12", "3.13", "3.14", "3.15"]


def test_a_minor_is_not_a_row_before_its_release(ci_versions: ModuleType) -> None:
    assert ci_versions.supported_python_minors(dt.date(2026, 10, 8))[-1] == (3, 14)
    assert ci_versions.supported_python_minors(dt.date(2026, 10, 9))[-1] == (3, 15)


def test_a_node_line_counts_from_lts_and_drops_at_eol(ci_versions: ModuleType) -> None:
    assert ci_versions.supported_node_lines(dt.date(2026, 10, 27)) == [22, 24]
    assert ci_versions.node_matrix(ROOT, dt.date(2027, 5, 1)) == ["24", "26"]


def test_node_matrix_follows_a_moved_engines_floor(ci_versions: ModuleType, tmp_path: Path) -> None:
    root = _package_with(tmp_path, ">=26.0.0")

    assert ci_versions.node_matrix(root, DAY) == ["26"]


def test_a_node_floor_on_an_end_of_life_line_drops_that_row(ci_versions: ModuleType, tmp_path: Path) -> None:
    root = _package_with(tmp_path, ">=20.0.0")

    assert ci_versions.node_matrix(root, DAY) == ["22", "24", "26"]


def test_a_node_floor_outside_the_table_fails_closed(ci_versions: ModuleType, tmp_path: Path) -> None:
    with pytest.raises(ci_versions.MatrixError, match="extend the table back"):
        ci_versions.node_matrix(_package_with(tmp_path, ">=18.0.0"), DAY)
    with pytest.raises(ci_versions.MatrixError, match="newer than the newest LTS line"):
        ci_versions.node_matrix(_package_with(tmp_path, ">=28.0.0"), DAY)


def test_a_node_spec_without_a_floor_fails_closed(ci_versions: ModuleType, tmp_path: Path) -> None:
    with pytest.raises(ci_versions.MatrixError, match="no '>=X' lower bound"):
        ci_versions.node_floor(_package_with(tmp_path, "^24.0.0"))


def test_a_package_without_engines_fails_closed(ci_versions: ModuleType, tmp_path: Path) -> None:
    package_dir = tmp_path / "packages" / "core-ts"
    package_dir.mkdir(parents=True)
    (package_dir / "package.json").write_text(json.dumps({"name": "x"}), encoding="utf-8")

    with pytest.raises(ci_versions.MatrixError, match=re.escape("declares no engines.node")):
        ci_versions.node_floor(tmp_path)


def test_version_tables_are_consistent(ci_versions: ModuleType) -> None:
    python_minors = [minor for minor, _, _ in ci_versions.PYTHON_MINORS]
    node_lines = [line for line, _, _ in ci_versions.NODE_LTS_LINES]

    assert len(python_minors) == len(set(python_minors))
    assert python_minors == sorted(python_minors)
    # CPython never skips a minor, so a gap means the table lost an entry.
    assert python_minors == [(3, python_minors[0][1] + index) for index in range(len(python_minors))]
    assert len(node_lines) == len(set(node_lines))
    assert node_lines == sorted(node_lines)
    assert all(line % 2 == 0 for line in node_lines), "odd Node majors never enter LTS"
    for _, start, end in (*ci_versions.PYTHON_MINORS, *ci_versions.NODE_LTS_LINES):
        assert dt.date.fromisoformat(start) < dt.date.fromisoformat(end)
    assert re.fullmatch(r"\d{4}-\d{2}-\d{2}", ci_versions.LAST_VERIFIED)


def test_today_is_utc(ci_versions: ModuleType) -> None:
    assert ci_versions._today() == dt.datetime.now(dt.UTC).date()


DECLARED_WINDOW = 'requires-python = ">=3.11.5"'


def _pyproject_with(tmp_path: Path, window: str) -> Path:
    """A copy of the real pyproject.toml carrying a different Python window."""
    source = (ROOT / "pyproject.toml").read_text(encoding="utf-8")
    mutated = source.replace(DECLARED_WINDOW, window, 1)
    assert mutated != source, f"pyproject.toml no longer declares {DECLARED_WINDOW}; update this test"
    (tmp_path / "pyproject.toml").write_text(mutated, encoding="utf-8")
    return tmp_path


def _package_with(tmp_path: Path, spec: str) -> Path:
    """A copy of the real core-ts package.json carrying a different engines.node."""
    source = json.loads((ROOT / "packages" / "core-ts" / "package.json").read_text(encoding="utf-8"))
    source["engines"]["node"] = spec
    package_dir = tmp_path / "packages" / "core-ts"
    package_dir.mkdir(parents=True, exist_ok=True)
    (package_dir / "package.json").write_text(json.dumps(source), encoding="utf-8")
    return tmp_path


def test_python_matrix_follows_a_moved_floor(ci_versions: ModuleType, tmp_path: Path) -> None:
    root = _pyproject_with(tmp_path, 'requires-python = ">=3.14"')

    assert ci_versions.python_matrix(root, DAY) == ["3.14", "3.15"]


def test_python_matrix_follows_a_moved_ceiling(ci_versions: ModuleType, tmp_path: Path) -> None:
    root = _pyproject_with(tmp_path, 'requires-python = ">=3.13,<3.15"')

    assert ci_versions.python_matrix(root, DAY) == ["3.13", "3.14"]


def test_a_floor_on_an_end_of_life_minor_drops_that_row(ci_versions: ModuleType, tmp_path: Path) -> None:
    root = _pyproject_with(tmp_path, 'requires-python = ">=3.10"')

    assert ci_versions.python_matrix(root, DAY) == ["3.11", "3.12", "3.13", "3.14", "3.15"]


def test_an_inclusive_ceiling_excludes_the_next_minor(ci_versions: ModuleType, tmp_path: Path) -> None:
    root = _pyproject_with(tmp_path, 'requires-python = ">=3.13,<=3.13"')

    assert ci_versions.python_ceiling(root) == (3, 14)
    assert ci_versions.python_matrix(root, DAY) == ["3.13"]


def test_a_major_only_ceiling_fails_closed(ci_versions: ModuleType, tmp_path: Path) -> None:
    root = _pyproject_with(tmp_path, 'requires-python = ">=3.13,<4"')

    with pytest.raises(ci_versions.MatrixError, match="caps on a major only"):
        ci_versions.python_ceiling(root)


def test_a_floor_below_the_table_fails_closed(ci_versions: ModuleType, tmp_path: Path) -> None:
    root = _pyproject_with(tmp_path, 'requires-python = ">=3.9"')

    with pytest.raises(ci_versions.MatrixError, match="extend the table back"):
        ci_versions.python_matrix(root, DAY)


def test_a_window_without_a_floor_fails_closed(ci_versions: ModuleType, tmp_path: Path) -> None:
    root = _pyproject_with(tmp_path, 'requires-python = "~=3.13"')

    with pytest.raises(ci_versions.MatrixError, match=re.escape("no '>=X.Y' lower bound")):
        ci_versions.python_floor(root)


def test_a_pyproject_without_requires_python_fails_closed(ci_versions: ModuleType, tmp_path: Path) -> None:
    (tmp_path / "pyproject.toml").write_text('[project]\nname = "x"\n', encoding="utf-8")

    with pytest.raises(ci_versions.MatrixError, match=re.escape("declares no project.requires-python")):
        ci_versions.python_floor(tmp_path)


def test_an_empty_window_fails_closed(ci_versions: ModuleType, tmp_path: Path) -> None:
    root = _pyproject_with(tmp_path, 'requires-python = ">=3.15,<3.15"')

    with pytest.raises(ci_versions.MatrixError, match="excludes everything"):
        ci_versions.python_matrix(root, DAY)


def test_a_ceiling_past_the_table_fails_closed(ci_versions: ModuleType, tmp_path: Path) -> None:
    root = _pyproject_with(tmp_path, 'requires-python = ">=3.13,<3.18"')

    with pytest.raises(ci_versions.MatrixError, match="extend the table"):
        ci_versions.python_matrix(root, DAY)


def test_a_window_with_no_supported_minor_fails_closed(ci_versions: ModuleType, tmp_path: Path) -> None:
    root = _pyproject_with(tmp_path, 'requires-python = ">=3.10,<3.11"')

    with pytest.raises(ci_versions.MatrixError, match="no upstream-supported interpreter"):
        ci_versions.python_matrix(root, DAY)


def test_a_span_wider_than_the_cap_keeps_the_ends(ci_versions: ModuleType) -> None:
    rows = [str(line) for line in range(10, 18)]

    assert ci_versions._cap(rows) == ["10", "14", "15", "16", "17"]


def test_github_output_carries_every_ci_key(ci_versions: ModuleType) -> None:
    rendered = ci_versions.github_output(ci_versions.matrix(ROOT, DAY))

    for key in ci_versions.GITHUB_KEYS:
        assert re.search(rf"^{re.escape(key)}(?:=|<<)", rendered, re.MULTILINE), rendered
    assert json.dumps(ci_versions.python_matrix(ROOT, DAY)) in rendered
    assert json.dumps(ci_versions.node_matrix(ROOT, DAY)) in rendered


def test_matrix_reports_the_day_it_resolved_for(ci_versions: ModuleType) -> None:
    resolved = ci_versions.matrix(ROOT, DAY)

    assert resolved["today"] == "2026-11-14"
    assert resolved["python-newest"] == "3.15"
    assert resolved["node-newest"] == "26"


@pytest.mark.parametrize("fmt", ["json", "text", "github"])
def test_main_renders_every_format(ci_versions: ModuleType, capsys: pytest.CaptureFixture[str], fmt: str) -> None:
    assert ci_versions.main(["--format", fmt, "--today", "2026-11-14", "--repo-root", str(ROOT)]) == 0
    out = capsys.readouterr().out

    assert "3.15" in out
    if fmt == "json":
        assert json.loads(out)["python"] == ["3.11", "3.12", "3.13", "3.14", "3.15"]


def test_main_reports_a_matrix_error(
    ci_versions: ModuleType, capsys: pytest.CaptureFixture[str], tmp_path: Path
) -> None:
    root = _pyproject_with(tmp_path, 'requires-python = ">=3.9"')
    _package_with(tmp_path, ">=24.0.0")

    assert ci_versions.main(["--repo-root", str(root), "--today", "2026-11-14"]) == 2
    assert "ci_versions:" in capsys.readouterr().err


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
