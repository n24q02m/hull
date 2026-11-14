#!/usr/bin/env python3
"""Derive the interpreter support matrix that CI has to cover.

The matrix is not a list somebody remembers to update: it is the overlap of two
windows, both computed rather than listed.

  * The upstream window: every interpreter line that upstream still supports
    on the day CI runs -- released, and not past its end-of-life date. A line
    drops out of CI on its EOL date without anybody editing a workflow.
  * The declared window: what this repository promises its consumers.
    Python reads ``project.requires-python`` (floor and optional ceiling; a
    ceiling such as ``<3.15`` says "3.15 is not supported", so 3.15 is not a
    CI row either). Node reads the ``engines.node`` floor in
    ``packages/core-ts/package.json``.

A row is a line inside both windows, oldest first, and each set is capped at
:data:`MAX_ROWS` rows. A declared floor that sits on an end-of-life line
simply loses that row: upstream no longer ships fixes for it, so CI stops
spending runners on it.

:func:`matrix` is the single source of truth for both. ``.github/workflows/
ci.yml`` feeds its output to ``actions/setup-python`` / ``actions/setup-node``
through ``fromJSON(needs.versions.outputs.*)``, and ``tests/test_ci_versions.py``
re-derives the same sets so that a literal version creeping back into the
workflow fails CI instead of silently shrinking coverage.

Freshness
---------
The two tables below are the only hand-maintained input, because "which
interpreter versions exist and until when upstream supports them" is not
recorded anywhere in this repository. Each entry carries its start date and
its end-of-life date, so the passage of time is handled by the code; only a
NEW line needs an edit. Re-verify both tables, and bump :data:`LAST_VERIFIED`
in the same commit, whenever

  * CPython ships a new minor -- every October,
    https://devguide.python.org/versions/ (machine-readable:
    https://endoflife.date/api/python.json)
  * a Node.js line is scheduled -- April and October,
    https://github.com/nodejs/Release/blob/main/schedule.json

A table older than six months is a defect: the guard test compares the workflow
against these same tables, so it cannot notice that the tables themselves went
stale.

Usage::

    python scripts/ci_versions.py                 # JSON, for humans and tests
    python scripts/ci_versions.py --format text   # the declared window + rows
    python scripts/ci_versions.py --format github >> "$GITHUB_OUTPUT"
    python scripts/ci_versions.py --today 2027-05-01   # the rows CI runs then

Stdlib only, and deliberately so: the workflow runs it on the runner's stock
``python3``, never on an interpreter this file selects.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import re
import sys
import tomllib
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]

# Last verified against the release calendars: 2026-11-14.
LAST_VERIFIED = "2026-11-14"

# CPython minors as (version, first release, end of life), oldest first, from
# the devguide status page. 3.10 stays after its EOL so that a floor at 3.10 is
# still recognised (and dropped by date) rather than refused. Extend this back
# if a manifest floor ever sits below the first entry -- the derivation refuses
# to guess.
PYTHON_MINORS: tuple[tuple[tuple[int, int], str, str], ...] = (
    ((3, 10), "2021-10-04", "2026-10-01"),
    ((3, 11), "2022-10-24", "2027-10-31"),
    ((3, 12), "2023-10-02", "2028-10-31"),
    ((3, 13), "2024-10-07", "2029-10-31"),
    ((3, 14), "2025-10-07", "2030-10-31"),
    ((3, 15), "2026-10-09", "2031-10-31"),
)

# Node.js LTS lines as (major, LTS start, end of life), oldest first, from the
# nodejs/Release schedule. A line counts from the day it enters LTS: before
# that it is "Current", and odd majors (23, 25, ...) never enter LTS, so
# neither belongs here. Node 20 reached end-of-life on 2026-04-30 and is kept
# only so that a floor at 20 is dropped by date, not refused.
NODE_LTS_LINES: tuple[tuple[int, str, str], ...] = (
    (20, "2023-10-24", "2026-04-30"),
    (22, "2024-10-29", "2027-04-30"),
    (24, "2025-10-28", "2028-04-30"),
    (26, "2026-10-28", "2029-04-30"),
)

# A support matrix wider than this stops being a test and becomes a sweep.
MAX_ROWS = 5

# Both bounds of the Python window: the floor is what still works, the ceiling
# (when the manifest declares one) is where support deliberately stops.
_PYTHON_FLOOR = re.compile(r">=\s*(\d+)\.(\d+)")
_PYTHON_CEILING = re.compile(r"(<=|<)\s*(\d+)\.(\d+)")
_NODE_FLOOR = re.compile(r">=\s*(\d+)(?:\.(\d+))?(?:\.(\d+))?")

# Output keys of --format github, in the order the workflow declares them.
GITHUB_KEYS = ("python", "python-newest", "node", "node-newest")


class MatrixError(RuntimeError):
    """A manifest declares a window the version tables cannot satisfy."""


def _today() -> dt.date:
    """The day CI runs, in UTC, so a runner's local clock cannot shift a row."""
    return dt.datetime.now(dt.UTC).date()


def _supported(start: str, end: str, today: dt.date) -> bool:
    """Upstream supports a line from ``start`` up to and including ``end``."""
    return dt.date.fromisoformat(start) <= today <= dt.date.fromisoformat(end)


def supported_python_minors(today: dt.date | None = None) -> list[tuple[int, int]]:
    """CPython minors upstream supports on ``today``, oldest first."""
    day = today or _today()
    return [minor for minor, released, eol in PYTHON_MINORS if _supported(released, eol, day)]


def supported_node_lines(today: dt.date | None = None) -> list[int]:
    """Node.js LTS lines upstream supports on ``today``, oldest first."""
    day = today or _today()
    return [line for line, lts, eol in NODE_LTS_LINES if _supported(lts, eol, day)]


def _pyproject(root: Path) -> dict[str, Any]:
    with (root / "pyproject.toml").open("rb") as handle:
        return tomllib.load(handle)


def _requires_python(root: Path) -> str:
    requires = _pyproject(root).get("project", {}).get("requires-python")
    if not isinstance(requires, str):
        raise MatrixError(f"{root / 'pyproject.toml'} declares no project.requires-python")
    return requires


def _node_spec(root: Path) -> str:
    manifest = root / "packages" / "core-ts" / "package.json"
    engines = json.loads(manifest.read_text(encoding="utf-8")).get("engines") or {}
    spec = engines.get("node")
    if not isinstance(spec, str):
        raise MatrixError(f"{manifest} declares no engines.node floor")
    return spec


def python_floor(root: Path = ROOT) -> tuple[int, int]:
    """The lowest CPython minor ``project.requires-python`` still advertises."""
    requires = _requires_python(root)
    found = _PYTHON_FLOOR.search(requires)
    if found is None:
        raise MatrixError(f"requires-python = {requires!r} carries no '>=X.Y' lower bound to derive rows from")
    return int(found.group(1)), int(found.group(2))


def python_ceiling(root: Path = ROOT) -> tuple[int, int] | None:
    """The first CPython minor the declared window excludes, if it caps one.

    ``"<3.15"`` and ``"<=3.14"`` describe the same window end, so both return
    ``(3, 15)``: the first minor outside support.
    """
    requires = _requires_python(root)
    found = _PYTHON_CEILING.search(requires)
    if found is None:
        if "<" in requires:
            raise MatrixError(
                f"requires-python = {requires!r} caps on a major only; the matrix is per minor, "
                "so write the ceiling as '<X.Y'"
            )
        return None
    major, minor = int(found.group(2)), int(found.group(3))
    return (major, minor + 1) if found.group(1) == "<=" else (major, minor)


def node_floor(root: Path = ROOT) -> tuple[int, int]:
    """The lowest Node.js major ``engines.node`` still advertises, plus its minor."""
    spec = _node_spec(root)
    found = _NODE_FLOOR.search(spec)
    if found is None:
        raise MatrixError(f"engines.node = {spec!r} carries no '>=X' lower bound to derive rows from")
    return int(found.group(1)), int(found.group(2) or 0)


def _cap(rows: list[str]) -> list[str]:
    """Keep the oldest and the newest rows when a span exceeds :data:`MAX_ROWS`."""
    if not rows:
        raise MatrixError("no upstream-supported interpreter satisfies the declared window")
    if len(rows) <= MAX_ROWS:
        return rows
    return [rows[0], *rows[1 - MAX_ROWS :]]


def python_matrix(root: Path = ROOT, today: dt.date | None = None) -> list[str]:
    """``["3.11", ..., "3.15"]``: the declared window, intersected with upstream support."""
    floor = python_floor(root)
    ceiling = python_ceiling(root)
    oldest = PYTHON_MINORS[0][0]
    newest = PYTHON_MINORS[-1][0]
    newest_exclusive = (newest[0], newest[1] + 1)

    if ceiling is not None and ceiling <= floor:
        raise MatrixError(
            f"requires-python window excludes everything: floor {floor[0]}.{floor[1]} is not below "
            f"the ceiling {ceiling[0]}.{ceiling[1]}"
        )
    if ceiling is not None and ceiling > newest_exclusive:
        raise MatrixError(
            f"the declared window reaches {ceiling[0]}.{ceiling[1]} but PYTHON_MINORS stops at "
            f"{newest[0]}.{newest[1]}; extend the table and bump LAST_VERIFIED"
        )
    if floor < oldest:
        raise MatrixError(
            f"PYTHON_MINORS starts at {oldest[0]}.{oldest[1]}, so it cannot cover the declared "
            f"floor {floor[0]}.{floor[1]}; extend the table back to the floor"
        )
    return _cap(
        [
            f"{major}.{minor}"
            for major, minor in supported_python_minors(today)
            if (major, minor) >= floor and (ceiling is None or (major, minor) < ceiling)
        ]
    )


def node_matrix(root: Path = ROOT, today: dt.date | None = None) -> list[str]:
    """``["22", "24", "26"]``: supported LTS lines at or above the engines.node floor.

    A line is selected by major only. ``setup-node`` resolves a line to its
    newest patch, and ``engines.node`` ">=22.12.0" is satisfied by that patch
    for every 22.x row, so the floor's minor never excludes its own line.
    """
    floor_major, floor_minor = node_floor(root)
    oldest = NODE_LTS_LINES[0][0]
    newest = NODE_LTS_LINES[-1][0]
    if floor_major > newest:
        raise MatrixError(
            f"engines.node floor {floor_major}.{floor_minor} is newer than the newest LTS line in "
            f"NODE_LTS_LINES ({newest}); extend the table and bump LAST_VERIFIED"
        )
    if floor_major < oldest:
        raise MatrixError(
            f"NODE_LTS_LINES starts at {oldest}, so it cannot cover the declared floor "
            f"{floor_major}.{floor_minor}; extend the table back to the floor"
        )
    return _cap([str(line) for line in supported_node_lines(today) if line >= floor_major])


def matrix(root: Path = ROOT, today: dt.date | None = None) -> dict[str, Any]:
    """Both matrices plus the declarations and the day they were resolved for."""
    day = today or _today()
    python_rows = python_matrix(root, day)
    node_rows = node_matrix(root, day)
    return {
        "python": python_rows,
        "python-newest": python_rows[-1],
        "node": node_rows,
        "node-newest": node_rows[-1],
        "declared": {"python": _requires_python(root), "node": _node_spec(root)},
        "verified": LAST_VERIFIED,
        "today": day.isoformat(),
    }


def github_output(resolved: dict[str, Any]) -> str:
    """The subset of :func:`matrix` that the workflow consumes, as $GITHUB_OUTPUT."""
    lines: list[str] = []
    for key in GITHUB_KEYS:
        value = resolved[key]
        if isinstance(value, list):
            lines.extend([f"{key}<<MATRIX_EOF", json.dumps(value), "MATRIX_EOF"])
        else:
            lines.append(f"{key}={value}")
    return "\n".join(lines) + "\n"


def format_text(resolved: dict[str, Any]) -> str:
    declared = resolved["declared"]
    return "\n".join(
        [
            f"python {', '.join(resolved['python'])} (declared {declared['python']}, {len(resolved['python'])} rows)",
            f"node   {', '.join(resolved['node'])} (declared {declared['node']}, {len(resolved['node'])} rows)",
            f"upstream support evaluated on {resolved['today']}; tables verified {resolved['verified']}",
        ]
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Derive the CI interpreter support matrix.")
    parser.add_argument(
        "--format",
        choices=("json", "text", "github"),
        default="json",
        help="json (default), text for a human, github for $GITHUB_OUTPUT",
    )
    parser.add_argument("--repo-root", type=Path, default=ROOT, help="repository root holding the manifests")
    parser.add_argument(
        "--today",
        type=dt.date.fromisoformat,
        default=None,
        help="evaluate upstream support on this ISO date instead of today (UTC)",
    )
    args = parser.parse_args(argv)

    try:
        resolved = matrix(args.repo_root, args.today)
    except MatrixError as error:
        print(f"ci_versions: {error}", file=sys.stderr)
        return 2

    if args.format == "github":
        sys.stdout.write(github_output(resolved))
    elif args.format == "text":
        print(format_text(resolved))
    else:
        print(json.dumps(resolved, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
