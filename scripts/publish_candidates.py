"""Weekly model-candidates snapshot publisher (public boards, scan-only).

Runs hull_core.model_selection.candidates() for every profile in TASKS and
writes data/model-candidates.json (schema_version 1, ``source_status`` added
additively).

Scope guarantees:
- Public leaderboards + the OpenRouter catalog only. No evals, no secrets,
  no paid calls (AA_API_KEY optional; absent key -> the AA sources are
  recorded ``missing`` in ``source_status``, never silent, never faked).
- Embedding/rerank profiles report ``or_backbone_empty: true`` when the
  OpenRouter backbone is thin (embedders/rerankers mostly do not route
  through OpenRouter); consumers keep those cells locally managed.

Consumer stacks (crg/wet/mnemo model-sync workflows) pull this file's raw
URL and propose per-task model config pins through label-gated PRs / sticky
issues. Model defaults in code stay empty and fail-closed everywhere.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import UTC, datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "packages" / "core-py" / "src"))

from hull_core.model_selection import TASKS, FileCache, candidates, get_task

TOP_N = 10
DEFAULT_OUT = Path(__file__).resolve().parents[1] / "data" / "model-candidates.json"
FIELDS = (
    "or_slug",
    "name",
    "litellm_id",
    "canonical_slug",
    "quality",
    "quality_ci",
    "cost_1m_blended",
    "context",
    "pareto_rank",
    "scores",
    "evidence",
    "weak_evidence",
    "modalities",
    "supported_parameters",
)


def _row(candidate: object) -> dict:
    data: dict = {}
    for field in FIELDS:
        value = getattr(candidate, field)
        if isinstance(value, tuple):
            value = list(value)
        data[field] = value
    return data


def _fill_unscanned_profiles(
    tasks: dict[str, dict],
    source_status: dict[str, dict],
    now: str,
) -> None:
    """Record sources a profile never attempted because its OR backbone was
    empty (the scan returns before touching sources) — honest bookkeeping
    instead of a silent gap in ``source_status``."""
    for name, block in tasks.items():
        if not block.get("or_backbone_empty"):
            continue
        profile = get_task(name)
        for src in (*profile.specialized_sources, *profile.aggregate_sources):
            if src not in source_status:
                source_status[src] = {
                    "status": "missing",
                    "fetched_at": now,
                    "row_count": 0,
                    "reason": "or_backbone_empty_profile_not_scanned",
                }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--out",
        type=Path,
        default=DEFAULT_OUT,
        help="output path (default: data/model-candidates.json; PREVIEW runs point this at a temp dir)",
    )
    parser.add_argument(
        "--refresh",
        action="store_true",
        help="bypass the per-source file cache and refetch every board",
    )
    args = parser.parse_args(argv)

    tasks: dict[str, dict] = {}
    source_status: dict[str, dict] = {}
    cache = FileCache(Path(os.environ.get("HULL_SNAPSHOT_CACHE", Path.home() / ".cache" / "hull" / "model_selection")))
    now = datetime.now(UTC).isoformat(timespec="seconds")
    for name in sorted(TASKS):
        ranked = candidates(name, cache=cache, refresh=args.refresh, status_out=source_status)
        tasks[name] = {
            "or_backbone_empty": not ranked,
            "candidates": [_row(c) for c in ranked[:TOP_N]],
        }
    _fill_unscanned_profiles(tasks, source_status, now)

    snapshot = {
        "schema_version": 1,
        "generated_at": now,
        "generator": (
            "hull_core.model_selection.candidates() — public boards + OpenRouter "
            "catalog, scan-only (no eval, no paid calls, no human-managed fields)"
        ),
        "top_n": TOP_N,
        # Additive block (schema_version stays 1): per-source health of this run.
        "source_status": dict(sorted(source_status.items())),
        "tasks": tasks,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(snapshot, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    rows = sum(t["candidates"].__len__() for t in tasks.values())
    degraded = sorted(n for n, s in source_status.items() if s.get("status") != "ok")
    print(f"wrote {args.out}: {rows} rows across {len(tasks)} task profiles")
    if degraded:
        print(f"DEGRADED SOURCES ({len(degraded)}):")
        for n in degraded:
            s = source_status[n]
            print(f"  - {n}: {s.get('status')} rows={s.get('row_count')} reason={s.get('reason')}")
    else:
        print("all sources ok")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
