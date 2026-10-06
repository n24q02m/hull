"""Weekly model-candidates snapshot publisher (public boards, scan-only).

Runs hull_core.model_selection.candidates() for every profile in TASKS and
writes data/model-candidates.json (schema_version 1).

Scope guarantees:
- Public leaderboards + the OpenRouter catalog only. No evals, no secrets,
  no paid calls — the snapshot carries no human-managed fields.
- Embedding/rerank profiles report ``or_backbone_empty: true`` when the
  OpenRouter backbone is thin (embedders/rerankers mostly do not route
  through OpenRouter); consumers keep those cells locally managed.

Consumer stacks (crg/wet/mnemo model-sync workflows) pull this file's raw
URL and propose per-task model config pins through label-gated PRs / sticky
issues. Model defaults in code stay empty and fail-closed everywhere.
"""

from __future__ import annotations

import json
import sys
from datetime import UTC, datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "packages" / "core-py" / "src"))

from hull_core.model_selection import TASKS, candidates

TOP_N = 10
OUT = Path(__file__).resolve().parents[1] / "data" / "model-candidates.json"
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


def main() -> int:
    tasks: dict[str, dict] = {}
    for name in sorted(TASKS):
        ranked = candidates(name)
        tasks[name] = {
            "or_backbone_empty": not ranked,
            "candidates": [_row(c) for c in ranked[:TOP_N]],
        }
    snapshot = {
        "schema_version": 1,
        "generated_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "generator": (
            "hull_core.model_selection.candidates() — public boards + OpenRouter "
            "catalog, scan-only (no eval, no paid calls, no human-managed fields)"
        ),
        "top_n": TOP_N,
        "tasks": tasks,
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(snapshot, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    rows = sum(t["candidates"].__len__() for t in tasks.values())
    print(f"wrote {OUT}: {rows} rows across {len(tasks)} task profiles")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
