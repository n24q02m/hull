"""Leaderboard data sources for model selection.

Each Source returns ``dict[join_key -> SourceRecord]``. Every fetcher is
FAIL-OPEN: network/parse/auth errors -> ``{}`` (warning logged), never raised
— a dead board only weakens the signal, it never crashes the pipeline.

Join key priority: OR ``canonical_slug`` -> OR ``id`` -> ``hugging_face_id`` ->
slug of display name (see ``normalize._alias_index``).

MTEB fetchers (``mteb_classification``, ``mteb_retrieval``, ``mteb_sts``,
``mteb_reranking``) aggregate the public CC0 ``mteb/results`` HF dataset (the
``mteb/leaderboard`` dataset is gated and NOT accessible without
credentials). The dataset ships as parquet shards (~300 MB total), read via
the optional ``pyarrow`` dependency (install extra ``hull-core[mteb]``);
without a parquet reader the fetchers degrade to ``{}`` like every other
fail-open source. Scores are the mean over tasks of per-task mean scores on
the dataset's ``test`` split, keyed by the raw ``model_name`` (an HF repo id,
which joins into the OR backbone through the alias index).
"""

from __future__ import annotations

import csv
import io
import json
import logging
import os
import re
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from pathlib import Path
from urllib.parse import quote
from typing import Any, Protocol, runtime_checkable

import httpx

from hull_core.model_selection.mteb_tasks import task_family

logger = logging.getLogger(__name__)

OR_MODELS_URL = "https://openrouter.ai/api/v1/models"
# Embedding/rerank candidates (2026-10-08 directive: embed/rerank picks MUST
# be OR-servable). The base /api/v1/models response lists only text chat
# models; the two output-modality categories are separate segments of the
# same endpoint and must be fetched explicitly.
OR_MODELS_EMBED_URL = OR_MODELS_URL + "?output_modalities=embeddings"
OR_MODELS_RERANK_URL = OR_MODELS_URL + "?output_modalities=rerank"
# Structured-decision segment (profile `decision`, 2026-10-09): System One
# endpoints that return a typed choice/score/yes-no instead of chat text. Same
# row shape as the other segments; rows are gated to the decision profile
# (see ``candidates``) so they never leak into chat pools.
OR_MODELS_DECISIONS_URL = OR_MODELS_URL + "?output_modalities=decisions"
OR_ENDPOINTS_URL = "https://openrouter.ai/api/v1/models/{slug}/endpoints"
OR_BENCH_URL = "https://openrouter.ai/benchmarks/{bench}"
# Free-tier AA list endpoint: a Free-plan key gets 403 "Language models list
# requires a Pro subscription" on /api/v2/language/models but 200 on /free
# (verified 2026-10-08). Same row shape; capability indexes ride along.
AA_MODELS_URL = "https://artificialanalysis.ai/api/v2/language/models/free"
# 2026-10-07: the DontPlanToEnd/UGI-Leaderboard DATASET went gated (401
# anonymous); the author's space ships the same leaderboard as CSV with a
# richer schema ("author/model_name", "UGI <trophy>") — still public.
UGI_CSV_URL = "https://huggingface.co/spaces/DontPlanToEnd/UGI-Leaderboard/resolve/main/ugi-leaderboard-data.csv"
# LMArena leaderboard releases (dataset, updated near-daily; the old
# spaces/.../arena-leaderboard/resolve/main/leaderboard_table.parquet path
# 404s — verified 2026-10-07). One parquet carries every category view;
# ``ArenaSource`` reads the ``overall`` category.
ARENA_PARQUET_URL = (
    "https://huggingface.co/datasets/lmarena-ai/leaderboard-dataset/resolve/main/text/latest-00000-of-00001.parquet"
)

VALS_BENCH_URL = "https://www.vals.ai/benchmarks/{board}"
BENCHLM_MD_URL = "https://benchlm.ai/md/index.md"
LLMSTATS_BENCH_URL = "https://llm-stats.com/benchmarks/{bench}"
BFCL_CSV_URL = "https://gorilla.cs.berkeley.edu/data_overall.csv"
EQBENCH4_DATA_URL = "https://eqbench.com/eqbench4/eqbench4_data.js"
JEVALS_HOME_URL = "https://jevals.com/"
# Release board: GET /data/releases/<release_id>/board.json (CC-BY-4.0). The
# catalog root /data/ serves the SPA shell (no JSON index), so the newest
# release_id is resolved from the server-rendered home page, which embeds
# `release_id:"YYYY-MM-DD"` in its payload (mechanism verified live 2026-10-09;
# /data/releases/latest/board.json is 404 — only dated releases exist).
JEVALS_BOARD_URL = "https://jevals.com/data/releases/{release_id}/board.json"
JEVALS_RELEASE_RE = re.compile(r'release_id\s*:\s*["\'](\d{4}-\d{2}-\d{2})["\']')
# Decision profile staleness guard (ticket 2026-10-09): jevals ships dated
# frozen releases; if the resolved release_id has not moved for 90 days the
# board is no longer "live" evidence and reports degraded (static anchor).
JEVALS_STATIC_ANCHOR_DAYS = 90
JEVALS_BENCHMARKS = ("banking77", "helpsteer2", "pubmedqa")
# JevBench (profile `decision`, 2026-10-09): independent second decision board
# (board #2 for the >=2-independent-boards strong rule). Pure JSON, one call;
# only ``api_flag == true`` systems are joinable (hosted routes) — self-hosted
# open-weight systems have no OR route and report unmatched, never mapped.
JEVBENCH_URL = "https://benchmarkheaven.com/api/jevbench/v1.6.1"
# Jev Decision Index (profile `decision`, 2026-10-09, user-named source): HF
# Space multimodalart/jev-decision-index (static SDK, HF-staff maintainer,
# lastModified 2026-10-07) — the reference board for open-weight Jev repros.
# Population is mostly NOT on OR: after the OR-servable hard filter only the
# decisions-segment repros survive (clef, clef-flash, kev-4b, tev1-4b) — the
# rest report unmatched. Thin post-filter by design; that is the honest state.
JEV_DECISION_INDEX_URL = "https://huggingface.co/spaces/multimodalart/jev-decision-index/raw/main/data/index.json"
# MangaVQA/MangaOCR project site (manga109 org GitHub Pages; the README's
# atsumiyai.github.io link 404s — manga109.github.io is the live surface,
# verified 2026-10-08).
MANGA_BENCH_URL = "https://manga109.github.io/MangaVQA_LMM/"
BRIDGE_LEADERBOARD_URL = (
    "https://huggingface.co/spaces/YLab-Open/BRIDGE-Medical-Leaderboard/"
    "resolve/main/leaderboards/{mode}_leaderboard.json"
)
GAIA_RESULTS_URL = (
    "https://huggingface.co/datasets/gaia-benchmark/results_public/"
    "resolve/refs%2Fconvert%2Fparquet/2023/test/0000.parquet"
)
OPEN_MEDICAL_RESULTS_TREE_URL = "https://huggingface.co/api/datasets/openlifescienceai/results/tree/main?recursive=true"
OPEN_MEDICAL_RESULT_FILE_URL = "https://huggingface.co/datasets/openlifescienceai/results/resolve/main/{path}"
MEDHELM_RELEASES_URL = (
    "https://storage.googleapis.com/storage/v1/b/crfm-helm-public/o"
    "?prefix=medhelm/benchmark_output/releases/&delimiter=/"
)
MEDHELM_GROUP_URL = (
    "https://storage.googleapis.com/crfm-helm-public/medhelm/benchmark_output/"
    "releases/{release}/groups/medhelm_scenarios.json"
)

TAU2_BENCHES = ("tau2-bench-airline", "tau2-bench-retail", "tau2-bench-telecom", "tau2-bench-banking")
BRIDGE_MODES = ("Zero-Shot", "Few-Shot", "CoT")

MTEB_PARQUET_LIST_URL = "https://datasets-server.huggingface.co/parquet?dataset=mteb%2Fresults"

DEFAULT_TIMEOUT = 30.0
DAY = 24 * 3600
WEEK = 7 * DAY

_SLUG_RE = re.compile(r"[^a-z0-9]+")


def slugify(text: str) -> str:
    """Normalize a model name to a slug: lowercase, keep [a-z0-9], '-' for other runs."""
    return _SLUG_RE.sub("-", text.lower()).strip("-")


@dataclass
class SourceRecord:
    """One score row from a leaderboard, before joining into the OR backbone."""

    key: str  # best join key the source has: slug / hf_id / raw name
    name: str = ""  # display name on the board
    score: float | None = None  # raw board score (not normalized)
    score_ci: float | None = None  # confidence interval +- when published
    board_version: str | None = None  # e.g. "v4.2" for AA index — drives version guard
    raw: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "key": self.key,
            "name": self.name,
            "score": self.score,
            "score_ci": self.score_ci,
            "board_version": self.board_version,
            "raw": self.raw,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> SourceRecord:
        return cls(
            key=d["key"],
            name=d.get("name", ""),
            score=d.get("score"),
            score_ci=d.get("score_ci"),
            board_version=d.get("board_version"),
            raw=d.get("raw") or {},
        )


@dataclass
class SourceStatus:
    """Per-source health of one snapshot run (publisher ``source_status`` block).

    ``status``:
    - ``ok`` — fetch (or fresh cache hit) returned rows.
    - ``stale`` — fetch failed/empty but a stale cached snapshot was served.
    - ``missing`` — nothing fetched and nothing stale to fall back on
      (key absent, anchor file not published, network dead, empty board).
    - ``unimplemented_access`` — the board has no fetchable public surface
      (probed); recorded honestly instead of faking rows.
    """

    status: str
    fetched_at: str | None = None  # ISO-8601 UTC of the data pull
    row_count: int = 0
    reason: str | None = None
    # Board data date from API/commit metadata (GitHub commits API, HF dataset
    # API, arXiv dateline — never a page footer). Served fresh only: a stale
    # cache hit cannot re-derive it, so cached snapshots leave it unset.
    board_updated_at: str | None = None
    # Verbatim credit line for licensed datasets (OR usage is CC BY 4.0).
    attribution: str | None = None

    def to_dict(self) -> dict[str, Any]:
        out: dict[str, Any] = {"status": self.status, "fetched_at": self.fetched_at, "row_count": self.row_count}
        if self.reason:
            out["reason"] = self.reason
        if self.board_updated_at:
            out["board_updated_at"] = self.board_updated_at
        if self.attribution:
            out["attribution"] = self.attribution
        return out


@runtime_checkable
class Source(Protocol):
    """Protocol for every leaderboard fetcher (including consumer-side plugins)."""

    name: str
    ttl_seconds: int

    def fetch(self) -> dict[str, SourceRecord]:
        """Returns {} on error — fail-open, never raises."""
        ...


def _get_json(url: str, *, headers: dict[str, str] | None = None, params: dict[str, str] | None = None) -> Any:
    resp = httpx.get(url, headers=headers, params=params, timeout=DEFAULT_TIMEOUT, follow_redirects=True)
    resp.raise_for_status()
    return resp.json()


def _get_bytes(url: str, *, headers: dict[str, str] | None = None) -> bytes:
    resp = httpx.get(url, headers=headers, timeout=DEFAULT_TIMEOUT, follow_redirects=True)
    resp.raise_for_status()
    return resp.content


def _float(value: Any) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _first_col(fieldnames: list[str], candidates: tuple[str, ...]) -> str | None:
    """Find a column by candidate names (case-insensitive)."""
    lowered = {f.lower().strip(): f for f in fieldnames}
    for cand in candidates:
        if cand in lowered:
            return lowered[cand]
    # fallback: column containing the substring
    for cand in candidates:
        for low, orig in lowered.items():
            if cand in low:
                return orig
    return None


def _parse_csv_records(
    text: str,
    *,
    name_cols: tuple[str, ...],
    score_cols: tuple[str, ...],
    ci_cols: tuple[str, ...] = (),
) -> dict[str, SourceRecord]:
    """Parse a defensive CSV leaderboard: auto-detect name + score columns."""
    reader = csv.DictReader(io.StringIO(text))
    if not reader.fieldnames:
        return {}
    name_col = _first_col(list(reader.fieldnames), name_cols)
    score_col = _first_col(list(reader.fieldnames), score_cols)
    ci_col = _first_col(list(reader.fieldnames), ci_cols) if ci_cols else None
    if not name_col or not score_col:
        logger.warning("model_selection csv: cannot detect columns: name_col=%s score_col=%s", name_col, score_col)
        return {}
    records: dict[str, SourceRecord] = {}
    for row in reader:
        name = (row.get(name_col) or "").strip()
        score = _float(row.get(score_col))
        if not name or score is None:
            continue
        rec = SourceRecord(
            key=slugify(name),
            name=name,
            score=score,
            score_ci=_float(row.get(ci_col)) if ci_col else None,
            raw=dict(row),
        )
        records[rec.key] = rec
    return records


def _strip_tags(html: str) -> str:
    """Collapse an HTML fragment to visible text with single spaces."""
    return re.sub(r"\s+", " ", re.sub(r"<[^>]+>", " ", html)).strip()


def _table_rows(html: str) -> list[list[str]]:
    """First HTML table in ``html`` -> list of rows of visible cell texts.

    Works on server-rendered fragments (vals.ai accessibility tables, OR
    benchmark pages, llm-stats boards). Header rows are included; callers
    validate shape defensively.
    """
    start = html.find("<table")
    if start < 0:
        return []
    end = html.find("</table>", start)
    frag = html[start : end + len("</table>")] if end >= 0 else html[start:]
    rows: list[list[str]] = []
    for tr in re.findall(r"<tr[^>]*>(.*?)</tr>", frag, re.S):
        cells = [_strip_tags(c) for c in re.findall(r"<t[hd][^>]*>(.*?)</t[hd]>", tr, re.S)]
        rows.append(cells)
    return rows


def _parse_pct(text: str) -> float | None:
    """``79.9%`` / ``77.47%`` -> 79.9; plain floats pass through; else None."""
    m = re.match(r"^(\d+(?:\.\d+)?)\s*%$", text.strip())
    if m:
        return float(m.group(1))
    return _float(text)


def _parse_pm(text: str) -> float | None:
    """``± 2.8 pp`` / ``+-1.2`` -> 2.8 (magnitude only); else None."""
    m = re.match(r"^[±+\-]\s*(\d+(?:\.\d+)?)", text.strip())
    return float(m.group(1)) if m else None


class _BaseSource:
    """Fail-open boilerplate: ``fetch`` wraps ``_fetch``, any exception -> {}.

    ``missing_reason`` is set by subclasses that return ``{}`` for a KNOWN,
    honest reason (key absent, anchor not published) so the publisher's
    ``source_status`` block can record it instead of a bare ``missing``.
    """

    name: str = "base"
    ttl_seconds: int = DAY

    def __init__(self) -> None:
        self.missing_reason: str | None = None

    def fetch(self) -> dict[str, SourceRecord]:
        try:
            records = self._fetch()
        except Exception as exc:
            logger.warning("model_selection source failed: source=%s error=%s", self.name, exc)
            self.missing_reason = f"error: {exc}"
            return {}
        if not records and self.missing_reason is None:
            self.missing_reason = "empty_fetch"
        elif records:
            self.missing_reason = None
        return records

    def _fetch(self) -> dict[str, SourceRecord]:
        raise NotImplementedError


class UnimplementedSource(_BaseSource):
    """A board the research named that has no fetchable public surface.

    Registered so TaskProfile names resolve and the snapshot records the gap
    as ``unimplemented_access`` (never silent, never faked rows).
    """

    reason: str = "no public fetchable surface"

    def __init__(self, name: str, reason: str) -> None:
        super().__init__()
        self.name = name
        self.reason = reason

    def _fetch(self) -> dict[str, SourceRecord]:
        self.missing_reason = self.reason
        return {}


class OpenRouterModelsSource(_BaseSource):
    """Mandatory backbone: GET /api/v1/models (+ embeddings/rerank/decisions
    segments).

    The base response lists only text chat models; embedding, rerank and
    structured-decision candidates live in dedicated ``?output_modalities=``
    segments of the same endpoint (2026-10-08 directive — embed/rerank picks
    must be OR-servable; decisions segment added for profile ``decision``,
    2026-10-09). All four segments are fetched and merged by id; a failing
    category segment degrades to the remaining rows (never kills the text
    backbone). Each merged record carries ``raw["_or_segments"]`` — the list
    of segments the id appeared in — which ``candidates()`` uses as the
    modality gate (decisions-only rows never leak into chat pools).

    Each record keeps the raw JSON row in ``raw`` — pricing (USD/token,
    including tiered ``overrides`` by ``min_prompt_tokens``),
    ``context_length``, ``architecture.modality``, ``supported_parameters``,
    plus AA scores embedded in ``benchmarks.artificial_analysis.*`` (present
    only on some rows; absent = missing, not 0).
    """

    name = "openrouter_models"
    ttl_seconds = DAY

    def _fetch(self) -> dict[str, SourceRecord]:
        segments = (
            (OR_MODELS_URL, "text"),
            (OR_MODELS_EMBED_URL, "embeddings"),
            (OR_MODELS_RERANK_URL, "rerank"),
            (OR_MODELS_DECISIONS_URL, "decisions"),
        )
        rows: list[tuple[dict, str]] = []
        for url, segment in segments:
            try:
                data = _get_json(url)
            except Exception as exc:
                logger.warning("model_selection: OR catalog segment failed url=%s error=%s", url, exc)
                continue
            segment_rows = data.get("data") if isinstance(data, dict) else data
            rows.extend((row, segment) for row in segment_rows or [] if isinstance(row, dict))
        records: dict[str, SourceRecord] = {}
        for row, segment in rows:
            slug = row.get("id")
            if not slug:
                continue
            existing = records.get(slug)
            if existing is not None:
                segs = existing.raw.setdefault("_or_segments", [])
                if segment not in segs:
                    segs.append(segment)
                continue
            row["_or_segments"] = [segment]
            records[slug] = SourceRecord(key=slug, name=row.get("name") or slug, raw=row)
        return records


class ArtificialAnalysisSource(_BaseSource):
    """AA Data API free tier: GET /api/v2/language/models/free, header
    ``x-api-key``.

    Key is REQUIRED (keyless requests get ``401 {"error": "API key is
    required"}``; a Free-plan key is 403 on the non-free sibling endpoint —
    verified 2026-10-08). Read from ``AA_API_KEY`` at fetch time; without a
    key -> {} with ``missing_reason=aa_api_key_required`` (fail-open skip,
    recorded missing — never silent, never scraped). Free rows carry the
    intelligence index at ``evaluations.artificial_analysis_intelligence_index``
    and its cost at top level, but NO ``*_index_version`` — ``board_version``
    stays None and the version guard never sees these rows (falsy versions
    are dropped at join).
    """

    name = "artificial_analysis"
    ttl_seconds = DAY

    def __init__(self, api_key: str | None = None) -> None:
        super().__init__()
        self._api_key = api_key

    def _fetch(self) -> dict[str, SourceRecord]:
        api_key = self._api_key or os.environ.get("AA_API_KEY")
        if not api_key:
            self.missing_reason = "aa_api_key_required"
            logger.info("model_selection: AA_API_KEY absent, skip artificial_analysis")
            return {}
        data = _get_json(AA_MODELS_URL, headers={"x-api-key": api_key})
        rows = data.get("data") if isinstance(data, dict) else data
        records: dict[str, SourceRecord] = {}
        for row in rows or []:
            if not isinstance(row, dict):
                continue
            name = row.get("name") or row.get("model") or row.get("slug") or ""
            slug = row.get("slug") or slugify(name)
            if not slug:
                continue
            evals = row.get("evaluations") if isinstance(row.get("evaluations"), dict) else row
            score = _float(
                evals.get("artificial_analysis_intelligence_index")
                or evals.get("intelligence_index")
                or evals.get("smart_index")
            )
            version = evals.get("artificial_analysis_index_version") or evals.get("index_version")
            records[slug] = SourceRecord(
                key=slugify(slug),
                name=name or slug,
                score=score,
                board_version=str(version) if version else None,
                raw=row,
            )
        return records


class UGISource(_BaseSource):
    """UGI leaderboard (HF CSV) — the ``permissive`` profile's specialized board.

    Registered in ``SOURCE_REGISTRY`` as ``"ugi"`` and wired into the
    ``permissive`` TaskProfile (spec 2026-09-28 D-KP6, phase H); consumers may
    still override it via ``candidates(..., sources={..., "ugi": <Source>})``
    with an in-house Source.
    """

    name = "ugi"
    ttl_seconds = WEEK

    def _fetch(self) -> dict[str, SourceRecord]:
        text = _get_bytes(UGI_CSV_URL).decode("utf-8", errors="replace")
        return _parse_csv_records(
            text,
            name_cols=("model", "model_name", "name"),
            score_cols=("ugi", "ugi_score", "score"),
        )


def _read_parquet_rows(blob: bytes) -> list[dict[str, Any]]:
    """Read a parquet blob into row dicts; needs ``pandas`` or ``pyarrow``."""
    try:
        import pandas as pd  # ty: ignore[unresolved-import]  # optional reader; absent -> pyarrow path
    except ImportError:
        pd = None
    if pd is not None:
        return pd.read_parquet(io.BytesIO(blob)).to_dict("records")
    try:
        import pyarrow.parquet as pq
    except ImportError as exc:
        raise RuntimeError("parquet source needs pandas or pyarrow") from exc
    return pq.read_table(io.BytesIO(blob)).to_pylist()


class ArenaSource(_BaseSource):
    """LMArena via the ``lmarena-ai/leaderboard-dataset`` HF release — the
    ``overall`` category Elo board.

    The previous mirror (``spaces/lmarena-ai/arena-leaderboard/.../leaderboard_table.parquet``)
    404s and the space only carries legacy MT-bench CSVs (verified 2026-10-07);
    the leaderboard dataset publishes ``model_name / rating / rating_lower /
    rating_upper / category`` parquet shards updated near-daily. Parquet needs
    ``pandas`` or ``pyarrow`` (optional, not a hull dependency) -> missing
    parser or schema change = {}.
    """

    name = "arena"
    ttl_seconds = WEEK

    def _fetch(self) -> dict[str, SourceRecord]:
        blob = _get_bytes(ARENA_PARQUET_URL)
        records: dict[str, SourceRecord] = {}
        for row in _read_parquet_rows(blob):
            if str(row.get("category") or "overall") != "overall":
                continue
            name = str(row.get("model_name") or "").strip()
            score = _float(row.get("rating"))
            if not name or score is None:
                continue
            lo, hi = _float(row.get("rating_lower")), _float(row.get("rating_upper"))
            ci = (hi - lo) / 2 if lo is not None and hi is not None else None
            rec = SourceRecord(
                key=slugify(name),
                name=name,
                score=score,
                score_ci=ci,
                board_version=str(row.get("leaderboard_publish_date")) if row.get("leaderboard_publish_date") else None,
                raw=dict(row),
            )
            records[rec.key] = rec
        return records


# --- boards-v2 fetchers (2026-10-09 directive: closed-inclusive boards + OR usage)
#
# Conventions: parsers are pure functions over fetched bytes/text (fixture-
# testable); every board captures ``board_updated_at`` from API/commit
# metadata (GitHub commits/repo API, HF dataset rows, arXiv dateline — never
# a page footer); join names for these boards are versioned in
# ``board_aliases.py`` and rows matching nothing are REPORTED, never guessed.

AGENTSET_EMBED_URL = "https://raw.githubusercontent.com/agentset-ai/embedding-leaderboard/main/results/benchmarks.json"
AGENTSET_EMBED_COMMIT_URL = (
    "https://api.github.com/repos/agentset-ai/embedding-leaderboard/commits?path=results/benchmarks.json&per_page=1"
)
AGENTSET_RERANK_URL = "https://agentset.ai/rerankers"
AGENTSET_RERANK_REPO_URL = "https://api.github.com/repos/agentset-ai/reranker-eval"
HINDSIGHT_CONTENTS_URL = (
    "https://api.github.com/repos/vectorize-io/hindsight-benchmarks/contents/results/leaderboard/{sub}"
)
HINDSIGHT_RAW_URL = (
    "https://raw.githubusercontent.com/vectorize-io/hindsight-benchmarks/main/results/leaderboard/{sub}/{name}"
)
HINDSIGHT_COMMIT_URL = (
    "https://api.github.com/repos/vectorize-io/hindsight-benchmarks/commits?path=results/leaderboard&per_page=1"
)
# LMArena agent arena: same parquet release infra as ``arena``, branch `agent`
# (verified 2026-10-09: rows carry tier suffixes (Max)/(High)/(xHigh) and a
# ``leaderboard_publish_date`` column).
ARENA_AGENT_PARQUET_URL = (
    "https://huggingface.co/datasets/lmarena-ai/leaderboard-dataset/resolve/main/agent/latest-00000-of-00001.parquet"
)
WMT25_HTML_URL = "https://arxiv.org/html/2508.14909v2"  # WMT25 General MT preliminary ranking
WMT25_ABS_URL = "https://arxiv.org/abs/2508.14909"
OR_USAGE_DAILY_URL = "https://openrouter.ai/api/v1/datasets/rankings-daily"
OR_USAGE_FRONTEND_URL = "https://openrouter.ai/api/frontend/v1/rankings/models"
OR_USAGE_WINDOW_DAYS = 30

_ARENA_TIER_SUFFIX = re.compile(r"\s*\((?:Max|High|xHigh)\)\s*$")
_ARENA_SNAPSHOT_SUFFIX = re.compile(r"\s*\(\d{3,4}\)\s*$")
_WMT25_LATEX_TOKEN = re.compile(r"\\[a-zA-Z]+")
_OR_PERMASLUG_DATE_TAIL = re.compile(r"-\d{8}$")


def _github_commit_date(url: str) -> str | None:
    """``YYYY-MM-DD`` of the most recent commit touching a path (GitHub REST)."""
    data = _get_json(url)
    row = data[0] if isinstance(data, list) and data else None
    date = row.get("commit", {}).get("committer", {}).get("date") if isinstance(row, dict) else None
    return str(date)[:10] or None if date else None


def _arxiv_last_revised(abs_url: str) -> str | None:
    """``YYYY-MM-DD`` of the arXiv paper version, from the abs-page dateline
    (arXiv's own API-served metadata; "last revised", falling back to the
    submission date)."""
    html = _get_bytes(abs_url).decode("utf-8", errors="replace")
    for pattern in (r"last revised (\d{1,2} \w+ \d{4})", r"Submitted on (\d{1,2} \w+ \d{4})"):
        m = re.search(pattern, html)
        if m:
            try:
                return datetime.strptime(m.group(1), "%d %b %Y").date().isoformat()
            except ValueError:
                continue
    return None


def _usage_base_slug(permaslug: str) -> str:
    """Permaslug -> base slug: drop ``:variant`` and the ``-YYYYMMDD`` date
    tail on the last segment (``voyageai/rerank-2.5-lite-20260727`` ->
    ``voyageai/rerank-2.5-lite``) so usage joins the OR catalog id."""
    base = permaslug.split(":", 1)[0]
    org, sep, part = base.rpartition("/")
    part = _OR_PERMASLUG_DATE_TAIL.sub("", part)
    return sep.join((org, part)) if sep else part


class AgentSetEmbedSource(_BaseSource):
    """AgentSet embeddings leaderboard: LLM-judge Elo over RAG retrieval
    (FiQa/MSMARCO/SciFact/DBPedia/...), 18 models incl. the closed vendors
    MTEB lacks. Mirror: ``agentset-ai/embedding-leaderboard`` results/benchmarks.json
    (a JSON list of ``{name, overall: {elo, ...}, by_dataset}``). Stale
    (~2026-03) but closed-inclusive — always paired with an independent board
    before promoting an embed pick. ``board_updated_at`` = GitHub commits API
    for the JSON path (the site prints no trustworthy date)."""

    name = "agentset_elo"
    ttl_seconds = WEEK

    def _fetch(self) -> dict[str, SourceRecord]:
        rows = _get_json(AGENTSET_EMBED_URL)
        try:
            self.board_updated_at = _github_commit_date(AGENTSET_EMBED_COMMIT_URL)
        except Exception as exc:  # commit metadata must never kill the board
            logger.info("model_selection agentset_elo commit date unavailable: %s", exc)
            self.board_updated_at = None
        records: dict[str, SourceRecord] = {}
        for row in rows if isinstance(rows, list) else []:
            if not isinstance(row, dict):
                continue
            name = str(row.get("name") or "").strip()
            overall = row.get("overall") if isinstance(row.get("overall"), dict) else {}
            elo = _float(overall.get("elo"))
            if not name or elo is None:
                continue
            rec = SourceRecord(
                key=slugify(name),
                name=name,
                score=elo,
                board_version=self.board_updated_at,
                raw={"overall": overall},
            )
            records[rec.key] = rec
        return records


def _parse_agentset_rerank_table(html: str) -> dict[str, SourceRecord]:
    """AgentSet rerankers static table: header carries an ``ELO`` column;
    one row per reranker (``Cohere Rerank 4 Pro``, ``Zerank 2``, ...)."""
    start = html.find("<table")
    if start < 0:
        return {}
    end = html.find("</table>", start)
    frag = html[start : end + len("</table>")] if end >= 0 else html[start:]
    parsed = [
        [_strip_tags(c) for c in re.findall(r"<t[hd][^>]*>(.*?)</t[hd]>", tr, re.S)]
        for tr in re.findall(r"<tr[^>]*>(.*?)</tr>", frag, re.S)
    ]
    if not parsed:
        return {}
    elo_col = next((i for i, cell in enumerate(parsed[0]) if "elo" in cell.lower()), None)
    if elo_col is None:
        return {}
    records: dict[str, SourceRecord] = {}
    for row in parsed[1:]:
        if len(row) <= elo_col:
            continue
        name = row[0].strip()
        elo = _float(row[elo_col])
        if not name or elo is None:
            continue
        rec = SourceRecord(key=slugify(name), name=name, score=elo)
        records[rec.key] = rec
    return records


class AgentSetRerankSource(_BaseSource):
    """AgentSet rerankers board: the static HTML table on agentset.ai/rerankers
    (server-rendered — no JS needed; verified 2026-10-09). Stale (~2026-02):
    newer routes (e.g. Voyage rerank-3) are absent, so OR usage stays the
    liveness signal for rerank picks. ``board_updated_at`` = the
    ``agentset-ai/reranker-eval`` repo ``pushed_at`` (the site prints no
    date). Fragile HTML: parse defensively, empty table -> ``{}``."""

    name = "agentset_rerank"
    ttl_seconds = WEEK

    def _fetch(self) -> dict[str, SourceRecord]:
        html = _get_bytes(AGENTSET_RERANK_URL).decode("utf-8", errors="replace")
        records = _parse_agentset_rerank_table(html)
        if not records:
            self.missing_reason = "agentset_rerank_table_empty"
            return {}
        try:
            data = _get_json(AGENTSET_RERANK_REPO_URL)
            pushed = str(data.get("pushed_at") or "")[:10] or None if isinstance(data, dict) else None
            self.board_updated_at = pushed
        except Exception as exc:
            logger.info("model_selection agentset_rerank repo date unavailable: %s", exc)
            self.board_updated_at = None
        for rec in records.values():
            rec.board_version = self.board_updated_at
        return records


class HindsightSource(_BaseSource):
    """Hindsight benchmarks (vectorize-io): one small JSON per model under
    ``results/leaderboard/{reranker,embeddings}/`` (MRR + recall@k on the
    LoComo ``recall()`` harness, n=165 — directional, self-admitted bias).
    Files are listed via the GitHub contents API and fetched raw; a file that
    fails to parse is skipped (fail-open), an empty listing -> ``{}``.
    ``board_updated_at`` = GitHub commits API for the leaderboard directory."""

    ttl_seconds = WEEK

    def __init__(self, name: str, sub: str, id_field: str) -> None:
        super().__init__()
        self.name = name
        self.sub = sub  # "reranker" | "embeddings"
        self.id_field = id_field  # "reranker_id" | "embedding_id"

    def _fetch(self) -> dict[str, SourceRecord]:
        listing = _get_json(HINDSIGHT_CONTENTS_URL.format(sub=self.sub))
        files = [
            str(e.get("name"))
            for e in (listing if isinstance(listing, list) else [])
            if isinstance(e, dict) and str(e.get("name", "")).endswith(".json")
        ]
        if not files:
            self.missing_reason = "hindsight_listing_empty"
            return {}
        records: dict[str, SourceRecord] = {}
        for fname in files:
            try:
                blob = _get_json(HINDSIGHT_RAW_URL.format(sub=self.sub, name=fname))
            except Exception as exc:  # one bad file must not kill the board
                logger.info("model_selection hindsight file skipped: sub=%s file=%s error=%s", self.sub, fname, exc)
                continue
            if not isinstance(blob, dict):
                continue
            key = str(blob.get(self.id_field) or Path(fname).stem).strip()
            score = _float(blob.get("mrr"))
            if not key or score is None:
                continue
            records[key] = SourceRecord(
                key=key,
                name=key,
                score=score,
                raw={
                    "recall_at_1": blob.get("recall_at_1"),
                    "recall_at_3": blob.get("recall_at_3"),
                    "recall_at_5": blob.get("recall_at_5"),
                },
            )
        try:
            self.board_updated_at = _github_commit_date(HINDSIGHT_COMMIT_URL)
        except Exception as exc:
            logger.info("model_selection hindsight commit date unavailable: %s", exc)
            self.board_updated_at = None
        for rec in records.values():
            rec.board_version = self.board_updated_at
        return records


class ArenaAgentSource(_BaseSource):
    """LMArena agent arena: the ``agent`` branch of the same
    ``lmarena-ai/leaderboard-dataset`` parquet release as ``arena`` (published
    2026-10-02, weekly cadence; verified 2026-10-09). Rows carry effort-tier
    suffixes ``(Max)/(High)/(xHigh)`` — settings of the SAME model, so they
    are stripped (snapshot markers like ``(0813)`` too) and a stripped-name
    collision collapses to the best observed score; the OR-side ladder still
    uniqueness-gates the join. ``board_updated_at`` = the parquet's
    ``leaderboard_publish_date`` column max."""

    name = "arena_agent"
    ttl_seconds = WEEK

    def _fetch(self) -> dict[str, SourceRecord]:
        rows = _read_parquet_rows(_get_bytes(ARENA_AGENT_PARQUET_URL))
        best: dict[str, tuple[float, SourceRecord]] = {}
        dates: list[str] = []
        for row in rows:
            if str(row.get("category") or "overall") != "overall":
                continue
            name = str(row.get("model_name") or "").strip()
            for _ in range(2):  # "(High) (0813)": snapshot then tier
                stripped = _ARENA_SNAPSHOT_SUFFIX.sub("", _ARENA_TIER_SUFFIX.sub("", name))
                if stripped == name:
                    break
                name = stripped
            score = _float(row.get("score"))
            if not name or score is None:
                continue
            date = str(row.get("leaderboard_publish_date") or "")
            if date:
                dates.append(date)
            lo, hi = _float(row.get("score_ci_lower")), _float(row.get("score_ci_upper"))
            rec = SourceRecord(
                key=slugify(name),
                name=name,
                score=score,
                score_ci=(hi - lo) / 2 if lo is not None and hi is not None else None,
                board_version=date or None,
                raw=dict(row),
            )
            prev = best.get(rec.key)
            if prev is None or score > prev[0]:
                best[rec.key] = (score, rec)
        self.board_updated_at = max(dates) if dates else None
        return {key: rec for key, (_, rec) in best.items()}


def _parse_wmt25_autorank(html: str) -> dict[str, SourceRecord]:
    """WMT25 per-language-pair ranking tables -> one record per system.

    Each table has a label row (language pair), a header row with an
    ``AutoRank`` column (lower = better), and one row per system; system cells
    carry LaTeX decoration (``\\blacktriangle``) and ``[M]``-style markers.
    Board score = NEGATIVE mean AutoRank across the pairs a system appears in,
    so the pipeline's higher-is-better min-max normalization stays honest.
    """
    per_system: dict[str, list[float]] = {}
    for table_m in re.finditer(r"<table.*?</table>", html, re.S):
        parsed = [
            [_strip_tags(c) for c in re.findall(r"<t[hd][^>]*>(.*?)</t[hd]>", tr, re.S)]
            for tr in re.findall(r"<tr[^>]*>(.*?)</tr>", table_m.group(0), re.S)
        ]
        if len(parsed) < 3:
            continue
        header_idx = next((i for i, row in enumerate(parsed) if any("AutoRank" in c for c in row)), None)
        if header_idx is None or header_idx + 1 >= len(parsed):
            continue
        rank_col = next(i for i, cell in enumerate(parsed[header_idx]) if "AutoRank" in cell)
        for row in parsed[header_idx + 1 :]:
            if len(row) <= rank_col:
                continue
            name = _WMT25_LATEX_TOKEN.sub(" ", row[0])
            name = re.sub(r"^[^A-Za-z0-9]+", "", name)
            name = re.sub(r"\[[A-Za-z]\]\s*$", "", name).strip()
            rank = _float(row[rank_col])
            if not name or rank is None:
                continue
            per_system.setdefault(name, []).append(rank)
    records: dict[str, SourceRecord] = {}
    for name, ranks in per_system.items():
        rec = SourceRecord(
            key=slugify(name),
            name=name,
            score=-sum(ranks) / len(ranks),
            raw={"n_lps": len(ranks), "mean_autorank": sum(ranks) / len(ranks)},
        )
        records[rec.key] = rec
    return records


class Wmt25GmtrSource(_BaseSource):
    """WMT25 General MT preliminary ranking (arXiv 2508.14909v2) — the annual
    closed-inclusive translation anchor. Mean AutoRank across the 31 language
    pairs (published negated: higher = better). Static per year (arXiv HTML
    re-fetched weekly is one ~3 MB request). ``board_updated_at`` = the
    arXiv dateline revision date (2025-08-24 for v2), never a page footer."""

    name = "wmt25_gmtr"
    ttl_seconds = WEEK

    def _fetch(self) -> dict[str, SourceRecord]:
        html = _get_bytes(WMT25_HTML_URL).decode("utf-8", errors="replace")
        records = _parse_wmt25_autorank(html)
        if not records:
            self.missing_reason = "wmt25_tables_empty"
            return {}
        try:
            self.board_updated_at = _arxiv_last_revised(WMT25_ABS_URL)
        except Exception as exc:
            logger.info("model_selection wmt25 revision date unavailable: %s", exc)
            self.board_updated_at = None
        for rec in records.values():
            rec.board_version = self.board_updated_at
        return records


def _usage_rows(data: Any) -> list[dict[str, Any]]:
    """Rankings payload -> row dicts (both feeds nest rows under ``data``)."""
    if isinstance(data, dict):
        data = data.get("data") or data.get("rows") or []
    return [row for row in (data or []) if isinstance(row, dict)]


def _usage_slug(row: dict[str, Any]) -> str | None:
    for key in ("model_permaslug", "permaslug", "slug", "model"):
        value = row.get(key)
        if value:
            return str(value)
    return None


def _usage_tokens(row: dict[str, Any]) -> float | None:
    for key in ("total_tokens", "rankingMetricValue", "tokens"):
        value = _float(row.get(key))
        if value is not None:
            return value
    return None


class OrUsageSource(_BaseSource):
    """OpenRouter real-world usage -> ``or_task_spend_share`` (tie-break ONLY).

    NOT a quality board: ``candidates()`` folds the aggregated totals into
    ``ModelCandidate.or_task_spend_share`` for the quadrant tie-break; usage
    never enters scores/evidence, and absence is neutral.

    Two feeds, both aggregated to total tokens per BASE permaslug (``:variant``
    and ``-YYYYMMDD`` date tails stripped):
    - PRIMARY ``GET /api/v1/datasets/rankings-daily`` (documented Data API,
      ``Authorization: Bearer $OPENROUTER_API_KEY``, CC BY 4.0 — the
      attribution line is recorded verbatim in the snapshot). Key absent ->
      skipped (fail-open, recorded).
    - SUPPLEMENT ``GET /api/frontend/v1/rankings/models`` (public, keyless,
      UNDOCUMENTED — verified 2026-10-09 to carry the embeddings/rerank
      long-tail the primary top-50/day ranking omits). May vanish without
      notice: fail-open.

    Both empty -> ``{}`` with ``or_usage_api_key_required`` /
    ``or_usage_unavailable``; ``board_updated_at`` = newest window date seen.
    """

    name = "or_usage"
    ttl_seconds = DAY

    def _fetch(self) -> dict[str, SourceRecord]:
        totals: dict[str, float] = {}
        dates: list[str] = []
        as_ofs: list[str] = []  # explicit meta.as_of from the documented feed
        api_key = os.environ.get("OPENROUTER_API_KEY")
        key_absent = not api_key
        if api_key:
            try:
                end = datetime.now(UTC).date()
                start = end - timedelta(days=OR_USAGE_WINDOW_DAYS)
                data = _get_json(
                    OR_USAGE_DAILY_URL,
                    headers={"Authorization": f"Bearer {api_key}"},
                    params={"start_date": start.isoformat(), "end_date": end.isoformat()},
                )
                meta = data.get("meta") if isinstance(data, dict) else None
                as_of = str((meta or {}).get("as_of") or "")[:10]
                if as_of:
                    as_ofs.append(as_of)
                for row in _usage_rows(data):
                    slug = _usage_slug(row)
                    tokens = _usage_tokens(row)
                    if not slug or tokens is None:
                        continue
                    base = _usage_base_slug(slug)
                    totals[base] = totals.get(base, 0.0) + tokens
                    date = str(row.get("date") or "")
                    if date:
                        dates.append(date[:10])
            except Exception as exc:  # primary feed must not kill the supplement
                logger.info("model_selection or_usage primary failed: %s", exc)
        try:
            data = _get_json(OR_USAGE_FRONTEND_URL)
            for row in _usage_rows(data):
                slug = _usage_slug(row)
                tokens = _usage_tokens(row)
                if not slug or tokens is None:
                    continue
                base = _usage_base_slug(slug)
                totals[base] = totals.get(base, 0.0) + tokens
                date = str(row.get("date") or "")
                if date:
                    dates.append(date[:10])
        except Exception as exc:
            logger.info("model_selection or_usage supplement failed: %s", exc)
        if not totals:
            self.missing_reason = "or_usage_api_key_required" if key_absent else "or_usage_unavailable"
            return {}
        self.board_updated_at = max(dates) if dates else None
        self.attribution = f"Source: OpenRouter (openrouter.ai/rankings), as of {self.board_updated_at or 'unknown'}"
        return {
            slug: SourceRecord(key=slug, name=slug, score=tokens, raw={"total_tokens": tokens})
            for slug, tokens in sorted(totals.items())
        }


# --- MTEB fetchers ------------------------------------------------------------
#
# All four families share one expensive step (list + download + aggregate the
# ``mteb/results`` parquet shards), memoized per process and per family result
# so e.g. ``candidates("embedding")`` pays for one download across
# mteb_classification / mteb_retrieval / mteb_sts.


def _mteb_parquet_urls() -> list[str]:
    """List parquet shard URLs for ``mteb/results`` via the datasets-server API."""
    data = _get_json(MTEB_PARQUET_LIST_URL)
    files = data.get("parquet_files") if isinstance(data, dict) else data
    urls = []
    for f in files or []:
        if isinstance(f, dict) and f.get("url"):
            urls.append(f["url"])
    return urls


def _read_mteb_table(blob: bytes) -> Any:
    """Read one parquet shard into a pyarrow Table (ImportError -> fail-open upstream)."""
    import pyarrow.parquet as pq

    return pq.read_table(
        io.BytesIO(blob),
        columns=["model_name", "task_name", "split", "score"],
    )


_MULTILINGUAL_MARKERS = ("Multilingual", "MIRACL", "Crosslingual")


def _is_multilingual_task(task_name: str) -> bool:
    """True for MTEB tasks whose evaluation is multilingual/cross-lingual."""
    return any(marker in task_name for marker in _MULTILINGUAL_MARKERS)


def _aggregate_tables(tables: list[Any]) -> dict[str, dict[str, float]]:
    """Aggregate pyarrow Tables into ``{family: {model_name: mean score}}``.

    Rows on splits other than ``test`` are ignored; scores are averaged per
    (model, task) first, then per family across tasks (mean of task means), so
    a task with many subset rows does not dominate its family. Task names that
    do not classify into one of the four families are ignored (fail-open).

    Multilingual-only (2026-10-07 directive): this stack selects embed/rerank
    models that must work across languages, so English-only MTEB tasks are
    excluded — only multilingual tasks (MIRACL*/``*Multilingual*``/
    ``*Crosslingual*``) count toward a family score.
    """
    import pyarrow.compute as pc

    sums: dict[str, dict[str, float]] = {}
    counts: dict[str, dict[str, int]] = {}
    for table in tables:
        filtered = table.filter(pc.equal(table.column("split"), "test"))  # ty: ignore[unresolved-attribute]  # stubs omit pc.equal
        grouped = filtered.group_by(["model_name", "task_name"]).aggregate([("score", "mean")])
        models = grouped.column("model_name").to_pylist()
        tasks = grouped.column("task_name").to_pylist()
        means = grouped.column("score_mean").to_pylist()
        for model, task_name, mean in zip(models, tasks, means, strict=True):
            family = task_family(task_name)
            if family is None or mean is None or not model:
                continue
            if not _is_multilingual_task(task_name):
                continue
            fam_sums = sums.setdefault(family, {})
            fam_counts = counts.setdefault(family, {})
            fam_sums[model] = fam_sums.get(model, 0.0) + float(mean)
            fam_counts[model] = fam_counts.get(model, 0) + 1
    return {
        family: {model: total / counts[family][model] for model, total in models.items()}
        for family, models in sums.items()
    }


_SHARED_FAMILY_SCORES: dict[str, dict[str, float]] | None = None


def _reset_mteb_memo() -> None:
    global _SHARED_FAMILY_SCORES
    _SHARED_FAMILY_SCORES = None


def _shared_family_scores(*, refresh: bool = False) -> dict[str, dict[str, float]]:
    """Family-level scores shared by all MTEB fetchers (one download per process)."""
    global _SHARED_FAMILY_SCORES
    if _SHARED_FAMILY_SCORES is not None and not refresh:
        return _SHARED_FAMILY_SCORES
    tables = [_read_mteb_table(_get_bytes(url)) for url in _mteb_parquet_urls()]
    _SHARED_FAMILY_SCORES = _aggregate_tables(tables)
    return _SHARED_FAMILY_SCORES


class _MtebFamilySource(_BaseSource):
    """Base for the four MTEB family fetchers; subclasses pick their family.

    Family scores are MULTILINGUAL-ONLY (MIRACL*/``*Multilingual*``/
    ``*Crosslingual*`` tasks) per the 2026-10-07 directive — embed/rerank
    picks must work across languages.
    """

    family = ""
    ttl_seconds = WEEK

    def _fetch(self) -> dict[str, SourceRecord]:
        scores = _shared_family_scores().get(self.family, {})
        return {
            model: SourceRecord(key=model, name=model, score=score)
            for model, score in scores.items()
            if score is not None
        }


class MtebClassificationSource(_MtebFamilySource):
    """Mean MTEB classification score per model (mteb/results, test split)."""

    name = "mteb_classification"
    family = "classification"


class MtebRetrievalSource(_MtebFamilySource):
    """Mean MTEB retrieval score per model (mteb/results, test split)."""

    name = "mteb_retrieval"
    family = "retrieval"


class MtebStsSource(_MtebFamilySource):
    """Mean MTEB STS score per model (mteb/results, test split)."""

    name = "mteb_sts"
    family = "sts"


class MtebRerankingSource(_MtebFamilySource):
    """Mean MTEB reranking score per model (mteb/results, test split)."""

    name = "mteb_reranking"
    family = "reranking"


# --- Wave-6 fetchers (2026-09-22 directive + wave6 leaderboard research) -------
#
# Conventions shared by the new boards: every parser is a pure function over
# the fetched bytes/text (fixture-testable), model identity goes into
# ``SourceRecord.key`` using the board's own slug when it publishes one (vals
# /models/<slug>, benchlm /models/<slug>, llm-stats /models/<slug>) and the
# slugified display name otherwise.


def _parse_vals_table(html: str) -> dict[str, SourceRecord]:
    """vals.ai accessibility table: rank, model link (/models/<slug>), accuracy%."""
    records: dict[str, SourceRecord] = {}
    for tr in re.findall(r"<tr[^>]*>(.*?)</tr>", html, re.S):
        m = re.search(r'<a[^>]+href="/models/([^"]+)"[^>]*>(.*?)</a>', tr, re.S)
        if not m:
            continue
        slug, name = m.group(1), _strip_tags(m.group(2))
        score = None
        for cell in (_strip_tags(c) for c in re.findall(r"<td[^>]*>(.*?)</td>", tr, re.S)):
            pct = _parse_pct(cell)
            if pct is not None:
                score = pct
                break
        if score is None or not slug:
            continue
        records[slug] = SourceRecord(key=slug, name=name, score=score)
    return records


class ValsSource(_BaseSource):
    """vals.ai self-run eval board — static Astro page, server-rendered table
    (research §1, LIVE). One fetcher per board slug: ``vals_index`` (composite),
    ``vals_medscribe`` (medical transcription), ``vals_cua_bench`` (computer-use
    agentic). Scores are accuracy percentages as published."""

    ttl_seconds = WEEK

    def __init__(self, name: str, board: str) -> None:
        super().__init__()
        self.name = name
        self.board = board

    def _fetch(self) -> dict[str, SourceRecord]:
        html = _get_bytes(VALS_BENCH_URL.format(board=self.board)).decode("utf-8", errors="replace")
        return _parse_vals_table(html)


def _parse_jevals_board(data: dict[str, Any]) -> dict[str, SourceRecord]:
    """jevals board.json -> one record per model, mean min-max decision score.

    Per benchmark (banking77/helpsteer2/pubmedqa — one per primitive
    choice/score/noul), the per-row ``decision_score`` (0-100, higher is
    better; Brier-based, published with bootstrap CIs) is min-max normalized
    across the listed models, then a model's score is the MEAN of its
    normalized per-benchmark scores. ``interface == "baseline"`` rows (the
    label prior, model_id null) are excluded — they are not models.
    """
    bench_rows: dict[str, list[dict]] = {b: [] for b in JEVALS_BENCHMARKS}
    primitive_of: dict[str, str] = {}
    for bench in data.get("benchmarks") or []:
        if isinstance(bench, dict) and bench.get("id") in bench_rows:
            primitive_of[str(bench["primitive"])] = str(bench["id"])
    for row in data.get("rows") or []:
        if not isinstance(row, dict):
            continue
        if row.get("interface") == "baseline" or row.get("model_id") is None:
            continue  # label-prior baseline, not a model
        bench_id = primitive_of.get(str(row.get("primitive")))
        if bench_id:
            bench_rows[bench_id].append(row)
    # min-max per benchmark; a flat benchmark (all scores equal) is neutral 50
    normalized: dict[str, dict[str, float]] = {b: {} for b in JEVALS_BENCHMARKS}
    for bench_id, rows in bench_rows.items():
        scores = [r.get("decision_score") for r in rows]
        values = [s for s in scores if isinstance(s, (int, float))]
        if not values:
            continue
        lo, hi = min(values), max(values)
        for row in rows:
            s = row.get("decision_score")
            if not isinstance(s, (int, float)):
                continue
            normalized[bench_id][str(row["model_id"])] = 50.0 if hi <= lo else (s - lo) / (hi - lo) * 100.0
    # per-model aggregation over the benchmarks the model was run on
    per_model: dict[str, dict] = {}
    for row in data.get("rows") or []:
        if isinstance(row, dict) and row.get("model_id") is not None:
            per_model[str(row["model_id"])] = row
    records: dict[str, SourceRecord] = {}
    for model_id, row in per_model.items():
        parts = [normalized[b][model_id] for b in JEVALS_BENCHMARKS if model_id in normalized[b]]
        if not parts:
            continue
        mean = sum(parts) / len(parts)
        lo, hi = _float(row.get("ci_low")), _float(row.get("ci_high"))
        records[slugify(model_id)] = SourceRecord(
            key=slugify(model_id),
            name=str(row.get("display_name") or model_id),
            score=mean,
            score_ci=(hi - lo) / 2 if lo is not None and hi is not None else None,
            board_version=str(data.get("release_id") or "") or None,
            raw=dict(row),
        )
    return records


class JevalsSource(_BaseSource):
    """jevals.com decision board (profile ``decision``, 2026-10-09).

    Frozen dated releases: GET ``/data/releases/<release_id>/board.json``
    (CC-BY-4.0). The SPA root serves no JSON index, so the newest release_id
    is resolved from the server-rendered home page payload
    (``release_id:"YYYY-MM-DD"``, verified live 2026-10-09); if that
    resolution fails the fetch is empty (fail-open) — never guessed.

    Staleness guard: when the resolved release has not changed for
    ``JEVALS_STATIC_ANCHOR_DAYS`` days, rows are still served but the source
    reports ``degraded`` (reason ``static_anchor``) — the board is then a
    frozen anchor, not live evidence. ``board_version`` is the release_id,
    ``board_updated_at`` the board's ``as_of``, and the CC-BY-4.0 citation
    from board.json is recorded verbatim as ``attribution``.
    """

    ttl_seconds = WEEK
    name = "jevals"

    def __init__(self) -> None:
        super().__init__()
        self.degraded_reason: str | None = None

    def _fetch(self) -> dict[str, SourceRecord]:
        self.degraded_reason = None
        home = _get_bytes(JEVALS_HOME_URL).decode("utf-8", errors="replace")
        m = JEVALS_RELEASE_RE.search(home)
        if not m:
            self.missing_reason = "jevals_release_unresolved"
            return {}
        release_id = m.group(1)
        data = _get_json(JEVALS_BOARD_URL.format(release_id=release_id))
        if not isinstance(data, dict):
            self.missing_reason = "jevals_board_unexpected_shape"
            return {}
        as_of = str(data.get("as_of") or release_id)
        self.board_updated_at = as_of
        citation = str(data.get("citation") or "")
        if citation:
            self.attribution = f"Source: {citation}"
        try:
            age_days = (datetime.now(UTC).date() - datetime.fromisoformat(as_of).date()).days
        except ValueError:
            age_days = 0
        if age_days > JEVALS_STATIC_ANCHOR_DAYS:
            self.degraded_reason = (
                f"static_anchor: release {release_id} unchanged for {age_days} days (> {JEVALS_STATIC_ANCHOR_DAYS})"
            )
        return _parse_jevals_board(data)


class JevbenchSource(_BaseSource):
    """JevBench decision board (profile ``decision``, 2026-10-09) — the
    independent second decision-domain board that lets native-decision models
    reach strong evidence (>= 2 boards) alongside jevals.

    One JSON call; ``systems`` rows carry ``jevbench_score`` (the board's
    composite, 0-100) with deep per-primitive calibration metrics in
    ``calibration.parts``. ALL systems are parsed as records; joinability is
    restricted by the versioned alias table to ``api_flag == true`` hosted
    routes only (two mapped today: Jev 1.13.0 -> typesafe/jev-1.13, Mercury
    Decide -> inception/mercury-decide). Self-hosted open-weight systems
    (Gemma/Qwen merges without an OR route) report ``unmatched_names`` —
    never mapped. ``board_version`` = revision, ``board_updated_at`` =
    ``v16.release``.
    """

    ttl_seconds = WEEK
    name = "jevbench"

    def _fetch(self) -> dict[str, SourceRecord]:
        data = _get_json(JEVBENCH_URL)
        if not isinstance(data, dict):
            self.missing_reason = "jevbench_unexpected_shape"
            return {}
        records: dict[str, SourceRecord] = {}
        for row in data.get("systems") or []:
            if not isinstance(row, dict):
                continue
            name = str(row.get("display") or "").strip()
            score = _float(row.get("jevbench_score"))
            if not name or score is None:
                continue
            key = slugify(name)
            records[key] = SourceRecord(
                key=key,
                name=name,
                score=score,
                board_version=str(data.get("revision") or "") or None,
                raw=dict(row),
            )
        if not records:
            self.missing_reason = "jevbench_empty_systems"
            return {}
        v16 = data.get("v16") if isinstance(data.get("v16"), dict) else {}
        release = str(v16.get("release") or "")
        self.board_updated_at = release or None
        revision = str(data.get("revision") or "")
        self.attribution = (
            f"Source: JevBench (benchmarkheaven.com), revision {revision}, release {release or 'unknown'}"
        )
        return records


class JevDecisionIndexSource(_BaseSource):
    """Jev Decision Index (profile ``decision``, 2026-10-09, user-named): the
    versioned JSON of HF Space ``multimodalart/jev-decision-index`` — the
    reference board for open-weight Jev repros (114 models).

    Score = ``scores.balanced_skill``: the board's 0-100 skill composite over
    the balanced (public + held-out) item set, i.e. accuracy with calibration
    gaps penalized. ``balanced_raw`` is the unpenalized twin (higher);
    ``breadth_skill``/``public_skill`` are subset views. ``board_updated_at``
    = the file's ``generated_utc``. JOIN CAVEAT: the population is mostly
    open repros with no OR route — only the decisions-segment repros (clef,
    clef-flash, kev-4b, tev1-4b; versioned alias table) survive the
    OR-servable filter, everything else reports ``unmatched_names`` — a thin
    board post-filter, by design. No license is declared in the space repo
    (LICENSE file 404, card license unset — probed 2026-10-09): the
    attribution line records space + generation date WITHOUT a license claim.
    """

    ttl_seconds = WEEK
    name = "jev_decision_index"

    def _fetch(self) -> dict[str, SourceRecord]:
        data = _get_json(JEV_DECISION_INDEX_URL)
        if not isinstance(data, dict):
            self.missing_reason = "jev_decision_index_unexpected_shape"
            return {}
        records: dict[str, SourceRecord] = {}
        for row in data.get("models") or []:
            if not isinstance(row, dict):
                continue
            engine = str(row.get("engine") or "").strip()
            scores = row.get("scores") if isinstance(row.get("scores"), dict) else {}
            score = _float(scores.get("balanced_skill"))
            if not engine or score is None:
                continue
            key = slugify(engine)
            records[key] = SourceRecord(
                key=key,
                name=str(row.get("name") or engine),
                score=score,
                board_version=str(data.get("generated_utc") or "")[:10] or None,
                raw=dict(row),
            )
        if not records:
            self.missing_reason = "jev_decision_index_empty_models"
            return {}
        generated = str(data.get("generated_utc") or "")
        self.board_updated_at = generated or None
        self.attribution = (
            "Source: Jev Decision Index (HF Space multimodalart/jev-decision-index),"
            f" generated {generated or 'unknown'}; no license declared in the space repo (probed 2026-10-09)"
        )
        return records


def _parse_benchlm_md(md: str) -> dict[str, SourceRecord]:
    """benchlm.ai markdown alternate: the ``## Overall rankings`` pipe table."""
    section = md.split("## Overall rankings", 1)
    if len(section) < 2:
        return {}
    records: dict[str, SourceRecord] = {}
    started = False
    for line in section[1].splitlines():
        line = line.strip()
        if not line.startswith("|"):
            if started:
                break  # table ended
            continue
        started = True
        cells = [c.strip() for c in line.strip("|").split("|")]
        if len(cells) < 6 or cells[0] in ("", "Rank") or set(cells[0]) <= set("- :"):
            continue
        name_m = re.search(r"\[([^\]]+)\]", cells[1])
        score = _float(cells[5])
        if not name_m or score is None:
            continue
        name = name_m.group(1)
        slug_m = re.search(r"\]\(/models/([^)\s]+)\)", cells[1])
        key = slug_m.group(1) if slug_m else slugify(name)
        records[key] = SourceRecord(key=key, name=name, score=score, raw={"cells": cells})
    return records


class BenchLMSource(_BaseSource):
    """benchlm.ai meta-aggregator via the markdown alternate ``/md/index.md``
    (research §1, LIVE: full ranked table, one row per model, BenchAlign score)."""

    name = "benchlm"
    ttl_seconds = WEEK

    def _fetch(self) -> dict[str, SourceRecord]:
        md = _get_bytes(BENCHLM_MD_URL).decode("utf-8", errors="replace")
        return _parse_benchlm_md(md)


def _parse_or_bench_table(html: str) -> dict[str, tuple[float, float | None]]:
    """One OR benchmark page -> ``{model_slug: (accuracy%, ci)}``.

    Model cells look like ``Vendor: Model Name  Pareto`` (the ``Pareto`` badge
    marks pareto-optimal endpoints, not part of the name).
    """
    out: dict[str, tuple[float, float | None]] = {}
    for cells in _table_rows(html):
        if len(cells) < 3 or not cells or not cells[0].strip().isdigit():
            continue
        label = re.sub(r"\s*Pareto\s*$", "", cells[1]).strip()
        model = label.partition(": ")[2].strip() or label
        acc = _parse_pct(cells[2])
        if acc is None or not model:
            continue
        out[slugify(model)] = (acc, _parse_pm(cells[3]) if len(cells) > 3 else None)
    return out


class Tau2BenchORSource(_BaseSource):
    """τ²-bench as run continuously on OpenRouter real endpoints
    (research §2.9: ``openrouter.ai/benchmarks/tau2-bench-*`` — accuracy,
    cost-per-task, speed). Score = mean accuracy across the domain boards that
    list the model; per-domain accuracies stay in ``raw``."""

    name = "tau2_bench_or"
    ttl_seconds = DAY

    def __init__(self, benches: tuple[str, ...] = TAU2_BENCHES) -> None:
        super().__init__()
        self.benches = benches

    def _fetch(self) -> dict[str, SourceRecord]:
        per_model: dict[str, dict[str, float]] = {}
        per_model_ci: dict[str, dict[str, float]] = {}
        names: dict[str, str] = {}
        for bench in self.benches:
            try:
                html = _get_bytes(OR_BENCH_URL.format(bench=bench)).decode("utf-8", errors="replace")
            except Exception as exc:  # one empty domain board must not kill the rest
                logger.info("model_selection tau2 domain unavailable: bench=%s error=%s", bench, exc)
                continue
            for slug, (acc, ci) in _parse_or_bench_table(html).items():
                per_model.setdefault(slug, {})[bench] = acc
                if ci is not None:
                    per_model_ci.setdefault(slug, {})[bench] = ci
        records: dict[str, SourceRecord] = {}
        for slug, domains in per_model.items():
            scores = list(domains.values())
            records[slug] = SourceRecord(
                key=slug,
                name=names.get(slug, slug),
                score=sum(scores) / len(scores),
                score_ci=max(per_model_ci.get(slug, {}).values()) if per_model_ci.get(slug) else None,
                raw={"domains": domains},
            )
        return records


def _parse_llmstats_table(html: str) -> dict[str, SourceRecord]:
    """llm-stats.com board table: model link (/models/<slug>) + 0-1 score."""
    records: dict[str, SourceRecord] = {}
    for tr in re.findall(r"<tr[^>]*>(.*?)</tr>", html, re.S):
        m = re.search(r'<a[^>]+href="/models/([^"]+)"[^>]*>(.*?)</a>', tr, re.S)
        if not m:
            continue
        slug, name = m.group(1), _strip_tags(m.group(2))
        cells = [_strip_tags(c) for c in re.findall(r"<td[^>]*>(.*?)</td>", tr, re.S)]
        # cells: [rank, model, score, size, context, cost, license]
        score = _float(cells[2]) if len(cells) > 2 else None
        if score is None or not slug:
            continue
        records[slug] = SourceRecord(key=slug, name=name, score=score)
    return records


class LLMStatsSource(_BaseSource):
    """llm-stats.com mirror board (server-rendered table; research §2.5/§2.6).
    Scores are 0-1 as published. ``healthbench`` and ``ocrbench`` boards."""

    ttl_seconds = WEEK

    def __init__(self, name: str, board: str) -> None:
        super().__init__()
        self.name = name
        self.board = board

    def _fetch(self) -> dict[str, SourceRecord]:
        html = _get_bytes(LLMSTATS_BENCH_URL.format(bench=self.board)).decode("utf-8", errors="replace")
        return _parse_llmstats_table(html)


def _latest_medhelm_release() -> str:
    """Newest ``v<major>.<minor>.<patch>`` release from the public GCS listing."""
    data = _get_json(MEDHELM_RELEASES_URL)
    prefixes = data.get("prefixes") if isinstance(data, dict) else data
    best, best_key = "", (-1, -1, -1)
    for p in prefixes or []:
        m = re.search(r"releases/(v(\d+)\.(\d+)\.(\d+))/", str(p))
        if not m:
            continue
        key = (int(m.group(2)), int(m.group(3)), int(m.group(4)))
        if key > best_key:
            best, best_key = m.group(1), key
    if not best:
        raise RuntimeError("no medhelm release found in GCS listing")
    return best


def _parse_medhelm_group(blob: bytes) -> dict[str, SourceRecord]:
    """MedHELM ``groups/medhelm_scenarios.json`` accuracy table.

    HELM publishes one column per task (no composite score) and the columns
    mix metric scales (0-1 accuracies next to 1-5 jury scores), so the score
    is the mean over columns whose EVERY published value is fraction-scale
    (<= 1.0, decided column-wise before scoring) — mixed-scale columns (jury
    scores etc.) stay in ``raw`` but do not contaminate the composite. If no
    column qualifies, the mean over all numeric task cells is the fallback.
    """
    tables = json.loads(blob.decode("utf-8", errors="replace"))
    if not isinstance(tables, list):
        return {}
    table = next((t for t in tables if isinstance(t, dict) and t.get("header") and t.get("rows")), None)
    if table is None:
        return {}
    header = table["header"]
    win_rate_cols = {i for i, h in enumerate(header) if "win rate" in str(h.get("value", "")).lower()}
    task_cols = [i for i in range(1, len(header)) if i not in win_rate_cols]

    parsed: list[tuple[str, dict[str, float], dict[int, float]]] = []
    col_vals: dict[int, list[float]] = {}
    for row in table["rows"]:
        cells = row if isinstance(row, list) else []
        if not cells or not isinstance(cells[0], dict):
            continue
        name = str(cells[0].get("value") or "").strip()
        if not name:
            continue
        values: dict[str, float] = {}
        by_index: dict[int, float] = {}
        for i in task_cols:
            if i >= len(cells) or not isinstance(cells[i], dict):
                continue
            val = _float(cells[i].get("value"))
            if val is not None:
                values[str(header[i].get("value"))] = val
                by_index[i] = val
                col_vals.setdefault(i, []).append(val)
        if values:
            parsed.append((name, values, by_index))
    fraction_cols = {i for i, vals in col_vals.items() if all(0.0 <= v <= 1.0 for v in vals)} if col_vals else set()

    records: dict[str, SourceRecord] = {}
    for name, values, by_index in parsed:
        scored = [by_index[i] for i in fraction_cols if i in by_index] if fraction_cols else list(values.values())
        if not scored:
            continue
        records[slugify(name)] = SourceRecord(
            key=slugify(name),
            name=name,
            score=sum(scored) / len(scored),
            raw={"tasks": values, "scored_tasks": len(scored)},
        )
    return records


class MedhelmSource(_BaseSource):
    """MedHELM (Stanford CRFM) via the public GCS release bucket: newest
    release's ``groups/medhelm_scenarios.json`` accuracy table. The site itself
    is a React app over the same bucket (config.js verified 2026-10-07)."""

    name = "medhelm"
    ttl_seconds = WEEK

    def _fetch(self) -> dict[str, SourceRecord]:
        release = _latest_medhelm_release()
        blob = _get_bytes(MEDHELM_GROUP_URL.format(release=release))
        records = _parse_medhelm_group(blob)
        for rec in records.values():
            rec.board_version = release
        return records


def _parse_bfcl_csv(text: str) -> dict[str, SourceRecord]:
    """BFCL ``data_overall.csv``: Rank, Overall Acc (%), Model, ... The model
    cell can carry a ``(FC)``/``(Prompt)`` suffix marking the serving mode."""
    records: dict[str, SourceRecord] = {}
    for row in csv.DictReader(io.StringIO(text)):
        name = (row.get("Model") or "").strip()
        mode_m = re.search(r"\((\w+)\)$", name)
        name = re.sub(r"\s*\(\w+\)$", "", name).strip()
        score = _parse_pct(row.get("Overall Acc") or "")
        if not name or score is None:
            continue
        rec = SourceRecord(
            key=slugify(name),
            name=name,
            score=score,
            raw={"mode": mode_m.group(1) if mode_m else None, "row": dict(row)},
        )
        records[rec.key] = rec
    return records


class BFCLSource(_BaseSource):
    """Berkeley Function-Calling Leaderboard overall board — the site's JS
    fetches ``data_overall.csv`` (traced in index_main.js, verified 2026-10-07)."""

    name = "bfcl"
    ttl_seconds = DAY

    def _fetch(self) -> dict[str, SourceRecord]:
        text = _get_bytes(BFCL_CSV_URL).decode("utf-8", errors="replace")
        return _parse_bfcl_csv(text)


class EqBenchCsvSource(_BaseSource):
    """EQ-Bench leaderboard whose data ships as a CSV inside a backtick
    template literal in the page's JS (tables are async-built client-side;
    verified 2026-10-07: ``leaderboardDataCreativeWritingV3`` in
    creative_writing.js, ``leaderboardDataLongformV3`` in
    creative_writing_longform.js). Leading ``*`` on model names is a legacy
    marker — slugify drops it."""

    ttl_seconds = WEEK

    def __init__(
        self,
        name: str,
        js_url: str,
        var_name: str,
        score_field: str,
        *,
        name_cols: tuple[str, ...] = ("model_name",),
    ) -> None:
        super().__init__()
        self.name = name
        self.js_url = js_url
        self.var_name = var_name
        self.score_field = score_field
        self.name_cols = name_cols

    def _fetch(self) -> dict[str, SourceRecord]:
        js = _get_bytes(self.js_url).decode("utf-8", errors="replace")
        m = re.search(re.escape(self.var_name) + r"\s*=\s*`(.*?)`", js, re.S)
        if not m:
            return {}
        # the template literal opens with blank lines (and may carry CRLF)
        # before the CSV header — strip before handing to the CSV parser
        return _parse_csv_records(m.group(1).strip(), name_cols=self.name_cols, score_cols=(self.score_field,))


class EqBench4Source(_BaseSource):
    """EQ-Bench 4 (multi-turn emotional/social intelligence Elo): data ships as
    ``const EQBENCH4_DATA = {json}`` in eqbench4/eqbench4_data.js; per-model
    ``elo`` with bootstrap ``ci_low``/``ci_high``."""

    name = "eqbench4"
    ttl_seconds = WEEK

    def _fetch(self) -> dict[str, SourceRecord]:
        js = _get_bytes(EQBENCH4_DATA_URL).decode("utf-8", errors="replace")
        start = js.find("{")
        end = js.rfind("}")
        if start < 0 or end <= start:
            return {}
        data = json.loads(js[start : end + 1])
        records: dict[str, SourceRecord] = {}
        for row in data.get("models") or []:
            if not isinstance(row, dict):
                continue
            name = str(row.get("model") or row.get("slug") or "").strip()
            score = _float(row.get("elo"))
            if not name or score is None:
                continue
            lo, hi = _float(row.get("ci_low")), _float(row.get("ci_high"))
            records[slugify(name)] = SourceRecord(
                key=slugify(name),
                name=name,
                score=score,
                score_ci=(hi - lo) / 2 if lo is not None and hi is not None else None,
                board_version=str(data.get("generated_at") or "")[:10] or None,
                raw=dict(row),
            )
        return records


def _parse_manga_bench_table(html: str, score_col: int) -> dict[str, SourceRecord]:
    """MangaVQA project-site results table -> one board's scores.

    The static page carries a single results table per section:
    ``Method | MangaOCR Hmean (%) | MangaVQA LLM (/10.0)``. ``score_col``
    picks the benchmark column (1 = MangaOCR Hmean, 2 = MangaVQA score);
    rows without a parseable float there are skipped. The dataset-count
    table (``Count Type | Total | ...``) is excluded by its header so its
    row counts can never leak in as scores. Method names are vendor-less
    display names (``GPT-4o``, ``MangaLMM (Ours)``) — the join layer's
    normalized matching handles the OR slug mapping.
    """
    records: dict[str, SourceRecord] = {}
    for table_m in re.finditer(r"<table.*?</table>", html, re.S):
        first_tr = re.search(r"<tr[^>]*>(.*?)</tr>", table_m.group(0), re.S)
        if not first_tr:
            continue
        head = [
            _strip_tags(c).strip().lower() for c in re.findall(r"<t[hd][^>]*>(.*?)</t[hd]>", first_tr.group(1), re.S)
        ]
        if not head or head[0] != "method":
            continue
        for tr in re.findall(r"<tr[^>]*>(.*?)</tr>", table_m.group(0), re.S):
            cells = [_strip_tags(c) for c in re.findall(r"<t[hd][^>]*>(.*?)</t[hd]>", tr, re.S)]
            if len(cells) <= score_col or not cells[0].strip():
                continue
            name = re.sub(r"\s*\(Ours\)\s*$", "", cells[0]).strip()
            score = _parse_pct(cells[score_col])
            if score is None or not name or name.lower() in ("method",):
                continue
            records[slugify(name)] = SourceRecord(
                key=slugify(name),
                name=name,
                score=score,
                board_version="2026",
                raw={"cells": cells},
            )
    return records


class MangaBenchSource(_BaseSource):
    """Manga109-org project-site board (research §2.5): the MangaVQA/MangaOCR
    results table on the static GitHub Pages site, one fetcher per benchmark
    column — ``manga109_v2026`` (MangaOCR Hmean % on Manga109-derived text
    tasks) and ``mangavqa`` (MangaVQA LLM score, 0-10 scale as published).
    Scan-only: no dataset download, no login, no paid calls."""

    ttl_seconds = WEEK

    def __init__(self, name: str, score_col: int) -> None:
        super().__init__()
        self.name = name
        self.score_col = score_col

    def _fetch(self) -> dict[str, SourceRecord]:
        html = _get_bytes(MANGA_BENCH_URL).decode("utf-8", errors="replace")
        records = _parse_manga_bench_table(html, self.score_col)
        if not records:
            self.missing_reason = "manga_bench_table_empty"
        return records


def _parse_bridge_blob(data: dict[str, Any]) -> dict[str, float]:
    """BRIDGE leaderboard JSON: column dicts keyed by model index; score =
    ``Average Performance`` (0-100)."""
    names = data.get("Model") or {}
    avg = data.get("Average Performance") or {}
    out: dict[str, float] = {}
    for idx, name in names.items():
        score = _float(avg.get(idx))
        if score is None or not str(name).strip():
            continue
        out[slugify(str(name))] = score
    return out


class BridgeSource(_BaseSource):
    """BRIDGE multilingual clinical leaderboard (HF space
    YLab-Open/BRIDGE-Medical-Leaderboard — research §2.6): per-mode
    ``leaderboards/<mode>_leaderboard.json``. Score = mean ``Average
    Performance`` across the eval modes that list the model."""

    name = "bridge"
    ttl_seconds = WEEK

    def __init__(self, modes: tuple[str, ...] = BRIDGE_MODES) -> None:
        super().__init__()
        self.modes = modes

    def _fetch(self) -> dict[str, SourceRecord]:
        per_model: dict[str, dict[str, float]] = {}
        for mode in self.modes:
            try:
                data = _get_json(BRIDGE_LEADERBOARD_URL.format(mode=mode))
            except Exception as exc:  # a missing mode file must not kill the rest
                logger.info("model_selection bridge mode unavailable: mode=%s error=%s", mode, exc)
                continue
            if not isinstance(data, dict):
                continue
            for slug, score in _parse_bridge_blob(data).items():
                per_model.setdefault(slug, {})[mode] = score
        records: dict[str, SourceRecord] = {}
        for slug, modes in per_model.items():
            vals = list(modes.values())
            records[slug] = SourceRecord(key=slug, name=slug, score=sum(vals) / len(vals), raw={"modes": modes})
        return records


class OpenMedicalLLMSource(_BaseSource):
    """Open Medical LLM Leaderboard (HF space openlifescienceai/...) — the
    space computes its table from the public ``openlifescienceai/results``
    dataset: one small JSON per model repo (``<org>/<model>/results_*.json``,
    lm-eval style ``results.<task>.acc,none``). Score = mean per-task accuracy;
    join key = the HF repo id (first two path segments). Bounded: at most
    ``max_files`` result files per pull (request budget, not quality)."""

    name = "open_medical_llm"
    ttl_seconds = WEEK
    max_files = 400

    def __init__(self, max_files: int = 400) -> None:
        super().__init__()
        self.max_files = max_files

    def _fetch(self) -> dict[str, SourceRecord]:
        data = _get_json(OPEN_MEDICAL_RESULTS_TREE_URL)
        entries = data if isinstance(data, list) else []
        files = sorted(
            str(e.get("path"))
            for e in entries
            if isinstance(e, dict) and e.get("type") == "file" and str(e.get("path", "")).endswith(".json")
        )
        if not files:
            self.missing_reason = "open_medical_results_empty"
            return {}
        records: dict[str, SourceRecord] = {}
        with httpx.Client(timeout=DEFAULT_TIMEOUT, follow_redirects=True) as client:
            for path in files[: self.max_files]:
                hf_id = "/".join(path.split("/")[:2])
                try:
                    resp = client.get(OPEN_MEDICAL_RESULT_FILE_URL.format(path=quote(path, safe="/")))
                    resp.raise_for_status()
                    blob = resp.json()
                except Exception as exc:  # one bad file must not kill the board
                    logger.info("model_selection open_medical file skipped: path=%s error=%s", path, exc)
                    continue
                score = self._mean_accuracy(blob)
                if score is None or not hf_id:
                    continue
                records[slugify(hf_id)] = SourceRecord(key=slugify(hf_id), name=hf_id, score=score, raw={"path": path})
        return records

    @staticmethod
    def _mean_accuracy(blob: Any) -> float | None:
        results = blob.get("results") if isinstance(blob, dict) else None
        if not isinstance(results, dict):
            return None
        vals = []
        for task in results.values():
            if not isinstance(task, dict):
                continue
            acc = _float(task.get("acc,none"))
            if acc is not None:
                vals.append(acc)
        return sum(vals) / len(vals) if vals else None


class GaiaSource(_BaseSource):
    """GAIA assistant benchmark board via the public results dataset
    ``gaia-benchmark/results_public`` (the leaderboard space reads the same;
    research §2.9). Rows are per-run; a model keeps its best score (0-1)."""

    name = "gaia"
    ttl_seconds = WEEK

    def _fetch(self) -> dict[str, SourceRecord]:
        rows = _read_parquet_rows(_get_bytes(GAIA_RESULTS_URL))
        best: dict[str, tuple[float, SourceRecord]] = {}
        for row in rows:
            name = str(row.get("model") or "").strip()
            score = _float(row.get("score"))
            if not name or score is None:
                continue
            rec = SourceRecord(
                key=slugify(name),
                name=name,
                score=score,
                board_version=str(row.get("date")) if row.get("date") else None,
                raw=dict(row),
            )
            prev = best.get(rec.key)
            if prev is None or score > prev[0]:
                best[rec.key] = (score, rec)
        return {key: rec for key, (_, rec) in best.items()}


class AaCapabilitySource(_BaseSource):
    """AA Capability Index (e.g. Healthcare & Medical) via the Data API free
    tier (same ``/free`` list endpoint as ``artificial_analysis``).

    Key comes from ``AA_API_KEY`` at fetch time; absent key -> ``{}`` with
    ``missing_reason=aa_api_key_required`` (recorded missing, never silent).
    Field names per the AA data-api docs (verified 2026-10-07/08):
    ``artificial_analysis_<capability>_index`` — present on free-tier rows
    for the models that have the capability measured (2026-10-08: 52 agentic
    / 23 healthcare of 200 rows); ``_version`` fields are absent on free
    rows, so ``board_version`` stays None and the version guard ignores them.
    """

    ttl_seconds = DAY

    def __init__(self, name: str, capability: str, api_key: str | None = None) -> None:
        super().__init__()
        self.name = name
        self.capability = capability
        self._api_key = api_key

    def _fetch(self) -> dict[str, SourceRecord]:
        api_key = self._api_key or os.environ.get("AA_API_KEY")
        if not api_key:
            self.missing_reason = "aa_api_key_required"
            return {}
        data = _get_json(AA_MODELS_URL, headers={"x-api-key": api_key})
        rows = data.get("data") if isinstance(data, dict) else data
        field = f"artificial_analysis_{self.capability}_index"
        records: dict[str, SourceRecord] = {}
        for row in rows or []:
            if not isinstance(row, dict):
                continue
            evals = row.get("evaluations") if isinstance(row.get("evaluations"), dict) else row
            score = _float(evals.get(field))
            if score is None:
                continue  # capability indexes are null for models missing any sub-evaluation
            name = row.get("name") or row.get("slug") or ""
            slug = row.get("slug") or slugify(name)
            if not slug:
                continue
            version = evals.get(f"{field}_version") or evals.get("artificial_analysis_index_version")
            records[slug] = SourceRecord(
                key=slugify(slug),
                name=name or slug,
                score=score,
                board_version=str(version) if version else None,
                raw=row,
            )
        return records


class AaAgenticIndexSource(AaCapabilitySource):
    """AA Agentic Index via the Data API (``AA_API_KEY``). Absent key ->
    ``missing_reason=aa_api_key_required`` (recorded gap). The OR-embedded
    ``benchmarks.artificial_analysis.agentic_index`` copy is catalog metadata,
    not a fetched board — it is never substituted for real AA evidence.
    """

    def __init__(self, api_key: str | None = None) -> None:
        super().__init__("aa_agentic_index", "agentic", api_key)


class InHouseAnchorSource(_BaseSource):
    """In-house eval anchor set (aiora_triage_eval) — self-run anchor per the
    wave6 research, NOT a scraped board. Reads ``<root>/<name>.json``::

        {"updated_at": "...", "records": [{"key": "...", "name": "...",
         "score": 0-100, "score_ci": optional, "board_version": optional}]}

    Absent file -> ``{}`` with ``missing_reason=in_house_anchor_not_published``
    (a recorded gap in the snapshot, not an error, not silent). Root defaults
    to ``data/anchors`` (override: ``HULL_ANCHOR_ROOT`` env or ``root=`` arg).
    """

    ttl_seconds = DAY

    def __init__(self, name: str, root: Path | str | None = None) -> None:
        super().__init__()
        self.name = name
        self.anchor_root = Path(root) if root else Path(os.environ.get("HULL_ANCHOR_ROOT", "data/anchors"))

    @property
    def path(self) -> Path:
        return self.anchor_root / f"{self.name}.json"

    def _fetch(self) -> dict[str, SourceRecord]:
        try:
            blob = json.loads(self.path.read_text(encoding="utf-8"))
        except FileNotFoundError:
            self.missing_reason = "in_house_anchor_not_published"
            return {}
        except (OSError, ValueError) as exc:
            self.missing_reason = f"anchor_unreadable: {exc}"
            return {}
        rows = blob.get("records") if isinstance(blob, dict) else None
        if not isinstance(rows, list):
            self.missing_reason = "anchor_schema_invalid"
            return {}
        records: dict[str, SourceRecord] = {}
        for row in rows:
            if not isinstance(row, dict):
                continue
            key = str(row.get("key") or "").strip()
            score = _float(row.get("score"))
            if not key or score is None:
                continue
            records[key] = SourceRecord(
                key=key,
                name=str(row.get("name") or key),
                score=score,
                score_ci=_float(row.get("score_ci")),
                board_version=str(row.get("board_version")) if row.get("board_version") else None,
                raw=dict(row),
            )
        if not records:
            self.missing_reason = "anchor_records_empty"
        return records


def fetch_endpoint_stats(or_slug: str) -> dict[str, Any]:
    """GET /api/v1/models/{slug}/endpoints — uptime/latency/zdr per provider.

    Used by ``enrich_uptime`` after shortlisting (1 request/model instead of a
    full sweep). Fail-open: error -> {}.
    """
    try:
        data = _get_json(OR_ENDPOINTS_URL.format(slug=or_slug))
    except Exception as exc:
        logger.warning("model_selection endpoints failed: slug=%s error=%s", or_slug, exc)
        return {}
    rows = data.get("data") if isinstance(data, dict) else data
    uptimes: list[float] = []
    zdr = False
    latency: float | None = None
    for row in rows or []:
        if not isinstance(row, dict):
            continue
        u = _float(row.get("uptime_last_1d"))
        if u is not None:
            uptimes.append(u)
        tag = str(row.get("tag") or "")
        if "zdr" in tag.lower():
            zdr = True
        lat = _float(row.get("latency_last_30m"))
        if lat is not None:
            latency = lat if latency is None else min(latency, lat)
    return {
        "uptime_1d": max(uptimes) if uptimes else None,
        "zdr_available": zdr,
        "latency_30m": latency,
    }


# Default registry. Every source name declared in a TaskProfile resolves here
# (live fetcher, InHouseAnchorSource, or UnimplementedSource recording an
# honestly-probed gap) — nothing silently skipped. Consumers may still supply
# plugin sources via ``candidates(..., sources={...})`` overrides.
SOURCE_REGISTRY: dict[str, Source] = {
    "openrouter_models": OpenRouterModelsSource(),
    "artificial_analysis": ArtificialAnalysisSource(),
    "arena": ArenaSource(),
    "ugi": UGISource(),
    "mteb_classification": MtebClassificationSource(),
    "mteb_retrieval": MtebRetrievalSource(),
    "mteb_sts": MtebStsSource(),
    "mteb_reranking": MtebRerankingSource(),
    # vals.ai boards (static Astro tables, research §1)
    "vals_index": ValsSource("vals_index", "vals_index"),
    "vals_medscribe": ValsSource("vals_medscribe", "medscribe"),
    "vals_cua_bench": ValsSource("vals_cua_bench", "cua_bench"),
    # boards-v2 (2026-10-09): LegalBench proxy board — classification has no
    # living specialized board (mteb_classification only joined embedding
    # models); this is an explicitly-labeled proxy, not a chat-classification
    # board.
    "vals_legal_bench": ValsSource("vals_legal_bench", "legal_bench"),
    # meta-aggregator (markdown alternate, research §1)
    "benchlm": BenchLMSource(),
    # agentic (research §2.9)
    "tau2_bench_or": Tau2BenchORSource(),
    "bfcl": BFCLSource(),
    "gaia": GaiaSource(),
    "aa_agentic_index": AaAgenticIndexSource(),
    # healthcare/medical (research §2.6)
    "aa_healthcare_index": AaCapabilitySource("aa_healthcare_index", "healthcare_and_medical"),
    "healthbench": LLMStatsSource("healthbench", "healthbench"),
    "medhelm": MedhelmSource(),
    "medarena": UnimplementedSource(
        "medarena",
        "clinician-voted board requires Doximity sign-in; no public leaderboard "
        "endpoint found (probed 2026-10-07: JS-only shell, no data table)",
    ),
    "open_medical_llm": OpenMedicalLLMSource(),
    "bridge": BridgeSource(),
    # vision/manga text-in-image (research §2.5)
    "ocrbench": LLMStatsSource("ocrbench", "ocrbench"),
    # story-gen / EQ-Bench suite (research §2.2)
    "eqbench_creative_v3": EqBenchCsvSource(
        "eqbench_creative_v3",
        "https://eqbench.com/creative_writing.js",
        "leaderboardDataCreativeWritingV3",
        "elo_score",
    ),
    "eqbench_longform": EqBenchCsvSource(
        "eqbench_longform",
        "https://eqbench.com/creative_writing_longform.js",
        "leaderboardDataLongformV3",
        "overall_score_100",
    ),
    "eqbench4": EqBench4Source(),
    # translation (research §2.1, boards-v2 2026-10-09): WMT24++ stays labeled
    # self-reported; WMT25 General MT preliminary ranking is the annual
    # closed-inclusive anchor. flores_speakleash was REMOVED (stale 9 months —
    # space SLEEPING since 2026-01-13; research verdict "optional or drop").
    "wmt24pp": LLMStatsSource("wmt24pp", "wmt24++"),
    "wmt25_gmtr": Wmt25GmtrSource(),
    # boards-v2 (2026-10-09): closed-inclusive embed/rerank boards, the
    # LMArena agent arena, and OR real-world usage. Join names for these
    # boards are versioned in board_aliases.py; unmapped rows are reported,
    # never guessed.
    "agentset_elo": AgentSetEmbedSource(),
    "agentset_rerank": AgentSetRerankSource(),
    "hindsight_reranker": HindsightSource("hindsight_reranker", "reranker", "reranker_id"),
    "hindsight_embeddings": HindsightSource("hindsight_embeddings", "embeddings", "embedding_id"),
    "arena_agent": ArenaAgentSource(),
    # OR real-world usage (not a quality board): candidates() folds it into
    # ModelCandidate.or_task_spend_share — quadrant tie-break ONLY. Primary
    # feed needs OPENROUTER_API_KEY; the public frontend supplement is
    # undocumented and fail-open. Absent both -> recorded missing.
    "or_usage": OrUsageSource(),
    # translation/manga specialized boards (public, scan-only; 2026-10-08):
    # Manga109-derived OCR + MangaVQA via the manga109 org project site.
    "manga109_v2026": MangaBenchSource("manga109_v2026", score_col=1),  # MangaOCR Hmean %
    "mangavqa": MangaBenchSource("mangavqa", score_col=2),  # MangaVQA LLM score (/10)
    # in-house anchor sets (research §2.5/§2.8: self-run, not scraped boards)
    "aiora_triage_eval": InHouseAnchorSource("aiora_triage_eval"),
    # decision profile (2026-10-09): jevals decision board (CC-BY-4.0, frozen
    # dated releases; join names versioned in board_aliases) + eqbench
    # Judgemark V4 — a creative-writing JUDGING domain proxy that may only
    # corroborate, never single-handedly rank a decision model (see TASKS).
    "jevals": JevalsSource(),
    # JevBench (2026-10-09): the independent second decision board —
    # native-decision models with rows on BOTH jevals and jevbench now reach
    # strong evidence under the >=2-boards rule (Mercury Decide, Jev 1.13).
    "jevbench": JevbenchSource(),
    # Jev Decision Index (2026-10-09, user-named): reference board for
    # open-weight Jev repros; thin after the OR-servable filter (4 survivors)
    # by design — still a live independent signal for those rows.
    "jev_decision_index": JevDecisionIndexSource(),
    "judgemark_v4": EqBenchCsvSource(
        "judgemark_v4",
        "https://eqbench.com/judgemark-v4.js",
        "leaderboardDataJudgemarkV4",
        "score",
        name_cols=("model",),
    ),
}
