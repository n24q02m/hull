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
from pathlib import Path
from urllib.parse import quote
from typing import Any, Protocol, runtime_checkable

import httpx

from hull_core.model_selection.mteb_tasks import task_family

logger = logging.getLogger(__name__)

OR_MODELS_URL = "https://openrouter.ai/api/v1/models"
OR_ENDPOINTS_URL = "https://openrouter.ai/api/v1/models/{slug}/endpoints"
OR_BENCH_URL = "https://openrouter.ai/benchmarks/{bench}"
AA_MODELS_URL = "https://artificialanalysis.ai/api/v2/language/models"
LIVEBENCH_CSV_URL = "https://raw.githubusercontent.com/live-bench/LiveBench/main/livebench/data/stats.csv"
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
FLORES_CSV_URL = "https://huggingface.co/spaces/speakleash/leaderboard-flores/resolve/main/results.csv"
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

    def to_dict(self) -> dict[str, Any]:
        out: dict[str, Any] = {"status": self.status, "fetched_at": self.fetched_at, "row_count": self.row_count}
        if self.reason:
            out["reason"] = self.reason
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
    """Mandatory backbone: GET /api/v1/models (no auth).

    Each record keeps the raw JSON row in ``raw`` — pricing (USD/token,
    including tiered ``overrides`` by ``min_prompt_tokens``),
    ``context_length``, ``architecture.modality``, ``supported_parameters``,
    plus AA scores embedded in ``benchmarks.artificial_analysis.*`` (present
    only on some rows; absent = missing, not 0).
    """

    name = "openrouter_models"
    ttl_seconds = DAY

    def _fetch(self) -> dict[str, SourceRecord]:
        data = _get_json(OR_MODELS_URL)
        rows = data.get("data") if isinstance(data, dict) else data
        records: dict[str, SourceRecord] = {}
        for row in rows or []:
            slug = row.get("id")
            if not slug:
                continue
            records[slug] = SourceRecord(key=slug, name=row.get("name") or slug, raw=row)
        return records


class ArtificialAnalysisSource(_BaseSource):
    """AA Data API: GET /api/v2/language/models, header ``x-api-key``.

    Key is OPTIONAL (only raises rate limits): read from ``AA_API_KEY`` at
    fetch time; without a key -> {} (fail-open, nothing to scrape instead).
    ``board_version`` = major version of the AA index for the version guard.
    """

    name = "artificial_analysis"
    ttl_seconds = DAY

    def __init__(self, api_key: str | None = None) -> None:
        super().__init__()
        self._api_key = api_key

    def _fetch(self) -> dict[str, SourceRecord]:
        api_key = self._api_key or os.environ.get("AA_API_KEY")
        if not api_key:
            self.missing_reason = "aa_api_key_absent"
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


class LiveBenchSource(_BaseSource):
    """LiveBench (contamination hedge) via the CSV release on GitHub live-bench/LiveBench."""

    name = "livebench"
    ttl_seconds = WEEK

    def _fetch(self) -> dict[str, SourceRecord]:
        text = _get_bytes(LIVEBENCH_CSV_URL).decode("utf-8", errors="replace")
        return _parse_csv_records(
            text,
            name_cols=("model", "model_name", "name"),
            score_cols=("global_average", "score", "average", "avg"),
            ci_cols=("ci", "confidence_interval", "stderr"),
        )


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


def _aggregate_tables(tables: list[Any]) -> dict[str, dict[str, float]]:
    """Aggregate pyarrow Tables into ``{family: {model_name: mean score}}``.

    Rows on splits other than ``test`` are ignored; scores are averaged per
    (model, task) first, then per family across tasks (mean of task means), so
    a task with many subset rows does not dominate its family. Task names that
    do not classify into one of the four families are ignored (fail-open).
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
    """Base for the four MTEB family fetchers; subclasses pick their family."""

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

    def __init__(self, name: str, js_url: str, var_name: str, score_field: str) -> None:
        super().__init__()
        self.name = name
        self.js_url = js_url
        self.var_name = var_name
        self.score_field = score_field

    def _fetch(self) -> dict[str, SourceRecord]:
        js = _get_bytes(self.js_url).decode("utf-8", errors="replace")
        m = re.search(re.escape(self.var_name) + r"\s*=\s*`(.*?)`", js, re.S)
        if not m:
            return {}
        # the template literal opens with blank lines (and may carry CRLF)
        # before the CSV header — strip before handing to the CSV parser
        return _parse_csv_records(m.group(1).strip(), name_cols=("model_name",), score_cols=(self.score_field,))


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


def _parse_flores_csv(text: str) -> dict[str, SourceRecord]:
    """speakleash FLORES leaderboard CSV: task rows, ``bleu``/``chrf`` metric
    sub-rows (the metric row repeats the task only implicitly — an empty task
    cell belongs to the previous task row), model scores as columns with
    decimal commas. Score = mean chrF over the ``ogx_flores200-trans-*`` tasks
    (FLORES' primary metric)."""
    rows = list(csv.reader(io.StringIO(text)))
    if len(rows) < 3 or len(rows[0]) < 3:
        return {}
    models = [m.strip() for m in rows[0][2:]]
    sums: dict[str, list[float]] = {m: [] for m in models}
    current_task = ""
    for row in rows[2:]:
        if len(row) < 3:
            continue
        task, metric = (row[0] or "").strip(), (row[1] or "").strip().lower()
        if task:
            current_task = task
        if not current_task.startswith("ogx_flores200-trans") or metric != "chrf":
            continue
        for i, model in enumerate(models):
            val = _float((row[2 + i] or "").replace(",", ".") if 2 + i < len(row) else None)
            if val is not None:
                sums[model].append(val)
    records: dict[str, SourceRecord] = {}
    for model, vals in sums.items():
        if not vals:
            continue
        records[slugify(model)] = SourceRecord(
            key=slugify(model), name=model, score=sum(vals) / len(vals), raw={"n_tasks": len(vals)}
        )
    return records


class FloresSpeakleashSource(_BaseSource):
    """FLORES-200 community leaderboard (HF space speakleash/leaderboard-flores,
    ``results.csv`` in the space repo — research §2.1)."""

    name = "flores_speakleash"
    ttl_seconds = WEEK

    def _fetch(self) -> dict[str, SourceRecord]:
        text = _get_bytes(FLORES_CSV_URL).decode("utf-8", errors="replace")
        return _parse_flores_csv(text)


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
    """AA Capability Index (e.g. Healthcare & Medical) via the Data API.

    Key comes from ``AA_API_KEY`` at fetch time; absent key -> ``{}`` with
    ``missing_reason=aa_api_key_absent`` (recorded missing, never silent).
    Field names per the AA data-api docs (verified 2026-10-07):
    ``artificial_analysis_<capability>_index`` (+ ``_version`` where published;
    the Intelligence Index version is the documented fallback marker).
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
            self.missing_reason = "aa_api_key_absent"
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
    """AA Agentic Index — Data API when ``AA_API_KEY`` is present, else the
    OR-embedded ``benchmarks.artificial_analysis.agentic_index`` already on
    ``/api/v1/models`` rows (research §2.9, LIVE), so the profile works with
    zero paid-API dependency."""

    def __init__(self, api_key: str | None = None) -> None:
        super().__init__("aa_agentic_index", "agentic", api_key)

    def _fetch(self) -> dict[str, SourceRecord]:
        if self._api_key or os.environ.get("AA_API_KEY"):
            return super()._fetch()
        self.missing_reason = None
        data = _get_json(OR_MODELS_URL)
        rows = data.get("data") if isinstance(data, dict) else data
        records: dict[str, SourceRecord] = {}
        for row in rows or []:
            if not isinstance(row, dict) or not row.get("id"):
                continue
            aa = (row.get("benchmarks") or {}).get("artificial_analysis")
            val = _float(aa.get("agentic_index")) if isinstance(aa, dict) else None
            if val is None:
                continue  # embedded scores present on some rows only; absent != 0
            records[row["id"]] = SourceRecord(key=row["id"], name=row.get("name") or row["id"], score=val, raw=row)
        if not records:
            self.missing_reason = "aa_agentic_or_embedded_absent"
        return records


class InHouseAnchorSource(_BaseSource):
    """In-house eval anchor sets (wmt24pp, manga109_v2026, mangavqa,
    aiora_triage_eval) — self-run anchors per the wave6 research, NOT scraped
    boards. Reads ``<root>/<name>.json``::

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
    "livebench": LiveBenchSource(),
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
    # translation (research §2.1)
    "flores_speakleash": FloresSpeakleashSource(),
    # in-house anchor sets (research §2.5/§2.8: self-run, not scraped boards)
    "wmt24pp": InHouseAnchorSource("wmt24pp"),
    "manga109_v2026": InHouseAnchorSource("manga109_v2026"),
    "mangavqa": InHouseAnchorSource("mangavqa"),
    "aiora_triage_eval": InHouseAnchorSource("aiora_triage_eval"),
}
