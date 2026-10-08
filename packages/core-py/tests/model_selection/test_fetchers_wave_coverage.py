"""Coverage completion for the wave-6 fetchers + gate plumbing.

Deterministic offline unit tests for the remaining uncovered statements:
parser guard rows, fail-open fallbacks, honest missing reasons, degenerate
pareto/knee shapes, pricing-parse failures, and eval-on-change skip. Every
network touch is monkeypatched; all fixtures are inline or from fixtures/.
"""

from __future__ import annotations

import json
import sys
import time
import types
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

import hull_core.model_selection.pareto as pareto_mod
from hull_core.model_selection import FileCache
from hull_core.model_selection import decide_promotion, refresh
from hull_core.model_selection import ModelRegistry
from hull_core.model_selection.normalize import ModelCandidate, passes_constraints
from hull_core.model_selection.pareto import knee_point, pick as pareto_pick, quadrants
from hull_core.model_selection.sources import (
    ArenaSource,
    AaAgenticIndexSource,
    AaCapabilitySource,
    BridgeSource,
    EqBench4Source,
    EqBenchCsvSource,
    FloresSpeakleashSource,
    GaiaSource,
    InHouseAnchorSource,
    MedhelmSource,
    OpenMedicalLLMSource,
    _BaseSource,
    _parse_benchlm_md,
    _parse_bfcl_csv,
    _parse_bridge_blob,
    _parse_flores_csv,
    _parse_llmstats_table,
    _parse_medhelm_group,
    _parse_or_bench_table,
    _parse_vals_table,
    _latest_medhelm_release,
    _read_parquet_rows,
    _table_rows,
)
from hull_core.model_selection.tasks import Constraints

FIXTURES = Path(__file__).parent / "fixtures"

# --- _BaseSource / table reader -------------------------------------------------


def test_base_source_fetch_is_abstract():
    with pytest.raises(NotImplementedError):
        _BaseSource()._fetch()


def test_table_rows_without_table_is_empty():
    assert _table_rows("<html><body><p>no tables here</p></body></html>") == []


# --- parquet readers (arena / gaia) ---------------------------------------------


def test_read_parquet_rows_prefers_pandas(monkeypatch):
    seen: dict[str, bytes] = {}

    class _FakeFrame:
        def to_dict(self, orient: str) -> list[dict]:
            assert orient == "records"
            return [{"a": 1.5}]

    def _fake_read_parquet(buf) -> _FakeFrame:
        seen["blob"] = buf.read()
        return _FakeFrame()

    fake_pd = types.SimpleNamespace(read_parquet=_fake_read_parquet)
    monkeypatch.setitem(sys.modules, "pandas", fake_pd)
    assert _read_parquet_rows(b"parquet-bytes") == [{"a": 1.5}]
    assert seen["blob"] == b"parquet-bytes"


def test_arena_filters_non_overall_and_malformed_rows():
    rows = [
        {"category": "math", "model_name": "M", "rating": 99.0},  # non-overall -> skip
        {"category": "overall", "model_name": "", "rating": 1.0},  # no name -> skip
        {"category": "overall", "model_name": "Good M", "rating": None},  # no score -> skip
        {
            "category": "overall",
            "model_name": "Good M",
            "rating": 42.5,
            "rating_lower": 40.0,
            "rating_upper": 45.0,
        },
    ]
    with (
        patch("hull_core.model_selection.sources._get_bytes", return_value=b""),
        patch("hull_core.model_selection.sources._read_parquet_rows", return_value=rows),
    ):
        recs = ArenaSource().fetch()
    assert set(recs) == {"good-m"}
    assert recs["good-m"].score == 42.5
    assert recs["good-m"].score_ci == pytest.approx(2.5)


def test_gaia_skips_nameless_rows_and_keeps_best_run():
    rows = [
        {"model": "", "score": 0.4},  # no name -> skip
        {"model": "Model Y", "score": None},  # no score -> skip
        {"model": "Model Y", "score": 0.3, "date": "2025-01-01"},
        {"model": "Model Y", "score": 0.2, "date": "2025-02-01"},  # worse run ignored
    ]
    with (
        patch("hull_core.model_selection.sources._get_bytes", return_value=b""),
        patch("hull_core.model_selection.sources._read_parquet_rows", return_value=rows),
    ):
        recs = GaiaSource().fetch()
    assert set(recs) == {"model-y"}
    assert recs["model-y"].score == 0.3
    assert recs["model-y"].board_version == "2025-01-01"


# --- vals.ai --------------------------------------------------------------------


def test_parse_vals_table_skips_rows_without_score():
    html = (
        "<table><tr><th>Model</th><th>Acc</th></tr>"
        '<tr><td><a href="/models/good-m">Good M</a></td><td>72.4%</td></tr>'
        '<tr><td><a href="/models/bad-m">Bad M</a></td><td>—</td><td>n/a</td></tr>'
        "</table>"
    )
    recs = _parse_vals_table(html)
    assert set(recs) == {"good-m"}
    assert recs["good-m"].score == 72.4


# --- benchlm --------------------------------------------------------------------


def test_parse_benchlm_md_without_rankings_section_is_empty():
    assert _parse_benchlm_md("# just prose\nno table") == {}


def test_parse_benchlm_md_skips_malformed_rows_and_stops_at_prose():
    md = (
        "## Overall rankings\n"
        "| Rank | Model | Acc | Cost | Ctx | Score |\n"
        "|------|-------|-----|------|-----|-------|\n"
        "| 1 | [Good M](/models/good-m) | - | - | - | 88.4 |\n"
        "| 2 | Plain Name | - | - | - | 77.0 |\n"  # no link -> skip
        "| 3 | [No Score](/models/no-score) | - | - | - | oops |\n"  # no score -> skip
        "\n"
        "post-table prose, table over\n"
        "| 4 | [After](/models/after) | - | - | - | 10.0 |\n"
    )
    recs = _parse_benchlm_md(md)
    assert set(recs) == {"good-m"}
    assert recs["good-m"].score == 88.4
    assert "after" not in recs  # prose ended the table


# --- OpenRouter benchmark pages -------------------------------------------------


def test_parse_or_bench_table_skips_rows_without_accuracy():
    html = (
        "<table><tr><th>Rank</th><th>Model</th><th>Acc</th></tr>"
        "<tr><td>1</td><td>Vendor: Model X</td><td>oops</td></tr>"
        "<tr><td>2</td><td>Model Y  Pareto</td><td>55.1%</td><td>± 2.8 pp</td></tr>"
        "</table>"
    )
    out = _parse_or_bench_table(html)
    assert out == {"model-y": (55.1, 2.8)}


# --- llm-stats ------------------------------------------------------------------


def test_parse_llmstats_table_skips_rows_without_score():
    html = (
        "<table>"
        '<tr><td>1</td><td><a href="/models/good-m">Good M</a></td><td>0.82</td></tr>'
        '<tr><td>2</td><td><a href="/models/bad-m">Bad M</a></td><td>nope</td></tr>'
        "</table>"
    )
    recs = _parse_llmstats_table(html)
    assert set(recs) == {"good-m"}
    assert recs["good-m"].score == pytest.approx(0.82)


# --- MedHELM --------------------------------------------------------------------


def test_medhelm_release_listing_ignores_non_release_prefixes():
    listing = {"prefixes": ["store/2026/", "releases/v1.10.0/", "releases/v1.2.3/"]}
    with patch("hull_core.model_selection.sources._get_json", return_value=listing):
        assert _latest_medhelm_release() == "v1.10.0"


def test_medhelm_fetch_stamps_board_version():
    listing = {"prefixes": ["releases/v0.4.2/"]}
    blob = (FIXTURES / "medhelm_groups.json").read_bytes()
    with (
        patch("hull_core.model_selection.sources._get_json", return_value=listing),
        patch("hull_core.model_selection.sources._get_bytes", return_value=blob),
    ):
        recs = MedhelmSource().fetch()
    assert recs
    assert all(r.board_version == "v0.4.2" for r in recs.values())


def test_medhelm_group_rejects_non_list_blob():
    assert _parse_medhelm_group(b'{"not": "a list"}') == {}


def test_medhelm_group_without_usable_table_is_empty():
    assert _parse_medhelm_group(b'[{"neither": "header nor rows"}]') == {}


def test_medhelm_group_skips_unusable_rows():
    blob = json.dumps(
        [
            {
                "header": [{"value": "Model"}, {"value": "accuracy"}],
                "rows": [
                    "not-a-list",  # non-list row -> skip
                    [],  # empty row -> skip
                    [{"value": ""}],  # empty name -> skip
                    [{"value": "M1"}],  # task cells missing -> skip
                ],
            }
        ]
    ).encode()
    assert _parse_medhelm_group(blob) == {}


def test_medhelm_group_mixed_scale_column_never_scored():
    """A column mixing 0-1 accuracies with 1-5 jury scores is excluded; a model
    holding ONLY excluded-column values gets no composite score."""
    blob = json.dumps(
        [
            {
                "header": [{"value": "Model"}, {"value": "accuracy"}, {"value": "jury_score"}],
                "rows": [
                    [{"value": "Model One"}, {"value": 0.8}, {"value": 4.5}],
                    [{"value": "Model Two"}, "n/a", {"value": 4.0}],
                ],
            }
        ]
    ).encode()
    recs = _parse_medhelm_group(blob)
    assert set(recs) == {"model-one"}
    assert recs["model-one"].score == pytest.approx(0.8)


# --- BFCL -----------------------------------------------------------------------


def test_parse_bfcl_csv_skips_unscored_and_nameless_rows():
    text = "Rank,Overall Acc,Model\n1,90.2%,Good M\n2,zzz,Bad M\n3,50.0%,\n"
    recs = _parse_bfcl_csv(text)
    assert set(recs) == {"good-m"}
    assert recs["good-m"].score == 90.2


# --- EQ-Bench -------------------------------------------------------------------


def test_eqbench_csv_missing_var_is_empty():
    src = EqBenchCsvSource("eqbench_x", "https://eqbench.com/x.js", "ABSENT_VAR", "score")
    with patch("hull_core.model_selection.sources._get_bytes", return_value=b"window.other = `a,b\n1,2`;"):
        assert src.fetch() == {}
    assert src.missing_reason == "empty_fetch"


def test_eqbench4_malformed_js_is_empty():
    src = EqBench4Source()
    with patch("hull_core.model_selection.sources._get_bytes", return_value=b"no braces here"):
        assert src.fetch() == {}
    assert src.missing_reason == "empty_fetch"


def test_eqbench4_skips_non_dict_and_unscored_rows():
    data = {
        "generated_at": "2026-10-07T00:00:00Z",
        "models": [
            "not-a-dict",
            {"model": "", "elo": 90.0},  # empty name -> skip
            {"slug": "no-elo", "elo": None},  # no score -> skip
            {"model": "Good M", "elo": 88.5, "ci_low": 85.0, "ci_high": 92.0},
        ],
    }
    with patch("hull_core.model_selection.sources._get_bytes", return_value=json.dumps(data).encode()):
        recs = EqBench4Source().fetch()
    assert set(recs) == {"good-m"}
    assert recs["good-m"].score == 88.5
    assert recs["good-m"].score_ci == pytest.approx(3.5)
    assert recs["good-m"].board_version == "2026-10-07"


# --- FLORES ---------------------------------------------------------------------


def test_parse_flores_csv_shape_guards():
    assert _parse_flores_csv("Task,Metric,M\n") == {}  # fewer than 3 rows
    assert _parse_flores_csv("Task,Metric,M\nx,y,z\nshort,row\n") == {}  # short data row only


def test_parse_flores_csv_skips_short_rows_and_scoreless_models():
    text = (
        "Task,Metric,ModelA,EmptyM\n"
        ",\n"  # short row -> skip
        "ogx_flores200-trans-eng,chrF,50.5,\n"  # EmptyM never scored
        "ogx_flores200-trans-ces,chrF,60.0,\n"
    )
    recs = _parse_flores_csv(text)
    assert set(recs) == {"modela"}
    assert recs["modela"].score == pytest.approx(55.25)
    assert recs["modela"].raw == {"n_tasks": 2}


def test_flores_source_fetch_uses_fixture_bytes():
    text = (FIXTURES / "flores_results.csv").read_text(encoding="utf-8")
    with patch("hull_core.model_selection.sources._get_bytes", return_value=text.encode()):
        recs = FloresSpeakleashSource().fetch()
    assert recs  # real fixture still parses end to end


# --- BRIDGE ---------------------------------------------------------------------


def test_parse_bridge_blob_skips_scoreless_and_nameless_entries():
    data = {
        "Model": {"0": "Model A", "1": "", "2": "Model C"},
        "Average Performance": {"0": 55.0, "1": 66.0, "2": None},
    }
    assert _parse_bridge_blob(data) == {"model-a": 55.0}


def test_bridge_fetch_skips_non_dict_mode_payloads():
    payloads = [
        {"Model": {"0": "Model A"}, "Average Performance": {"0": 55.0}},
        "not-a-dict",  # broken mode file -> skip, board survives
    ]
    with patch("hull_core.model_selection.sources._get_json", side_effect=payloads):
        recs = BridgeSource(modes=("mode1", "mode2")).fetch()
    assert set(recs) == {"model-a"}
    assert recs["model-a"].score == 55.0
    assert recs["model-a"].raw["modes"] == {"mode1": 55.0}


# --- Open Medical LLM -----------------------------------------------------------


def _open_medical_client(side_effects: list) -> MagicMock:
    client = MagicMock()
    client.__enter__ = MagicMock(return_value=client)
    client.__exit__ = MagicMock(return_value=False)
    client.get.side_effect = side_effects
    return client


def _open_medical_resp(payload) -> MagicMock:
    r = MagicMock()
    r.json.return_value = payload
    r.raise_for_status.return_value = None
    return r


def test_open_medical_skips_failing_and_scoreless_files():
    tree = [
        {"type": "file", "path": "org/bad/results.json"},
        {"type": "file", "path": "org/good/results.json"},
        {"type": "file", "path": "org/noscore/results.json"},
    ]
    client = _open_medical_client(
        [
            RuntimeError("boom"),  # fetch error -> skip file
            _open_medical_resp({"results": {"task_a": {"acc,none": 0.5}}}),
            _open_medical_resp({}),  # no results dict -> no score -> skip
        ]
    )
    with (
        patch("hull_core.model_selection.sources._get_json", return_value=tree),
        patch("hull_core.model_selection.sources.httpx.Client", return_value=client),
    ):
        recs = OpenMedicalLLMSource().fetch()
    assert set(recs) == {"org-good"}
    assert recs["org-good"].score == pytest.approx(0.5)


def test_open_medical_mean_accuracy_guards():
    assert OpenMedicalLLMSource._mean_accuracy({}) is None  # results not a dict
    assert OpenMedicalLLMSource._mean_accuracy({"results": "nope"}) is None
    blob = {"results": {"broken": "not-a-dict", "ok": {"acc,none": 0.4}}}
    assert OpenMedicalLLMSource._mean_accuracy(blob) == pytest.approx(0.4)


# --- AA capability / agentic ----------------------------------------------------


def test_aa_capability_skips_non_dict_and_slugless_rows(monkeypatch):
    monkeypatch.delenv("AA_API_KEY", raising=False)
    field = "artificial_analysis_healthcare_and_medical_index"
    payload = {
        "data": [
            "not-a-dict",  # -> skip
            {"name": "", "slug": "", "evaluations": {field: 51.2}},  # no slug -> skip
            {"slug": "GPT-6", "name": "GPT-6", "evaluations": {field: 51.2}},
        ]
    }
    with patch("hull_core.model_selection.sources.httpx.get", return_value=_open_medical_resp(payload)):
        recs = AaCapabilitySource("aa_healthcare_index", "healthcare_and_medical", api_key="k").fetch()
    assert set(recs) == {"GPT-6"}
    assert recs["GPT-6"].score == 51.2


def test_aa_agentic_with_key_delegates_to_capability_fetch(monkeypatch):
    monkeypatch.delenv("AA_API_KEY", raising=False)
    payload = {
        "data": [{"slug": "agent-pro", "name": "Agent Pro", "evaluations": {"artificial_analysis_agentic_index": 70.0}}]
    }
    with patch("hull_core.model_selection.sources.httpx.get", return_value=_open_medical_resp(payload)) as get:
        recs = AaAgenticIndexSource(api_key="k").fetch()
    assert get.call_args.kwargs["headers"] == {"x-api-key": "k"}
    assert set(recs) == {"agent-pro"}
    assert recs["agent-pro"].score == 70.0


def test_aa_agentic_without_key_is_honest_missing(monkeypatch):
    """No AA_API_KEY -> recorded gap; the OR-embedded copy is NEVER board evidence."""
    monkeypatch.delenv("AA_API_KEY", raising=False)
    src = AaAgenticIndexSource()
    assert src.fetch() == {}
    assert src.missing_reason == "aa_api_key_absent"


# --- In-house anchors -----------------------------------------------------------


def test_in_house_anchor_invalid_schema_records_reason(tmp_path):
    src = InHouseAnchorSource("anchor_x", root=tmp_path)
    (tmp_path / "anchor_x.json").write_text(json.dumps({"records": "not-a-list"}), encoding="utf-8")
    assert src.fetch() == {}
    assert src.missing_reason == "anchor_schema_invalid"


def test_in_house_anchor_all_rows_unusable_records_reason(tmp_path):
    src = InHouseAnchorSource("anchor_y", root=tmp_path)
    (tmp_path / "anchor_y.json").write_text(
        json.dumps({"records": ["not-a-dict", {"name": "no-key"}, {"key": "", "score": 1.0}]}),
        encoding="utf-8",
    )
    assert src.fetch() == {}
    assert src.missing_reason == "anchor_records_empty"


# --- cache ----------------------------------------------------------------------


def test_file_cache_fetched_at_absent_corrupt_and_valid(tmp_path):
    cache = FileCache(tmp_path)
    assert cache.fetched_at("absent") is None  # missing file
    cache._path("corrupt").write_text("{not json", encoding="utf-8")
    assert cache.fetched_at("corrupt") is None  # unparsable blob
    cache.set("good", {"x": 1})
    assert cache.fetched_at("good") == pytest.approx(time.time(), abs=30)
    assert cache.get("good", ttl_seconds=60) == {"x": 1}


# --- constraints ----------------------------------------------------------------


def test_passes_constraints_structured_outputs_fail_closed():
    def _cand(params: tuple[str, ...]) -> ModelCandidate:
        return ModelCandidate(
            or_slug="m",
            quality=80,
            context=32_000,
            modalities=("text",),
            supported_parameters=params,
        )

    without = _cand(("tools",))
    assert passes_constraints(without, Constraints(require_structured_outputs=True, require_tools=True)) is False
    with_so = _cand(("tools", "structured_outputs"))
    assert passes_constraints(with_so, Constraints(require_structured_outputs=True, require_tools=True)) is True


def test_passes_constraints_uptime_only_filters_when_known():
    def _cand(uptime: float | None) -> ModelCandidate:
        return ModelCandidate(
            or_slug="m",
            quality=80,
            context=32_000,
            modalities=("text",),
            supported_parameters=("tools",),
            uptime_1d=uptime,
        )

    up97 = _cand(97.0)
    assert passes_constraints(up97, Constraints(min_uptime_1d=99.0, require_tools=True)) is False
    assert passes_constraints(up97, Constraints(min_uptime_1d=95.0, require_tools=True)) is True
    assert passes_constraints(_cand(None), Constraints(min_uptime_1d=99.0)) is True  # unknown uptime stays


# --- pareto / knee --------------------------------------------------------------


def test_knee_quality_cost_tie_returns_cheapest_point():
    tied = [
        ModelCandidate(or_slug="a", quality=50, cost_1m_blended=1.0),
        ModelCandidate(or_slug="b", quality=50, cost_1m_blended=1.0),
        ModelCandidate(or_slug="c", quality=50, cost_1m_blended=1.0),
    ]
    front = knee_point(tied)
    assert front is tied[0]  # all-tie frontier: knee is undefined -> thrift


def test_knee_degenerate_chord_returns_cheapest(monkeypatch):
    monkeypatch.setattr(pareto_mod.math, "hypot", lambda *coords: 0.0)
    front = [
        ModelCandidate(or_slug="cheap", quality=50, cost_1m_blended=1.0),
        ModelCandidate(or_slug="mid", quality=95, cost_1m_blended=2.0),
        ModelCandidate(or_slug="best", quality=100, cost_1m_blended=10.0),
    ]
    assert knee_point(front) is front[0]


def test_quadrants_empty_input():
    assert quadrants([]) == {"attractive": [], "premium": [], "budget": [], "avoid": []}


def test_pick_all_unknown_cost_is_none():
    unknown = [
        ModelCandidate(or_slug="a", quality=90, cost_1m_blended=None),
        ModelCandidate(or_slug="b", quality=50, cost_1m_blended=None),
    ]
    assert pareto_pick(unknown, "knee") is None
    assert pareto_pick(unknown, "quadrant") is None


# --- registry: pricing parse + dominance + eval-on-change -----------------------


def _registry(tmp_path: Path) -> ModelRegistry:
    data = {
        "snapshot_meta": {"generated_at": "test"},
        "models": [
            {
                "litellm_id": "openrouter/m-incumbent",
                "display_name": "Incumbent",
                "provider": "openrouter",
                "openrouter_slug": "m-incumbent",
                "prompt_usd_per_1m": 1.0,
                "completion_usd_per_1m": 2.0,
                "nsfw_capable": False,
                "tiers": ["echo"],
                "chain_position": 0,
                "active": True,
            }
        ],
    }
    path = tmp_path / "model_rankings.json"
    path.write_text(json.dumps(data), encoding="utf-8")
    return ModelRegistry.load(path)


def _cand(slug: str, q: float, cost: float | None, rank: int | None = 0) -> ModelCandidate:
    c = ModelCandidate(or_slug=slug, quality=q, cost_1m_blended=cost, pareto_rank=rank)
    c.raw = {"pricing": {"prompt": str(cost / 1e6) if cost is not None else "0", "completion": "0.000002"}}
    return c


def test_promote_unparsable_pricing_leaves_fields_unset(tmp_path):
    reg = _registry(tmp_path)
    bad = ModelCandidate(or_slug="m/badprice", quality=95, cost_1m_blended=0.4)
    bad.raw = {"pricing": {"prompt": "abc", "completion": None}}  # ValueError + TypeError
    entry = reg.promote("m/badprice", candidate=bad)
    assert "prompt_usd_per_1m" not in entry
    assert "completion_usd_per_1m" not in entry

    sentinel = ModelCandidate(or_slug="m/sentinel", quality=94, cost_1m_blended=0.5)
    sentinel.raw = {"pricing": {"prompt": "-1", "completion": "0.000002"}}  # -1 = unknown, not free
    entry2 = reg.promote("m/sentinel", candidate=sentinel)
    assert "prompt_usd_per_1m" not in entry2
    assert entry2["completion_usd_per_1m"] == pytest.approx(2.0)


def test_decide_promotion_second_dominant_challenger_is_not_best():
    grid = [_cand("m/first", 90, 0.5), _cand("m/second", 90, 0.5)]
    rows = [
        {"litellm_id": "openrouter/m/first", "score": 85.0, "n_scored": 5, "n_items": 5},
        {"litellm_id": "openrouter/m/second", "score": 85.0, "n_scored": 5, "n_items": 5},
    ]
    inc = {"litellm_id": "openrouter/m-incumbent", "score": 80.0, "secondary": None, "n_scored": 5, "n_items": 5}
    d = decide_promotion(grid, rows, incumbent_row=inc, incumbent_cost_1m=1.0)
    assert d.promote and d.winner_slug == "m/first"
    assert "m/second: dominant but not best" in d.per_challenger


def test_refresh_skips_eval_when_no_new_challengers(tmp_path):
    """All challengers already evaluated with unchanged signals -> eval skipped."""
    reg = _registry(tmp_path)
    cands = [_cand("m/weak", 95, 0.4, rank=0)]
    calls: list[list[str]] = []

    def eval_fn(ids: list[str]) -> list[dict]:
        calls.append(ids)
        return [
            {"litellm_id": ids[0], "score": 90.0, "n_scored": 5, "n_items": 5},
            {"litellm_id": ids[1], "score": 50.0, "n_scored": 5, "n_items": 5},
        ]

    r1 = refresh("translation", reg, cands, eval_fn=eval_fn)
    assert r1.gate == "no_dominance"  # challenger scored below incumbent -> not promoted
    r2 = refresh("translation", reg, cands, eval_fn=eval_fn)
    assert r2.gate == "eval_skipped_no_change"
    assert len(calls) == 1  # second pass spent no eval calls
