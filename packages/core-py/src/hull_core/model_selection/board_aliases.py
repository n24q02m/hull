"""Versioned board-name -> OpenRouter slug mapping table (2026-10-09).

Boards added in the boards-v2 wave publish display names (or vendor-flavored
ids) that the uniqueness-gated join ladder cannot always rescue ("Openai Text
Embedding 3 Large" vs ``openai/text-embedding-3-large``). Every NEW board
must declare its explicit name->or_slug mapping here; rows that resolve
neither through this table nor through the ladder are REPORTED (snapshot
``unmatched_names``), never guessed.

Rules for entries:
- The key is the exact ``SourceRecord.key`` the source's parser produces
  (slugified display name for HTML/JSON boards).
- The value is the OpenRouter catalog ``id`` observed live on 2026-10-09
  (catalog segments /api/v1/models?output_modalities=embeddings|rerank and
  the base text catalog). Ambiguous display names ("Mistral-Medium" — OR
  lists three mistral-medium generations; "Claude-4" — no unique claude-4
  route) stay OUT; they are reported unmapped instead.
- Removing an entry requires a board research note (unmapped rows change the
  pick evidence), not a silent edit.

``BOARD_ALIASES`` maps source name -> {record key: or_slug}. Boards absent
from the mapping (all pre-v2 sources) join through the ladder only, exactly
as before.
"""

from __future__ import annotations

BOARD_ALIASES_VERSION = "2026-10-09"

BOARD_ALIASES: dict[str, dict[str, str]] = {
    # AgentSet embeddings leaderboard (Elo, 18 models; results/benchmarks.json).
    # Mapped: 7 of 18 rows are OR-servable today. Unmapped (not on OR):
    # zembed-1, Jina V5 Small / V3, Voyage 3 Large / 3.5 / 3.5 Lite,
    # Cohere Embed V3 / Multilingual V3, Kanon 2, Qwen3 0.6B, Gemini 004.
    "agentset_elo": {
        "gemini-embedding-2": "google/gemini-embedding-2",
        "voyage-4": "voyageai/voyage-4",
        "openai-text-embedding-3-large": "openai/text-embedding-3-large",
        "openai-text-embedding-3-small": "openai/text-embedding-3-small",
        "qwen3-embedding-8b-deepinfra": "qwen/qwen3-embedding-8b",
        "qwen3-embedding-4b-deepinfra": "qwen/qwen3-embedding-4b",
        "bge-m3-deepinfra": "baai/bge-m3",
    },
    # AgentSet rerankers board (static HTML, 10 rows). Mapped: 6 of 10.
    # Unmapped (not on OR): zerank-2, zerank-1, zerank-1-small,
    # contextual-ai-rerank-v2-instruct.
    "agentset_rerank": {
        "cohere-rerank-4-pro": "cohere/rerank-4-pro",
        "cohere-rerank-4-fast": "cohere/rerank-4-fast",
        "cohere-rerank-3-5": "cohere/rerank-v3.5",
        "voyage-ai-rerank-2-5": "voyageai/rerank-2.5",
        "voyage-ai-rerank-2-5-lite": "voyageai/rerank-2.5-lite",
        "qwen3-reranker-8b": "qwen/qwen3-reranker-8b",
    },
    # Hindsight benchmarks (vectorize-io), per-model JSON, key = file stem /
    # model id inside the JSON. Mapped: 2 of 10 rerankers, 4 of 7 embeddings;
    # the rest are legacy generations (rerank v2/v3-multilingual, voyage-2) or
    # local baselines (flashrank, minilm, rrf) absent from the OR catalog.
    "hindsight_reranker": {
        "cohere-rerank-v4-pro": "cohere/rerank-4-pro",
        "cohere-rerank-v4-fast": "cohere/rerank-4-fast",
    },
    "hindsight_embeddings": {
        "text-embedding-3-small": "openai/text-embedding-3-small",
        "bge-base-en-v1-5": "baai/bge-base-en-v1.5",
        "bge-large-en-v1-5": "baai/bge-large-en-v1.5",
    },
    # WMT25 General MT preliminary ranking (arXiv 2508.14909v2, mean AutoRank
    # across 31 language pairs). Only systems verifiably present on today's OR
    # text catalog are mapped; anonymous baselines (ONLINE-*), team systems
    # (Shy-hunyuan-MT, Wenyiil, ...), and ambiguous vendor names
    # (Mistral-Medium, Claude-4, DeepSeek-V3 — delisted generations) report
    # unmapped.
    "wmt25_gmtr": {
        "gemini-2-5-pro": "google/gemini-2.5-pro",
        "gpt-4-1": "openai/gpt-4.1",
        "qwen3-235b": "qwen/qwen3-235b-a22b",
        "llama-4-maverick": "meta-llama/llama-4-maverick",
        "gemma-3-27b": "google/gemma-3-27b-it",
        "gemma-3-12b": "google/gemma-3-12b-it",
        "commanda": "cohere/command-a",
        "llama-3-1-8b": "meta-llama/llama-3.1-8b-instruct",
        # "mistral-7b" -> mistralai/mistral-7b-instruct was removed 2026-10-09:
        # OR delisted the route; the stale mapping reported the row unmatched
        # (Mistral-7B), which is the honest state — never re-guessed.
        "qwen2-5-7b": "qwen/qwen-2.5-7b-instruct",
    },
    # LMArena agent arena (HF leaderboard-dataset branch `agent`). Tier/snapshot
    # suffixes are stripped by the fetcher, then the uniqueness-gated ladder
    # resolves the bare display names against OR parts — no explicit aliases
    # needed today. Rows for models absent from the OR catalog (gemini-4-argon,
    # solar-pro-4, qwen3-8-max, ...) report unmapped.
    "arena_agent": {},
}

# Unmatched-name reports are capped so a fully renamed board cannot bloat the
# snapshot; the count of the remainder is recorded alongside.
UNMATCHED_REPORT_LIMIT = 40
