"""HTTP API for shared embedding daemon.

Exposes /embed (text to vector), /rerank (query + docs to scores), /health.
Used by the wet / crg / mnemo products (via hull) to share a single
ONNX/GGUF model instance instead of loading per-server.

v0.1.0 alpha: /health works. /embed and /rerank return 501 Not Implemented
with a pointer to the roadmap because the ONNX + GGUF backends ship as
thin adapters around qwen3-embed in a follow-up release.
"""

from __future__ import annotations

from hull_embedding_daemon._extra import require_embedding_extra

# Must run before the extra's own imports below: fail with the install hint
# instead of a bare ModuleNotFoundError on a base `hull-core` install.
require_embedding_extra()

from fastapi import FastAPI, HTTPException, status  # noqa: E402
from pydantic import BaseModel  # noqa: E402

from hull_embedding_daemon import __version__  # noqa: E402


class EmbedRequest(BaseModel):
    model: str = "qwen3-0.6b"
    input: list[str]
    dims: int = 768


class EmbedResponse(BaseModel):
    data: list[list[float]]
    model: str
    dims: int


class RerankRequest(BaseModel):
    model: str = "qwen3-rerank-0.6b"
    query: str
    documents: list[str]
    top_n: int | None = None


class RerankResponse(BaseModel):
    results: list[dict]
    model: str


class HealthResponse(BaseModel):
    status: str
    version: str


app = FastAPI(title="hull-embedding-daemon", version=__version__)


NOT_IMPLEMENTED_DETAIL = (
    "Embedding backend (ONNX / GGUF) is not yet wired in v0.1.0. "
    "Track progress at https://github.com/n24q02m/hull/issues"
)


@app.get("/health", response_model=HealthResponse)
async def health() -> HealthResponse:
    return HealthResponse(status="ok", version=__version__)


@app.post("/embed", response_model=EmbedResponse)
async def embed(req: EmbedRequest) -> EmbedResponse:
    raise HTTPException(
        status_code=status.HTTP_501_NOT_IMPLEMENTED,
        detail=NOT_IMPLEMENTED_DETAIL,
    )


@app.post("/rerank", response_model=RerankResponse)
async def rerank(req: RerankRequest) -> RerankResponse:
    raise HTTPException(
        status_code=status.HTTP_501_NOT_IMPLEMENTED,
        detail=NOT_IMPLEMENTED_DETAIL,
    )
