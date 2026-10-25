"""Real-model smoke for hull-embedding-daemon (/health, /models, /embed, /rerank).

Starts the daemon as a subprocess, downloads the small q4f16 ONNX exports from
Hugging Face into a temp cache, and asserts the documented HTTP contract
(docs/HANDOVER.md "Embedding daemon"). Stdlib only; needs the ``embedding``
extra installed. Exit code 0 = every check passed.
"""

from __future__ import annotations

import json
import math
import os
import subprocess
import sys
import tempfile
import time
import urllib.error
import urllib.request

PORT = int(os.environ.get("HULL_SMOKE_PORT", "9811"))
BASE = f"http://127.0.0.1:{PORT}"
EMBED_MODEL = "qwen3-0.6b-q4f16"
RERANK_MODEL = "qwen3-rerank-0.6b-q4f16"
READY_TIMEOUT_S = 120
REQUEST_TIMEOUT_S = 900  # first call downloads the model artifacts


def call(method: str, path: str, body: dict | None = None) -> tuple[int, dict]:
    data = None if body is None else json.dumps(body).encode()
    req = urllib.request.Request(BASE + path, data=data, method=method, headers={"content-type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=REQUEST_TIMEOUT_S) as resp:
            return resp.status, json.loads(resp.read())
    except urllib.error.HTTPError as exc:
        return exc.code, json.loads(exc.read() or b"{}")


def check(cond: bool, label: str, detail: object = "") -> None:
    print(f"{'PASS' if cond else 'FAIL'}  {label}  {detail}")
    if not cond:
        raise SystemExit(1)


def cosine(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b, strict=True))
    return dot / (math.sqrt(sum(x * x for x in a)) * math.sqrt(sum(y * y for y in b)))


def wait_ready(proc: subprocess.Popen[bytes]) -> None:
    deadline = time.monotonic() + READY_TIMEOUT_S
    while time.monotonic() < deadline:
        if proc.poll() is not None:
            raise SystemExit(f"daemon exited early with code {proc.returncode}")
        try:
            status, _ = call("GET", "/health")
            if status == 200:
                return
        except OSError:
            time.sleep(1)
    raise SystemExit("daemon did not become ready")


def run_checks() -> None:
    status, body = call("GET", "/health")
    check(status == 200 and body.get("status") == "ok" and bool(body.get("version")), "GET /health", body)

    status, body = call("GET", "/models")
    check(
        status == 200 and EMBED_MODEL in body["embed"] and RERANK_MODEL in body["rerank"],
        "GET /models lists the aliases",
        sorted(body.get("embed", {})),
    )

    query = "How do I reset my account password?"
    relevant = "To reset your password, open Settings and choose Forgot password."
    unrelated = "The quarterly revenue grew twelve percent on strong cloud demand."
    status, body = call("POST", "/embed", {"model": EMBED_MODEL, "input": [query, relevant, unrelated], "dims": 512})
    check(status == 200, "POST /embed 200", status)
    vecs = body["data"]
    check(len(vecs) == 3 and all(len(v) == 512 for v in vecs) and body["dims"] == 512, "/embed shape + dims=512 MRL")
    norm = math.sqrt(sum(x * x for x in vecs[0]))
    check(abs(norm - 1.0) < 1e-2, "/embed vectors are L2-normalised", round(norm, 4))
    sim_rel, sim_irr = cosine(vecs[0], vecs[1]), cosine(vecs[0], vecs[2])
    check(sim_rel > sim_irr, "/embed relevant doc is closer than unrelated", (round(sim_rel, 3), round(sim_irr, 3)))

    status, _ = call("POST", "/embed", {"model": EMBED_MODEL, "input": [query], "dims": 8})
    check(status == 400, "/embed dims below MRL floor -> 400", status)
    status, _ = call("POST", "/embed", {"model": EMBED_MODEL, "input": [query], "dims": 2048})
    check(status == 400, "/embed dims above native width -> 400", status)

    docs = [unrelated, relevant, "The cafeteria menu changes every Tuesday."]
    status, body = call("POST", "/rerank", {"model": RERANK_MODEL, "query": query, "documents": docs})
    check(status == 200, "POST /rerank 200", status)
    results = body["results"]
    scores = [r["relevance_score"] for r in results]
    check(len(results) == 3 and scores == sorted(scores, reverse=True), "/rerank returns all docs, scores descending")
    check(results[0]["index"] == 1, "/rerank ranks the relevant doc first", results)
    status, body = call("POST", "/rerank", {"model": RERANK_MODEL, "query": query, "documents": docs, "top_n": 1})
    check(status == 200 and len(body["results"]) == 1 and body["results"][0]["index"] == 1, "/rerank top_n=1")

    status, body = call("POST", "/embed", {"model": "qwen3-0.6b-gguf", "input": [query]})
    check(
        status == 503 and "embedding-gguf" in json.dumps(body),
        "/embed GGUF without the extra -> 503 + install hint",
        status,
    )


def main() -> int:
    with tempfile.TemporaryDirectory(prefix="hull-embed-smoke-") as cache:
        proc = subprocess.Popen(
            [sys.executable, "-m", "hull_embedding_daemon", "--port", str(PORT), "--cache-dir", cache],
        )
        try:
            wait_ready(proc)
            run_checks()
        finally:
            proc.terminate()
            try:
                proc.wait(timeout=30)
            except subprocess.TimeoutExpired:
                proc.kill()
    print("embedding daemon smoke: all checks passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
