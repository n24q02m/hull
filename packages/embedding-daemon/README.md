# hull_embedding_daemon

Shared local ONNX/GGUF embedding server for the wet / crg / mnemo products
(via hull). One daemon serves embeddings over HTTP so every product shares a
single model instance instead of loading its own.

Ships in the `n24q02m-hull` dist; the server stack is the `[embedding]` extra:

```bash
pip install "n24q02m-hull[embedding]"
hull-embedding-daemon --help
```

## Endpoints

| Route | Purpose |
|-------|---------|
| `GET /health` | liveness + version |
| `GET /models` | public alias table -> canonical model names |
| `POST /embed` | `{input: [str], model?, dims?}` -> `{data: [[float]], model, dims}` |
| `POST /rerank` | `{query, documents: [str], model?, top_n?}` -> `{results: [{index, relevance_score}]}` sorted desc |

`/embed` honours MRL truncation: `dims` accepts 32..1024 (default 1024).
`/rerank` defaults to the full ranking; `top_n >= 1` caps it.

## Models

Inference runs through [qwen3-embed](https://github.com/n24q02m/qwen3-embed)
over the `n24q02m/Qwen3-*` exports. Public aliases:

| Alias | Resolves to | Kind |
|-------|-------------|------|
| `qwen3-0.6b` | `n24q02m/Qwen3-Embedding-0.6B-ONNX` | embed |
| `qwen3-0.6b-q4f16` | `n24q02m/Qwen3-Embedding-0.6B-ONNX-Q4F16` | embed |
| `qwen3-0.6b-gguf` | `n24q02m/Qwen3-Embedding-0.6B-GGUF` | embed |
| `qwen3-rerank-0.6b` | `n24q02m/Qwen3-Reranker-0.6B-ONNX` | rerank |
| `qwen3-rerank-0.6b-q4f16` | `n24q02m/Qwen3-Reranker-0.6B-ONNX-Q4F16` | rerank |
| `qwen3-rerank-0.6b-yesno` | `n24q02m/Qwen3-Reranker-0.6B-ONNX-YesNo` | rerank |
| `qwen3-rerank-0.6b-gguf` | `n24q02m/Qwen3-Reranker-0.6B-GGUF` | rerank |

Canonical `n24q02m/...` names pass through unchanged. Model artifacts
download from Hugging Face into the user cache lazily on first use.

## Configuration

| Knob | Flag | Env |
|------|------|-----|
| Bind | `--host` / `--port` (127.0.0.1:9800) | — |
| Model cache | `--cache-dir` | `HULL_EMBEDDING_CACHE_DIR` (falls back to `QWEN3_EMBED_CACHE_PATH` / platform user cache) |
| Intra-op threads | `--threads` | `HULL_EMBEDDING_THREADS` |

Backends: ONNX (default). CUDA: replace `onnxruntime` with
`onnxruntime-gpu` in the target environment (the two wheels ship the same
`onnxruntime` module and must not be installed side by side).

GGUF: `*-GGUF` names run on llama.cpp via `llama-cpp-python`, which ships
source-only on PyPI and needs a C++ toolchain — opt in with:

```bash
pip install "n24q02m-hull[embedding,embedding-gguf]"
```

Without it, GGUF requests fail `503` with the install hint; ONNX requests
are unaffected.
