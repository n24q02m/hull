# hull_embedding_daemon

Shared local ONNX/GGUF embedding server for the wet / crg / mnemo products
(via hull). One daemon serves embeddings over HTTP so every product shares a
single model instance instead of loading its own.

Ships in the `hull-core` dist; the server stack is the `[embedding]` extra:

```bash
pip install "hull-core[embedding]"
hull-embedding-daemon --help
```

Backends: ONNX (default), GGUF (stub). CUDA: replace `onnxruntime` with
`onnxruntime-gpu` in the target environment (the two wheels ship the same
`onnxruntime` module and must not be installed side by side).
