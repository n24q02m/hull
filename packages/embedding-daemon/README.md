# hull-embedding-daemon

Shared local ONNX/GGUF embedding server for the wet / crg / mnemo products
(via hull). One daemon serves embeddings over HTTP so every product shares a
single model instance instead of loading its own.

```bash
hull-embedding-daemon --help
```

Backends: ONNX (default), GGUF (stub). CUDA via the `cuda` extra.
