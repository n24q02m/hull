"""Review Focus 4 boundary: hull must not own or import a model runtime.

The embed/rerank service lives in the ``fastretrieval`` dist
(``fastretrieval[server]`` serving ``/embed`` + ``/rerank``). hull is a
consumer at most: an HTTP client, never an importer of the runtime. This test
fails on a reintroduced model runtime, and the reverse direction is asserted in
fastretrieval's own ``tests/test_server_boundary.py``.
"""

from __future__ import annotations

import ast
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

SOURCE_ROOTS = (
    ROOT / "packages" / "core-py" / "src",
    ROOT / "packages" / "web" / "src",
)

# Import names that would mean hull owns or links a model runtime again. This
# covers the runtime itself (fastretrieval, its legacy dist name qwen3-embed),
# the ONNX/llama.cpp execution layers, and the PyTorch model stacks the
# converter uses. A caller reaches embeddings over HTTP, so none of these may
# appear anywhere in hull's shipped sources.
FORBIDDEN = frozenset(
    {
        "fastretrieval",
        "qwen3_embed",
        "onnxruntime",
        "llama_cpp",
        "torch",
        "transformers",
        "sentence_transformers",
        "optimum",
    }
)


def _imported_roots(path: Path) -> set[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    roots: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            roots.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
            roots.add(node.module.split(".")[0])
    return roots


def test_shipped_sources_never_import_a_model_runtime() -> None:
    files = sorted(p for root in SOURCE_ROOTS for p in root.rglob("*.py"))
    assert files, "no shipped sources found"
    offenders = []
    for path in files:
        for name in sorted(_imported_roots(path) & FORBIDDEN):
            offenders.append(f"{path.relative_to(ROOT)}: {name}")
    assert not offenders, "hull must not import a model runtime: " + ", ".join(offenders)


def test_no_embedding_daemon_package_remains() -> None:
    # Scoped to packages/ on purpose: sibling worktrees and virtualenvs under
    # the repo root are not this checkout's source.
    assert not (ROOT / "packages" / "embedding-daemon").exists()
    assert not list((ROOT / "packages").rglob("hull_embedding_daemon"))


def test_pyproject_declares_no_model_runtime() -> None:
    text = (ROOT / "pyproject.toml").read_text(encoding="utf-8").lower()
    for name in ("qwen3", "onnxruntime", "llama-cpp", "embedding-daemon", "hull-embedding-daemon"):
        assert name not in text, f"pyproject.toml still declares {name}"


def test_importing_the_core_pulls_no_model_runtime() -> None:
    code = (
        "import sys, hull_core; "
        f"print(sorted(m.split('.')[0] for m in sys.modules if m.split('.')[0] in {sorted(FORBIDDEN)!r}))"
    )
    proc = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=False)
    assert proc.returncode == 0, proc.stderr
    assert proc.stdout.strip() == "[]", proc.stdout
