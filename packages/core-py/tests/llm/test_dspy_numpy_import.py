"""Guard for the dspy lazy-numpy import bug behind the [dspy] extra ceiling.

dspy >=3.3 binds ``np = require("numpy")`` at module level, which parks a lazy
proxy in ``sys.modules["numpy"]`` without running numpy's ``__init__``. A later
``import numpy.typing`` (pyarrow, pandas, ...) then loads the real numpy from
inside the half-initialized ``numpy._typing`` and fails with a circular
import. 3.3.0 and 3.4.0 both do this.

Each order runs in a fresh interpreter: pytest has already imported numpy, so
an in-process check would always pass. A dspy bump that still carries the bug
fails here, with the cause in the message, before it fails in some unrelated
test's collection.
"""

from __future__ import annotations

import subprocess
import sys

import pytest

pytest.importorskip("dspy")


@pytest.mark.parametrize(
    "code",
    [
        "import dspy; import numpy.typing",
        "import numpy.typing; import dspy",
    ],
)
def test_dspy_and_numpy_import_in_either_order(code):
    proc = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, timeout=120)
    assert proc.returncode == 0, (
        f"`{code}` failed in a fresh interpreter. If this is a dspy bump, the lazy numpy "
        f"proxy is still there; keep the [dspy] extra below it.\n{proc.stderr[-2000:]}"
    )
