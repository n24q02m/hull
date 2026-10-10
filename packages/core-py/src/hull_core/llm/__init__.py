"""hull_core.llm: unified LLM transport (chat / embed / rerank / DSPy).

Ships in the ``n24q02m-hull`` dist; needs the ``[llm]`` extra
(``pip install "n24q02m-hull[llm]"``); ``dspy_lm`` additionally needs ``[dspy]``.
"""

# Must stay the first import: raises ImportError with the install hint when the
# [llm] extra is missing.
import hull_core.llm._extra  # noqa: F401
from hull_core.llm.transport import (
    acompletion,
    acompletion_text,
    aembedding,
    arerank,
    completion,
    completion_text,
    provider_params,
)

__all__ = [
    "acompletion",
    "acompletion_text",
    "aembedding",
    "arerank",
    "completion",
    "completion_text",
    "provider_params",
]
