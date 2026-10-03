"""hull_core.llm: unified LLM transport (chat / embed / rerank / DSPy).

Ships in the ``hull-core`` dist; needs the ``[llm]`` extra
(``pip install "hull-core[llm]"``); ``dspy_lm`` additionally needs ``[dspy]``.
"""

# Must stay the first import: raises ImportError with the install hint when the
# [llm] extra is missing.
import hull_core.llm._extra  # noqa: F401
from hull_core.llm.transport import (
    acompletion,
    acompletion_text,
    aembedding,
    arerank,
    cohere_routing,
    completion,
    completion_text,
    provider_params,
)
from hull_core.llm.vertex_express import (
    VERTEX_EXPRESS_PREFIX,
    VertexExpressError,
    acompletion_express,
    completion_express,
)

__all__ = [
    "VERTEX_EXPRESS_PREFIX",
    "VertexExpressError",
    "acompletion",
    "acompletion_express",
    "acompletion_text",
    "aembedding",
    "arerank",
    "cohere_routing",
    "completion",
    "completion_express",
    "completion_text",
    "provider_params",
]
