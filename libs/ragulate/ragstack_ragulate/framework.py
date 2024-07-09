from enum import Enum


class Framework(Enum):
    """Frameworks supported by RagStack Ragulate."""

    LANG_CHAIN = "langChain"
    LLAMA_INDEX = "llamaIndex"
