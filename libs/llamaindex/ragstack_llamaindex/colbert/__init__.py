try:
    from ragstack_colbert.base_retriever import BaseRetriever  # noqa: F401
except (ImportError, ModuleNotFoundError) as e:
    msg = (
        "Could not import ragstack-ai-colbert. "
        "Please install it with `pip install ragstack-ai-llamaindex[colbert]`."
    )
    raise ImportError(msg) from e

from .colbert_retriever import ColbertRetriever

__all__ = [
    "ColbertRetriever",
]
