try:
    from ragstack_colbert.base_retriever import BaseRetriever
except (ImportError, ModuleNotFoundError):
    raise ImportError(
        "Could not import ragstack-ai-colbert. "
        "Please install it with `pip install ragstack-ai-llamaindex[colbert]`."
    )

from .colbert_retriever import ColbertRetriever

__all__ = [
    "ColbertRetriever",
]
