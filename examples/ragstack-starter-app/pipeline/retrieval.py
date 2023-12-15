from typing import Optional

from langchain_core.vectorstores import VectorStoreRetriever
from langchain_core.vectorstores import VectorStore


def as_retriever(vstore: VectorStore, k: Optional[int] = 4) -> VectorStoreRetriever:
    """
    Convert a VectorStore into a VectorStoreRetriever.

    Args:
        vstore (VectorStore): The VectorStore to be converted into a retriever.
        k (Optional[int]): Amount of documents to return
            Default is 4 if not specified.

    Returns:
        VectorStoreRetriever: A retriever instance.
    """
    return vstore.as_retriever(search_kwargs={"k": k})
