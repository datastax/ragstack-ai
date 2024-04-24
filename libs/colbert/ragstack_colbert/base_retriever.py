"""
This module defines abstract base classes for implementing retrieval mechanisms for text chunk
embeddings, specifically designed to work with ColBERT or similar embedding models.
"""

from abc import ABC, abstractmethod
from typing import Any, List, Optional

from .objects import RetrievedChunk


class BaseRetriever(ABC):
    """
    Abstract base class (ABC) for a retrieval system that operates on a ColBERT vector store, facilitating
    the search and retrieval of text chunks based on query embeddings.
    """

    @abstractmethod
    def close(self) -> None:
        """
        Closes the retriever, releasing any resources or connections used during operation.
        Implementations should ensure that all necessary cleanup is performed to avoid resource leaks.
        """

    @abstractmethod
    def retrieve(
        self,
        query: str,
        k: Optional[int] = None,
        query_maxlen: Optional[int] = None,
        **kwargs: Any
    ) -> List[RetrievedChunk]:
        """
        Retrieves a list of text chunks relevant to a given query from the vector store, ranked by
        relevance or other metrics.

        Parameters:
            query (str): The query text to search for relevant text chunks.
            k (Optional[int]): The number of top results to retrieve.
            query_maxlen (Optional[int]): The maximum length of the query to consider. If None, the
                                          maxlen will be dynamically generated.
            **kwargs (Any): Additional parameters that implementations might require for customized
                            retrieval operations.

        Returns:
            List[RetrievedChunk]: A list of `RetrievedChunk` instances representing the retrieved
                                  text chunks, ranked by their relevance to the query.
        """
