"""Base retriever module.

This module defines abstract base classes for implementing retrieval mechanisms for
text chunk embeddings, specifically designed to work with ColBERT or similar embedding
models.
"""

from abc import ABC, abstractmethod
from typing import Any, List, Optional, Tuple

from .objects import Chunk, Embedding


class BaseRetriever(ABC):
    """Base Retriever abstract class for ColBERT.

    Abstract base class (ABC) for a retrieval system that operates on a ColBERT
    vector store, facilitating the search and retrieval of text chunks based on query
    embeddings.
    """

    # handles LlamaIndex query
    @abstractmethod
    def embedding_search(
        self,
        query_embedding: Embedding,
        k: Optional[int] = None,
        include_embedding: bool = False,
        **kwargs: Any,
    ) -> List[Tuple[Chunk, float]]:
        """Search for relevant text chunks based on a query embedding.

        Retrieves a list of text chunks relevant to a given query from the vector
        store, ranked by relevance or other metrics.

        Args:
            query_embedding: The query embedding to search for relevant
                text chunks.
            k: The number of top results to retrieve.
            include_embedding: Optional (default False) flag to
                include the embedding vectors in the returned chunks
            **kwargs: Additional parameters that implementations might require
                for customized retrieval operations.

        Returns:
            A list of retrieved Chunk, float Tuples,
                each representing a text chunk that is relevant to the query,
                along with its similarity score.
        """

    # handles LlamaIndex async query
    @abstractmethod
    async def aembedding_search(
        self,
        query_embedding: Embedding,
        k: Optional[int] = None,
        include_embedding: bool = False,
        **kwargs: Any,
    ) -> List[Tuple[Chunk, float]]:
        """Search for relevant text chunks based on a query embedding.

        Retrieves a list of text chunks relevant to a given query from the vector
        store, ranked by relevance or other metrics.

        Args:
            query_embedding: The query embedding to search for relevant
                text chunks.
            k: The number of top results to retrieve.
            include_embedding: Optional (default False) flag to
                include the embedding vectors in the returned chunks
            **kwargs: Additional parameters that implementations might require
                for customized retrieval operations.

        Returns:
            A list of retrieved Chunk, float Tuples,
                each representing a text chunk that is relevant to the query,
                along with its similarity score.
        """

    # handles LangChain search
    @abstractmethod
    def text_search(
        self,
        query_text: str,
        k: Optional[int] = None,
        query_maxlen: Optional[int] = None,
        include_embedding: bool = False,
        **kwargs: Any,
    ) -> List[Tuple[Chunk, float]]:
        """Search for relevant text chunks based on a query text.

        Retrieves a list of text chunks relevant to a given query from the vector
        store, ranked by relevance or other metrics.

        Args:
            query_text: The query text to search for relevant text chunks.
            k: The number of top results to retrieve.
            query_maxlen: The maximum length of the query to consider.
                If None, the maxlen will be dynamically generated.
            include_embedding: Optional (default False) flag to
                include the embedding vectors in the returned chunks
            **kwargs: Additional parameters that implementations might require
                for customized retrieval operations.

        Returns:
            A list of retrieved Chunk, float Tuples,
                each representing a text chunk that is relevant to the query,
                along with its similarity score.
        """

    # handles LangChain async search
    @abstractmethod
    async def atext_search(
        self,
        query_text: str,
        k: Optional[int] = None,
        query_maxlen: Optional[int] = None,
        include_embedding: bool = False,
        **kwargs: Any,
    ) -> List[Tuple[Chunk, float]]:
        """Search for relevant text chunks based on a query text.

        Retrieves a list of text chunks relevant to a given query from the vector
        store, ranked by relevance or other metrics.

        Args:
            query_text: The query text to search for relevant text chunks.
            k: The number of top results to retrieve.
            query_maxlen: The maximum length of the query to consider.
                If None, the maxlen will be dynamically generated.
            include_embedding: Optional (default False) flag to
                include the embedding vectors in the returned chunks
            **kwargs: Additional parameters that implementations might require
                for customized retrieval operations.

        Returns:
            A list of retrieved Chunk, float Tuples,
                each representing a text chunk that is relevant to the query,
                along with its similarity score.
        """
