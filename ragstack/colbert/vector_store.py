"""
This module contains the VectorStore class, which is used to store vectors.
"""

from abc import ABC, abstractmethod
from typing import Any, List, Optional

from .chunks import EmbeddedChunk, RetrievedChunk


class ColbertVectorStore(ABC):
    """Interface for a colbert vector store."""

    @abstractmethod
    def put_chunks(
        self, chunks: List[EmbeddedChunk], delete_existing: Optional[bool] = False
    ) -> None:
        """Put a document into the store."""

    @abstractmethod
    def delete_documents(self, ids: List[str]) -> None:
        """Delete a document from the store."""


class ColbertVectorStoreRetriever(ABC):
    """Interface for a colbert vector store retriever."""

    @abstractmethod
    def close(self) -> None:
        """Close the store."""

    @abstractmethod
    def retrieve(
        self, query: str, k: Optional[int], query_maxlen: Optional[int], **kwargs: Any
    ) -> List[RetrievedChunk]:
        """Retrieve chunks from the store"""
