"""
This module contains the VectorStore class, which is used to store vectors.
"""

import dataclasses
from abc import ABC, abstractmethod
from numbers import Number
from typing import List, Optional, Any
from .token_embedding import EmbeddedChunk


class ColbertVectorStore(ABC):
    """Interface for a vector store."""

    @abstractmethod
    def put_chunks(
        self, chunks: List[EmbeddedChunk], delete_existing: bool = False
    ) -> None:
        """Put a document into the store."""

    @abstractmethod
    def delete_documents(self, ids: List[str]) -> None:
        """Delete a document from the store."""


@dataclasses.dataclass
class RetrievedChunk:
    doc_id: str
    chunk_id: int
    text: str
    rank: int
    score: Number


class ColbertVectorStoreRetriever(ABC):
    @abstractmethod
    def close(self) -> None:
        """Close the store."""

    @abstractmethod
    def retrieve(
        self, query: str, k: Optional[int], query_maxlen: Optional[int], **kwargs
    ) -> List[RetrievedChunk]:
        """Retrieve chunks from the store"""
