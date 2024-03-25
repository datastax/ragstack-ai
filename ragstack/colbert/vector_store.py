"""
This module contains the VectorStore class, which is used to store vectors.
"""

import dataclasses
from abc import ABC, abstractmethod
from numbers import Number
from typing import List, Optional


class ColBERTVectorStore(ABC):
    """Interface for a vector store."""

    @abstractmethod
    def close(self):
        """Close the store."""
        pass

    @abstractmethod
    def put_document(self, document: str, metadata: dict):
        """Put a document into the store."""
        pass

    @abstractmethod
    def delete_documents(self, titles: List[str]):
        """Delete a document from the store."""
        pass


@dataclasses.dataclass
class Document:
    title: str
    body: str
    rank: int
    score: Number


class ColBERTVectorStoreRetriever(ABC):
    @abstractmethod
    def close(self):
        """Close the store."""
        pass

    @abstractmethod
    def retrieve(
        self, query: str, k: Optional[int], query_maxlen: Optional[int], **kwargs
    ) -> List[Document]:
        """Retrieve documents from the store"""
        pass
