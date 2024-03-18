"""
This module contains the VectorStore class, which is used to store vectors.
"""

from abc import ABC, abstractmethod

class ColBERTVectorStore(ABC):
    """Interface for a vector store."""
    @abstractmethod
    def create_store(self):
        """Create the store."""
        pass

    @abstractmethod
    def health_check(self):
        """Check the health of the store."""
        pass

    @abstractmethod
    def close(self):
        """Close the store."""
        pass

    @abstractmethod
    def put_document(self, document: str, metadata: dict):
        """Put a document into the store."""
        pass

    @abstractmethod
    def delete_documents(self, title: str):
        """Delete a document from the store."""
        pass


