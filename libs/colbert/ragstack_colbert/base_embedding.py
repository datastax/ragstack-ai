"""
This module defines an abstract base class (ABC) for generating token-based embeddings for text.
"""

from abc import ABC, abstractmethod
from typing import List, Optional

from torch import Tensor

from .objects import ChunkData, EmbeddedChunk


class BaseEmbedding(ABC):
    """
    Abstract base class (ABC) for token-based embedding models.

    This class defines the interface for models that generate embeddings for text chunks and queries.
    It's designed to be subclassed by specific token embedding implementations, such as ColBERT token
    embeddings. Subclasses must implement the `embed_chunks` and `embed_query` abstract methods.
    """

    @abstractmethod
    def embed_chunks(
        self, chunks: List[ChunkData], doc_id: Optional[str] = None
    ) -> List[EmbeddedChunk]:
        """
        Embeds a list of text chunks into their corresponding vector representations.

        This method takes multiple chunks of text and optionally their associated document identifier,
        returning a list of `EmbeddedChunk` instances containing the embeddings.

        Parameters:
            chunks (List[ChunkData]): A list of chunks including document text and any associated metadata.
            doc_id (Optional[str], optional): An optional document identifier that all chunks belong to.
                                               This can be used for tracing back embeddings to their
                                               source document. If not passed, an uuid will be generated.

        Returns:
            List[EmbeddedChunk]: A list of `EmbeddedChunks` instances with embeddings populated,
                                  corresponding to the input text chunks, ready for insertion into
                                  a vector store.
        """

    @abstractmethod
    def embed_query(self, text: str) -> Tensor:
        """
        Embeds a single query text into its vector representation.

        This method processes a query string, converting it into a tensor of embeddings that
        represent the query in the embedded space. This is typically used for matching against
        embedded documents or chunks in retrieval tasks.

        Parameters:
            text (str): The query text to be embedded.

        Returns:
            Tensor: A tensor representing the embedded query.
        """
