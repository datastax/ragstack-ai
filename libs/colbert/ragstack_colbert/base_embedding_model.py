"""Base embedding for ColBERT.

This module defines an abstract base class (ABC) for generating token-based
embeddings for text.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .objects import Embedding


class BaseEmbeddingModel(ABC):
    """Abstract base class (ABC) for token-based embedding models.

    This class defines the interface for models that generate embeddings for text
    chunks and queries.
    It's designed to be subclassed by specific token embedding implementations,
    such as ColBERT token embeddings. Subclasses must implement the `embed_chunks`
    and `embed_query` abstract methods.
    """

    @abstractmethod
    def embed_texts(self, texts: list[str]) -> list[Embedding]:
        """Embeds a list of texts into their vector embedding representations.

        Args:
            texts (List[str]): A list of string texts.

        Returns:
            List[Embedding]: A list of embeddings, in the order of the input list
        """

    @abstractmethod
    def embed_query(
        self,
        query: str,
        full_length_search: bool = False,
        query_maxlen: int | None = None,
    ) -> Embedding:
        """Embeds a single query text into its vector representation.

        If the query has fewer than query_maxlen tokens it will be padded with BERT
        special [mast] tokens.

        Args:
            query: The query text to encode.
            full_length_search: Indicates whether to encode the
                query for a full-length search. Defaults to False.
            query_maxlen: The fixed length for the query token embedding.
                If None, uses a dynamically calculated value.

        Returns:
            A vector embedding representation of the query text
        """
