"""
This is the base class for token based embedding
ColBERT token embeddings is an example of a class that inherits from this class
"""

from abc import ABC, abstractmethod
from typing import List, Optional

from torch import Tensor

from .chunks import EmbeddedChunk


class TokenEmbeddings(ABC):
    """Interface for token embedding models.
    This is the base class for token based embedding
    ColBERT token embeddings is an example of a class that inherits from this class.
    """

    @abstractmethod
    def embed_chunks(
        self, texts: List[str], doc_id: Optional[str] = None
    ) -> List[EmbeddedChunk]:
        """Embed search chunks."""

    @abstractmethod
    def embed_query(self, text: str) -> Tensor:
        """Embed query text."""
