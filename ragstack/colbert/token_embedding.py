#
# this is a base class for ColBERT per token based embedding

from abc import ABC, abstractmethod
from typing import List
from torch import Tensor


class EmbeddedChunk:
    __vectors: Tensor
    __text: str
    __doc_id: str
    __chunk_id: int

    def __init__(
        self,
        text: str,
        doc_id: str,
        chunk_id: int,
        vectors: Tensor
    ):
        self.__text = text
        self.__doc_id = doc_id
        self.__chunk_id = chunk_id
        self.__vectors = vectors

    def __len__(self):
        return len(self.__vectors)

    def doc_id(self):
        return self.__doc_id

    def chunk_id(self):
        return self.__chunk_id

    def text(self) -> str:
        return self.__text

    def vectors(self) -> Tensor:
        return self.__vectors


"""
This is the base class for token based embedding
ColBERT token embeddings is an example of a class that inherits from this class
"""
class TokenEmbeddings(ABC):
    """Interface for token embedding models.
    This is the base class for token based embedding
    ColBERT token embeddings is an example of a class that inherits from this class.
    """

    @abstractmethod
    def embed_chunks(self, chunk_texts: List[str], doc_id: str = None) -> List[EmbeddedChunk]:
        """Embed search chunks."""

    @abstractmethod
    def embed_query(self, text: str) -> Tensor:
        """Embed query text."""
