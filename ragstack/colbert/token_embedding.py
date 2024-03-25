#
# this is a base class for ColBERT per token based embedding

from abc import ABC, abstractmethod
from typing import List, Optional
from .constant import DEFAULT_COLBERT_DIM, DEFAULT_COLBERT_MODEL
import uuid


class PerTokenEmbeddings:
    __embeddings: List[float]
    __embedding_id: int
    __part_id: int

    def __init__(self, embedding_id: int, part_id: int):
        self.__embeddings = []
        self.__embedding_id = embedding_id
        self.__part_id = part_id

    def embedding_id(self) -> int:
        return self.__embedding_id

    def part_id(self) -> int:
        return self.__part_id

    def add_embeddings(self, embeddings: List[float]) -> None:
        self.__embeddings = embeddings

    def get_embeddings(self) -> List[float]:
        return self.__embeddings


class PassageEmbeddings:
    __token_embeddings: List[PerTokenEmbeddings]
    __text: str
    __doc_id: str
    __part_id: int

    def __init__(
        self,
        text: str,
        doc_id: str = None,
        part_id: int = 0,
    ):
        self.__text = text
        self.__token_embeddings = []
        self.__doc_id = doc_id
        self.__part_id = part_id

    def doc_id(self) -> str:
        return self.__doc_id

    def part_id(self) -> int:
        return self.__part_id

    def text(self) -> str:
        return self.__text

    def add_token_embeddings(self, token_embeddings: PerTokenEmbeddings) -> None:
        self.__token_embeddings.append(token_embeddings)

    def get_all_token_embeddings(self) -> List[PerTokenEmbeddings]:
        return self.__token_embeddings


class TokenEmbeddings(ABC):
    """Interface for token embedding models.
    This is the base class for token based embedding
    ColBERT token embeddings is an example of a class that inherits from this class.
    """

    @abstractmethod
    def embed_documents(self, texts: List[str]) -> List[PassageEmbeddings]:
        """Embed search docs."""

    @abstractmethod
    def embed_query(self, text: str) -> PassageEmbeddings:
        """Embed query text."""
