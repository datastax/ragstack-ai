#
# this is a base class for ColBERT per token based embedding

from abc import ABC, abstractmethod
from typing import List
from .constant import DEFAULT_COLBERT_DIM, DEFAULT_COLBERT_MODEL
import uuid


class PerTokenEmbeddings:
    __embeddings: List[float]

    def __init__(
        self,
        id: int,
        part: int,
        parent_id: uuid.UUID = None,
        title: str = "",
    ):
        self.id = id
        self.parent_id = parent_id
        self.__embeddings = []
        self.title = title
        self.part = part

    def add_embeddings(self, embeddings: List[float]):
        self.__embeddings = embeddings

    def get_embeddings(self) -> List[float]:
        return self.__embeddings

    def id(self):
        return self.id

    def parent_id(self):
        return self.parent_id

    def part(self):
        return self.part


class PassageEmbeddings:
    __token_embeddings: List[PerTokenEmbeddings]
    __text: str
    __title: str
    __id: uuid.UUID

    def __init__(
        self,
        text: str,
        title: str = "",
        part: int = 0,
        id: uuid.UUID = None,
        model: str = DEFAULT_COLBERT_MODEL,
        dim: int = DEFAULT_COLBERT_DIM,
    ):
        self.__text = text
        self.__token_embeddings = []
        if id is None:
            self.__id = uuid.uuid4()
        else:
            self.__id = id
        self.__model = model
        self.__dim = dim
        self.__title = title
        self.__part = part

    def model(self):
        return self.__model

    def dim(self):
        return self.__dim

    def token_size(self):
        return len(self.token_ids)

    def title(self):
        return self.__title

    def __len__(self):
        return len(self.embeddings)

    def id(self):
        return self.__id

    def part(self):
        return self.__part

    def add_token_embeddings(self, token_embeddings: PerTokenEmbeddings):
        self.__token_embeddings.append(token_embeddings)

    def get_token_embeddings(self, token_id: int) -> PerTokenEmbeddings:
        for token in self.__token_embeddings:
            if token.token_id == token_id:
                return token
        return None

    def get_all_token_embeddings(self) -> List[PerTokenEmbeddings]:
        return self.__token_embeddings

    def get_text(self):
        return self.__text


"""
This is the base class for token based embedding
ColBERT token embeddings is an example of a class that inherits from this class
"""


class TokenEmbeddings(ABC):
    """Interface for token embedding models."""

    @abstractmethod
    def embed_documents(self, texts: List[str]) -> List[PassageEmbeddings]:
        """Embed search docs."""

    @abstractmethod
    def embed_query(self, text: str) -> PassageEmbeddings:
        """Embed query text."""
