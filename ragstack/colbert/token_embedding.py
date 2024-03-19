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
        id: int,  # id is the embedding id
        part: int,  # part is the part id of the passage
        title: str = "",  # title is the title of the passage
    ):
        self.id = id
        self.__embeddings = []
        self.title = title
        self.part = part

    def add_embeddings(self, embeddings: List[float]):
        self.__embeddings = embeddings

    def get_embeddings(self) -> List[float]:
        return self.__embeddings

    def id(self) -> id:
        return self.id

    def title(self):
        return self.title

    def part(self):
        return self.part


class PassageEmbeddings:
    __token_embeddings: List[PerTokenEmbeddings]
    __text: str
    __title: str
    __id: str
    __model: str
    __dim: int

    def __init__(
        self,
        text: str,
        title: str = "",  # keep this as backward compatibility, use id instead
        part: int = -1,
        id: str = "",
        model: str = DEFAULT_COLBERT_MODEL,
        dim: int = DEFAULT_COLBERT_DIM,
    ):
        self.__text = text
        self.__token_embeddings = []
        if title == "" and id == "":
           self.__title = str(uuid.uuid4())
           self.__id = self.__title
        elif id != "":
           self.__title = id
           self.__id = id
        elif title != "":
            self.__id = title
            self.__title = title
        self.__model = model
        self.__dim = dim
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
