from abc import ABC, abstractmethod
from typing import List


class EmbeddingModel(ABC):
    """Embedding model."""

    @abstractmethod
    def embed_texts(self, texts: List[str]) -> List[List[float]]:
        """Embed texts."""

    @abstractmethod
    def embed_query(self, text: str) -> List[float]:
        """Embed query text."""

    @abstractmethod
    async def aembed_texts(self, texts: List[str]) -> List[List[float]]:
        """Embed texts."""

    @abstractmethod
    async def aembed_query(self, text: str) -> List[float]:
        """Embed query text."""
