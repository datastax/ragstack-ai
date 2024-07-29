from __future__ import annotations

from abc import ABC, abstractmethod


class EmbeddingModel(ABC):
    """Embedding model."""

    @abstractmethod
    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        """Embed texts."""

    @abstractmethod
    def embed_query(self, text: str) -> list[float]:
        """Embed query text."""

    @abstractmethod
    async def aembed_texts(self, texts: list[str]) -> list[list[float]]:
        """Embed texts."""

    @abstractmethod
    async def aembed_query(self, text: str) -> list[float]:
        """Embed query text."""
