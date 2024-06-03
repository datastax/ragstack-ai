from __future__ import annotations

import typing
from abc import ABC, abstractmethod
from typing import Any, Dict, Iterable, List, Set

from langchain_core.runnables import run_in_executor

if typing.TYPE_CHECKING:
    from ragstack_knowledge_store.cassandra import CassandraKnowledgeStore


class EdgeExtractor(ABC):
    """Extension defining how edges should be created."""

    @property
    @abstractmethod
    def kind(self) -> str:
        """Return the kind of edge extracted by this."""

    def tags(self, text: str, metadata: Dict[str, Any]) -> Set[str]:
        """Return the set of tags to add for this extraction."""
        return set()

    @abstractmethod
    def extract_edges(
        self,
        store: CassandraKnowledgeStore,
        texts: Iterable[str],
        text_embeddings: Iterable[List[float]],
        metadatas: Iterable[Dict[str, Any]],
    ) -> int:
        """Add edges for the given nodes.

        The nodes have already been persisted.

        Args:
            store: CassandraKnowledgeStore edges are being extracted for.
            texts: The texts of the nodes to be processed.
            text_embeddings: The embeddings of the text nodes.
            metadatas: The metadatas of the nodes to be processed.

        Returns:
            Number of edges extracted involving the given nodes.
        """

    async def aextract_edges(
        self,
        store: CassandraKnowledgeStore,
        texts: Iterable[str],
        text_embeddings: Iterable[List[float]],
        metadatas: Iterable[Dict[str, Any]],
    ) -> int:
        """Add edges for the given nodes.

        The nodes have already been persisted.

        Args:
            store: CassandraKnowledgeStore edges are being extracted for.
            texts: The texts of the nodes to be processed.
            text_embedings: The embeddings of the text nodes.
            metadatas: The metadatas of the nodes to be processed.

        Returns:
            Number of edges extracted involving the given nodes.
        """
        return await run_in_executor(
            None,
            self._extract_edges,
            store,
            texts,
            text_embeddings,
            metadatas,
        )
