from abc import ABC
from typing import Any, Dict, Iterable, Set

from knowledge_store import KnowledgeStore


class EdgeExtractor(ABC):
    @property
    def kind(self) -> str:
        """Return the kind of edge extracted by this."""
        ...

    def tags(self, text: str, metadata: Dict[str, Any]) -> Set[str]:
        """Return the set of tags to add for this extraction."""
        return set()

    def extract_edges(
        self, store: KnowledgeStore, texts: Iterable[str], metadatas: Iterable[Dict[str, Any]]
    ) -> int:
        """Add edges for the given nodes.

        The nodes have already been persisted.

        Parameters:
        - store: KnowledgeStore edges are being extracted for.
        - texts: The texts of the nodes to be processed.
        - metadatas: The metadatas of the nodes to be processed.

        Returns:
        Number of edges extracted involving the given nodes.
        """
        ...

    def aextract_edges(
        self, store: KnowledgeStore, texts: Iterable[str], metadatas: Iterable[Dict[str, Any]]
    ) -> int:
        """Add edges for the given nodes.

        The nodes have already been persisted.

        Parameters:
        - store: KnowledgeStore edges are being extracted for.
        - texts: The texts of the nodes to be processed.
        - metadatas: The metadatas of the nodes to be processed.

        Returns:
        Number of edges extracted involving the given nodes.
        """
        return self.extract_edges(store, texts, metadatas)
