from abc import ABC
from typing import Any, Dict, Iterable

from knowledge_store import KnowledgeStore

class EdgeExtractor(ABC):
    def extract_edges(self, store: KnowledgeStore, texts: Iterable[str], metadatas: Iterable[Dict[str, Any]]) -> int:
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

    def aextract_edges(self, store: KnowledgeStore, texts: Iterable[str], metadatas: Iterable[Dict[str, Any]]) -> int:
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
