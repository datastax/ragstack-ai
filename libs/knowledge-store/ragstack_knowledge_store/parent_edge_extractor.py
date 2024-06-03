from typing import Any, Dict, Iterable

from ragstack_knowledge_store.edge_extractor import EdgeExtractor
from ragstack_knowledge_store.knowledge_store import CONTENT_ID, KnowledgeStore


class ParentEdgeExtractor(EdgeExtractor):
    """Extract an edge from a node to the `parent_content_id`."""

    def __init__(self, parent_field: str = "parent_content_id") -> None:
        """Create a new ParentEdgeExtractor.

        Params:
        - parent_field: The metadata field containing the parent content ID.
        """

        # TODO: Allow specifying how properties should be added to the edge.
        # For instance, `links_to`.
        self._parent_field = parent_field

    @property
    def kind(self) -> str:
        return "has_parent"

    def extract_edges(
        self, store: KnowledgeStore, texts: Iterable[str], metadatas: Iterable[Dict[str, Any]]
    ) -> int:
        num_edges = 0
        with store._concurrent_queries() as cq:
            for md in metadatas:
                if (parent_content_id := md.get(self._parent_field)) is not None:
                    id = md[CONTENT_ID]
                    cq.execute(store._insert_edge, (id, str(parent_content_id)))
                    num_edges += 1
        return num_edges
