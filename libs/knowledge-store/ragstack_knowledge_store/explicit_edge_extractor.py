from typing import Any, Dict, Iterable

from ragstack_knowledge_store.edge_extractor import EdgeExtractor
from ragstack_knowledge_store.knowledge_store import CONTENT_ID, KnowledgeStore


class ExplicitEdgeExtractor(EdgeExtractor):
    def __init__(self,
                 edges_field: str,
                 kind: str,
                 bidir: bool = False) -> None:
        """Extract edges from explicit IDs in the metadata.

        This extraction is faster than using a `DirectedEdgeExtractor` when the IDs
        are available since it doesn't need to look-up the nodes associated with a
        given tag.

        Note: This extractor does not check whether the target ID exists. Edges
        will be created even if the target does not exist. This means traversals
        over graphs using this extractor may discover nodes that do not exist.
        Such "phantom IDs" will be filtered out when loading content for the
        nodes.

        Args:
            edges_field: The metadata field containing the IDs of nodes to link to.
            kind: The `kind` to apply to edges created by this extractor.
            bidir: If true, creates edges in both directions.
        """

        self._edges_field = edges_field
        self._kind = kind
        self._bidir = bidir

    @property
    def kind(self) -> str:
        return self._kind

    def extract_edges(
        self, store: KnowledgeStore, texts: Iterable[str], metadatas: Iterable[Dict[str, Any]]
    ) -> int:
        num_edges = 0
        with store._concurrent_queries() as cq:
            for md in metadatas:
                if (edges := md.get(self._edges_field, None)) is not None:
                    id = md[CONTENT_ID]
                    for target in set(edges):
                        cq.execute(store._insert_edge, (id, target))
                        num_edges += 1
                        if self._bidir:
                            cq.execute(store._insert_edge, (target, id))
                            num_edges += 1
        return num_edges
