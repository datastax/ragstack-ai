from typing import Any, Dict, Iterable, List, Optional, Set

from cassandra.cluster import ResponseFuture
from cachetools import LRUCache
from ragstack_knowledge_store.edge_extractor import EdgeExtractor
from ragstack_knowledge_store.cassandra import CONTENT_ID, CassandraKnowledgeStore


class ExplicitEdgeExtractor(EdgeExtractor):
    def __init__(self, edges_field: str, kind: str, bidir: bool = False) -> None:
        """Extract edges from explicit IDs in the metadata.

        This extraction is faster than using a `DirectedEdgeExtractor` when the IDs
        are available since it doesn't need to look-up the nodes associated with a
        given tag.

        NOTE: If no edges are specified, no edges will be created. If edge(s)
        are specified and the target node doesn't exist, an error will be
        raised. This means the target must be added in the same or an earlier
        batch of nodes.

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

    def _get_edges(self, md: Dict[str, Any]) -> Set[str]:
        if (edges := md.get(self._edges_field, None)) is not None:
            if isinstance(edges, Set):
                return edges
            elif isinstance(edges, Iterable) and not isinstance(edges, str):
                return set(edges)
            else:
                return set([edges])
        else:
            return set()

    def extract_edges(
        self,
        store: CassandraKnowledgeStore,
        texts: Iterable[str],
        text_embeddings: Iterable[List[float]],
        metadatas: Iterable[Dict[str, Any]],
    ) -> int:
        num_edges = 0

        ids_to_embeddings = { md[CONTENT_ID]: embedding for (md, embedding) in zip(metadatas, text_embeddings, strict=True) }

        needed_embeddings = { target_id for md in metadatas for target_id in self._get_edges(md) }
        needed_embeddings.difference_update(ids_to_embeddings.keys())

        # TODO: Test that embeddings are retrieved. Also verify the cases:
        # - If the node is in the batch, it doesn't query embeddings
        # - If the node is in an earlier batch, it queries the embeddings
        # - If the node doesn't exist an error is raised.

        with store._concurrent_queries() as cq:
            def add_embeddings(rows):
                    for row in rows:
                        ids_to_embeddings[row.content_id] = row.text_embedding

            for needed_id in needed_embeddings:
                cq.execute(store._query_embedding_by_id, (needed_id, ), callback=add_embeddings)

        # Drop the concurrent queries so we know all the embeddings are available.
        # We could try to do continue concurrently executing, but would need to deal with
        # more synchronization conditions. Better to have a true `async` implementation,
        # and use that where possible.

        with store._concurrent_queries() as cq:
            for (md, text_embedding) in zip(metadatas, text_embeddings, strict=True):
                if edges := self._get_edges(md):
                    id = md[CONTENT_ID]
                    for target_id in set(edges):
                        target_embedding = ids_to_embeddings.get(target_id, None)
                        if target_embedding is None:
                            # Rather than ignoring this case (edges to a node that won't be linked up)
                            # raise an error. We *could* do a search to find references to nodes that
                            # are being added, but that would require more book-keeping and queries,
                            # so wait for this to be a common issue.
                            raise ValueError(f"Node '{id}' has edge to non-existent node '{target_id}' in '{self._edges_field}'")

                        cq.execute(store._insert_edge, (id, target_id, self.kind, target_embedding))
                        num_edges += 1
                        if self._bidir:
                            cq.execute(store._insert_edge, (target_id, id, self.kind, text_embedding))
                            num_edges += 1
        return num_edges
