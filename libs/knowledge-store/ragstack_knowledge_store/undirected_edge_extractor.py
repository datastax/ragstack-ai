from typing import Any, Dict, Iterable
from ragstack_knowledge_store.edge_extractor import EdgeExtractor
from ragstack_knowledge_store.knowledge_store import KnowledgeStore


class UndirectedEdgeExtractor(EdgeExtractor):
    """Extract undirected edges between nodes with common keywords."""

    def __init__(self,
                 keyword_field: str = "keywords") -> None:
        """Create a new UndirectedEdgeExtractor.

        Params:
        - keyword_field: The metadata field to read keywords from.
        """
        # TODO: Allow specifying how properties should be added to the edge.
        # For instance, `has_keyword: <keyword>`.
        #
        # TODO: Allow configuring a minimum keyword interesection > 1, to only
        # connect nodes with multiple keywords in common?
        self._keyword_field = keyword_field

    def extract_edges(self,
                      store: KnowledgeStore,
                      texts: Iterable[str],
                      metadatas: Iterable[Dict[str, Any]]) -> int:
        # First, iterate over the new nodes, collecting the keywords that are referenced
        # and which IDs contain those.
        keywords_to_new_ids = {}
        for md in metadatas:
            keywords = set(md.get(self._keyword_field, []))
            if not keywords:
                continue
            id = md["content_id"]
            for kw in keywords:
                keywords_to_new_ids.setdefault(kw, set()).add(id)

        # Then, retrieve the set of persisted items for each of those keywords
        # and link them to the new items as needed.
        added_edges = 0
        with store._concurrent_queries() as cq:
            def handle_keywords(rows, new_ids):
                nonlocal added_edges
                for row in rows:
                    found_id = row.content_id

                    # We link all IDs to all of the new IDs except for the loop-back.
                    for new_id in new_ids:
                        if found_id != new_id:
                            # Create edge from `found_id -> new_id`.
                            # Since the new IDs are already persisted, we'll also find
                            # them and create the back-edge there.
                            cq.execute(
                                store._insert_edge,
                                (found_id, new_id),
                            )
                            added_edges += 1

            for kw, new_ids in keywords_to_new_ids.items():
                cq.execute(
                    # TODO: For full generality, we should either prefix what we write
                    # to this column, or allow per-extractor columns.
                    store._query_ids_by_keyword,
                    (kw, ),
                    callback=lambda rows, new_ids=new_ids: handle_keywords(rows, new_ids)
                )

        return added_edges
