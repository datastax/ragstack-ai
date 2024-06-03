from typing import Any, Dict, Iterable, Set

from ragstack_knowledge_store.edge_extractor import EdgeExtractor
from ragstack_knowledge_store.knowledge_store import CONTENT_ID, KnowledgeStore


class UndirectedEdgeExtractor(EdgeExtractor):
    def __init__(
        self,
        keyword_field: str = "keywords",
        kind: str = "kw",
    ) -> None:
        """Extract undirected edges between nodes with common keywords.

        Args:
            keyword_field: The metadata field to read keywords from.
            kind: The kind label to apply to created edges. Must be unique.
        """
        # TODO: Allow specifying how properties should be added to the edge.
        # For instance, `has_keyword: <keyword>`.
        #
        # TODO: Allow configuring a minimum keyword interesection > 1, to only
        # connect nodes with multiple keywords in common?
        self._keyword_field = keyword_field
        self._kind = kind

    @property
    def kind(self) -> str:
        return self._kind

    def _keywords(self, metadata: Dict[str, Any]) -> Set[str]:
        keywords = metadata.get(self._keyword_field)
        if not keywords:
            return set()
        elif isinstance(keywords, str):
            return set({keywords})
        else:
            return set(keywords)

    def tags(self, text: str, metadata: Dict[str, Any]) -> Set[str]:
        return {f"{self._kind}:{kw}" for kw in self._keywords(metadata)}

    def extract_edges(
        self, store: KnowledgeStore, texts: Iterable[str], metadatas: Iterable[Dict[str, Any]]
    ) -> int:
        # First, iterate over the new nodes, collecting the keywords that are referenced
        # and which IDs contain those.
        keywords_to_new_ids = {}
        for md in metadatas:
            keywords = self._keywords(md)
            if not keywords:
                continue

            id = md[CONTENT_ID]
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
                    store._query_ids_by_tag,
                    (f"{self._kind}:{kw}",),
                    callback=lambda rows, new_ids=new_ids: handle_keywords(rows, new_ids),
                )

        return added_edges
