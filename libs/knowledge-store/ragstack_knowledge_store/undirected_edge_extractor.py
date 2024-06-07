from typing import Any, Dict, Iterable, List, Set

from ragstack_knowledge_store._utils import strict_zip
from ragstack_knowledge_store.cassandra import CONTENT_ID, CassandraKnowledgeStore
from ragstack_knowledge_store.edge_extractor import EdgeExtractor


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
        self,
        store: CassandraKnowledgeStore,
        texts: Iterable[str],
        text_embeddings: Iterable[List[float]],
        metadatas: Iterable[Dict[str, Any]],
    ) -> int:
        # First, iterate over the new nodes, collecting the keywords that are referenced
        # and which IDs contain those.
        keywords_to_new_id_embeddings = {}
        for md, embedding in strict_zip(metadatas, text_embeddings):
            keywords = self._keywords(md)
            if not keywords:
                continue

            id = md[CONTENT_ID]
            for kw in keywords:
                keywords_to_new_id_embeddings.setdefault(kw, dict())[id] = embedding

        # Then, retrieve the set of persisted items for each of those keywords
        # and link them to the new items as needed.
        added_edges = 0
        with store._concurrent_queries() as cq:

            def handle_keywords(rows, new_id_embedings: Dict[str, List[float]]):
                nonlocal added_edges
                for row in rows:
                    old_id = row.content_id
                    if old_id not in new_id_embedings:
                        for new_id, new_embedding in new_id_embeddings.items():
                            # We link each found ID to all of the new IDs in both directions.
                            cq.execute(
                                store._insert_edge,
                                (old_id, new_id, self._kind, new_embedding),
                            )
                            cq.execute(
                                store._insert_edge,
                                (new_id, old_id, self._kind, row.text_embedding),
                            )
                            added_edges += 2

            for kw, new_id_embeddings in keywords_to_new_id_embeddings.items():
                # Add edges for new ids.
                for source_id in new_id_embeddings.keys():
                    for target_id, target_embedding in new_id_embeddings.items():
                        if source_id == target_id:
                            # Don't create cyclic edges
                            continue

                        cq.execute(
                            store._insert_edge,
                            (source_id, target_id, self._kind, target_embedding),
                        )
                        added_edges += 1

                # Find "old" IDs (already persisted, not in new set)
                cq.execute(
                    store._query_ids_and_embedding_by_tag,
                    (f"{self._kind}:{kw}",),
                    callback=lambda rows, new_id_embeddings=new_id_embeddings: handle_keywords(
                        rows, new_id_embeddings
                    ),
                )

        return added_edges
