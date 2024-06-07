from __future__ import annotations

from typing import Any, Dict, Iterable, List, Set

from ragstack_knowledge_store._utils import strict_zip
from ragstack_knowledge_store.cassandra import CONTENT_ID, CassandraKnowledgeStore
from ragstack_knowledge_store.edge_extractor import EdgeExtractor


def _rows_to_sources(rows) -> Iterable[str]:
    return [row.content_id for row in rows]


def _rows_to_targets(rows) -> Dict[str, List[float]]:
    return {row.content_id: row.text_embedding for row in rows}


class DirectedEdgeExtractor(EdgeExtractor):
    def __init__(self, sources_field: str, targets_field: str, kind: str) -> None:
        """Extract directed edges between uses and definitions.
        While `UndirectedEdgeExtractor` links nodes in both directions if they share
        a keyword, this only creates links from nodes with a "source" to nodes with
        a matching "target". For example, uses may be the `href` of `a` tags in the
        chunk and definitions may be the URLs that the chunk is accessible at.

        This may also be used for other forms of references, such as Wikipedia
        article IDs, etc.

        Args:
            sources_field: The metadata field to read sources from.
            targets_field: The metadata field to read targets from.
            kind: The kind label to apply to created edges. Must be unique.
        """

        # TODO: Assert the kind matches some reasonable regex?

        # TODO: Allow specifying how properties should be added to the edge.
        # For instance, `links_to`.
        self._sources_field = sources_field
        self._targets_field = targets_field
        self._kind = kind

    @property
    def kind(self) -> str:
        return self._kind

    @staticmethod
    def for_hrefs_to_urls() -> DirectedEdgeExtractor:
        return DirectedEdgeExtractor(sources_field="urls", targets_field="hrefs", kind="link")

    def _sources(self, metadata: Dict[str, Any]) -> Set[str]:
        sources = metadata.get(self._sources_field)
        if not sources:
            return set()
        elif isinstance(sources, str):
            return set({sources})
        else:
            return set(sources)

    def _targets(self, metadata: Dict[str, Any]) -> Set[str]:
        targets = metadata.get(self._targets_field)
        if not targets:
            return set()
        elif isinstance(targets, str):
            return set({targets})
        else:
            return set(targets)

    def tags(self, text: str, metadata: Dict[str, Any]) -> Set[str]:
        sources = {f"{self._kind}_s:{source}" for source in self._sources(metadata)}
        targets = {f"{self._kind}_t:{target}" for target in self._targets(metadata)}
        return sources.union(targets)

    def extract_edges(
        self,
        store: CassandraKnowledgeStore,
        texts: Iterable[str],
        text_embeddings: Iterable[str],
        metadatas: Iterable[Dict[str, Any]],
    ) -> int:
        # First, iterate over the new nodes, collecting the sources/targets that
        # are referenced and which IDs contain those.
        new_ids = set()
        resource_to_new_defs_embs = {}
        resource_to_new_refs = {}
        for md, embedding in strict_zip(metadatas, text_embeddings):
            id = md[CONTENT_ID]

            new_ids.add(id)
            for resource in self._sources(md):
                resource_to_new_defs_embs.setdefault(resource, dict())[id] = embedding
            for target_id in self._targets(md):
                resource_to_new_refs.setdefault(target_id, set()).add(id)

        # Then, retrieve the set of persisted items for each of those
        # source/targets and link them to the new items as needed.
        # Remembering that the the *new* nodes will have been added.
        source_target_pairs = dict()
        with store._concurrent_queries() as cq:

            def add_source_target_pairs(
                source_ids: Iterable[str], target_id_embeddings: Dict[str, List[float]]
            ):
                for source_id in source_ids:
                    for target_id, target_embedding in target_id_embeddings.items():
                        source_target_pairs[(source_id, target_id)] = target_embedding

            for resource, new_defs_embs in resource_to_new_defs_embs.items():
                cq.execute(
                    store._query_ids_by_tag,
                    parameters=(f"{self._kind}_t:{resource}",),
                    # Weird syntax to capture each `source_ids` instead of the last iteration.
                    callback=lambda sources, targets=new_defs_embs: add_source_target_pairs(
                        _rows_to_sources(sources), targets
                    ),
                )

            for resource, new_refs in resource_to_new_refs.items():
                cq.execute(
                    store._query_ids_and_embedding_by_tag,
                    parameters=(f"{self._kind}_s:{resource}",),
                    # Weird syntax to capture each `target_ids` instead of the last iteration.
                    callback=lambda targets, sources=new_refs: add_source_target_pairs(
                        sources, _rows_to_targets(targets)
                    ),
                )

        # TODO: we should allow passing in the concurent queries, and figure out
        # how to start sending these before everyting previously finished.
        with store._concurrent_queries() as cq:
            for (source_id, target_id), target_embedding in source_target_pairs.items():
                cq.execute(
                    store._insert_edge, (source_id, target_id, self._kind, target_embedding)
                )

        return len(source_target_pairs)
