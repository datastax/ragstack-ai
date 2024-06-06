from __future__ import annotations

from typing import Any, Dict, Iterable, Set

from ragstack_knowledge_store.cassandra import CONTENT_ID, CassandraKnowledgeStore
from ragstack_knowledge_store.edge_extractor import EdgeExtractor


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
        return DirectedEdgeExtractor(
            sources_field="hrefs", targets_field="urls", kind="link"
        )

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
        results = set()
        for source in self._sources(metadata):
            results.add(f"{self._kind}_s:{source}")
        for target in self._targets(metadata):
            results.add(f"{self._kind}_t:{target}")
        return results

    def extract_edges(
        self,
        store: CassandraKnowledgeStore,
        texts: Iterable[str],
        metadatas: Iterable[Dict[str, Any]],
    ) -> int:
        # First, iterate over the new nodes, collecting the sources/targets that
        # are referenced and which IDs contain those.
        new_ids = set()
        new_sources_to_ids = {}
        new_targets_to_ids = {}
        for md in metadatas:
            id = md[CONTENT_ID]

            new_ids.add(id)
            for resource in self._sources(md):
                new_sources_to_ids.setdefault(resource, set()).add(id)
            for target in self._targets(md):
                new_targets_to_ids.setdefault(target, set()).add(id)

        # Then, retrieve the set of persisted items for each of those
        # source/targets and link them to the new items as needed.
        # Remembering that the the *new* nodes will have been added.
        source_target_pairs = set()
        with store._concurrent_queries() as cq:

            def add_source_target_pairs(href_ids, url_ids):
                for href_id in href_ids:
                    if not isinstance(href_id, str):
                        href_id = href_id.content_id

                    for url_id in url_ids:
                        if not isinstance(url_id, str):
                            url_id = url_id.content_id
                    source_target_pairs.add((href_id, url_id))

            for resource, source_ids in new_sources_to_ids.items():
                cq.execute(
                    store._query_ids_by_tag,
                    parameters=(f"{self._kind}_t:{resource}",),
                    # Weird syntax to capture each `source_ids` instead of the last iteration.
                    callback=lambda targets, sources=source_ids: add_source_target_pairs(
                        sources, targets
                    ),
                )

            for resource, target_ids in new_targets_to_ids.items():
                cq.execute(
                    store._query_ids_by_tag,
                    parameters=(f"{self._kind}_s:{resource}",),
                    # Weird syntax to capture each `target_ids` instead of the last iteration.
                    callback=lambda sources, targets=target_ids: add_source_target_pairs(
                        sources, targets
                    ),
                )

        # TODO: we should allow passing in the concurent queries, and figure out
        # how to start sending these before everyting previously finished.
        with store._concurrent_queries() as cq:
            for source, target in source_target_pairs:
                cq.execute(store._insert_edge, (source, target))

        return len(source_target_pairs)
