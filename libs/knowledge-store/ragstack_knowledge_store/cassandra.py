import secrets
from typing import (
    Any,
    Iterable,
    List,
    NamedTuple,
    Optional,
    Sequence,
    Type,
)

from cassandra.cluster import ResponseFuture, Session
from cassio.config import check_resolve_keyspace, check_resolve_session
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings

from ragstack_knowledge_store.edge_extractor import EdgeExtractor

from .concurrency import ConcurrentQueries
from .content import Kind
from .base import KnowledgeStore, Node, TextNode

CONTENT_ID = "content_id"


def _row_to_document(row) -> Document:
    return Document(
        page_content=row.text_content,
        metadata={
            CONTENT_ID: row.content_id,
            "kind": row.kind,
        },
    )


def _results_to_documents(results: Optional[ResponseFuture]) -> Iterable[Document]:
    if results:
        for row in results:
            yield _row_to_document(row)


def _results_to_ids(results: Optional[ResponseFuture]) -> Iterable[str]:
    if results:
        for row in results:
            yield row.content_id


class CassandraKnowledgeStore(KnowledgeStore):
    def __init__(
        self,
        embedding: Embeddings,
        edge_extractors: List[EdgeExtractor],
        *,
        node_table: str = "knowledge_nodes",
        edge_table: str = "knowledge_edges",
        session: Optional[Session] = None,
        keyspace: Optional[str] = None,
        apply_schema: bool = True,
        concurrency: int = 20,
    ):
        """
        Create the hybrid knowledge store.

        Parameters configure the ways that edges should be added between
        documents. Many take `Union[bool, Set[str]]`, with `False` disabling
        inference, `True` enabling it globally between all documents, and a set
        of metadata fields defining a scope in which to enable it. Specifically,
        passing a set of metadata fields such as `source` only links documents
        with the same `source` metadata value.

        Args:
            embedding: The embeddings to use for the document content.
            edge_extractors: Edge extractors to use for linking knowledge chunks.
            concurrency: Maximum number of queries to have concurrently executing.
            apply_schema: If true, the schema will be created if necessary. If false,
                the schema must have already been applied.
        """
        session = check_resolve_session(session)
        keyspace = check_resolve_keyspace(keyspace)

        self._concurrency = concurrency
        self._embedding = embedding
        self._node_table = node_table
        self._edge_table = edge_table
        self._session = session
        self._keyspace = keyspace

        if apply_schema:
            self._apply_schema()

        assert len(edge_extractors) == len(set([e.kind for e in edge_extractors]))
        self._edge_extractors = edge_extractors

        # TODO: Metadata
        # TODO: Parent ID / source ID / etc.
        self._insert_passage = session.prepare(
            f"""
            INSERT INTO {keyspace}.{node_table} (
                content_id, kind, text_content, text_embedding, tags
            ) VALUES (?, '{Kind.passage}', ?, ?, ?)
            """
        )

        self._insert_edge = session.prepare(
            f"""
            INSERT INTO {keyspace}.{edge_table} (
                source_content_id, target_content_id
            ) VALUES (?, ?)
            """
        )

        self._query_by_id = session.prepare(
            f"""
            SELECT content_id, kind, text_content
            FROM {keyspace}.{node_table}
            WHERE content_id = ?
            """
        )

        self._query_by_embedding = session.prepare(
            f"""
            SELECT content_id, kind, text_content
            FROM {keyspace}.{node_table}
            ORDER BY text_embedding ANN OF ?
            LIMIT ?
            """
        )

        self._query_ids_by_embedding = session.prepare(
            f"""
            SELECT content_id
            FROM {keyspace}.{node_table}
            ORDER BY text_embedding ANN OF ?
            LIMIT ?
            """
        )

        self._query_linked_ids = session.prepare(
            f"""
            SELECT target_content_id AS content_id
            FROM {keyspace}.{edge_table}
            WHERE source_content_id = ?
            """
        )

        self._query_ids_by_tag = session.prepare(
            f"""
            SELECT content_id
            FROM {keyspace}.{node_table}
            WHERE tags CONTAINS ?
            """
        )

    def _apply_schema(self):
        """Apply the schema to the database."""
        embedding_dim = len(self._embedding.embed_query("Test Query"))
        self._session.execute(
            f"""CREATE TABLE IF NOT EXISTS {self._keyspace}.{self._node_table} (
                content_id TEXT,
                kind TEXT,
                text_content TEXT,
                text_embedding VECTOR<FLOAT, {embedding_dim}>,

                tags SET<TEXT>,

                PRIMARY KEY (content_id)
            )
            """
        )

        self._session.execute(
            f"""CREATE TABLE IF NOT EXISTS {self._keyspace}.{self._edge_table} (
                source_content_id TEXT,
                target_content_id TEXT,
                -- Kind of edge.
                kind TEXT,

                PRIMARY KEY (source_content_id, target_content_id)
            )
            """
        )

        # Index on text_embedding (for similarity search)
        self._session.execute(
            f"""CREATE CUSTOM INDEX IF NOT EXISTS {self._node_table}_text_embedding_index
            ON {self._keyspace}.{self._node_table}(text_embedding)
            USING 'StorageAttachedIndex';
            """
        )

        # Index on tags
        # TODO: Case insensitivity?
        self._session.execute(
            f"""
            CREATE CUSTOM INDEX IF NOT EXISTS {self._node_table}_tags_index
            ON {self._keyspace}.{self._node_table} (tags)
            USING 'StorageAttachedIndex';
            """
        )

    @property
    def embeddings(self) -> Optional[Embeddings]:
        return self._embedding

    def _concurrent_queries(self) -> ConcurrentQueries:
        return ConcurrentQueries(self._session, concurrency=self._concurrency)

    def add_nodes(
        self,
        nodes: Iterable[Node] = None,
        **kwargs: Any,
    ):
        texts = []
        metadatas = []
        for node in nodes:
            if not isinstance(node, TextNode):
                raise ValueError("Only adding TextNode is supported at the moment")
            texts.append(node.text)
            metadatas.append(node.metadata)

        text_embeddings = self._embedding.embed_documents(texts)

        ids = []
        with self._concurrent_queries() as cq:
            tuples = zip(texts, text_embeddings, metadatas, strict=True)
            for text, text_embedding, metadata in tuples:
                if CONTENT_ID not in metadata:
                    metadata[CONTENT_ID] = secrets.token_hex(8)
                id = metadata[CONTENT_ID]
                ids.append(id)

                tags = set()
                tags.update(*[e.tags(text, metadata) for e in self._edge_extractors])

                cq.execute(self._insert_passage, (id, text, text_embedding, tags))

        for extractor in self._edge_extractors:
            extractor.extract_edges(self, texts, metadatas)

        return ids

    @classmethod
    def from_texts(
        cls: Type["CassandraKnowledgeStore"],
        texts: Iterable[str],
        embedding: Embeddings,
        metadatas: Optional[List[dict]] = None,
        ids: Optional[Iterable[str]] = None,
        **kwargs: Any,
    ) -> "CassandraKnowledgeStore":
        """Return CassandraKnowledgeStore initialized from texts and embeddings."""
        store = cls(embedding, **kwargs)
        store.add_texts(texts, metadatas, ids=ids)
        return store

    @classmethod
    def from_documents(
        cls: Type["CassandraKnowledgeStore"],
        documents: Iterable[Document],
        embedding: Embeddings,
        ids: Optional[Iterable[str]] = None,
        **kwargs: Any,
    ) -> "CassandraKnowledgeStore":
        """Return CassandraKnowledgeStore initialized from documents and embeddings."""
        store = cls(embedding, **kwargs)
        store.add_documents(documents, ids=ids)
        return store

    def similarity_search(
        self, query: str, k: int = 4, **kwargs: Any
    ) -> List[Document]:
        embedding_vector = self._embedding.embed_query(query)
        return self.similarity_search_by_vector(
            embedding_vector,
            k=k,
        )

    def similarity_search_by_vector(
        self, embedding: List[float], k: int = 4, **kwargs: Any
    ) -> List[Document]:
        results = self._session.execute(self._query_by_embedding, (embedding, k))
        return list(_results_to_documents(results))

    def _query_by_ids(
        self,
        ids: Iterable[str],
    ) -> Iterable[Document]:
        results = []
        with self._concurrent_queries() as cq:
            for id in ids:

                def add_documents(rows):
                    results.extend(_results_to_documents(rows))

                cq.execute(
                    self._query_by_id,
                    parameters=(id,),
                    callback=lambda rows: add_documents(rows),
                )
        return results

    def _linked_ids(
        self,
        source_id: str,
    ) -> Iterable[str]:
        results = self._session.execute(self._query_linked_ids, (source_id,))
        return _results_to_ids(results)

    def traversing_retrieve(
        self,
        query: str,
        *,
        k: int = 4,
        depth: int = 1,
        **kwargs: Any,
    ) -> Iterable[Document]:
        with self._concurrent_queries() as cq:
            visited = {}

            def visit(d: int, nodes: Sequence[NamedTuple]):
                nonlocal visited
                for node in nodes:
                    content_id = node.content_id
                    if d <= visited.get(content_id, depth):
                        visited[content_id] = d
                        # We discovered this for the first time, or at a shorter depth.
                        if d + 1 <= depth:
                            cq.execute(
                                self._query_linked_ids,
                                parameters=(content_id,),
                                callback=lambda n, _d=d: visit(_d + 1, n),
                            )

            query_embedding = self._embedding.embed_query(query)
            cq.execute(
                self._query_ids_by_embedding,
                parameters=(query_embedding, k),
                callback=lambda nodes: visit(0, nodes),
            )

        return self._query_by_ids(visited.keys())
