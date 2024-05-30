import secrets
from typing import Any, Coroutine, Dict, Iterable, List, NamedTuple, Optional, Sequence, Union

from cassandra.cluster import ResponseFuture, Session
from cassio.config import check_resolve_keyspace, check_resolve_session
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.runnables import Runnable, RunnableLambda
from langchain_core.vectorstores import VectorStore
from ragstack_knowledge_store.edge_extractor import EdgeExtractor

from .concurrency import ConcurrentQueries
from .content import Kind

CONTENT_ID = "content_id"
PARENT_CONTENT_ID = "parent_content_id"
KEYWORDS = "keywords"
HREFS = "hrefs"
URLS = "urls"


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


class KnowledgeStore(VectorStore):
    """A hybrid vector-and-graph knowledge store.

    Document chunks support vector-similarity search as well as edges linking
    chunks based on structural and semantic properties.
    """

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

        self._edge_extractors = edge_extractors

        # TODO: Metadata
        # TODO: Parent ID / source ID / etc.
        self._insert_passage = session.prepare(
            f"""
            INSERT INTO {keyspace}.{node_table} (
                content_id, kind, text_content, text_embedding, keywords, urls, hrefs
            ) VALUES (?, '{Kind.passage}', ?, ?, ?, ?, ?)
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

        self._query_ids_by_keyword = session.prepare(
            f"""
            SELECT content_id
            FROM {keyspace}.{node_table}
            WHERE keywords CONTAINS ?
            """
        )

        self._query_ids_by_href = session.prepare(
            f"""
            SELECT content_id
            FROM {keyspace}.{node_table}
            WHERE hrefs CONTAINS ?
            """
        )

        self._query_ids_by_url = session.prepare(
            f"""
            SELECT content_id
            FROM {keyspace}.{node_table}
            WHERE urls CONTAINS ?
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

                keywords SET<TEXT>,
                hrefs SET<TEXT>,
                urls SET<TEXT>,

                PRIMARY KEY (content_id)
            )
            """
        )

        self._session.execute(
            f"""CREATE TABLE IF NOT EXISTS {self._keyspace}.{self._edge_table} (
                source_content_id TEXT,
                target_content_id TEXT,
                -- Denormalized target kind for filtering.
                target_kind TEXT,

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

        # Index on keywords
        # TODO: Case insensitivity?
        self._session.execute(
            f"""
            CREATE CUSTOM INDEX IF NOT EXISTS {self._node_table}_keywords_index
            ON {self._keyspace}.{self._node_table} (keywords)
            USING 'StorageAttachedIndex';
            """
        )

        self._session.execute(
            f"""
            CREATE CUSTOM INDEX IF NOT EXISTS {self._node_table}_hrefs_index
            ON {self._keyspace}.{self._node_table} (hrefs)
            USING 'StorageAttachedIndex';
            """
        )

        self._session.execute(
            f"""
            CREATE CUSTOM INDEX IF NOT EXISTS {self._node_table}_urls_index
            ON {self._keyspace}.{self._node_table} (urls)
            USING 'StorageAttachedIndex';
            """
        )

    @property
    def embeddings(self) -> Optional[Embeddings]:
        """Access the query embedding object if available."""
        return self._embedding

    def _concurrent_queries(self) -> ConcurrentQueries:
        return ConcurrentQueries(self._session, concurrency=self._concurrency)

    async def aadd_texts(self, texts: Iterable[str], metadatas: List[Dict] | None = None, **kwargs: Any) -> List[str]:
        # TODO: Batch texts in case it is too long? Or assume it fits in memory.
        text_embeddings_future = self._embedding.aembed_documents(texts)

        texts = list(texts)
        metadatas: Iterable[Dict[str, str]] = (
            [{} for _ in texts] if metadatas is None else metadatas
        )

        return super().aadd_texts(texts, metadatas, **kwargs)

    def add_texts(
        self,
        texts: Iterable[str],
        metadatas: Optional[Iterable[Dict[str, Any]]] = None,
        **kwargs: Any,
    ) -> List[str]:
        """Run more texts through the embeddings and add to the vectorstore.

        Args:
            texts: Iterable of strings to add to the vectorstore.
            metadatas: Optional list of metadatas associated with the texts.
            kwargs: vectorstore specific parameters

        Returns:
            List of ids from adding the texts into the vectorstore.
        """
        # TODO: Batch texts in case it is too long? Or assume it fits in memory.
        texts = list(texts)

        metadatas: Iterable[Dict[str, str]] = (
            [{} for _ in texts] if metadatas is None else metadatas
        )
        text_embeddings = self._embedding.embed_documents(texts)

        ids = []
        with self._concurrent_queries() as cq:
            tuples = zip(texts, text_embeddings, metadatas, strict=True)
            for text, text_embedding, metadata in tuples:
                if CONTENT_ID not in metadata:
                    metadata[CONTENT_ID] = secrets.token_hex(8)
                id = metadata[CONTENT_ID]
                ids.append(id)
                keywords = metadata.get(KEYWORDS, set())
                urls = metadata.get(URLS, set())
                hrefs = metadata.get(HREFS, set())

                cq.execute(
                    self._insert_passage, (id, text, text_embedding, keywords, urls, hrefs)
                )

        for extractor in self._edge_extractors:
            extractor.extract_edges(self, texts, metadatas)

        return ids

    # TODO: Async
    @classmethod
    def from_texts(
        cls,
        texts: List[str],
        embedding: Embeddings,
        metadatas: Optional[List[dict]] = None,
        **kwargs: Any,
    ) -> "KnowledgeStore":
        """Return VectorStore initialized from texts and embeddings."""

        ids = kwargs.pop("ids")
        knowledge_store = KnowledgeStore(embedding, **kwargs)
        knowledge_store.add_texts(texts, metadatas, ids=ids)
        return knowledge_store

    # TODO: Async
    def similarity_search(
        self,
        query: str,
        *,
        k: int = 4,
        metadata_filter: Dict[str, str] = {},
    ) -> List[Document]:
        """Return docs most similar to query.

        Args:
            query: Text to look up documents similar to.
            k: Number of Documents to return. Defaults to 4.
            filter: Filter on the metadata to apply.
        Returns:
            List of Document, the most similar to the query vector.
        """
        embedding_vector = self._embedding.embed_query(query)
        return self.similarity_search_by_vector(
            embedding_vector,
            k=k,
            metadata_filter=metadata_filter,
        )

    # TODO: Async
    def similarity_search_by_vector(
        self,
        query_vector: List[float],
        *,
        k: int = 4,
        metadata_filter: Dict[str, str] = {},
    ) -> List[Document]:
        """Return docs most similar to query_vector.

        Args:
            query_vector: Embeding to lookup documents similar to.
            k: Number of Documents to return. Defaults to 4.
            filter: Filter on the metadata to apply.
        Returns:
            List of Document, the most simliar to the query vector.
        """
        results = self._session.execute(self._query_by_embedding, (query_vector, k))
        return _results_to_documents(results)

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

    def retrieve(
        self, query: Union[str, Iterable[str]], *, k: int = 4, depth: int = 1
    ) -> Iterable[Document]:
        """Return a runnable for retrieving from this knowledge store.

        First, `k` nodes are retrieved using a vector search for each `query` string.
        Then, additional nodes are discovered up to the given `depth` from those starting
        nodes.

        Args:
            query: The query string or collection fo query strings.
            k: The number of Documents to return from the initial vector search.
                Defaults to 4. Applies to each of the query strings.
            depth: The maximum depth of edges to traverse. Defaults to 1.
        Returns:
            Collection of retrieved documents.
        """
        if isinstance(query, str):
            query = {query}
        else:
            query = set(query)

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
                                callback=lambda nodes, d=d: visit(d + 1, nodes),
                            )

            for q in query:
                query_embedding = self._embedding.embed_query(q)
                cq.execute(
                    self._query_ids_by_embedding,
                    parameters=(query_embedding, k),
                    callback=lambda nodes: visit(0, nodes),
                )

        return self._query_by_ids(visited.keys())

    def as_retriever(
        self,
        *,
        k: int = 4,
        depth: int = 1,
    ) -> Runnable[Union[str | Iterable[str]], Iterable[Document]]:
        """Return a runnable for retrieving from this knowldege store.

        The initial nodes are retrieved using a vector search.
        Additional nodes are discovered up to the given depth from those starting nodes.

        Args:
            k: The number of Documents to return from the initial vector search.
                Defaults to 4. Applies to each of the query strings.
            depth: The maximum depth of edges to traverse. Defaults to 1.
        Returns:
            Runnable accepting a query string or collection of query strings and
            returning corresponding the documents.
        """
        # TODO: Async version
        retriever = RunnableLambda(func=self.retrieve, name="Knowledge Store Retriever")
        return retriever.bind(k=k, depth=depth)
