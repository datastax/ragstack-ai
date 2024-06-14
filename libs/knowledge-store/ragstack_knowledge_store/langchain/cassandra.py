from typing import (
    Any,
    Iterable,
    List,
    Optional,
    Type,
)

from cassandra.cluster import ResponseFuture, Session
from langchain_community.utilities.cassandra import SetupMode
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings

from .base import GraphStore, Node, TextNode
from ragstack_knowledge_store import EmbeddingModel, graph_store


class _EmbeddingModelAdapter(EmbeddingModel):
    def __init__(self, embeddings: Embeddings):
        self.embeddings = embeddings

    def embed_texts(self, texts: List[str]) -> List[List[float]]:
        return self.embeddings.embed_documents(texts)

    def embed_query(self, text: str) -> List[float]:
        return self.embeddings.embed_query(text)

    async def aembed_texts(self, texts: List[str]) -> List[List[float]]:
        return await self.embeddings.aembed_documents(texts)

    async def aembed_query(self, text: str) -> List[float]:
        return await self.embeddings.aembed_query(text)


def _row_to_document(row) -> Document:
    return Document(
        page_content=row.text_content,
        metadata={
            graph_store.CONTENT_ID: row.content_id,
            "kind": row.kind,
        },
    )


def _results_to_documents(results: Optional[ResponseFuture]) -> Iterable[Document]:
    if results:
        for row in results:
            yield _row_to_document(row)


class CassandraGraphStore(GraphStore):
    def __init__(
        self,
        embedding: Embeddings,
        *,
        node_table: str = "graph_nodes",
        edge_table: str = "graph_edges",
        session: Optional[Session] = None,
        keyspace: Optional[str] = None,
        setup_mode: SetupMode = SetupMode.SYNC,
        concurrency: int = 20,
    ):
        """
        Create the hybrid graph store.
        Parameters configure the ways that edges should be added between
        documents. Many take `Union[bool, Set[str]]`, with `False` disabling
        inference, `True` enabling it globally between all documents, and a set
        of metadata fields defining a scope in which to enable it. Specifically,
        passing a set of metadata fields such as `source` only links documents
        with the same `source` metadata value.
        Args:
            embedding: The embeddings to use for the document content.
            concurrency: Maximum number of queries to have concurrently executing.
            setup_mode: Mode used to create the Cassandra table (SYNC,
                ASYNC or OFF).
        """
        self._embedding = embedding
        _setup_mode = getattr(graph_store.SetupMode, setup_mode.name)

        self.store = graph_store.GraphStore(
            embedding=_EmbeddingModelAdapter(embedding),
            node_table=node_table,
            edge_table=edge_table,
            session=session,
            keyspace=keyspace,
            setup_mode=_setup_mode,
            concurrency=concurrency,
        )

    @property
    def embeddings(self) -> Optional[Embeddings]:
        return self._embedding

    def add_nodes(
        self,
        nodes: Iterable[Node] = None,
        **kwargs: Any,
    ):
        _nodes = []
        for node in nodes:
            if not isinstance(node, TextNode):
                raise ValueError("Only adding TextNode is supported at the moment")
            _nodes.append(
                graph_store.TextNode(id=node.id, text=node.text, metadata=node.metadata)
            )
        return self.store.add_nodes(_nodes)

    @classmethod
    def from_texts(
        cls: Type["CassandraGraphStore"],
        texts: Iterable[str],
        embedding: Embeddings,
        metadatas: Optional[List[dict]] = None,
        ids: Optional[Iterable[str]] = None,
        **kwargs: Any,
    ) -> "CassandraGraphStore":
        """Return CassandraGraphStore initialized from texts and embeddings."""
        store = cls(embedding, **kwargs)
        store.add_texts(texts, metadatas, ids=ids)
        return store

    @classmethod
    def from_documents(
        cls: Type["CassandraGraphStore"],
        documents: Iterable[Document],
        embedding: Embeddings,
        ids: Optional[Iterable[str]] = None,
        **kwargs: Any,
    ) -> "CassandraGraphStore":
        """Return CassandraGraphStore initialized from documents and embeddings."""
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
        for node in self.store.similarity_search(embedding, k=k):
            yield Document(
                page_content=node.text,
                metadata=node.metadata,
            )

    def traversal_search(
        self,
        query: str,
        *,
        k: int = 4,
        depth: int = 1,
        **kwargs: Any,
    ) -> Iterable[Document]:
        for node in self.store.traversal_search(query, k=k, depth=depth):
            yield Document(
                page_content=node.text,
                metadata=node.metadata,
            )

    def mmr_traversal_search(
        self,
        query: str,
        *,
        k: int = 4,
        depth: int = 2,
        fetch_k: int = 100,
        lambda_mult: float = 0.5,
        score_threshold: float = float("-inf"),
        **kwargs: Any,
    ) -> Iterable[Document]:
        for node in self.store.mmr_traversal_search(
            query,
            k=k,
            depth=depth,
            fetch_k=fetch_k,
            lambda_mult=lambda_mult,
            score_threshold=score_threshold,
        ):
            yield Document(
                page_content=node.text,
                metadata=node.metadata,
            )
