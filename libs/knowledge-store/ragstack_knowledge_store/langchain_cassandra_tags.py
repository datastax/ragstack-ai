from __future__ import annotations

from typing import (
    TYPE_CHECKING,
    Any,
    Iterable,
    List,
    Optional,
    Type,
)

from langchain_core._api import beta
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.graph_vectorstores.base import (
    GraphVectorStore,
    nodes_to_documents,
)

from langchain_core.graph_vectorstores.base import Node as BaseNode
from langchain_community.utilities.cassandra import SetupMode
from langchain_core.graph_vectorstores import GraphVectorStoreRetriever


if TYPE_CHECKING:
    from cassandra.cluster import Session

from ragstack_knowledge_store import graph_store_tags, Node
from ragstack_knowledge_store.embedding_model import EmbeddingModel

@beta()
class CassandraGraphVectorStore(GraphVectorStore):
    def __init__(
        self,
        embedding: Embeddings,
        *,
        node_table: str = "graph_nodes",
        session: Optional[Session] = None,
        keyspace: Optional[str] = None,
        setup_mode: SetupMode = SetupMode.SYNC,
        **kwargs: Any,
    ):
        """
        Create the hybrid graph store.

        Args:
            embedding: The embeddings to use for the document content.
            setup_mode: Mode used to create the Cassandra table (SYNC,
                ASYNC or OFF).
        """
        # try:
        #     from ragstack_knowledge_store import EmbeddingModel, graph_store
        # except (ImportError, ModuleNotFoundError):
        #     raise ImportError(
        #         "Could not import ragstack_knowledge_store python package. "
        #         "Please install it with `pip install ragstack-ai-knowledge-store`."
        #     )

        self._embedding = embedding
        _setup_mode = getattr(graph_store_tags.SetupMode, setup_mode.name)

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

        self.store = graph_store_tags.GraphStore(
            embedding=_EmbeddingModelAdapter(embedding),
            node_table=node_table,
            session=session,
            keyspace=keyspace,
            setup_mode=_setup_mode,
            **kwargs,
        )

    @property
    def embeddings(self) -> Optional[Embeddings]:
        return self._embedding

    def add_nodes(
        self,
        nodes: Iterable[BaseNode],
        **kwargs: Any,
    ) -> Iterable[str]:
        converted_nodes = [Node(text=n.text, id=n.id, metadata=n.metadata, links=n.links) for n in nodes]
        return self.store.add_nodes(converted_nodes)

    @classmethod
    def from_texts(
        cls: Type["CassandraGraphVectorStore"],
        texts: Iterable[str],
        embedding: Embeddings,
        metadatas: Optional[List[dict]] = None,
        ids: Optional[Iterable[str]] = None,
        **kwargs: Any,
    ) -> "CassandraGraphVectorStore":
        """Return CassandraGraphVectorStore initialized from texts and embeddings."""
        store = cls(embedding, **kwargs)
        store.add_texts(texts, metadatas, ids=ids)
        return store

    @classmethod
    def from_documents(
        cls: Type["CassandraGraphVectorStore"],
        documents: Iterable[Document],
        embedding: Embeddings,
        ids: Optional[Iterable[str]] = None,
        **kwargs: Any,
    ) -> "CassandraGraphVectorStore":
        """Return CassandraGraphVectorStore initialized from documents and
        embeddings."""
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
        nodes = self.store.similarity_search(embedding, k=k)
        return list(nodes_to_documents(nodes))

    def traversal_search(
        self,
        query: str,
        *,
        k: int = 4,
        depth: int = 1,
        **kwargs: Any,
    ) -> Iterable[Document]:
        nodes = self.store.traversal_search(query, k=k, depth=depth)
        return nodes_to_documents(nodes)

    def mmr_traversal_search(
        self,
        query: str,
        *,
        k: int = 4,
        depth: int = 2,
        fetch_k: int = 100,
        adjacent_k: int = 10,
        lambda_mult: float = 0.5,
        score_threshold: float = float("-inf"),
        **kwargs: Any,
    ) -> Iterable[Document]:
        nodes = self.store.mmr_traversal_search(
            query,
            k=k,
            depth=depth,
            fetch_k=fetch_k,
            adjacent_k=adjacent_k,
            lambda_mult=lambda_mult,
            score_threshold=score_threshold,
        )
        return nodes_to_documents(nodes)
