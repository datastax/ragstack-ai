from typing import Any, List, Optional, Tuple, Iterable, Type, TypeVar

from llama_index.core.vector_stores.types import VectorStore, VectorStoreQuery, VectorStoreQueryResult
from llama_index.core.schema import TextNode

from ragstack_colbert.base_database import BaseDatabase as ColbertBaseDatabase
from ragstack_colbert.base_vector_store import BaseVectorStore as ColbertBaseVectorStore
from ragstack_colbert import ColbertVectorStore, Chunk, Embedding

from .retriever import ColbertRetriever

class ColbertTextNode(TextNode):
    doc_id: str
    chunk_id: int
    embedding: Embedding # need to overshadow the embedding in BaseNode, since it is 1-dimensional


class ColbertVectorStore(VectorStore):


    _vector_store: ColbertBaseVectorStore

    def __init__(
            self,
            database: ColbertBaseDatabase,
    ):
        self._vector_store = ColbertVectorStore(database=database)

    def add(
        self,
        nodes: List[ColbertTextNode],
        **add_kwargs: Any,
    ) -> List[str]:
        """Add nodes with embedding to vector store."""

        chunks = [Chunk(doc_id=n.doc_id, chunk_id=n.chunk_id, text=n.text, metadata=n.metadata, embedding=n.embedding) for n in nodes]

        ids:Tuple[str, int] = self._vector_store.add_chunks(chunks)

        return ???

    async def async_add(
        self,
        nodes: List[TextNode],
        **kwargs: Any,
    ) -> List[str]:
        """
        Asynchronously add nodes with embedding to vector store.
        NOTE: this is not implemented for all vector stores. If not implemented,
        it will just call add synchronously.
        """
        return self.add(nodes)

    def delete(self, ref_doc_id: str, **delete_kwargs: Any) -> None:
        """
        Delete nodes using with ref_doc_id."""
        self._vector_store.delete_chunks(doc_ids=[ref_doc_id])

    async def adelete(self, ref_doc_id: str, **delete_kwargs: Any) -> None:
        """
        Delete nodes using with ref_doc_id.
        NOTE: this is not implemented for all vector stores. If not implemented,
        it will just call delete synchronously.
        """
        self.delete(ref_doc_id, **delete_kwargs)

    def query(self, query: VectorStoreQuery, **kwargs: Any) -> VectorStoreQueryResult:
        """Query vector store."""
        ...
        IMPLEMENT THIS SSTILL

    async def aquery(
        self, query: VectorStoreQuery, **kwargs: Any
    ) -> VectorStoreQueryResult:
        """
        Asynchronously query vector store.
        NOTE: this is not implemented for all vector stores. If not implemented,
        it will just call query synchronously.
        """
        IMPLEMENT THIS SSTILL