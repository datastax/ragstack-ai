from typing import Any, Iterable, List, Optional, Tuple, Type, TypeVar

from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
from langchain_core.runnables.config import run_in_executor
from langchain_core.vectorstores import VectorStore
from ragstack_colbert import Chunk
from ragstack_colbert import ColbertVectorStore as RagstackColbertVectorStore
from ragstack_colbert.base_database import BaseDatabase as ColbertBaseDatabase
from ragstack_colbert.base_embedding_model import (
    BaseEmbeddingModel as ColbertBaseEmbeddingModel,
)
from ragstack_colbert.base_retriever import BaseRetriever as ColbertBaseRetriever
from ragstack_colbert.base_vector_store import BaseVectorStore as ColbertBaseVectorStore

from .colbert_retriever import ColbertRetriever

CVS = TypeVar("CVS", bound="ColbertVectorStore")


class ColbertVectorStore(VectorStore):

    _vector_store: ColbertBaseVectorStore
    _retriever: ColbertBaseRetriever

    def __init__(
        self,
        database: ColbertBaseDatabase,
        embedding_model: ColbertBaseEmbeddingModel,
    ):
        self._initialize(database=database, embedding_model=embedding_model)

    def _initialize(
        self,
        database: ColbertBaseDatabase,
        embedding_model: ColbertBaseEmbeddingModel,
    ):
        self._vector_store = RagstackColbertVectorStore(
            database=database, embedding_model=embedding_model
        )
        self._retriever = self._vector_store.as_retriever()

    def add_texts(
        self,
        texts: Iterable[str],
        metadatas: Optional[List[dict]] = None,
        doc_id: Optional[str] = None,
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
        return self._vector_store.add_texts(
            texts=list(texts), metadatas=metadatas, doc_id=doc_id
        )

    async def aadd_texts(
        self,
        texts: Iterable[str],
        metadatas: Optional[List[dict]] = None,
        doc_id: Optional[str] = None,
        concurrent_inserts: Optional[int] = 100,
        **kwargs: Any,
    ) -> List[str]:
        """Run more texts through the embeddings and add to the vectorstore.

        Args:
            texts: Iterable of strings to add to the vectorstore.
            metadatas: Optional list of metadatas associated with the texts.
            concurrent_inserts (Optional[int]): How many concurrent inserts to make to the database. Defaults to 100.
            kwargs: vectorstore specific parameters

        Returns:
            List of ids from adding the texts into the vectorstore.
        """
        return await self._vector_store.aadd_texts(
            texts=list(texts),
            metadatas=metadatas,
            doc_id=doc_id,
            concurrent_inserts=concurrent_inserts,
        )

    def delete(self, ids: Optional[List[str]] = None, **kwargs: Any) -> Optional[bool]:
        """Delete by vector ID or other criteria.

        Args:
            ids: List of ids to delete.
            **kwargs: Other keyword arguments that subclasses might use.

        Returns:
            Optional[bool]: True if deletion is successful,
            False otherwise, None if not implemented.
        """
        return None if ids is None else self._vector_store.delete(ids=ids)

    async def adelete(
        self,
        ids: Optional[List[str]] = None,
        concurrent_deletes: Optional[int] = 100,
        **kwargs: Any
    ) -> Optional[bool]:
        """Delete by vector ID or other criteria.

        Args:
            ids: List of ids to delete.
            concurrent_deletes (Optional[int]): How many concurrent deletes to make to the database. Defaults to 100.
            **kwargs: Other keyword arguments that subclasses might use.

        Returns:
            Optional[bool]: True if deletion is successful,
            False otherwise, None if not implemented.
        """
        return None if ids is None else await self._vector_store.adelete(ids=ids, concurrent_deletes=concurrent_deletes)

    def similarity_search(
        self,
        query: str,
        k: Optional[int] = 5,
        query_maxlen: Optional[int] = None,
        **kwargs: Any,
    ) -> List[Document]:
        """Return docs most similar to query."""
        chunk_scores: List[Tuple[Chunk, float]] = self._retriever.text_search(
            query_text=query, k=k, query_maxlen=query_maxlen
        )

        return [
            Document(page_content=c.text, metadata=c.metadata)
            for (c, _) in chunk_scores
        ]

    def similarity_search_with_score(
        self,
        query: str,
        k: Optional[int] = 5,
        query_maxlen: Optional[int] = None,
        **kwargs: Any,
    ) -> List[Tuple[Document, float]]:
        """Run similarity search with distance."""
        chunk_scores: List[Tuple[Chunk, float]] = self._retriever.text_search(
            query_text=query, k=k, query_maxlen=query_maxlen
        )

        return [
            (Document(page_content=c.text, metadata=c.metadata), s)
            for (c, s) in chunk_scores
        ]

    async def asimilarity_search(
        self,
        query: str,
        k: Optional[int] = 5,
        query_maxlen: Optional[int] = None,
        **kwargs: Any,
    ) -> List[Document]:
        """Return docs most similar to query."""
        chunk_scores: List[Tuple[Chunk, float]] = await self._retriever.atext_search(
            query_text=query, k=k, query_maxlen=query_maxlen
        )

        return [
            Document(page_content=c.text, metadata=c.metadata)
            for (c, _) in chunk_scores
        ]

    async def asimilarity_search_with_score(
        self,
        query: str,
        k: Optional[int] = 5,
        query_maxlen: Optional[int] = None,
        **kwargs: Any,
    ) -> List[Tuple[Document, float]]:
        """Run similarity search with distance."""
        chunk_scores: List[Tuple[Chunk, float]] = await self._retriever.atext_search(
            query_text=query, k=k, query_maxlen=query_maxlen
        )

        return [
            (Document(page_content=c.text, metadata=c.metadata), s)
            for (c, s) in chunk_scores
        ]

    @classmethod
    def from_documents(
        cls,
        documents: List[Document],
        database: ColbertBaseDatabase,
        embedding_model: ColbertBaseEmbeddingModel,
        **kwargs: Any,
    ) -> CVS:
        """Return VectorStore initialized from documents and embeddings."""
        texts = [d.page_content for d in documents]
        metadatas = [d.metadata for d in documents]
        return cls.from_texts(
            texts=texts,
            database=database,
            embedding_model=embedding_model,
            metadatas=metadatas,
            **kwargs,
        )

    @classmethod
    async def afrom_documents(
        cls: Type[CVS],
        documents: List[Document],
        database: ColbertBaseDatabase,
        embedding_model: ColbertBaseEmbeddingModel,
        concurrent_inserts: Optional[int] = 100,
        **kwargs: Any,
    ) -> CVS:
        """Return VectorStore initialized from documents and embeddings."""
        texts = [d.page_content for d in documents]
        metadatas = [d.metadata for d in documents]
        return await cls.afrom_texts(
            texts=texts,
            database=database,
            embedding_model=embedding_model,
            metadatas=metadatas,
            concurrent_inserts=concurrent_inserts,
            **kwargs,
        )

    @classmethod
    def from_texts(
        cls: Type[CVS],
        texts: List[str],
        database: ColbertBaseDatabase,
        embedding_model: ColbertBaseEmbeddingModel,
        metadatas: Optional[List[dict]] = None,
        **kwargs: Any,
    ) -> CVS:
        """Return VectorStore initialized from texts and embeddings."""

        instance = super().__new__(cls)
        instance._initialize(database=database, embedding_model=embedding_model)
        instance.add_texts(texts=texts, metadatas=metadatas)
        return instance

    @classmethod
    async def afrom_texts(
        cls: Type[CVS],
        texts: List[str],
        database: ColbertBaseDatabase,
        embedding_model: ColbertBaseEmbeddingModel,
        metadatas: Optional[List[dict]] = None,
        concurrent_inserts: Optional[int] = 100,
        **kwargs: Any,
    ) -> CVS:
        """Return VectorStore initialized from texts and embeddings."""
        instance = super().__new__(cls)
        instance._initialize(database=database, embedding_model=embedding_model)
        await instance.aadd_texts(texts=texts, metadatas=metadatas, concurrent_inserts=concurrent_inserts)
        return instance

    def as_retriever(self, k: Optional[int] = 5, **kwargs: Any) -> BaseRetriever:
        """Return a VectorStoreRetriever initialized from this VectorStore."""
        return ColbertRetriever(
            retriever=self._vector_store.as_retriever(), k=k, **kwargs
        )
