from __future__ import annotations

from typing import TYPE_CHECKING, Any, Dict, Iterable, List, Optional, Tuple

from langchain_core.documents import Document
from langchain_core.vectorstores import VectorStore, VectorStoreRetriever
from ragstack_colbert import Chunk
from ragstack_colbert import ColbertVectorStore as RagstackColbertVectorStore
from typing_extensions import Self, override

from ragstack_langchain.colbert.embedding import TokensEmbeddings

if TYPE_CHECKING:
    from langchain_core.embeddings import Embeddings
    from ragstack_colbert.base_database import BaseDatabase as ColbertBaseDatabase
    from ragstack_colbert.base_embedding_model import (
        BaseEmbeddingModel as ColbertBaseEmbeddingModel,
    )
    from ragstack_colbert.base_retriever import BaseRetriever as ColbertBaseRetriever
    from ragstack_colbert.base_vector_store import (
        BaseVectorStore as ColbertBaseVectorStore,
    )


class ColbertVectorStore(VectorStore):
    """VectorStore for ColBERT."""

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
    ) -> None:
        self._vector_store = RagstackColbertVectorStore(
            database=database, embedding_model=embedding_model
        )
        self._retriever = self._vector_store.as_retriever()

    @override
    def add_texts(
        self,
        texts: Iterable[str],
        metadatas: Optional[List[Dict[str, Any]]] = None,
        doc_id: Optional[str] = None,
        **kwargs: Any,
    ) -> List[str]:
        """Run more texts through the embeddings and add to the vectorstore.

        Args:
            texts: Iterable of strings to add to the vectorstore.
            metadatas: Optional list of metadatas associated with the texts.
            doc_id: Optional document ID to associate with the texts.
            kwargs: vectorstore specific parameters

        Returns:
            List of ids from adding the texts into the vectorstore.
        """
        results = self._vector_store.add_texts(
            texts=list(texts), metadatas=metadatas, doc_id=doc_id
        )
        return [results[0][0]] if results else []

    @override
    async def aadd_texts(
        self,
        texts: Iterable[str],
        metadatas: Optional[List[Dict[str, Any]]] = None,
        doc_id: Optional[str] = None,
        concurrent_inserts: int = 100,
        **kwargs: Any,
    ) -> List[str]:
        """Run more texts through the embeddings and add to the vectorstore.

        Args:
            texts: Iterable of strings to add to the vectorstore.
            metadatas: Optional list of metadatas associated with the texts.
            doc_id: Optional document ID to associate with the texts.
            concurrent_inserts: How many concurrent inserts to make to the database.
                Defaults to 100.
            kwargs: vectorstore specific parameters

        Returns:
            List of ids from adding the texts into the vectorstore.
        """
        results = await self._vector_store.aadd_texts(
            texts=list(texts),
            metadatas=metadatas,
            doc_id=doc_id,
            concurrent_inserts=concurrent_inserts,
        )
        return [results[0][0]] if results else []

    @override
    def delete(self, ids: Optional[List[str]] = None, **kwargs: Any) -> Optional[bool]:
        return None if ids is None else self._vector_store.delete_chunks(doc_ids=ids)

    @override
    async def adelete(
        self,
        ids: Optional[List[str]] = None,
        concurrent_deletes: int = 100,
        **kwargs: Any,
    ) -> Optional[bool]:
        return (
            None
            if ids is None
            else await self._vector_store.adelete_chunks(
                doc_ids=ids, concurrent_deletes=concurrent_deletes
            )
        )

    @override
    def similarity_search(
        self,
        query: str,
        k: Optional[int] = 5,
        query_maxlen: Optional[int] = None,
        **kwargs: Any,
    ) -> List[Document]:
        """Return docs most similar to query."""
        chunk_scores: List[Tuple[Chunk, float]] = self._retriever.text_search(
            query_text=query, k=k, query_maxlen=query_maxlen, **kwargs
        )

        return [
            Document(page_content=c.text, metadata=c.metadata)
            for (c, _) in chunk_scores
        ]

    @override
    def similarity_search_with_score(
        self,
        query: str,
        k: Optional[int] = 5,
        query_maxlen: Optional[int] = None,
        **kwargs: Any,
    ) -> List[Tuple[Document, float]]:
        """Run similarity search with distance."""
        chunk_scores: List[Tuple[Chunk, float]] = self._retriever.text_search(
            query_text=query, k=k, query_maxlen=query_maxlen, **kwargs
        )

        return [
            (Document(page_content=c.text, metadata=c.metadata), s)
            for (c, s) in chunk_scores
        ]

    @override
    async def asimilarity_search(
        self,
        query: str,
        k: Optional[int] = 5,
        query_maxlen: Optional[int] = None,
        **kwargs: Any,
    ) -> List[Document]:
        """Return docs most similar to query."""
        chunk_scores: List[Tuple[Chunk, float]] = await self._retriever.atext_search(
            query_text=query, k=k, query_maxlen=query_maxlen, **kwargs
        )

        return [
            Document(page_content=c.text, metadata=c.metadata)
            for (c, _) in chunk_scores
        ]

    @override
    async def asimilarity_search_with_score(
        self,
        query: str,
        k: Optional[int] = 5,
        query_maxlen: Optional[int] = None,
        **kwargs: Any,
    ) -> List[Tuple[Document, float]]:
        """Run similarity search with distance."""
        chunk_scores: List[Tuple[Chunk, float]] = await self._retriever.atext_search(
            query_text=query, k=k, query_maxlen=query_maxlen, **kwargs
        )

        return [
            (Document(page_content=c.text, metadata=c.metadata), s)
            for (c, s) in chunk_scores
        ]

    @classmethod
    @override
    def from_documents(
        cls,
        documents: List[Document],
        embedding: Embeddings,
        *,
        database: Optional[ColbertBaseDatabase] = None,
        **kwargs: Any,
    ) -> Self:
        """Return VectorStore initialized from documents and embeddings."""
        texts = [d.page_content for d in documents]
        metadatas = [d.metadata for d in documents]
        return cls.from_texts(
            texts=texts,
            database=database,
            embedding=embedding,
            metadatas=metadatas,
            **kwargs,
        )

    @classmethod
    @override
    async def afrom_documents(
        cls,
        documents: List[Document],
        embedding: Embeddings,
        *,
        database: Optional[ColbertBaseDatabase] = None,
        concurrent_inserts: int = 100,
        **kwargs: Any,
    ) -> Self:
        """Return VectorStore initialized from documents and embeddings."""
        texts = [d.page_content for d in documents]
        metadatas = [d.metadata for d in documents]
        return await cls.afrom_texts(
            texts=texts,
            database=database,
            embedding=embedding,
            metadatas=metadatas,
            concurrent_inserts=concurrent_inserts,
            **kwargs,
        )

    @classmethod
    @override
    def from_texts(
        cls,
        texts: List[str],
        embedding: Embeddings,
        metadatas: Optional[List[Dict[str, Any]]] = None,
        *,
        database: Optional[ColbertBaseDatabase] = None,
        **kwargs: Any,
    ) -> Self:
        if not isinstance(embedding, TokensEmbeddings):
            msg = "ColbertVectorStore requires a TokensEmbeddings embedding."
            raise TypeError(msg)
        if database is None:
            msg = "ColbertVectorStore requires a ColbertBaseDatabase database."
            raise ValueError(msg)
        instance = cls(
            database=database, embedding_model=embedding.get_embedding_model(), **kwargs
        )
        instance.add_texts(texts=texts, metadatas=metadatas)
        return instance

    @classmethod
    @override
    async def afrom_texts(
        cls,
        texts: List[str],
        embedding: Embeddings,
        metadatas: Optional[List[Dict[str, Any]]] = None,
        *,
        database: Optional[ColbertBaseDatabase] = None,
        concurrent_inserts: int = 100,
        **kwargs: Any,
    ) -> Self:
        if not isinstance(embedding, TokensEmbeddings):
            msg = "ColbertVectorStore requires a TokensEmbeddings embedding."
            raise TypeError(msg)
        if database is None:
            msg = "ColbertVectorStore requires a ColbertBaseDatabase database."
            raise ValueError(msg)
        instance = cls(
            database=database, embedding_model=embedding.get_embedding_model(), **kwargs
        )
        await instance.aadd_texts(
            texts=texts, metadatas=metadatas, concurrent_inserts=concurrent_inserts
        )
        return instance

    @override
    def as_retriever(self, k: Optional[int] = 5, **kwargs: Any) -> VectorStoreRetriever:
        """Return a VectorStoreRetriever initialized from this VectorStore."""
        search_kwargs = kwargs.pop("search_kwargs", {})
        search_kwargs["k"] = k
        search_type = kwargs.get("search_type", "similarity")
        if search_type != "similarity":
            msg = f"Unsupported search type: {search_type}"
            raise ValueError(msg)
        return super().as_retriever(search_kwargs=search_kwargs, **kwargs)
