from __future__ import annotations

import itertools
from abc import abstractmethod
from typing import (
    Any,
    Iterable,
    List,
    Optional,
    Union,
    Iterator,
    ClassVar,
    Collection,
)

from langchain_core.callbacks import CallbackManagerForRetrieverRun
from langchain_core.documents import Document
from langchain_core.load import Serializable
from langchain_core.vectorstores import VectorStore, VectorStoreRetriever
from pydantic import Field


class Node(Serializable):
    id: Optional[str]
    metadata: dict = Field(default_factory=dict)


class TextNode(Node):
    text: str


def _texts_to_nodes(
    texts: Iterator[str],
    metadatas: Iterator[dict],
    ids: Iterator[str],
) -> Iterator[Node]:
    for text, metadata, node_id in zip(texts, metadatas, ids):
        yield TextNode(
            id=node_id,
            metadata=metadata,
            text=text,
        )


def _documents_to_nodes(
    documents: Iterator[Document], ids: Iterator[str]
) -> Iterator[Node]:
    for (
        doc,
        node_id,
    ) in zip(documents, ids):
        yield TextNode(
            id=node_id,
            metadata=doc.metadata,
            text=doc.page_content,
        )


class KnowledgeStore(VectorStore):
    """A hybrid vector-and-graph knowledge store.

    Document chunks support vector-similarity search as well as edges linking
    chunks based on structural and semantic properties.
    """

    @abstractmethod
    def add_nodes(
        self,
        nodes: Iterable[Node],
        **kwargs: Any,
    ) -> List[str]:
        """Add nodes to the graph"""

    def add_texts(
        self,
        texts: Iterable[str],
        metadatas: Optional[Iterable[dict]] = None,
        *,
        ids: Optional[Iterable[str]] = None,
        **kwargs: Any,
    ) -> List[str]:
        metadatas_it = iter(metadatas) if metadatas else itertools.repeat({})
        ids_it = iter(ids) if ids else itertools.repeat(None)
        nodes = _texts_to_nodes(iter(texts), metadatas_it, ids_it)
        return self.add_nodes(nodes, **kwargs)

    def add_documents(
        self,
        documents: Iterable[Document] = None,
        *,
        ids: Optional[Iterable[str]] = None,
        **kwargs: Any,
    ) -> List[str]:
        ids_it = iter(ids) if ids else itertools.repeat(None)
        nodes = _documents_to_nodes(iter(documents), ids_it)
        return self.add_nodes(nodes, **kwargs)

    @abstractmethod
    def traversing_retrieve(
        self,
        query: Union[str, Iterable[str]],
        *,
        k: int = 4,
        depth: int = 1,
        **kwargs: Any,
    ) -> Iterable[Document]:
        """Retrieve documents from traversing this knowledge store.

        First, `k` nodes are retrieved using a search for each `query` string.
        Then, additional nodes are discovered up to the given `depth` from those
        starting nodes.

        Args:
            query: The query string or collection of query strings.
            k: The number of Documents to return from the initial search.
                Defaults to 4. Applies to each of the query strings.
            depth: The maximum depth of edges to traverse. Defaults to 1.
        Returns:
            Retrieved documents.
        """

    def similarity_search(
        self, query: str, k: int = 4, **kwargs: Any
    ) -> List[Document]:
        return list(self.traversing_retrieve(query, k=k, depth=0))

    def search(self, query: str, search_type: str, **kwargs: Any) -> List[Document]:
        if search_type == "similarity":
            return self.similarity_search(query, **kwargs)
        elif search_type == "similarity_score_threshold":
            docs_and_similarities = self.similarity_search_with_relevance_scores(
                query, **kwargs
            )
            return [doc for doc, _ in docs_and_similarities]
        elif search_type == "mmr":
            return self.max_marginal_relevance_search(query, **kwargs)
        elif search_type == "traversal":
            return list(self.traversing_retrieve(query, **kwargs))
        else:
            raise ValueError(
                f"search_type of {search_type} not allowed. Expected "
                "search_type to be 'similarity', 'similarity_score_threshold', "
                "'mmr' or 'traversal'."
            )

    def as_retriever(self, **kwargs: Any) -> "KnowledgeStoreRetriever":
        """Return a Retriever for retrieving from this knowledge store."""
        return KnowledgeStoreRetriever(vectorstore=self, **kwargs)


class KnowledgeStoreRetriever(VectorStoreRetriever):
    """Retriever class for KnowledgeStore."""

    vectorstore: KnowledgeStore
    """KnowledgeStore to use for retrieval."""
    search_type: str = "traversal"
    """Type of search to perform. Defaults to "traversal"."""
    allowed_search_types: ClassVar[Collection[str]] = (
        "similarity",
        "similarity_score_threshold",
        "mmr",
        "traversal",
    )

    def _get_relevant_documents(
        self, query: str, *, run_manager: CallbackManagerForRetrieverRun
    ) -> List[Document]:
        if self.search_type == "traversal":
            return list(
                self.vectorstore.traversing_retrieve(query, **self.search_kwargs)
            )
        else:
            return super()._get_relevant_documents(query, run_manager=run_manager)
