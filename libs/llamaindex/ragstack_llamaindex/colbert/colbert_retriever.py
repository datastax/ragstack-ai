from __future__ import annotations

from typing import TYPE_CHECKING, Any

from llama_index.core.constants import DEFAULT_SIMILARITY_TOP_K
from llama_index.core.retrievers import BaseRetriever
from llama_index.core.schema import NodeWithScore, QueryBundle, TextNode

if TYPE_CHECKING:
    from llama_index.core.callbacks.base import CallbackManager
    from ragstack_colbert import Chunk
    from ragstack_colbert.base_retriever import BaseRetriever as ColbertBaseRetriever


class ColbertRetriever(BaseRetriever):
    """ColBERT vector store retriever.

    Args:
        retriever (BaseRetriever): vector store index.
        similarity_top_k (int): number of top k results to return.
    """

    _retriever: ColbertBaseRetriever
    _k: int
    _query_maxlen: int | None

    def __init__(
        self,
        retriever: ColbertBaseRetriever,
        similarity_top_k: int = DEFAULT_SIMILARITY_TOP_K,
        callback_manager: CallbackManager | None = None,
        object_map: dict[str, Any] | None = None,
        verbose: bool = False,
        query_maxlen: int = -1,
    ) -> None:
        """Initialize params."""
        self._retriever = retriever
        self._k = similarity_top_k
        self._query_maxlen = query_maxlen
        super().__init__(
            callback_manager=callback_manager,
            object_map=object_map,
            verbose=verbose,
        )

    def _retrieve(
        self,
        query_bundle: QueryBundle,
    ) -> list[NodeWithScore]:
        chunk_scores: list[tuple[Chunk, float]] = self._retriever.text_search(
            query_text=query_bundle.query_str,
            k=self._k,
            query_maxlen=self._query_maxlen,
        )
        return [
            NodeWithScore(node=TextNode(text=c.text, extra_info=c.metadata), score=s)
            for (c, s) in chunk_scores
        ]
