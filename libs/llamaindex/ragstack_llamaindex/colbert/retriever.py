from typing import Any, List, Optional, Tuple

from llama_index.core.callbacks.base import CallbackManager
from llama_index.core.constants import DEFAULT_SIMILARITY_TOP_K
from llama_index.core.retrievers import BaseRetriever
from llama_index.core.schema import NodeWithScore, QueryBundle, TextNode
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
    _query_maxlen: Optional[int]

    def __init__(
        self,
        retriever: ColbertBaseRetriever,
        similarity_top_k: int = DEFAULT_SIMILARITY_TOP_K,
        callback_manager: Optional[CallbackManager] = None,
        object_map: Optional[dict] = None,
        verbose: bool = False,
        query_maxlen: int = -1,
        **kwargs: Any,
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
    ) -> List[NodeWithScore]:
        chunk_scores: List[Tuple[Chunk, float]] = self._retriever.text_search(
            query_text=query_bundle.query_str,
            k=self._k,
            query_maxlen=self._query_maxlen,
        )
        return [
            NodeWithScore(node=TextNode(text=c.text, metadata=c.metadata), score=s)
            for (c, s) in chunk_scores
        ]
