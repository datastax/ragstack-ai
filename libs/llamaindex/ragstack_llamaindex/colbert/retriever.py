
from llama_index.core.schema import NodeWithScore, QueryBundle, TextNode
from llama_index.core.callbacks.base import CallbackManager
from llama_index.core.retrievers import BaseRetriever as LlamaIndexBaseRetriever
from llama_index.core.constants import DEFAULT_SIMILARITY_TOP_K
from typing import Any, List, Optional

from ragstack_colbert.base_retriever import BaseRetriever


class ColbertLIRetriever(LlamaIndexBaseRetriever):
    """ColBERT vector store retriever.

    Args:
        retriever (BaseRetriever): vector store index.
        similarity_top_k (int): number of top k results to return.
    """

    def __init__(
        self,
        retriever: BaseRetriever,
        similarity_top_k: int = DEFAULT_SIMILARITY_TOP_K,
        callback_manager: Optional[CallbackManager] = None,
        object_map: Optional[dict] = None,
        verbose: bool = False,
        query_maxlen: int = -1,
        **kwargs: Any,
    ) -> None:
        """Initialize params."""
        self._retriever = retriever
        self._similarity_top_k = similarity_top_k
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
        nodes: List[NodeWithScore] = []

        chunks = self._retriever.retrieve(query_bundle.query_str, k=self._similarity_top_k, query_maxlen=self._query_maxlen)
        for chunk in chunks:
            text = chunk.data.text
            metadata=chunk.data.metadata
            metadata["rank"] = chunk.rank

            node = TextNode(text=text, metadata=metadata)
            nodes.append(NodeWithScore(node=node, score=chunk.score))
        return nodes
