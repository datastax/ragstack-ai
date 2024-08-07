from __future__ import annotations

from typing import TYPE_CHECKING, List, Optional, Tuple

from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
from typing_extensions import override

if TYPE_CHECKING:
    from langchain_core.callbacks.manager import (
        AsyncCallbackManagerForRetrieverRun,
        CallbackManagerForRetrieverRun,
    )
    from ragstack_colbert import Chunk
    from ragstack_colbert.base_retriever import BaseRetriever as ColbertBaseRetriever


class ColbertRetriever(BaseRetriever):
    """Chain for langchain retrieve using ColBERT vector store.

    Example:
        .. code-block:: python

        from langchain.chains import RetrievalQA
        from langchain_openai import AzureChatOpenAI

        llm = AzureChatOpenAI()
        retriever = ColbertLCRetriever(colbert_retriever, k=5)
        qa = RetrievalQA.from_chain_type(
            llm=llm,
            chain_type="stuff",
            retriever=retriever
        )
        qa.run("what happened on June 4th?")
    """

    retriever: ColbertBaseRetriever
    k: int
    query_maxlen: Optional[int]

    def __init__(
        self,
        retriever: ColbertBaseRetriever,
        k: int = 5,
        query_maxlen: Optional[int] = None,
    ):
        super().__init__()
        self.retriever = retriever
        self.k = k
        self.query_maxlen = query_maxlen

    @override
    def _get_relevant_documents(
        self,
        query: str,
        *,
        run_manager: CallbackManagerForRetrieverRun,
    ) -> List[Document]:
        """Get documents relevant to a query.

        Args:
            query: String to find relevant documents for
            run_manager: The callbacks handler to use
        Returns:
            List of relevant documents
        """
        chunk_scores: List[Tuple[Chunk, float]] = self.retriever.text_search(
            query_text=query, k=self.k, query_maxlen=self.query_maxlen
        )

        return [
            Document(page_content=c.text, metadata=c.metadata)
            for (c, _) in chunk_scores
        ]

    @override
    async def _aget_relevant_documents(
        self,
        query: str,
        *,
        run_manager: AsyncCallbackManagerForRetrieverRun,
    ) -> List[Document]:
        """Asynchronously get documents relevant to a query.

        Args:
            query: String to find relevant documents for
            run_manager: The callbacks handler to use
        Returns:
            List of relevant documents
        """
        chunk_scores: List[Tuple[Chunk, float]] = await self.retriever.atext_search(
            query_text=query, k=self.k, query_maxlen=self.query_maxlen
        )

        return [
            Document(page_content=c.text, metadata=c.metadata)
            for (c, _) in chunk_scores
        ]
