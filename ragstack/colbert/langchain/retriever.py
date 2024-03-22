from pydantic import Field
from typing import List

from langchain_core.callbacks.manager import CallbackManagerForRetrieverRun
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
from ..vector_store import ColbertVectorStoreRetriever


class ColbertVectorStoreLangChainRetriever(BaseRetriever):
    """Chain for langchain retrieve using ColBERT vector store.

    Example:
        .. code-block:: python

        from langchain.chains import RetrievalQA
        from langchain_openai import AzureChatOpenAI

        llm = AzureChatOpenAI()
        retriever = ColbertVectorStoreLangChainRetriever(colbertCassandraRetriever, k=5)
        qa = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=retriever)
        qa.run("what happened on June 4th?")
    """
    retriever: ColbertVectorStoreRetriever = Field(default=None)
    kwargs: dict = {}
    k: int = 10

    class Config:
        """Configuration for this pydantic object."""

        arbitrary_types_allowed = True

    def __init__(self, retriever: ColbertVectorStoreRetriever, k: int = 10, **kwargs):
        super().__init__(retriever=retriever, k=k, **kwargs)
        self.retriever = retriever
        self.k = k

    def _get_relevant_documents(
        self,
        query: str,
        *,
        run_manager: CallbackManagerForRetrieverRun,  # noqa
    ) -> List[Document]:
        """Get documents relevant to a query."""
        chunks = self.retriever.retrieve(query, self.k)
        return [
            Document(metadata={"id": c.doc_id, "rank": c.rank}, page_content=c.text)
            for c in chunks
        ]
