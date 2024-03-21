from pydantic import Field
from typing import List

from langchain_core.callbacks.manager import CallbackManagerForRetrieverRun
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
from ..vector_store import ColBERTVectorStoreRetriever


class ColBERTVectorStoreLangChainRetriever(BaseRetriever):
    """Chain for langchain retrieve using ColBERT vector store.

    Example:
        .. code-block:: python

        from langchain.chains import RetrievalQA
        from langchain_openai import AzureChatOpenAI

        llm = AzureChatOpenAI()
        retriever = ColBERTVectorStoreLangChainRetriever(colbertCassandraRetriever, k=2)
        qa = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=retriever)
        qa.run("what happened on June 4th?")
    """
    retriever: ColBERTVectorStoreRetriever = Field(default=None)
    kwargs: dict = {}
    k: int = 10

    class Config:
        """Configuration for this pydantic object."""

        arbitrary_types_allowed = True

    def __init__(self, retriever: ColBERTVectorStoreRetriever, k: int = 10, **kwargs):
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
        answers = self.retriever.retrieve(query, self.k)
        return [
            Document(metadata={"id": d.id, "rank": d.rank}, page_content=d.body)
            for d in answers
        ]
