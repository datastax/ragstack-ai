from typing import Any, List

from langchain_core.callbacks.manager import CallbackManagerForRetrieverRun
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever as LangChainBaseRetriever
from pydantic import Field

from ragstack_colbert.base_retriever import BaseRetriever

class ColbertLCRetriever(LangChainBaseRetriever):
    """Chain for langchain retrieve using ColBERT vector store.

    Example:
        .. code-block:: python

        from langchain.chains import RetrievalQA
        from langchain_openai import AzureChatOpenAI

        llm = AzureChatOpenAI()
        retriever = ColbertLCRetriever(colbert_retriever, k=5)
        qa = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=retriever)
        qa.run("what happened on June 4th?")
    """

    retriever: BaseRetriever = Field(default=None)
    kwargs: dict = {}
    k: int = 10

    class Config:
        """Configuration for this pydantic object."""

        arbitrary_types_allowed = True

    def __init__(
        self, retriever: BaseRetriever, k: int = 10, **kwargs: Any
    ):
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

        output: List[Document] = []
        for chunk in chunks:
            page_content = chunk.data.text
            metadata=chunk.data.metadata
            metadata["rank"] = chunk.rank
            output.append(Document(page_content=page_content, metadata=metadata))

        return output
