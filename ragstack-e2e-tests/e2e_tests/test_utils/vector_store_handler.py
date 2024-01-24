from enum import Enum
from typing import List

import pytest
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.vectorstores import VectorStore as LangChainVectorStore
from llama_index.vector_stores.types import VectorStore as LLamaIndexVectorStore


class VectorStoreImplementation(Enum):
    ASTRADB = "astradb"
    CASSANDRA = "cassandra"


class EnhancedLangChainVectorStore(LangChainVectorStore):
    def put_document(
        self, doc_id: str, document: str, metadata: dict, vector: List[float]
    ) -> None:
        raise NotImplementedError()

    def search_documents(self, vector: List[float], limit: int) -> List[str]:
        raise NotImplementedError()


class EnhancedLlamaIndexVectorStore(LLamaIndexVectorStore):
    def put_document(
        self, doc_id: str, document: str, metadata: dict, vector: List[float]
    ) -> None:
        raise NotImplementedError()

    def search(self, vector: List[float], limit: int) -> List[str]:
        raise NotImplementedError()


class VectorStoreTestContext:
    def new_langchain_vector_store(self, **kwargs) -> EnhancedLangChainVectorStore:
        raise NotImplementedError()

    def new_langchain_chat_memory(self, **kwargs) -> BaseChatMessageHistory:
        raise NotImplementedError()

    def new_llamaindex_vector_store(self, **kwargs) -> EnhancedLlamaIndexVectorStore:
        raise NotImplementedError()


class VectorStoreHandler:
    def __init__(self, supported_implementations: List[VectorStoreImplementation]):
        self.supported_implementations = supported_implementations

    def before_test(
        self, implementation: VectorStoreImplementation
    ) -> VectorStoreTestContext:
        if implementation not in self.supported_implementations:
            pytest.skip(f"Skipping test because {implementation} is not configured")

    def after_test(self, implementation: VectorStoreImplementation):
        pass
