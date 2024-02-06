from abc import ABC, abstractmethod
from enum import Enum
from typing import List

from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.vectorstores import VectorStore as LangChainVectorStore
from llama_index.vector_stores.types import VectorStore as LLamaIndexVectorStore

from e2e_tests.test_utils import skip_test_due_to_implementation_not_supported


class VectorStoreImplementation(Enum):
    ASTRADB = "astradb"
    CASSANDRA = "cassandra"
    UNINITIALIZED = ""


class EnhancedVectorStore(ABC):
    @abstractmethod
    def put_document(
        self, doc_id: str, document: str, metadata: dict, vector: List[float]
    ) -> None:
        """Put a document"""

    @abstractmethod
    def search_documents(self, vector: List[float], limit: int) -> List[str]:
        """Search documents"""


class EnhancedLangChainVectorStore(LangChainVectorStore, EnhancedVectorStore, ABC):
    """Enhanced LangChain vector store"""


class EnhancedLlamaIndexVectorStore(LLamaIndexVectorStore, EnhancedVectorStore, ABC):
    """Enhanced Llama-Index vector store"""


class VectorStoreTestContext(ABC):
    @abstractmethod
    def new_langchain_vector_store(self, **kwargs) -> EnhancedLangChainVectorStore:
        """Create a new LangChain VectorStore"""

    @abstractmethod
    def new_langchain_chat_memory(self, **kwargs) -> BaseChatMessageHistory:
        """Create a new LangChain Chat Memory"""

    @abstractmethod
    def new_llamaindex_vector_store(self, **kwargs) -> EnhancedLlamaIndexVectorStore:
        """Create a new LLama-Index VectorStore"""


class VectorStoreHandler(ABC):
    def __init__(
        self,
        supported_implementations: List[VectorStoreImplementation],
    ):
        self.implementation = VectorStoreImplementation.UNINITIALIZED
        self.supported_implementations = supported_implementations

    def check_implementation(self):
        if self.implementation == VectorStoreImplementation.UNINITIALIZED:
            raise ValueError("Implementation not initialized")

        if self.implementation not in self.supported_implementations:
            skip_test_due_to_implementation_not_supported(self.implementation.value)

    @abstractmethod
    def before_test(
        self, implementation: VectorStoreImplementation
    ) -> VectorStoreTestContext:
        pass

    def after_test(self):
        pass
