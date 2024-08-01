from __future__ import annotations

from abc import ABC, abstractmethod
from enum import Enum
from typing import TYPE_CHECKING

from langchain_core.vectorstores import VectorStore as LangChainVectorStore

from e2e_tests.test_utils import skip_test_due_to_implementation_not_supported

if TYPE_CHECKING:
    from langchain_core.chat_history import BaseChatMessageHistory


class VectorStoreImplementation(Enum):
    ASTRADB = "astradb"
    CASSANDRA = "cassandra"


class EnhancedVectorStore(ABC):
    @abstractmethod
    def put_document(
        self, doc_id: str, document: str, metadata: dict, vector: list[float]
    ) -> None:
        """Put a document"""

    @abstractmethod
    def search_documents(self, vector: list[float], limit: int) -> list[str]:
        """Search documents"""


class EnhancedLangChainVectorStore(LangChainVectorStore, EnhancedVectorStore, ABC):
    """Enhanced LangChain vector store"""


# we can't use the VectorStore type here from llama_index.vector_stores.types
# because AstraDBVectorStore is based on BasePydanticVectorStore.
class EnhancedLlamaIndexVectorStore(EnhancedVectorStore, ABC):
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
        implementation: VectorStoreImplementation,
        supported_implementations: list[VectorStoreImplementation],
    ):
        self.implementation = implementation
        self.supported_implementations = supported_implementations

    def check_implementation(self) -> None:
        if self.implementation not in self.supported_implementations:
            skip_test_due_to_implementation_not_supported(self.implementation.value)

    @abstractmethod
    def before_test(self) -> VectorStoreTestContext:
        pass

    def after_test(self) -> None:
        return
