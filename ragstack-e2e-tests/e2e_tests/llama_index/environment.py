import logging
from typing import List

import pytest

from e2e_tests.conftest import (
    get_required_env,
    is_astra,
)
from llama_index import (
    ServiceContext,
    StorageContext,
)
from llama_index.embeddings import BaseEmbedding
from llama_index.llms import OpenAI, LLM
from llama_index.node_parser import SimpleNodeParser
from llama_index.vector_stores import AstraDBVectorStore

from e2e_tests.test_utils import skip_test_due_to_implementation_not_supported
from e2e_tests.test_utils.astradb_vector_store_handler import AstraDBVectorStoreHandler
from e2e_tests.test_utils.vector_store_handler import VectorStoreImplementation


class Environment:
    def __init__(
        self, vectorstore: AstraDBVectorStore, llm: LLM, embedding: BaseEmbedding
    ):
        self.vectorstore = vectorstore
        self.llm = llm
        self.embedding = embedding
        self.service_context = ServiceContext.from_defaults(
            embed_model=self.embedding, llm=self.llm
        )
        basic_node_parser = SimpleNodeParser.from_defaults(
            chunk_size=100000000, include_prev_next_rel=False, include_metadata=True
        )
        self.service_context_no_splitting = ServiceContext.from_defaults(
            embed_model=self.embedding,
            llm=self.llm,
            transformations=[basic_node_parser],
        )
        self.storage_context = StorageContext.from_defaults(vector_store=vectorstore)


@pytest.fixture
def environment() -> Environment:
    if not is_astra:
        skip_test_due_to_implementation_not_supported("astradb")
    embeddings = MockEmbeddings()
    handler = AstraDBVectorStoreHandler(VectorStoreImplementation.ASTRADB)
    vector_db = handler.before_test().new_llamaindex_vector_store(embedding_dimension=3)
    llm = OpenAI(
        api_key=get_required_env("OPEN_AI_KEY"),
        model="gpt-3.5-turbo-16k",
        streaming=False,
        temperature=0,
    )
    yield Environment(vectorstore=vector_db, llm=llm, embedding=embeddings)
    handler.after_test()


class MockEmbeddings(BaseEmbedding):
    def _get_query_embedding(self, query: str) -> List[float]:
        return self.mock_embedding(query)

    async def _aget_query_embedding(self, query: str) -> List[float]:
        return self.mock_embedding(query)

    def _get_text_embedding(self, text: str) -> List[float]:
        return self.mock_embedding(text)

    @staticmethod
    def mock_embedding(text: str):
        res = [len(text) / 2, len(text) / 5, len(text) / 10]
        logging.debug("mock_embedding for " + text + " : " + str(res))
        return res
