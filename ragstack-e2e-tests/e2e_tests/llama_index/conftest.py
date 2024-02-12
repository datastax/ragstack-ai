import pytest

from llama_index.embeddings import OpenAIEmbedding
from llama_index.llms import OpenAI

from e2e_tests.conftest import (
    get_required_env,
    get_vector_store_handler,
)

from e2e_tests.test_utils.vector_store_handler import (
    VectorStoreImplementation,
)


@pytest.fixture
def openai_llm():
    return "openai", OpenAI(api_key=get_required_env("OPEN_AI_KEY"))


@pytest.fixture
def openai_embedding():
    return "openai", 1536, OpenAIEmbedding(api_key=get_required_env("OPEN_AI_KEY"))


@pytest.fixture
def astra_db():
    handler = get_vector_store_handler(VectorStoreImplementation.ASTRADB)
    context = handler.before_test()
    yield context
    handler.after_test()


@pytest.fixture
def cassandra():
    handler = get_vector_store_handler(VectorStoreImplementation.CASSANDRA)
    context = handler.before_test()
    yield context
    handler.after_test()
