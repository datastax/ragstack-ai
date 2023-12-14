import logging
from typing import List

from astrapy.db import AstraDB as LibAstraDB
import pytest
from httpx import ConnectError
from e2e_tests.conftest import get_required_env, get_astra_ref
from llama_index import (
    OpenAIEmbedding,
    ServiceContext,
    StorageContext,
    VectorStoreIndex,
    Document,
)
from llama_index.embeddings import BaseEmbedding
from llama_index.llms import OpenAI, LLM
from llama_index.vector_stores import AstraDBVectorStore


def test_basic_vector_search(environment):
    print("Running test_basic_vector_search")
    documents = [
        Document(text="RAGStack is a framework to run LangChain in production")
    ]

    index = VectorStoreIndex.from_documents(
        documents,
        storage_context=environment.storage_context,
        service_context=environment.service_context,
    )

    # Verify that the document is in the vector store
    retriever = index.as_retriever()
    assert len(retriever.retrieve("RAGStack")) > 0


def test_ingest_errors(environment):
    print("Running test_ingestion")

    empty_text = ""

    try:
        # empty text computes embeddings vector as all zeroes and this is not allowed
        documents = [Document(text=empty_text)]

        VectorStoreIndex.from_documents(
            documents,
            storage_context=environment.storage_context,
            service_context=environment.service_context,
        )
    except ValueError as e:
        print("Error:", e)
        # API Exception while running bulk insertion: [{'message': "Failed to insert document with _id 'b388435404254c17b720816ee9e0ddc4': Zero vectors cannot be indexed or queried with cosine similarity"}]
        if (
            "Zero vectors cannot be indexed or queried with cosine similarity"
            not in e.args[0]
        ):
            pytest.fail(
                f"Should have thrown ValueError with Zero vectors cannot be indexed or queried with cosine similarity but it was {e}"
            )

    very_long_text = "RAGStack is a framework to run LangChain in production. " * 1000
    try:
        documents = [Document(text=very_long_text)]

        VectorStoreIndex.from_documents(
            documents,
            storage_context=environment.storage_context,
            service_context=environment.service_context,
        )
        pytest.fail("Should have thrown ValueError")
    except ValueError as e:
        print("Error:", e)
        # API Exception while running bulk insertion: {'errors': [{'message': 'Document size limitation violated: String value length (56000) exceeds maximum allowed (16000)', 'errorCode': 'SHRED_DOC_LIMIT_VIOLATION'}]}
        if "SHRED_DOC_LIMIT_VIOLATION" not in e.args[0]:
            pytest.fail(
                f"Should have thrown ValueError with SHRED_DOC_LIMIT_VIOLATION but it was {e}"
            )

    very_very_long_text = (
        "RAGStack is a framework to run LangChain in production. " * 10000
    )
    try:
        documents = [Document(text=very_very_long_text)]

        VectorStoreIndex.from_documents(
            documents,
            storage_context=environment.storage_context,
            service_context=environment.service_context,
        )
        pytest.fail("Should have thrown ValueError")
    except ValueError as e:
        print("Error:", e)
        # API Exception while running bulk insertion: {'errors': [{'message': 'Document size limitation violated: String value length (560000) exceeds maximum allowed (16000)', 'errorCode': 'SHRED_DOC_LIMIT_VIOLATION'}]}
        if "SHRED_DOC_LIMIT_VIOLATION" not in e.args[0]:
            pytest.fail(
                f"Should have thrown ValueError with SHRED_DOC_LIMIT_VIOLATION but it was {e}"
            )


def test_wrong_connection_parameters():
    # This is expected to be a valid endpoint, because we want to test an AUTHENTICATION error
    astra_ref = get_astra_ref()
    api_endpoint = astra_ref.api_endpoint

    try:
        AstraDBVectorStore(
            token="xxxxx",
            # we assume that post 1234 is not open locally
            api_endpoint="https://locahost:1234",
            collection_name="something",
            embedding_dimension=1536,
        )
        pytest.fail("Should have thrown exception")
    except ConnectError as e:
        print("Error:", e)
        pass

    try:
        print("api_endpoint:", api_endpoint)
        AstraDBVectorStore(
            token="this-is-a-wrong-token",
            api_endpoint=api_endpoint,
            collection_name="something",
            embedding_dimension=1536,
        )
        pytest.fail("Should have thrown exception")
    except ValueError as e:
        print("Error:", e)
        if "AUTHENTICATION ERROR" not in e.args[0]:
            pytest.fail(
                f"Should have thrown ValueError with AUTHENTICATION ERROR but it was {e}"
            )


def init_vector_db() -> AstraDBVectorStore:
    astra_ref = get_astra_ref()
    collection = astra_ref.collection
    token = astra_ref.token
    api_endpoint = astra_ref.api_endpoint

    raw_client = LibAstraDB(api_endpoint=api_endpoint, token=token)
    collections = raw_client.get_collections().get("status").get("collections")
    logging.info(f"Existing collections: {collections}")
    for collection_info in collections:
        try:
            logging.info(f"Deleting collection: {collection_info}")
            raw_client.delete_collection(collection_info)
        except Exception as e:
            logging.error(f"Error while deleting collection {collection_info}: {e}")

    vector_db = AstraDBVectorStore(
        token=token,
        api_endpoint=api_endpoint,
        collection_name=collection,
        embedding_dimension=3,
    )

    return vector_db


class Environment:
    def __init__(
        self, vectorstore: AstraDBVectorStore, llm: LLM, embedding: BaseEmbedding
    ):
        self.vectorstore = vectorstore
        self.llm = llm
        self.embedding = embedding
        self.service_context = ServiceContext.from_defaults()
        self.service_context = ServiceContext.from_defaults(
            embed_model=self.embedding, llm=self.llm
        )
        self.storage_context = StorageContext.from_defaults(vector_store=vectorstore)


@pytest.fixture
def environment():
    embeddings_impl = init_embeddings()
    vector_db_impl = init_vector_db()
    llm_impl = init_llm()
    yield Environment(
        vectorstore=vector_db_impl, llm=llm_impl, embedding=embeddings_impl
    )
    close_vector_db(vector_db_impl)


def close_vector_db(vector_store: AstraDBVectorStore):
    vector_store._astra_db.delete_collection(
        vector_store._astra_db_collection.collection_name
    )


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
        logging.info("mock_embedding for " + text + " : " + str(res))
        return res


def init_embeddings() -> BaseEmbedding:
    return MockEmbeddings()


def init_llm() -> LLM:
    openai_key = get_required_env("OPEN_AI_KEY")
    return OpenAI(
        api_key=openai_key, model="gpt-3.5-turbo-16k", streaming=False, temperature=0
    )
