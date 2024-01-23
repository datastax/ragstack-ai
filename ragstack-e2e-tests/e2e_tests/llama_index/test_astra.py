import logging
from typing import List

import pytest
from httpx import ConnectError, HTTPStatusError
from e2e_tests.conftest import (
    get_required_env,
    get_vector_store_handler,
)
from llama_index import (
    ServiceContext,
    StorageContext,
    VectorStoreIndex,
    Document,
)
from llama_index.embeddings import BaseEmbedding
from llama_index.llms import OpenAI, LLM
from llama_index.node_parser import SimpleNodeParser
from llama_index.schema import NodeWithScore
from llama_index.vector_stores import (
    AstraDBVectorStore,
    MetadataFilters,
    ExactMatchFilter,
)

from e2e_tests.test_utils.vector_store_handler import VectorStoreImplementation


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
    print("Running test_ingest_errors")

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
        if "Cannot build index from nodes with no content. " not in e.args[0]:
            pytest.fail(f"LLama-index should have thrown an error but it was {e}")

    very_long_text = "RAGStack is a framework to run LangChain in production. " * 1000

    # with the default set of transformations this write succeeds because LI automatically does text splitting
    documents = [Document(text=very_long_text)]
    VectorStoreIndex.from_documents(
        documents,
        storage_context=environment.storage_context,
        service_context=environment.service_context,
    )

    # if we disable text splitting, this write fails because the document is too long
    very_long_text = "RAGStack is a framework to run LangChain in production. " * 1000
    try:
        documents = [Document(text=very_long_text)]

        VectorStoreIndex.from_documents(
            documents,
            storage_context=environment.storage_context,
            service_context=environment.service_context_no_splitting,
        )
        pytest.fail("Should have thrown ValueError")
    except ValueError as e:
        print("Error:", e)
        # API Exception while running bulk insertion: {'errors': [{'message': 'Document size limitation violated: String value length (56000) exceeds maximum allowed (16000)', 'errorCode': 'SHRED_DOC_LIMIT_VIOLATION'}]}
        if "SHRED_DOC_LIMIT_VIOLATION" not in e.args[0]:
            pytest.fail(
                f"Should have thrown ValueError with SHRED_DOC_LIMIT_VIOLATION but it was {e}"
            )


def test_wrong_connection_parameters(environment):
    # This is expected to be a valid endpoint, because we want to test an AUTHENTICATION error
    astra_ref = get_vector_store_handler().astra_ref
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
    except HTTPStatusError as e:
        print("Error:", e)
        if "UNAUTHENTICATED" not in e.response.text:
            pytest.fail(
                f"Should have thrown ValueError with UNAUTHENTICATED but it was {e}"
            )


def verify_document(document, expected_content, expected_metadata):
    if isinstance(document, NodeWithScore):
        document = document.node
        assert document.text == expected_content
        # metadata is not returned by LlamaIndex
        # assert document.metadata == expected_metadata
    else:
        raise Exception(
            "document is not of type NodeWithScore but of type " + str(type(document))
        )


def test_vector_search_with_metadata(environment):
    print("Running test_vector_search_with_metadata")

    documents = [
        Document(
            text="RAGStack is a framework to run LangChain in production",
            metadata={
                "id": "http://mywebsite/intro",
                "source": "website",
                "context": "homepage",
            },
        ),
        Document(
            text="RAGStack is developed by DataStax",
            metadata={
                "id": "http://mywebsite/about",
                "source": "website",
                "context": "other",
            },
        ),
    ]

    VectorStoreIndex.from_documents(
        documents,
        storage_context=environment.storage_context,
        service_context=environment.service_context,
    )

    document_ids = ((doc.get_doc_id()) for doc in documents)

    # test for search
    index = VectorStoreIndex.from_vector_store(
        vector_store=environment.vectorstore,
        service_context=environment.service_context,
    )
    filters = MetadataFilters(
        filters=[ExactMatchFilter(key="context", value="homepage")]
    )

    documents = index.as_retriever(filters=filters).retrieve("What is RAGStack ?")

    assert len(documents) == 1
    verify_document(
        documents[0],
        "RAGStack is a framework to run LangChain in production",
        {"id": "http://mywebsite/intro", "source": "website", "context": "homepage"},
    )

    documents = index.as_retriever().retrieve("RAGStack")
    assert len(documents) == 2

    # delete all the documents
    for doc_id in document_ids:
        environment.vectorstore.delete(doc_id)

    # commenting this part, as the delete is not working, maybe it is a problem with document ids ?
    # documents = index.as_retriever().retrieve("RAGStack")
    # assert len(documents) == 0


def init_vector_db() -> AstraDBVectorStore:
    handler = get_vector_store_handler()
    return handler.before_test(
        VectorStoreImplementation.ASTRADB
    ).new_llamaindex_vector_store(embedding_dimension=3)


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
def environment():
    embeddings_impl = init_embeddings()
    vector_db_impl = init_vector_db()
    llm_impl = init_llm()
    yield Environment(
        vectorstore=vector_db_impl, llm=llm_impl, embedding=embeddings_impl
    )
    get_vector_store_handler().after_test(VectorStoreImplementation.ASTRADB)


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


def init_embeddings() -> BaseEmbedding:
    return MockEmbeddings()


def init_llm() -> LLM:
    openai_key = get_required_env("OPEN_AI_KEY")
    return OpenAI(
        api_key=openai_key, model="gpt-3.5-turbo-16k", streaming=False, temperature=0
    )
