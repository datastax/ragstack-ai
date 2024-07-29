from __future__ import annotations

import json
import logging
from typing import TYPE_CHECKING

import pytest
from astrapy.api import APIRequestError
from httpx import ConnectError, HTTPStatusError
from langchain.schema.embeddings import Embeddings
from langchain_astradb import AstraDBVectorStore
from langchain_core.documents import Document
from langchain_core.runnables import RunnableConfig

from e2e_tests.conftest import (
    is_astra,
)
from e2e_tests.test_utils import skip_test_due_to_implementation_not_supported
from e2e_tests.test_utils.astradb_vector_store_handler import AstraDBVectorStoreHandler
from e2e_tests.test_utils.vector_store_handler import VectorStoreImplementation

if TYPE_CHECKING:
    from langchain_core.vectorstores import VectorStore

MINIMUM_ACCEPTABLE_SCORE = 0.1


def test_basic_vector_search(vectorstore: AstraDBVectorStore):
    print("Running test_basic_vector_search")
    vectorstore.add_texts(["RAGStack is a framework to run LangChain in production"])
    retriever = vectorstore.as_retriever()
    assert len(retriever.get_relevant_documents("RAGStack")) > 0


def test_ingest_errors(vectorstore: AstraDBVectorStore):
    print("Running test_ingestion")

    empty_text = ""

    try:
        # empty text computes embeddings vector as all zeroes and this is not allowed
        vectorstore.add_texts([empty_text])
    except ValueError as e:
        print("Error:", e)
        # API Exception while running bulk insertion: [{'message': "Failed to insert document with _id 'b388435404254c17b720816ee9e0ddc4': Zero vectors cannot be indexed or queried with cosine similarity"}]  # noqa: E501
        if (
            "Zero and near-zero vectors cannot be indexed "
            "or queried with cosine similarity" not in e.args[0]
        ):
            pytest.fail(
                f"Should have thrown ValueError with Zero vectors cannot be indexed "
                f"or queried with cosine similarity but it was {e}"
            )
    very_long_text = "RAGStack is a framework to run LangChain in production. " * 10_000
    # body is not indexed by default, but metadata is
    vectorstore.add_texts([very_long_text])

    vectorstore.add_documents([Document(page_content=very_long_text, metadata={})])
    try:
        vectorstore.add_documents(
            [
                Document(
                    page_content="some short text", metadata={"text": very_long_text}
                )
            ]
        )
        pytest.fail("Should have thrown ValueError")
    except ValueError as e:
        print("Error:", e)
        # API Exception while running bulk insertion: {'errors': [{'message': 'Document size limitation violated: String value length (56000) exceeds maximum allowed (16000)', 'errorCode': 'SHRED_DOC_LIMIT_VIOLATION'}]}  # noqa: E501
        if "SHRED_DOC_LIMIT_VIOLATION" not in e.args[0]:
            pytest.fail(
                f"Should have thrown ValueError with SHRED_DOC_LIMIT_VIOLATION "
                f"but it was {e}"
            )


def test_wrong_connection_parameters(vectorstore: AstraDBVectorStore):
    try:
        AstraDBVectorStore(
            collection_name="something",
            embedding=MockEmbeddings(),
            token="xxxxx",  # noqa: S106
            # we assume that post 1234 is not open locally
            api_endpoint="https://locahost:1234",
        )
        pytest.fail("Should have thrown exception")
    except ConnectError as e:
        print("Error:", e)

    # This is expected to be a valid endpoint,
    # because we want to test an AUTHENTICATION error
    api_endpoint = vectorstore.api_endpoint
    try:
        print("api_endpoint:", api_endpoint)
        AstraDBVectorStore(
            collection_name="something",
            embedding=MockEmbeddings(),
            token="this-is-a-wrong-token",  # noqa: S106
            api_endpoint=api_endpoint,
        )
        pytest.fail("Should have thrown exception")
    except HTTPStatusError as e:
        print("Error:", e)
        if "401 Unauthorized" not in str(e):
            pytest.fail(
                f"Should have thrown HTTPStatusError with '401 Unauthorized' "
                f"but it was {e}"
            )


def test_basic_metadata_filtering_no_vector(vectorstore: AstraDBVectorStore):
    collection = vectorstore.collection
    vectorstore.add_texts(
        texts=["RAGStack is a framework to run LangChain in production"],
        metadatas=[
            {
                "id": "http://mywebsite",
                "language": "en",
                "source": "website",
                "name": "Homepage",
            }
        ],
    )

    response = collection.find_one(filter={}).get("data").get("document")
    print("Response:", response)
    verify_document(
        response,
        "RAGStack is a framework to run LangChain in production",
        {
            "id": "http://mywebsite",
            "language": "en",
            "source": "website",
            "name": "Homepage",
        },
    )

    response = (
        collection.find_one(filter={"metadata.source": "website"})
        .get("data")
        .get("document")
    )
    print("Response:", response)
    verify_document(
        response,
        "RAGStack is a framework to run LangChain in production",
        {
            "id": "http://mywebsite",
            "language": "en",
            "source": "website",
            "name": "Homepage",
        },
    )

    response = (
        collection.find_one(
            filter={
                "$and": [{"metadata.language": "en"}, {"metadata.source": "website"}]
            }
        )
        .get("data")
        .get("document")
    )
    print("Response:", response)
    verify_document(
        response,
        "RAGStack is a framework to run LangChain in production",
        {
            "id": "http://mywebsite",
            "language": "en",
            "source": "website",
            "name": "Homepage",
        },
    )

    try:
        collection.find_one(filter={"metadata.chunks": {"$invalid": 2}})
        pytest.fail("Should have thrown ValueError")
    except APIRequestError as e:
        print("Error:", e)

        # Parse the error message
        errors = json.loads(e.response.text)

        # Check that the errors field has been properly retrieved
        assert "errors" in errors
        errors = errors["errors"]

        if len(errors) == 1:
            error = errors[0]
            assert error.get("errorCode") == "UNSUPPORTED_FILTER_OPERATION"
        elif len(errors) > 1:
            assert (
                errors[0].get("errorCode") == "UNSUPPORTED_FILTER_OPERATION"
                or errors[1].get("errorCode") == "UNSUPPORTED_FILTER_OPERATION"
            )
        else:
            pytest.fail(
                f"Should have thrown ValueError with UNSUPPORTED_FILTER_OPERATION "
                f"but it was {e}"
            )


def verify_document(document, expected_content, expected_metadata):
    if isinstance(document, Document):
        assert document.page_content == expected_content
        assert document.metadata == expected_metadata
    else:
        assert document.get("content") == expected_content
        assert document.get("metadata") == expected_metadata


def test_vector_search_with_metadata(vectorstore: VectorStore):
    print("Running test_vector_search_with_metadata")

    document_ids = vectorstore.add_texts(
        texts=[
            "RAGStack is a framework to run LangChain in production",
            "RAGStack is developed by DataStax",
        ],
        metadatas=[
            {
                "id": "http://mywebsite/intro",
                "source": "website",
                "context": "homepage",
            },
            {"id": "http://mywebsite/about", "source": "website", "context": "other"},
        ],
    )

    # test for search

    documents = vectorstore.search(
        "RAGStack", "similarity", filter={"context": "homepage"}
    )
    assert len(documents) == 1
    verify_document(
        documents[0],
        "RAGStack is a framework to run LangChain in production",
        {"id": "http://mywebsite/intro", "source": "website", "context": "homepage"},
    )

    documents = vectorstore.search("RAGStack", "similarity")
    assert len(documents) == 2  # noqa: PLR2004

    documents = vectorstore.search(
        "RAGStack", "similarity", filter={"context": "homepage"}
    )
    assert len(documents) == 1
    verify_document(
        documents[0],
        "RAGStack is a framework to run LangChain in production",
        {"id": "http://mywebsite/intro", "source": "website", "context": "homepage"},
    )

    documents = vectorstore.search("RAGStack", "mmr")
    assert len(documents) == 2  # noqa: PLR2004

    documents = vectorstore.search("RAGStack", "mmr", filter={"context": "homepage"})
    assert len(documents) == 1
    verify_document(
        documents[0],
        "RAGStack is a framework to run LangChain in production",
        {"id": "http://mywebsite/intro", "source": "website", "context": "homepage"},
    )

    documents = vectorstore.similarity_search(
        "RAGStack", filter={"context": "homepage"}
    )
    assert len(documents) == 1
    verify_document(
        documents[0],
        "RAGStack is a framework to run LangChain in production",
        {"id": "http://mywebsite/intro", "source": "website", "context": "homepage"},
    )

    documents = vectorstore.similarity_search(
        "RAGStack", distance_threshold=0.9, filter={"context": "homepage"}
    )
    assert len(documents) == 1
    verify_document(
        documents[0],
        "RAGStack is a framework to run LangChain in production",
        {"id": "http://mywebsite/intro", "source": "website", "context": "homepage"},
    )

    # test for similarity_search_with_score

    documents_with_score = vectorstore.similarity_search_with_score(
        "RAGStack", filter={"context": "homepage"}
    )
    assert len(documents_with_score) == 1
    # th elements are Tuple(document, score)
    score = documents_with_score[0][1]
    assert score > MINIMUM_ACCEPTABLE_SCORE

    verify_document(
        documents_with_score[0][0],
        "RAGStack is a framework to run LangChain in production",
        {"id": "http://mywebsite/intro", "source": "website", "context": "homepage"},
    )

    # test for similarity_search_with_relevance_scores

    documents_with_score = vectorstore.similarity_search_with_relevance_scores(
        query="RAGStack", k=1, filter={"context": "homepage"}
    )
    assert len(documents_with_score) == 1
    # the elements are Tuple(document, score)
    score = documents_with_score[0][1]
    assert score > MINIMUM_ACCEPTABLE_SCORE

    verify_document(
        documents_with_score[0][0],
        "RAGStack is a framework to run LangChain in production",
        {"id": "http://mywebsite/intro", "source": "website", "context": "homepage"},
    )

    documents_with_score = vectorstore.similarity_search_with_relevance_scores(
        query="RAGStack", k=1
    )
    assert len(documents_with_score) == 1
    # the elements are Tuple(document, score)
    score = documents_with_score[0][1]
    assert score > MINIMUM_ACCEPTABLE_SCORE

    # test for max_marginal_relevance_search_by_vector

    embeddings: Embeddings = vectorstore.embeddings
    vector = embeddings.embed_query("RAGStack")

    documents = vectorstore.max_marginal_relevance_search_by_vector(
        embedding=vector, k=1
    )
    assert len(documents) == 1

    documents = vectorstore.max_marginal_relevance_search_by_vector(
        embedding=vector, k=1, filter={"context": "none"}
    )
    assert len(documents) == 0

    documents = vectorstore.max_marginal_relevance_search_by_vector(
        embedding=vector, k=1, filter={"context": "homepage"}
    )
    assert len(documents) == 1

    verify_document(
        documents[0],
        "RAGStack is a framework to run LangChain in production",
        {"id": "http://mywebsite/intro", "source": "website", "context": "homepage"},
    )

    documents = vectorstore.similarity_search_by_vector(embedding=vector, k=1)
    assert len(documents) == 1

    documents = vectorstore.similarity_search_by_vector(embedding=vector, k=2)
    assert len(documents) == 2  # noqa: PLR2004

    documents = vectorstore.similarity_search_by_vector(
        embedding=vector, k=1, filter={"context": "none"}
    )
    assert len(documents) == 0

    documents = vectorstore.similarity_search_by_vector(
        embedding=vector, k=1, filter={"context": "homepage"}
    )
    assert len(documents) == 1

    verify_document(
        documents[0],
        "RAGStack is a framework to run LangChain in production",
        {"id": "http://mywebsite/intro", "source": "website", "context": "homepage"},
    )

    # Use Retriever

    retriever = vectorstore.as_retriever(
        search_kwargs={"filter": {"context": "homepage"}}
    )
    documents = retriever.get_relevant_documents("RAGStack")
    assert len(documents) == 1
    verify_document(
        documents[0],
        "RAGStack is a framework to run LangChain in production",
        {"id": "http://mywebsite/intro", "source": "website", "context": "homepage"},
    )

    retriever = vectorstore.as_retriever()
    documents = retriever.get_relevant_documents("RAGStack")
    assert len(documents) == 2  # noqa: PLR2004

    documents = retriever.invoke("RAGStack", RunnableConfig(tags=["custom_retriever"]))
    assert len(documents) == 2  # noqa: PLR2004

    retriever = vectorstore.as_retriever(search_kwargs={"k": 1})
    documents = retriever.get_relevant_documents("RAGStack")
    assert len(documents) == 1

    retriever = vectorstore.as_retriever(search_kwargs={"k": 2})
    documents = retriever.get_relevant_documents("RAGStack")
    assert len(documents) == 2  # noqa: PLR2004

    # delete all the documents
    vectorstore.delete(document_ids)

    documents = vectorstore.search("RAGStack", "similarity")
    assert len(documents) == 0


@pytest.mark.skip()
def test_stress_astra():
    handler = AstraDBVectorStoreHandler(VectorStoreImplementation.ASTRADB)
    while True:
        context = handler.before_test()
        logging.info("mocking test")
        vstore = context.new_langchain_vector_store(embedding=MockEmbeddings())
        vstore.add_texts(["hello world, im a document"])
        result = vstore.search("hello", search_type="similarity")
        print(str(result))
        logging.info("test finished")
        handler.after_test()


class MockEmbeddings(Embeddings):
    def __init__(self):
        self.embedded_documents = None
        self.embedded_query = None

    @staticmethod
    def mock_embedding(text: str):
        return [len(text) / 2, len(text) / 5, len(text) / 10]

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        self.embedded_documents = texts
        return [self.mock_embedding(text) for text in texts]

    def embed_query(self, text: str) -> list[float]:
        self.embedded_query = text
        return self.mock_embedding(text)


@pytest.fixture()
def vectorstore() -> AstraDBVectorStore:
    if not is_astra:
        skip_test_due_to_implementation_not_supported("astradb")
    handler = AstraDBVectorStoreHandler(VectorStoreImplementation.ASTRADB)
    context = handler.before_test()
    vector_db = context.new_langchain_vector_store(embedding=MockEmbeddings())
    yield vector_db
    handler.after_test()
