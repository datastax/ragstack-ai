import json
import logging
from typing import List

from astrapy.db import AstraDB as LibAstraDB
import pytest
from httpx import ConnectError

from langchain.schema.embeddings import Embeddings
from langchain.vectorstores import AstraDB
from langchain.chat_models import ChatOpenAI
from langchain.schema.language_model import BaseLanguageModel
from e2e_tests.conftest import get_required_env, get_astra_ref
from langchain_core.documents import Document
from langchain_core.runnables import RunnableConfig
from langchain_core.vectorstores import VectorStore


def test_basic_vector_search(environment):
    print("Running test_basic_vector_search")
    vectorstore = environment.vectorstore
    vectorstore.add_texts(["RAGStack is a framework to run LangChain in production"])
    retriever = vectorstore.as_retriever()
    assert len(retriever.get_relevant_documents("RAGStack")) > 0


def test_ingest_errors(environment):
    print("Running test_ingestion")
    vectorstore = environment.vectorstore

    empty_text = ""

    try:
        # empty text computes embeddings vector as all zeroes and this is not allowed
        vectorstore.add_texts([empty_text])
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

    # with Llama Index this is not an error because the document is automatically split into chunks (nodes)
    very_long_text = "RAGStack is a framework to run LangChain in production. " * 1000
    vectorstore.add_texts([very_long_text])

    try:
        vectorstore.add_texts([very_long_text])
        pytest.fail("Should have thrown ValueError")
    except ValueError as e:
        print("Error:", e)
        # API Exception while running bulk insertion: {'errors': [{'message': 'Document size limitation violated: String value length (56000) exceeds maximum allowed (16000)', 'errorCode': 'SHRED_DOC_LIMIT_VIOLATION'}]}
        if "SHRED_DOC_LIMIT_VIOLATION" not in e.args[0]:
            pytest.fail(
                f"Should have thrown ValueError with SHRED_DOC_LIMIT_VIOLATION but it was {e}"
            )


def test_wrong_connection_parameters():
    # This is expected to be a valid endpoint, because we want to test an AUTHENTICATION error
    astra_ref = get_astra_ref()
    api_endpoint = astra_ref.api_endpoint

    try:
        AstraDB(
            collection_name="something",
            embedding=init_embeddings(),
            token="xxxxx",
            # we assume that post 1234 is not open locally
            api_endpoint="https://locahost:1234",
        )
        pytest.fail("Should have thrown exception")
    except ConnectError as e:
        print("Error:", e)
        pass

    try:
        print("api_endpoint:", api_endpoint)
        AstraDB(
            collection_name="something",
            embedding=init_embeddings(),
            token="this-is-a-wrong-token",
            api_endpoint=api_endpoint,
        )
        pytest.fail("Should have thrown exception")
    except ValueError as e:
        print("Error:", e)
        if "AUTHENTICATION ERROR" not in e.args[0]:
            pytest.fail(
                f"Should have thrown ValueError with AUTHENTICATION ERROR but it was {e}"
            )


def test_basic_metadata_filtering_no_vector(environment):
    print("Running test_basic_metadata_filtering_no_vector")

    vectorstore = environment.vectorstore
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
        collection.find_one(filter={"metadata.chunks": {"$gt": 2}})
        pytest.fail("Should have thrown ValueError")
    except ValueError as e:
        print("Error:", e)
        if "UNSUPPORTED_FILTER_OPERATION" not in e.args[0]:
            pytest.fail(
                f"Should have thrown ValueError with UNSUPPORTED_FILTER_OPERATION but it was {e}"  # noqa: E501
            )

        # This looks very ugly, but it's the only way to get the error message
        # reference ticket on Astrapy https://github.com/datastax/astrapy/issues/126
        errors = json.loads(e.args[0])
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
                f"Should have thrown ValueError with UNSUPPORTED_FILTER_OPERATION but it was {e}"  # noqa: E501
            )


def verify_document(document, expected_content, expected_metadata):
    if isinstance(document, Document):
        assert document.page_content == expected_content
        assert document.metadata == expected_metadata
    else:
        assert document.get("content") == expected_content
        assert document.get("metadata") == expected_metadata


def test_vector_search_with_metadata(environment):
    print("Running test_vector_search_with_metadata")

    vectorstore: VectorStore = environment.vectorstore

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
    assert len(documents) == 2

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
    assert len(documents) == 2

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
    assert score > 0.1

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
    assert score > 0.1

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
    assert score > 0.1

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
    assert len(documents) == 2

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
    assert len(documents) == 2

    documents = retriever.invoke("RAGStack", RunnableConfig(tags=["custom_retriever"]))
    assert len(documents) == 2

    retriever = vectorstore.as_retriever(search_kwargs={"k": 1})
    documents = retriever.get_relevant_documents("RAGStack")
    assert len(documents) == 1

    retriever = vectorstore.as_retriever(search_kwargs={"k": 2})
    documents = retriever.get_relevant_documents("RAGStack")
    assert len(documents) == 2

    # delete all the documents
    vectorstore.delete(document_ids)

    documents = vectorstore.search("RAGStack", "similarity")
    assert len(documents) == 0


class MockEmbeddings(Embeddings):
    def __init__(self):
        self.embedded_documents = None
        self.embedded_query = None

    @staticmethod
    def mock_embedding(text: str):
        return [len(text) / 2, len(text) / 5, len(text) / 10]

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        self.embedded_documents = texts
        return [self.mock_embedding(text) for text in texts]

    def embed_query(self, text: str) -> List[float]:
        self.embedded_query = text
        return self.mock_embedding(text)


def init_vector_db(embedding: Embeddings) -> VectorStore:
    astra_ref = get_astra_ref()
    collection = astra_ref.collection
    token = astra_ref.token
    api_endpoint = astra_ref.api_endpoint

    raw_client = LibAstraDB(api_endpoint=api_endpoint, token=token)
    collections = raw_client.get_collections().get("status").get("collections")
    logging.info(f"Existing collections: {collections}")
    for collection_info in collections:
        logging.info(f"Deleting collection: {collection_info}")
        raw_client.delete_collection(collection_info)

    vector_db = AstraDB(
        collection_name=collection,
        embedding=embedding,
        token=token,
        api_endpoint=api_endpoint,
    )
    return vector_db


class Environment:
    def __init__(
        self, vectorstore: VectorStore, llm: BaseLanguageModel, embedding: Embeddings
    ):
        self.vectorstore = vectorstore
        self.llm = llm
        self.embedding = embedding


@pytest.fixture
def environment():
    embeddings_impl = init_embeddings()
    vector_db_impl = init_vector_db(embeddings_impl)
    llm_impl = init_llm()
    yield Environment(
        vectorstore=vector_db_impl, llm=llm_impl, embedding=embeddings_impl
    )
    close_vector_db(vector_db_impl)


def close_vector_db(vector_store: VectorStore):
    vector_store.astra_db.delete_collection(vector_store.collection_name)


def init_embeddings() -> Embeddings:
    return MockEmbeddings()


def init_llm() -> BaseLanguageModel:
    key = get_required_env("OPEN_AI_KEY")
    return ChatOpenAI(
        openai_api_key=key,
        model="gpt-3.5-turbo-16k",
        streaming=True,
        temperature=0,
    )
