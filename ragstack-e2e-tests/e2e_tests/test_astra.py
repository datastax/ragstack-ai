import json
from typing import List

import pytest
from langchain.schema.embeddings import Embeddings
from langchain.schema.vectorstore import VectorStore
from langchain.vectorstores import AstraDB
from langchain.chat_models import ChatOpenAI
from langchain.schema.language_model import BaseLanguageModel
from e2e_tests.conftest import get_required_env


def test_basic_vector_search(environment):
    print("Running test_basic_vector_search")
    vectorstore = environment.vectorstore
    vectorstore.add_texts(["RAGStack is a framework to run LangChain in production"])
    retriever = vectorstore.as_retriever()
    assert len(retriever.get_relevant_documents("RAGStack")) > 0


def test_basic_metadata_filtering(environment):
    print("Running test_basic_metadata_filtering")

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
        if not ("UNSUPPORTED_FILTER_OPERATION" in e.args[0]):
            pytest.fail(
                f"Should have thrown ValueError with UNSUPPORTED_FILTER_OPERATION but it was {e}"
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
                f"Should have thrown ValueError with UNSUPPORTED_FILTER_OPERATION but it was {e}"
            )


def verify_document(document, expected_content, expected_metadata):
    assert document.get("content") == expected_content
    assert document.get("metadata").get("id") == expected_metadata.get("id")
    assert document.get("metadata").get("source") == expected_metadata.get("source")
    assert document.get("metadata").get("language") == expected_metadata.get("language")
    assert document.get("metadata").get("name") == expected_metadata.get("name")


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
    collection = get_required_env("ASTRA_PROD_TABLE_NAME")
    token = get_required_env("ASTRA_PROD_DB_TOKEN")
    api_endpoint = get_required_env("ASTRA_PROD_DB_ENDPOINT")
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
