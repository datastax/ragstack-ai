import logging
from typing import List

import pytest
from langchain.schema.embeddings import Embeddings
from langchain.schema.vectorstore import VectorStore
from langchain.vectorstores import AstraDB
from langchain.chat_models import ChatOpenAI
from langchain.schema.language_model import BaseLanguageModel
from conftest import get_required_env

VECTOR_ASTRADB_PROD = "astradb-prod"
VECTOR_ASTRADB_DEV = "astradb-dev"


class MockEmbeddings(Embeddings):
    def __init__(self):
        self.embedded_documents = None
        self.embedded_query = None

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        self.embedded_documents = texts
        return [[1.0, 2.0, 3.0]] * len(texts)

    def embed_query(self, text: str) -> List[float]:
        self.embedded_query = text
        return [1.0, 2.0, 3.0]


def init_vector_db(impl, embedding: Embeddings) -> VectorStore:
    if impl in [VECTOR_ASTRADB_DEV, VECTOR_ASTRADB_PROD]:
        if impl == VECTOR_ASTRADB_DEV:
            collection = get_required_env("ASTRA_DEV_TABLE_NAME")
            token = get_required_env("ASTRA_DEV_DB_TOKEN")
            api_endpoint = get_required_env("ASTRA_DEV_DB_ENDPOINT")
        else:
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
    else:
        raise Exception("Unknown vector db implementation: " + impl)


def close_vector_db(impl: str, vector_store: VectorStore):
    if impl in [VECTOR_ASTRADB_DEV, VECTOR_ASTRADB_PROD]:
        vector_store.astra_db.delete_collection(vector_store.collection_name)
    else:
        raise Exception("Unknown vector db implementation: " + impl)


def init_embeddings() -> Embeddings:
    return MockEmbeddings()


def close_embeddings(embeddings: Embeddings):
    pass


def init_llm() -> BaseLanguageModel:
    key = get_required_env("OPEN_AI_KEY")
    return ChatOpenAI(
        openai_api_key=key,
        model="gpt-3.5-turbo-16k",
        streaming=True,
        temperature=0,
    )


def close_llm(llm: BaseLanguageModel):
    pass


def vector_dbs():
    return [
        # VECTOR_ASTRADB_DEV,
        VECTOR_ASTRADB_PROD
    ]


@pytest.mark.parametrize("vector_db", vector_dbs())
def test_basic_vector_search(vector_db: str):
    print("Running test_basic_vector_search")

    def test(vectorstore, llm, embedding):
        vectorstore.add_texts(
            ["RAGStack is a framework to run LangChain in production"]
        )
        retriever = vectorstore.as_retriever()
        assert len(retriever.get_relevant_documents("RAGStack")) > 0

    _run_test(vector_db, test)


@pytest.mark.parametrize("vector_db", vector_dbs())
def test_basic_metadata_filtering(vector_db: str):
    print("Running test_basic_metadata_filtering")

    def test(vectorstore, llm, embedding):
        collection = vectorstore.collection

        vectorstore.add_texts(
            texts=["RAGStack is a framework to run LangChain in production"],
            metadatas=[{"id": "http://mywebsite", "language": "en", "source": "website", "name": "Homepage"}],
        )

        response = collection.find_one(filter={}).get("data").get("document")
        print("Response:", response)
        assert response.get('content') == "RAGStack is a framework to run LangChain in production"
        assert response.get('metadata').get('id') == "http://mywebsite"
        assert response.get('metadata').get('source') == "website"

        response = collection.find_one(filter={"metadata.source": "website"}).get("data").get("document")
        print("Response:", response)
        assert response.get('content') == "RAGStack is a framework to run LangChain in production"
        assert response.get('metadata').get('id') == "http://mywebsite"
        assert response.get('metadata').get('source') == "website"

        response = collection.find_one(filter={
            "$and":
                [{"metadata.language": "en"},
                 {"metadata.source": "website"}]
        }).get("data").get("document")
        print("Response:", response)
        assert response.get('content') == "RAGStack is a framework to run LangChain in production"
        assert response.get('metadata').get('id') == "http://mywebsite"
        assert response.get('metadata').get('source') == "website"

    _run_test(vector_db, test)


def _run_test(vector_db: str, func: callable):
    embeddings_impl = init_embeddings()
    vector_db_impl = init_vector_db(vector_db, embeddings_impl)
    llm_impl = init_llm()
    try:
        func(vectorstore=vector_db_impl, llm=llm_impl, embedding=embeddings_impl)
    finally:
        if vector_db_impl:
            close_vector_db(vector_db, vector_db_impl)
        if embeddings_impl:
            close_embeddings(embeddings_impl)
        if llm_impl:
            close_llm(llm_impl)
