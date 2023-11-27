import pytest

from langchain.schema.embeddings import Embeddings
from langchain.schema.vectorstore import VectorStore
from langchain.vectorstores import AstraDB
from langchain.chat_models import ChatOpenAI
from langchain.embeddings import OpenAIEmbeddings
from langchain.schema.language_model import BaseLanguageModel
from conftest import get_required_env

VECTOR_ASTRADB_PROD = "astradb-prod"
VECTOR_ASTRADB_DEV = "astradb-dev"


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
    key = get_required_env("OPEN_AI_KEY")
    return OpenAIEmbeddings(openai_api_key=key)


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
