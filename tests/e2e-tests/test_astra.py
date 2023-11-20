import pytest
from chat_application import run_application
import os
from langchain.schema.embeddings import Embeddings
from langchain.schema.vectorstore import VectorStore
from langchain.vectorstores import AstraDB
from langchain.chat_models import ChatOpenAI, AzureChatOpenAI
from langchain.embeddings import OpenAIEmbeddings
from langchain.schema.language_model import BaseLanguageModel


def get_required_env(name) -> str:
    if name not in os.environ:
        raise Exception(f"Missing required environment variable: {name}")
    return os.environ[name]


def init_vector_db(impl, embedding: Embeddings) -> VectorStore:
    if impl == "astradb":
        collection = get_required_env("ASTRA_TABLE_NAME")
        vector_db = AstraDB(
            collection_name=collection,
            embedding=embedding,
            namespace=get_required_env("ASTRA_KEYSPACE"),
            token=get_required_env("ASTRA_DB_TOKEN"),
            api_endpoint=get_required_env("ASTRA_DB_ENDPOINT"),
        )
        return vector_db
    else:
        raise Exception("Unknown vector db implementation: " + impl)


def close_vector_db(impl: str, vector_store: VectorStore):
    if impl == "astradb":
        vector_store.astra_db.delete_collection(vector_store.collection_name)
    else:
        raise Exception("Unknown vector db implementation: " + impl)


def init_embeddings(impl) -> Embeddings:
    if impl == "openai":
        key = get_required_env("OPEN_AI_KEY")
        return OpenAIEmbeddings(openai_api_key=key)
    elif impl == "openai-azure":
        model_and_deployment = get_required_env(
            "AZURE_OPEN_AI_EMBEDDINGS_MODEL_DEPLOYMENT"
        )
        return OpenAIEmbeddings(
            model=model_and_deployment,
            deployment=model_and_deployment,
            openai_api_key=get_required_env("AZURE_OPEN_AI_KEY"),
            openai_api_base=get_required_env("AZURE_OPEN_AI_ENDPOINT"),
            openai_api_type="azure",
            openai_api_version="2023-05-15",
            chunk_size=1,
        )
    else:
        raise Exception("Unknown embedding implementation: " + impl)


def close_embeddings(impl, embeddings: Embeddings):
    if impl in ("openai", "openai-azure"):
        pass
    else:
        raise Exception("Unknown embedding implementation: " + impl)


def init_llm(impl) -> BaseLanguageModel:
    if impl == "openai":
        key = get_required_env("OPEN_AI_KEY")
        return ChatOpenAI(
            openai_api_key=key,
            model="gpt-3.5-turbo-16k",
            streaming=True,
            temperature=0,
        )
    elif impl == "openai-azure":
        model_and_deployment = get_required_env("AZURE_OPEN_AI_CHAT_MODEL_DEPLOYMENT")
        azure_open_ai = AzureChatOpenAI(
            azure_deployment=model_and_deployment,
            openai_api_base=get_required_env("AZURE_OPEN_AI_ENDPOINT"),
            openai_api_key=get_required_env("AZURE_OPEN_AI_KEY"),
            openai_api_type="azure",
            openai_api_version="2023-07-01-preview",
        )
        return azure_open_ai
    else:
        raise Exception("Unknown llm implementation: " + impl)


def close_llm(impl, llm: BaseLanguageModel):
    if impl in ("openai", "openai-azure"):
        pass
    else:
        raise Exception("Unknown llm implementation: " + impl)


def vector_dbs():
    return ["astradb"]


@pytest.mark.parametrize("vector_db", vector_dbs())
def test_openai_azure(vector_db: str):
    _run_test(vector_db=vector_db, embedding="openai-azure", llm="openai-azure")


@pytest.mark.parametrize("vector_db", vector_dbs())
def test_openai(vector_db: str):
    _run_test(vector_db=vector_db, embedding="openai", llm="openai")


def _run_test(vector_db: str, embedding: str, llm: str):
    embeddings_impl = init_embeddings(embedding)
    vector_db_impl = init_vector_db(vector_db, embeddings_impl)
    llm_impl = init_llm(llm)
    try:
        response = run_application(
            question="When was released MyFakeProductForTesting for the first time ?",
            vector_store=vector_db_impl,
            llm=llm_impl,
        )
        print(f"Got response ${response}")
        assert "2020" in response
    finally:
        if vector_db_impl:
            close_vector_db(vector_db, vector_db_impl)
        if embeddings_impl:
            close_embeddings(embedding, embeddings_impl)
        if llm_impl:
            close_llm(llm, llm_impl)
