import logging

from astrapy.db import AstraDB as LibAstraDB
from e2e_tests.conftest import (
    set_current_test_info_simple_rag,
    get_required_env,
)
from e2e_tests.chat_application import run_application
from langchain.llms.huggingface_hub import HuggingFaceHub

import pytest
import cassio

from langchain.schema.embeddings import Embeddings
from langchain.schema.vectorstore import VectorStore
from langchain.vectorstores import AstraDB, Cassandra
from langchain.chat_models import ChatOpenAI, AzureChatOpenAI, ChatVertexAI, BedrockChat
from langchain.embeddings import (
    OpenAIEmbeddings,
    VertexAIEmbeddings,
    BedrockEmbeddings,
    HuggingFaceInferenceAPIEmbeddings,
)
from langchain.embeddings.azure_openai import AzureOpenAIEmbeddings
from langchain.schema.language_model import BaseLanguageModel

VECTOR_ASTRADB_PROD = "astradb-prod"
VECTOR_ASTRADB_DEV = "astradb-dev"
VECTOR_CASSANDRA = "cassandra"


def vector_dbs():
    return [
        VECTOR_ASTRADB_PROD,
        VECTOR_CASSANDRA,
    ]


def init_vector_db(impl, embedding: Embeddings) -> VectorStore:
    if impl == VECTOR_ASTRADB_DEV:
        collection = get_required_env("ASTRA_DEV_TABLE_NAME")
        token = get_required_env("ASTRA_DEV_DB_TOKEN")
        api_endpoint = get_required_env("ASTRA_DEV_DB_ENDPOINT")

        # Ensure collections from previous runs are cleared
        delete_collections(api_endpoint, token)

        return AstraDB(
            collection_name=collection,
            embedding=embedding,
            token=token,
            api_endpoint=api_endpoint,
        )
    elif impl == VECTOR_ASTRADB_PROD:
        collection = get_required_env("ASTRA_PROD_TABLE_NAME")
        token = get_required_env("ASTRA_PROD_DB_TOKEN")
        api_endpoint = get_required_env("ASTRA_PROD_DB_ENDPOINT")

        # Ensure collections from previous runs are cleared
        delete_collections(api_endpoint, token)

        return AstraDB(
            collection_name=collection,
            embedding=embedding,
            token=token,
            api_endpoint=api_endpoint,
        )
    elif impl == VECTOR_CASSANDRA:
        table_name = get_required_env("ASTRA_PROD_TABLE_NAME")
        token = get_required_env("ASTRA_PROD_DB_TOKEN")
        id = get_required_env("ASTRA_PROD_DB_ID")
        api_endpoint = get_required_env("ASTRA_PROD_DB_ENDPOINT")

        # Ensure collections from previous runs are cleared
        delete_collections(api_endpoint, token)

        cassio.init(token=token, database_id=id)
        return Cassandra(
            embedding=embedding,
            session=None,
            keyspace="default_keyspace",
            table_name=table_name,
        )
    else:
        raise Exception("Unknown vector db implementation: " + impl)


def delete_collections(api_endpoint, token):
    """
    Deletes all collections.

    Current AstraDB has a limit of 5 collections, meaning orphaned collections
    will cause subsequent tests to fail if the limit is reached.
    """
    raw_client = LibAstraDB(api_endpoint=api_endpoint, token=token)
    collections = raw_client.get_collections().get("status").get("collections")
    logging.info(f"Existing collections: {collections}")
    for collection_info in collections:
        logging.info(f"Deleting collection: {collection_info}")
        raw_client.delete_collection(collection_info)


def close_vector_db(impl: str, vector_store: VectorStore):
    if impl in [VECTOR_ASTRADB_DEV, VECTOR_ASTRADB_PROD]:
        vector_store.astra_db.delete_collection(vector_store.collection_name)
    elif impl == VECTOR_CASSANDRA:
        vector_store.table.session.execute(f"DROP TABLE IF EXISTS {vector_store.keyspace}.{vector_store.table_name};")
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
        return AzureOpenAIEmbeddings(
            model=model_and_deployment,
            deployment=model_and_deployment,
            openai_api_key=get_required_env("AZURE_OPEN_AI_KEY"),
            openai_api_base=get_required_env("AZURE_OPEN_AI_ENDPOINT"),
            openai_api_type="azure",
            openai_api_version="2023-05-15",
            chunk_size=1,
        )
    elif impl == "vertex-ai":
        return VertexAIEmbeddings(model_name="textembedding-gecko")
    elif impl == "bedrock-titan":
        return BedrockEmbeddings(
            model_id="amazon.titan-embed-text-v1",
            region_name=get_required_env("BEDROCK_AWS_REGION"),
        )
    elif impl == "bedrock-cohere":
        return BedrockEmbeddings(
            model_id="cohere.embed-english-v3",
            region_name=get_required_env("BEDROCK_AWS_REGION"),
        )
    elif impl == "huggingface-hub":
        return HuggingFaceInferenceAPIEmbeddings(
            api_key=get_required_env("HUGGINGFACE_HUB_KEY"),
            model_name="sentence-transformers/all-MiniLM-l6-v2",
        )
    else:
        raise Exception("Unknown embedding implementation: " + impl)


def close_embeddings(impl, embeddings: Embeddings):
    pass


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
    elif impl == "vertex-ai":
        return ChatVertexAI()
    elif impl == "bedrock-anthropic":
        return BedrockChat(
            model_id="anthropic.claude-v2",
            region_name=get_required_env("BEDROCK_AWS_REGION"),
        )
    elif impl == "bedrock-meta":
        return BedrockChat(
            model_id="meta.llama2-13b-chat-v1",
            region_name=get_required_env("BEDROCK_AWS_REGION"),
        )
    elif impl == "huggingface-hub":
        return HuggingFaceHub(
            repo_id="google/flan-t5-xxl",
            huggingfacehub_api_token=get_required_env("HUGGINGFACE_HUB_KEY"),
            model_kwargs={"temperature": 0.5, "max_length": 64},
        )
    else:
        raise Exception("Unknown llm implementation: " + impl)


def close_llm(impl, llm: BaseLanguageModel):
    pass


# def test_openai_azure_astra_dev():
#     _run_test(
#         vector_db=VECTOR_ASTRADB_DEV, embedding="openai-azure", llm="openai-azure"
#     )


@pytest.mark.parametrize("vector_db", vector_dbs())
def test_openai_azure(vector_db: str):
    _run_test(vector_db=vector_db, embedding="openai-azure", llm="openai-azure")


@pytest.mark.parametrize("vector_db", vector_dbs())
def test_openai(vector_db: str):
    _run_test(vector_db=vector_db, embedding="openai", llm="openai")


@pytest.mark.parametrize("vector_db", vector_dbs())
def test_vertex_ai(vector_db: str):
    _run_test(vector_db=vector_db, embedding="vertex-ai", llm="vertex-ai")


@pytest.mark.parametrize("vector_db", vector_dbs())
def test_bedrock_anthropic(vector_db: str):
    _run_test(
        vector_db=vector_db,
        embedding="bedrock-titan",
        llm="bedrock-anthropic",
    )


@pytest.mark.parametrize("vector_db", vector_dbs())
def test_bedrock_meta(vector_db: str):
    _run_test(vector_db=vector_db, embedding="bedrock-cohere", llm="bedrock-meta")


def test_huggingface_hub():
    _run_test(
        vector_db=VECTOR_ASTRADB_PROD,
        embedding="huggingface-hub",
        llm="huggingface-hub",
    )


def _run_test(vector_db: str, embedding: str, llm: str):
    set_current_test_info_simple_rag(llm=llm, embedding=embedding, vector_db=vector_db)

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
