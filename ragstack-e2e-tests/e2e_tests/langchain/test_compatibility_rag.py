import logging
import os
import pathlib
import random
import time
from typing import List

import cassio
import pytest
from e2e_tests.conftest import (
    set_current_test_info,
    get_required_env,
    get_astra_ref,
    delete_all_astra_collections_with_client,
    delete_astra_collection,
)
from e2e_tests.langchain.rag_application import (
    run_rag_custom_chain,
    run_conversational_rag,
)
from langchain.chat_models import ChatOpenAI, AzureChatOpenAI, ChatVertexAI, BedrockChat
from langchain.embeddings import (
    OpenAIEmbeddings,
    VertexAIEmbeddings,
    BedrockEmbeddings,
    HuggingFaceInferenceAPIEmbeddings,
)
from langchain.embeddings.azure_openai import AzureOpenAIEmbeddings
from langchain.llms.huggingface_hub import HuggingFaceHub
from langchain.memory import AstraDBChatMessageHistory
from langchain.vectorstores import AstraDB, Cassandra
from astrapy.db import AstraDB as AstraDBClient
from langchain_core.embeddings import Embeddings
from langchain_core.messages import HumanMessage
from vertexai.vision_models import MultiModalEmbeddingModel, Image


def astra_db_client():
    astra_ref = get_astra_ref()
    return AstraDBClient(
        token=astra_ref.token,
        api_endpoint=astra_ref.api_endpoint,
    )


@pytest.fixture
def astra_db():
    astra_ref = get_astra_ref()
    client = astra_db_client()
    delete_all_astra_collections_with_client(astra_db_client())
    session_id = "test_session_id" + str(random.randint(0, 1000000))
    yield (
        "astradb",
        lambda embedding: AstraDB(
            collection_name=astra_ref.collection,
            embedding=embedding,
            astra_db_client=client,
        ),
        AstraDBChatMessageHistory(
            session_id=session_id,
            astra_db_client=astra_db_client(),
            collection_name=astra_ref.collection + "_chat_memory",
        ),
    )
    delete_all_astra_collections_with_client(astra_db_client())


@pytest.fixture
def cassandra():
    astra_ref = get_astra_ref()

    session_id = "test_session_id" + str(random.randint(0, 1000000))
    cassio.init(token=astra_ref.token, database_id=astra_ref.id)

    yield "cassandra", lambda embedding: Cassandra(
        embedding=embedding,
        session=None,
        keyspace="default_keyspace",
        table_name=astra_ref.collection,
    ), AstraDBChatMessageHistory(
        session_id=session_id,
        api_endpoint=astra_ref.api_endpoint,
        token=astra_ref.token,
        collection_name=astra_ref.collection + "_chat_memory",
    )
    delete_astra_collection(astra_ref)


@pytest.fixture
def openai_llm():
    return "openai", ChatOpenAI(
        openai_api_key=get_required_env("OPEN_AI_KEY"),
        model="gpt-3.5-turbo-16k",
        streaming=True,
        temperature=0,
    )


@pytest.fixture
def openai_embedding():
    return "openai", OpenAIEmbeddings(openai_api_key=get_required_env("OPEN_AI_KEY"))


@pytest.fixture
def azure_openai_llm():
    return "azure-openai", AzureChatOpenAI(
        azure_deployment=get_required_env("AZURE_OPEN_AI_CHAT_MODEL_DEPLOYMENT"),
        openai_api_base=get_required_env("AZURE_OPEN_AI_ENDPOINT"),
        openai_api_key=get_required_env("AZURE_OPEN_AI_KEY"),
        openai_api_type="azure",
        openai_api_version="2023-07-01-preview",
    )


@pytest.fixture
def azure_openai_embedding():
    model_and_deployment = get_required_env("AZURE_OPEN_AI_EMBEDDINGS_MODEL_DEPLOYMENT")
    return "azure-openai", AzureOpenAIEmbeddings(
        model=model_and_deployment,
        deployment=model_and_deployment,
        openai_api_key=get_required_env("AZURE_OPEN_AI_KEY"),
        openai_api_base=get_required_env("AZURE_OPEN_AI_ENDPOINT"),
        openai_api_type="azure",
        openai_api_version="2023-05-15",
        chunk_size=1,
    )


@pytest.fixture
def vertex_llm():
    return "vertex-ai", ChatVertexAI()


@pytest.fixture
def vertex_embedding():
    return "vertex-ai", VertexAIEmbeddings(model_name="textembedding-gecko")


@pytest.fixture
def bedrock_anthropic_llm():
    return "bedrock-anthropic", BedrockChat(
        model_id="anthropic.claude-v2",
        region_name=get_required_env("BEDROCK_AWS_REGION"),
    )


@pytest.fixture
def bedrock_meta_llm():
    return "bedrock-meta", BedrockChat(
        model_id="meta.llama2-13b-chat-v1",
        region_name=get_required_env("BEDROCK_AWS_REGION"),
    )


@pytest.fixture
def bedrock_titan_embedding():
    return "bedrock-titan", BedrockEmbeddings(
        model_id="amazon.titan-embed-text-v1",
        region_name=get_required_env("BEDROCK_AWS_REGION"),
    )


@pytest.fixture
def bedrock_cohere_embedding():
    return "bedrock-cohere", BedrockEmbeddings(
        model_id="cohere.embed-english-v3",
        region_name=get_required_env("BEDROCK_AWS_REGION"),
    )


@pytest.fixture
def huggingface_hub_llm():
    return "huggingface-hub", HuggingFaceHub(
        repo_id="google/flan-t5-xxl",
        huggingfacehub_api_token=get_required_env("HUGGINGFACE_HUB_KEY"),
        model_kwargs={"temperature": 1, "max_length": 256},
    )


@pytest.fixture
def huggingface_hub_embedding():
    return "huggingface-hub", HuggingFaceInferenceAPIEmbeddings(
        api_key=get_required_env("HUGGINGFACE_HUB_KEY"),
        model_name="sentence-transformers/all-MiniLM-l6-v2",
    )


@pytest.mark.parametrize(
    "test_case",
    ["rag_custom_chain", "conversational_rag"],
)
@pytest.mark.parametrize(
    "vector_store",
    ["astra_db", "cassandra"],
)
@pytest.mark.parametrize(
    "embedding,llm",
    [
        ("openai_embedding", "openai_llm"),
        ("azure_openai_embedding", "azure_openai_llm"),
        ("vertex_embedding", "vertex_llm"),
        ("bedrock_titan_embedding", "bedrock_anthropic_llm"),
        ("bedrock_cohere_embedding", "bedrock_meta_llm"),
        ("huggingface_hub_embedding", "huggingface_hub_llm"),
    ],
)
def test_rag(test_case, vector_store, embedding, llm, request):
    set_current_test_info(
        "langchain::" + test_case,
        f"{llm},{embedding},{vector_store}",
    )
    start = time.perf_counter_ns()
    resolved_vector_store = request.getfixturevalue(vector_store)
    logging.info(
        "Vector store initialized in %s seconds", (time.perf_counter_ns() - start) / 1e9
    )
    start = time.perf_counter_ns()
    resolved_embedding = request.getfixturevalue(embedding)
    logging.info(
        "Embedding initialized in %s seconds", (time.perf_counter_ns() - start) / 1e9
    )
    start = time.perf_counter_ns()
    resolved_llm = request.getfixturevalue(llm)
    logging.info(
        "LLM initialized in %s seconds", (time.perf_counter_ns() - start) / 1e9
    )
    _run_test(
        test_case,
        resolved_vector_store,
        resolved_embedding,
        resolved_llm,
    )


def _run_test(test_case: str, vector_store, embedding, llm):
    embedding_name, embedding = embedding
    vector_store_name, vector_store, chat_memory = vector_store
    vector_store = vector_store(embedding)
    llm_name, llm = llm
    if test_case == "rag_custom_chain":
        run_rag_custom_chain(
            vector_store=vector_store,
            llm=llm,
        )
    elif test_case == "conversational_rag":
        run_conversational_rag(
            vector_store=vector_store, llm=llm, chat_memory=chat_memory
        )
    else:
        raise ValueError(f"Unknown test case: {test_case}")


@pytest.fixture
def vertex_gemini_multimodal_embedding():
    return "gemini-multimodalembedding@001", MultiModalEmbeddingModel.from_pretrained("multimodalembedding@001"), 1408


@pytest.fixture
def vertex_gemini_pro_vision():
    return "gemini-pro-vision", ChatVertexAI(model_name="gemini-pro-vision")


@pytest.mark.parametrize(
    "vector_store",
    [
        "astra_db",
        "cassandra"
    ],
)
@pytest.mark.parametrize(
    "embedding,llm",
    [
        ("vertex_gemini_multimodal_embedding", "vertex_gemini_pro_vision"),
    ],
)
def test_multimodal(vector_store, embedding, llm, request):
    set_current_test_info(
        "langchain::multimodal",
        f"{llm},{embedding},{vector_store}",
    )

    _, resolved_embedding, embedding_size = request.getfixturevalue(embedding)

    class FakeEmbeddings(Embeddings):

        def embed_documents(self, texts: List[str]) -> List[List[float]]:
            return [[0.0] * embedding_size] * len(texts)

        def embed_query(self, text: str) -> List[float]:
            return [0.0] * embedding_size

    resolved_vector_store = request.getfixturevalue(vector_store)
    resolved_vector_store = resolved_vector_store[1]
    resolved_vector_store = resolved_vector_store(FakeEmbeddings())
    _, resolved_llm = request.getfixturevalue(llm)

    products = [
        {"name": "Coffee Machine Ultra Cool", "image": (pathlib.Path(__file__).parent / 'coffee_machine.jpeg').name},
        {"name": "Tree", "image": (pathlib.Path(__file__).parent / 'tree.jpeg').name},
        {"name": "Another Tree", "image": (pathlib.Path(__file__).parent / 'tree.jpeg').name},
        {"name": "Another Tree 2", "image": (pathlib.Path(__file__).parent / 'tree.jpeg').name},
        {"name": "Another Tree 3", "image": (pathlib.Path(__file__).parent / 'tree.jpeg').name},
    ]

    for p in products:
        img = Image.load_from_file(p["image"])
        embeddings = resolved_embedding.get_embeddings(image=img, contextual_text=p["name"])
        p["$vector"] = embeddings.image_embedding

    resolved_vector_store.collection.insert_many(products)

    query_image_path = (pathlib.Path(__file__).parent / 'coffee_maker_part.png').name
    img = Image.load_from_file(query_image_path)
    embeddings = resolved_embedding.get_embeddings(image=img, contextual_text="Coffee Maker Part")

    documents = resolved_vector_store.collection.vector_find(
        embeddings.image_embedding,
        limit=3,
    )

    image_message = {
        "type": "image_url",
        "image_url": {"url": query_image_path},
    }
    text_message = {
        "type": "text",
        "text": f"What is this image? Tell me which one of these products it is part of: {', '.join([p['name'] for p in documents])}"
    }
    message = HumanMessage(content=[text_message, image_message])
    response = resolved_llm([message])
    assert "Coffee Machine Ultra Cool" in response.content
