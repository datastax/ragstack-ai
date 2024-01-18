import logging
import os
import random
import time
from abc import abstractmethod
from typing import List

import cassio
import pytest
from e2e_tests.conftest import (
    set_current_test_info,
    get_required_env,
    get_astra_ref,
    delete_all_astra_collections_with_client,
    delete_astra_collection,
    AstraRef,
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
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.embeddings import Embeddings
from langchain_core.messages import HumanMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.vectorstores import VectorStore
from langchain_google_genai import ChatGoogleGenerativeAI
from sqlalchemy import Float
from vertexai.vision_models import MultiModalEmbeddingModel, Image


def astra_db_client():
    astra_ref = get_astra_ref()
    return AstraDBClient(
        token=astra_ref.token,
        api_endpoint=astra_ref.api_endpoint,
    )


class VectorStoreWrapper:
    @abstractmethod
    def init(self, embedding: Embeddings) -> str:
        pass

    @abstractmethod
    def as_vector_store(self) -> VectorStore:
        pass

    @abstractmethod
    def create_chat_history(self) -> BaseChatMessageHistory:
        pass

    @abstractmethod
    def put(
        self, doc_id: str, document: str, metadata: dict, vector: List[Float]
    ) -> None:
        pass

    @abstractmethod
    def search(self, vector: List[float], limit: int) -> List[str]:
        pass


class CassandraVectorStoreWrapper(VectorStoreWrapper):
    def __init__(self, astra_ref: AstraRef):
        self.astra_ref = astra_ref
        self.session_id = "test_session_id" + str(random.randint(0, 1000000))
        self.vector_store = None

    def init(self, embedding: Embeddings):
        if self.astra_ref.env == "dev":
            bundle_url_template = "https://api.dev.cloud.datastax.com/v2/databases/{database_id}/secureBundleURL"
            cassio.init(
                token=self.astra_ref.token,
                database_id=self.astra_ref.id,
                bundle_url_template=bundle_url_template,
            )
        else:
            cassio.init(token=self.astra_ref.token, database_id=self.astra_ref.id)
        self.vector_store = Cassandra(
            embedding=embedding,
            session=None,
            keyspace="default_keyspace",
            table_name=self.astra_ref.collection,
        )

    def as_vector_store(self) -> VectorStore:
        return self.vector_store

    def create_chat_history(self) -> BaseChatMessageHistory:
        return AstraDBChatMessageHistory(
            session_id=self.session_id,
            api_endpoint=self.astra_ref.api_endpoint,
            token=self.astra_ref.token,
            collection_name=self.astra_ref.collection + "_chat_memory",
        )

    def put(
        self, doc_id: str, document: str, metadata: dict, vector: List[Float]
    ) -> None:
        self.vector_store.table.table.put(
            row_id=doc_id,
            body_blob=document,
            vector=vector,
            metadata=metadata or {},
        )

    def search(self, vector: List[float], limit: int) -> List[str]:
        return map(
            lambda doc: doc["document"],
            self.vector_store.table.search(embedding_vector=vector, top_k=limit),
        )


class AstraDBVectorStoreWrapper(VectorStoreWrapper):
    def __init__(self, astra_ref: AstraRef):
        self.astra_ref = astra_ref
        self.session_id = "test_session_id" + str(random.randint(0, 1000000))
        self.vector_store = None

    def init(self, embedding: Embeddings):
        self.vector_store = AstraDB(
            collection_name=self.astra_ref.collection,
            embedding=embedding,
            astra_db_client=astra_db_client(),
        )

    def as_vector_store(self) -> VectorStore:
        return self.vector_store

    def create_chat_history(self) -> BaseChatMessageHistory:
        return AstraDBChatMessageHistory(
            session_id=self.session_id,
            astra_db_client=astra_db_client(),
            collection_name=self.astra_ref.collection + "_chat_memory",
        )

    def put(
        self, doc_id: str, document: str, metadata: dict, vector: List[Float]
    ) -> None:
        self.vector_store.collection.insert_one(
            {
                "_id": doc_id,
                "document": document,
                "metadata": metadata or {},
                "$vector": vector,
            }
        )

    def search(self, vector: List[float], limit: int) -> List[str]:
        return map(
            lambda doc: doc["document"],
            self.vector_store.collection.vector_find(
                vector,
                limit=limit,
            ),
        )


@pytest.fixture
def astra_db():
    astra_ref = get_astra_ref()
    delete_all_astra_collections_with_client(astra_db_client())
    yield AstraDBVectorStoreWrapper(astra_ref)
    delete_astra_collection(astra_ref)
    delete_all_astra_collections_with_client(astra_db_client())


@pytest.fixture
def cassandra():
    astra_ref = get_astra_ref()
    delete_all_astra_collections_with_client(astra_db_client())
    yield CassandraVectorStoreWrapper(astra_ref)
    delete_astra_collection(astra_ref)
    delete_all_astra_collections_with_client(astra_db_client())


@pytest.fixture
def openai_llm():
    return ChatOpenAI(
        openai_api_key=get_required_env("OPEN_AI_KEY"),
        model="gpt-3.5-turbo-16k",
        streaming=True,
        temperature=0,
    )


@pytest.fixture
def openai_embedding():
    return OpenAIEmbeddings(openai_api_key=get_required_env("OPEN_AI_KEY"))


@pytest.fixture
def azure_openai_llm():
    return AzureChatOpenAI(
        azure_deployment=get_required_env("AZURE_OPEN_AI_CHAT_MODEL_DEPLOYMENT"),
        openai_api_base=get_required_env("AZURE_OPEN_AI_ENDPOINT"),
        openai_api_key=get_required_env("AZURE_OPEN_AI_KEY"),
        openai_api_type="azure",
        openai_api_version="2023-07-01-preview",
    )


@pytest.fixture
def azure_openai_embedding():
    model_and_deployment = get_required_env("AZURE_OPEN_AI_EMBEDDINGS_MODEL_DEPLOYMENT")
    return AzureOpenAIEmbeddings(
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
    return ChatVertexAI()


@pytest.fixture
def vertex_embedding():
    return VertexAIEmbeddings(model_name="textembedding-gecko")


@pytest.fixture
def bedrock_anthropic_llm():
    return BedrockChat(
        model_id="anthropic.claude-v2",
        region_name=get_required_env("BEDROCK_AWS_REGION"),
    )


@pytest.fixture
def bedrock_meta_llm():
    return BedrockChat(
        model_id="meta.llama2-13b-chat-v1",
        region_name=get_required_env("BEDROCK_AWS_REGION"),
    )


@pytest.fixture
def bedrock_titan_embedding():
    return BedrockEmbeddings(
        model_id="amazon.titan-embed-text-v1",
        region_name=get_required_env("BEDROCK_AWS_REGION"),
    )


@pytest.fixture
def bedrock_cohere_embedding():
    return BedrockEmbeddings(
        model_id="cohere.embed-english-v3",
        region_name=get_required_env("BEDROCK_AWS_REGION"),
    )


@pytest.fixture
def huggingface_hub_llm():
    return HuggingFaceHub(
        repo_id="google/flan-t5-xxl",
        huggingfacehub_api_token=get_required_env("HUGGINGFACE_HUB_KEY"),
        model_kwargs={"temperature": 1, "max_length": 256},
    )


@pytest.fixture
def huggingface_hub_embedding():
    return HuggingFaceInferenceAPIEmbeddings(
        api_key=get_required_env("HUGGINGFACE_HUB_KEY"),
        model_name="sentence-transformers/all-MiniLM-l6-v2",
    )


@pytest.fixture
def nvidia_embedding():
    get_required_env("NVIDIA_API_KEY")
    from langchain_nvidia_ai_endpoints import NVIDIAEmbeddings

    return NVIDIAEmbeddings(model="nvolve-40k")


@pytest.fixture
def nvidia_mixtral_llm():
    get_required_env("NVIDIA_API_KEY")
    from langchain_nvidia_ai_endpoints import ChatNVIDIA

    return ChatNVIDIA(model="mixtral_8x7b")


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
        ("nvidia_embedding", "nvidia_mixtral_llm"),
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


def _run_test(test_case: str, vector_store_wrapper, embedding, llm):
    vector_store_wrapper.init(embedding=embedding)
    vector_store = vector_store_wrapper.as_vector_store()
    if test_case == "rag_custom_chain":
        run_rag_custom_chain(
            vector_store=vector_store,
            llm=llm,
        )
    elif test_case == "conversational_rag":
        run_conversational_rag(
            vector_store=vector_store,
            llm=llm,
            chat_memory=vector_store_wrapper.create_chat_history(),
        )
    else:
        raise ValueError(f"Unknown test case: {test_case}")


@pytest.fixture
def vertex_gemini_multimodal_embedding():
    return MultiModalEmbeddingModel.from_pretrained("multimodalembedding@001"), 1408


@pytest.fixture
def vertex_gemini_pro_vision_llm():
    return ChatVertexAI(model_name="gemini-pro-vision")


@pytest.fixture
def vertex_gemini_pro_llm():
    return ChatVertexAI(model_name="gemini-pro")


@pytest.fixture
def gemini_pro_vision_llm():
    return ChatGoogleGenerativeAI(
        model="gemini-pro-vision", google_api_key=get_required_env("GOOGLE_API_KEY")
    )


@pytest.fixture
def gemini_pro_llm():
    return ChatGoogleGenerativeAI(
        model="gemini-pro", google_api_key=get_required_env("GOOGLE_API_KEY")
    )


@pytest.mark.parametrize(
    "vector_store",
    ["astra_db", "cassandra"],
)
@pytest.mark.parametrize(
    "embedding,llm",
    [
        ("vertex_gemini_multimodal_embedding", "vertex_gemini_pro_vision_llm"),
        ("vertex_gemini_multimodal_embedding", "gemini_pro_vision_llm"),
    ],
)
def test_multimodal(vector_store, embedding, llm, request):
    set_current_test_info(
        "langchain::multimodal_rag",
        f"{llm},{embedding},{vector_store}",
    )

    resolved_embedding, embedding_size = request.getfixturevalue(embedding)

    class FakeEmbeddings(Embeddings):
        def embed_documents(self, texts: List[str]) -> List[List[float]]:
            return [[0.0] * embedding_size] * len(texts)

        def embed_query(self, text: str) -> List[float]:
            return [0.0] * embedding_size

    vector_store_wrapper = request.getfixturevalue(vector_store)
    vector_store_wrapper.init(embedding=FakeEmbeddings())
    resolved_llm = request.getfixturevalue(llm)

    tree_image = get_local_resource_path("tree.jpeg")
    products = [
        {
            "name": "Coffee Machine Ultra Cool",
            "image": get_local_resource_path("coffee_machine.jpeg"),
        },
        {"name": "Tree", "image": tree_image},
        {"name": "Another Tree", "image": tree_image},
        {"name": "Another Tree 2", "image": tree_image},
        {"name": "Another Tree 3", "image": tree_image},
    ]

    for p in products:
        img = Image.load_from_file(p["image"])
        embeddings = resolved_embedding.get_embeddings(
            image=img, contextual_text=p["name"]
        )
        p["$vector"] = embeddings.image_embedding

        vector_store_wrapper.put(p["name"], p["name"], {}, embeddings.image_embedding)

    query_image_path = get_local_resource_path("coffee_maker_part.png")
    img = Image.load_from_file(query_image_path)
    embeddings = resolved_embedding.get_embeddings(
        image=img, contextual_text="Coffee Maker Part"
    )

    documents = vector_store_wrapper.search(embeddings.image_embedding, 3)
    image_message = {
        "type": "image_url",
        "image_url": {"url": query_image_path},
    }
    text_message = {
        "type": "text",
        "text": f"What is this image? Tell me which one of these products it is part of: {', '.join([p for p in documents])}",
    }
    message = HumanMessage(content=[text_message, image_message])
    response = resolved_llm([message])
    assert "Coffee Machine Ultra Cool" in response.content


def get_local_resource_path(filename: str):
    dirname = os.path.dirname(__file__)
    e2e_tests_dir = os.path.dirname(dirname)
    return os.path.join(e2e_tests_dir, "resources", filename)


@pytest.mark.parametrize(
    "chat",
    ["vertex_gemini_pro_llm", "gemini_pro_llm"],
)
def test_chat(chat, request):
    set_current_test_info(
        "langchain::chat",
        chat,
    )
    chat_model = request.getfixturevalue(chat)
    prompt = ChatPromptTemplate.from_messages(
        [("human", "Hello! Where Archimede was born?")]
    )
    chain = prompt | chat_model
    response = chain.invoke({})
    assert "Syracuse" in response.content
