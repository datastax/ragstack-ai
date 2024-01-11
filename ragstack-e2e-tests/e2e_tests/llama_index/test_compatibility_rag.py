import os
import random
from abc import abstractmethod
from typing import List

import cassio
import pytest
from langchain.embeddings import VertexAIEmbeddings, HuggingFaceInferenceAPIEmbeddings
from llama_index import (
    VectorStoreIndex,
    StorageContext,
    ServiceContext,
    Document,
)
from llama_index.embeddings import (
    OpenAIEmbedding,
    AzureOpenAIEmbedding,
    BedrockEmbedding,
)
from llama_index.llms import (
    OpenAI,
    AzureOpenAI,
    Vertex,
    Bedrock,
    HuggingFaceInferenceAPI,
    ChatMessage,
)
from llama_index.schema import TextNode
from llama_index.vector_stores import AstraDBVectorStore, CassandraVectorStore

from e2e_tests.conftest import (
    set_current_test_info,
    get_required_env,
    get_astra_ref,
    delete_all_astra_collections,
    delete_astra_collection,
    AstraRef,
)
from llama_index.vector_stores.types import VectorStore, VectorStoreQuery
from sqlalchemy import Float
from vertexai.vision_models import MultiModalEmbeddingModel, Image


class VectorStoreWrapper:
    @abstractmethod
    def init(self, embedding_dimension: int) -> str:
        pass

    @abstractmethod
    def as_vector_store(self) -> VectorStore:
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

    def init(self, embedding_dimension: int):
        cassio.init(token=self.astra_ref.token, database_id=self.astra_ref.id)
        self.vector_store = CassandraVectorStore(
            embedding_dimension=embedding_dimension,
            session=None,
            keyspace="default_keyspace",
            table=self.astra_ref.collection,
        )

    def as_vector_store(self) -> VectorStore:
        return self.vector_store

    def put(
        self, doc_id: str, document: str, metadata: dict, vector: List[Float]
    ) -> None:
        self.vector_store.add(
            [TextNode(text=document, metadata=metadata, id_=doc_id, embedding=vector)]
        )

    def search(self, vector: List[float], limit: int) -> List[str]:
        return map(
            lambda doc: doc,
            self.vector_store.query(
                VectorStoreQuery(query_embedding=vector, similarity_top_k=limit)
            ).ids,
        )


class AstraDBVectorStoreWrapper(VectorStoreWrapper):
    def __init__(self, astra_ref: AstraRef):
        self.astra_ref = astra_ref
        self.session_id = "test_session_id" + str(random.randint(0, 1000000))
        self.vector_store = None

    def init(self, embedding_dimension: int):
        self.vector_store = AstraDBVectorStore(
            collection_name=self.astra_ref.collection,
            embedding_dimension=embedding_dimension,
            token=self.astra_ref.token,
            api_endpoint=self.astra_ref.api_endpoint,
        )

    def as_vector_store(self) -> VectorStore:
        return self.vector_store

    def put(
        self, doc_id: str, document: str, metadata: dict, vector: List[Float]
    ) -> None:
        self.vector_store.client.insert_one(
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
            self.vector_store.client.vector_find(
                vector,
                limit=limit,
            ),
        )


@pytest.fixture
def astra_db():
    astra_ref = get_astra_ref()
    delete_all_astra_collections(astra_ref)

    yield AstraDBVectorStoreWrapper(astra_ref)
    delete_astra_collection(astra_ref)


@pytest.fixture
def cassandra():
    astra_ref = get_astra_ref()
    delete_all_astra_collections(astra_ref)
    yield CassandraVectorStoreWrapper(astra_ref)
    delete_astra_collection(astra_ref)


@pytest.fixture
def openai_llm():
    return "openai", OpenAI(api_key=get_required_env("OPEN_AI_KEY"))


@pytest.fixture
def openai_embedding():
    return "openai", 1536, OpenAIEmbedding(api_key=get_required_env("OPEN_AI_KEY"))


@pytest.fixture
def azure_openai_llm():
    return "azure-openai", AzureOpenAI(
        azure_deployment=get_required_env("AZURE_OPEN_AI_CHAT_MODEL_DEPLOYMENT"),
        azure_endpoint=get_required_env("AZURE_OPEN_AI_ENDPOINT"),
        api_key=get_required_env("AZURE_OPEN_AI_KEY"),
        api_version="2023-07-01-preview",
    )


@pytest.fixture
def azure_openai_embedding():
    model_and_deployment = get_required_env("AZURE_OPEN_AI_EMBEDDINGS_MODEL_DEPLOYMENT")
    return (
        "azure-openai",
        1536,
        AzureOpenAIEmbedding(
            model=model_and_deployment,
            deployment_name=model_and_deployment,
            api_key=get_required_env("AZURE_OPEN_AI_KEY"),
            azure_endpoint=get_required_env("AZURE_OPEN_AI_ENDPOINT"),
            api_version="2023-05-15",
            embed_batch_size=1,
        ),
    )


@pytest.fixture
def vertex_llm():
    return "vertex-ai", Vertex(model="chat-bison")


@pytest.fixture
def vertex_embedding():
    # Llama-Index doesn't have Vertex AI embedding
    # so we use LangChain's wrapped one
    return "vertex-ai", 768, VertexAIEmbeddings(model_name="textembedding-gecko")


@pytest.fixture
def bedrock_anthropic_llm():
    return "bedrock-anthropic", Bedrock(
        model="anthropic.claude-v2",
        aws_access_key_id=get_required_env("AWS_ACCESS_KEY_ID"),
        aws_secret_access_key=get_required_env("AWS_SECRET_ACCESS_KEY"),
        aws_region_name=get_required_env("BEDROCK_AWS_REGION"),
    )


@pytest.fixture
def bedrock_meta_llm():
    return "bedrock-meta", Bedrock(
        model="meta.llama2-13b-chat-v1",
        aws_access_key_id=get_required_env("AWS_ACCESS_KEY_ID"),
        aws_secret_access_key=get_required_env("AWS_SECRET_ACCESS_KEY"),
        aws_region_name=get_required_env("BEDROCK_AWS_REGION"),
    )


@pytest.fixture
def bedrock_titan_embedding():
    return (
        "bedrock-titan",
        1536,
        BedrockEmbedding.from_credentials(
            model_name="amazon.titan-embed-text-v1",
            aws_region=get_required_env("BEDROCK_AWS_REGION"),
        ),
    )


@pytest.fixture
def bedrock_cohere_embedding():
    return (
        "bedrock-cohere",
        1024,
        BedrockEmbedding.from_credentials(
            model_name="cohere.embed-english-v3",
            aws_region=get_required_env("BEDROCK_AWS_REGION"),
        ),
    )


@pytest.fixture
def huggingface_hub_llm():
    return "huggingface-hub", HuggingFaceInferenceAPI(
        model_name="google/flan-t5-xxl",
        token=get_required_env("HUGGINGFACE_HUB_KEY"),
    )


@pytest.fixture
def huggingface_hub_embedding():
    # There's a bug in Llama-Index HuggingFace Hub embedding
    # so we use LangChain's wrapped one for now
    return (
        "huggingface-hub",
        384,
        HuggingFaceInferenceAPIEmbeddings(
            api_key=get_required_env("HUGGINGFACE_HUB_KEY"),
            model_name="sentence-transformers/all-MiniLM-l6-v2",
        ),
    )


@pytest.mark.parametrize("vector_store", ["cassandra", "astra_db"])
@pytest.mark.parametrize(
    "embedding,llm",
    [
        ("openai_embedding", "openai_llm"),
        ("azure_openai_embedding", "azure_openai_llm"),
        ("vertex_embedding", "vertex_llm"),
        # ("bedrock_titan_embedding", "bedrock_anthropic_llm"),
        # Deactivated for now because of
        # https://github.com/run-llama/llama_index/pull/9396
        # ("bedrock_cohere_embedding", "bedrock_meta_llm"),
        ("huggingface_hub_embedding", "huggingface_hub_llm"),
    ],
)
def test_rag(vector_store, embedding, llm, request):
    embedding_name, embedding_dimensions, embedding = request.getfixturevalue(embedding)
    vector_store_wrapper = request.getfixturevalue(vector_store)
    llm_name, llm = request.getfixturevalue(llm)
    set_current_test_info(
        "llama_index::rag",
        f"{llm_name},{embedding_name},{vector_store}",
    )
    vector_store_wrapper.init(embedding_dimensions)
    vector_store = vector_store_wrapper.as_vector_store()
    storage_context = StorageContext.from_defaults(vector_store=vector_store)
    service_context = ServiceContext.from_defaults(llm=llm, embed_model=embedding)

    documents = [
        Document(
            text="MyFakeProductForTesting is a versatile testing tool designed to streamline the testing process for software developers, quality assurance professionals, and product testers. It provides a comprehensive solution for testing various aspects of applications and systems, ensuring robust performance and functionality."  # noqa: E501
        ),
        Document(
            text="MyFakeProductForTesting comes equipped with an advanced dynamic test scenario generator. This feature allows users to create realistic test scenarios by simulating various user interactions, system inputs, and environmental conditions. The dynamic nature of the generator ensures that tests are not only diverse but also adaptive to changes in the application under test."  # noqa: E501
        ),
        Document(
            text="The product includes an intelligent bug detection and analysis module. It not only identifies bugs and issues but also provides in-depth analysis and insights into the root causes. The system utilizes machine learning algorithms to categorize and prioritize bugs, making it easier for developers and testers to address critical issues first."  # noqa: E501
        ),
        Document(text="MyFakeProductForTesting first release happened in June 2020."),
    ]

    index = VectorStoreIndex.from_documents(
        documents, storage_context=storage_context, service_context=service_context
    )

    query_engine = index.as_query_engine()
    response = query_engine.query(
        "When was released MyFakeProductForTesting for the first time ?"
    ).response
    print(f"Got response ${response}")
    assert "2020" in response


@pytest.fixture
def vertex_gemini_multimodal_embedding():
    # this is from LangChain, LLamaIndex does not have Vertex Embeddings support yet
    return MultiModalEmbeddingModel.from_pretrained("multimodalembedding@001"), 1408


@pytest.fixture
def vertex_gemini_pro_vision():
    return Vertex(model="gemini-pro-vision")


@pytest.mark.parametrize(
    "vector_store",
    ["astra_db", "cassandra"],
)
@pytest.mark.parametrize(
    "embedding,llm",
    [
        ("vertex_gemini_multimodal_embedding", "vertex_gemini_pro_vision"),
    ],
)
def test_multimodal(vector_store, embedding, llm, request):
    set_current_test_info(
        "llama_index::multimodal",
        f"{llm},{embedding},{vector_store}",
    )

    resolved_embedding, embedding_size = request.getfixturevalue(embedding)

    vector_store_wrapper = request.getfixturevalue(vector_store)
    vector_store_wrapper.init(embedding_dimension=embedding_size)
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

    history = [
        ChatMessage(
            role="user",
            content=[
                {
                    "type": "text",
                    "text": f"What is this image? Tell me which one of these products it is part of: {', '.join([p for p in documents])}",
                },
                {
                    "type": "image_url",
                    "image_url": query_image_path,
                },
            ],
        ),
    ]
    response = resolved_llm.chat(history).message.content
    assert "Coffee Machine Ultra Cool" in response


def get_local_resource_path(filename: str):
    dirname = os.path.dirname(__file__)
    e2e_tests_dir = os.path.dirname(dirname)
    return os.path.join(e2e_tests_dir, "resources", filename)
