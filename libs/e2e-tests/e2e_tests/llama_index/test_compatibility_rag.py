import logging

import pytest
from langchain.embeddings import HuggingFaceInferenceAPIEmbeddings
from langchain_google_vertexai import VertexAIEmbeddings

from llama_index.core import (
    ServiceContext,
    StorageContext,
    VectorStoreIndex,
    Document,
)
from llama_index.core.base.llms.types import ChatMessage
from llama_index.core.schema import ImageNode
from llama_index.llms.openai import OpenAI
from llama_index.llms.azure_openai import AzureOpenAI
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.embeddings.azure_openai import AzureOpenAIEmbedding
from llama_index.embeddings.bedrock import BedrockEmbedding
from llama_index.llms.vertex import Vertex
from llama_index.llms.bedrock import Bedrock
from llama_index.llms.gemini import Gemini
from llama_index.multi_modal_llms.gemini import GeminiMultiModal

from e2e_tests.conftest import (
    set_current_test_info,
    get_required_env,
)
from vertexai.vision_models import MultiModalEmbeddingModel, Image

from e2e_tests.test_utils import get_local_resource_path
from e2e_tests.test_utils.vector_store_handler import (
    VectorStoreTestContext,
)


def _openai_llm(**kwargs) -> OpenAI:
    return OpenAI(api_key=get_required_env("OPENAI_API_KEY"), **kwargs)


@pytest.fixture
def openai_gpt35turbo_llm():
    return _openai_llm(model="gpt-3.5-turbo")


@pytest.fixture
def openai_gpt4_llm():
    return _openai_llm(model="gpt-4")


def _openai_embeddings(**kwargs) -> OpenAIEmbedding:
    return OpenAIEmbedding(api_key=get_required_env("OPENAI_API_KEY"), **kwargs)


@pytest.fixture
def openai_ada002_embedding():
    return 1536, _openai_embeddings(model="text-embedding-ada-002")


@pytest.fixture
def openai_3small_embedding():
    return 1536, _openai_embeddings(model="text-embedding-3-small")


@pytest.fixture
def openai_3large_embedding():
    return 3072, _openai_embeddings(model="text-embedding-3-large")


@pytest.fixture
def azure_openai_gpt35turbo_llm():
    # model is configurable because it can be different from the deployment
    # but the targeting model must be gpt-35-turbo
    return AzureOpenAI(
        azure_deployment=get_required_env("AZURE_OPEN_AI_CHAT_MODEL_DEPLOYMENT"),
        azure_endpoint=get_required_env("AZURE_OPENAI_ENDPOINT"),
        api_key=get_required_env("AZURE_OPENAI_API_KEY"),
        api_version="2023-07-01-preview",
    )


@pytest.fixture
def azure_openai_ada002_embedding():
    # model is configurable because it can be different from the deployment
    # but the targeting model must be ada-002

    model_and_deployment = get_required_env("AZURE_OPEN_AI_EMBEDDINGS_MODEL_DEPLOYMENT")
    return (
        1536,
        AzureOpenAIEmbedding(
            model=model_and_deployment,
            deployment_name=model_and_deployment,
            api_key=get_required_env("AZURE_OPENAI_API_KEY"),
            azure_endpoint=get_required_env("AZURE_OPENAI_ENDPOINT"),
            api_version="2023-05-15",
            embed_batch_size=1,
        ),
    )


@pytest.fixture
def vertex_bison_llm():
    return Vertex(model="chat-bison")


@pytest.fixture
def vertex_gecko_langchain_embedding():
    # Llama-Index doesn't have Vertex AI embedding
    # so we use LangChain's wrapped one
    return 768, VertexAIEmbeddings(model_name="textembedding-gecko")


def _bedrock_llm(**kwargs):
    return Bedrock(
        aws_access_key_id=get_required_env("AWS_ACCESS_KEY_ID"),
        aws_secret_access_key=get_required_env("AWS_SECRET_ACCESS_KEY"),
        region_name=get_required_env("BEDROCK_AWS_REGION"),
        **kwargs,
    )


@pytest.fixture
def bedrock_anthropic_claudev2_llm():
    return _bedrock_llm(
        model="anthropic.claude-v2",
    )


@pytest.fixture
def bedrock_ai21_jurassic2mid_llm():
    return _bedrock_llm(
        model="ai21.j2-mid-v1",
    )


@pytest.fixture
def bedrock_meta_llama2_llm():
    return _bedrock_llm(model="meta.llama2-13b-chat-v1")


@pytest.fixture
def bedrock_titan_embedding():
    return (
        1536,
        BedrockEmbedding.from_credentials(
            model_name="amazon.titan-embed-text-v1",
            aws_region=get_required_env("BEDROCK_AWS_REGION"),
        ),
    )


@pytest.fixture
def bedrock_cohere_embedding():
    import boto3

    return (
        1024,
        BedrockEmbedding(
            client=boto3.Session(
                region_name=get_required_env("BEDROCK_AWS_REGION")
            ).client("bedrock-runtime"),
            model="cohere.embed-english-v3",
        ),
    )


@pytest.fixture
def huggingface_hub_flant5xxl_llm():
    # lazy import to supported disabling test
    from llama_index.llms.huggingface import HuggingFaceInferenceAPI

    return HuggingFaceInferenceAPI(
        model_name="google/flan-t5-xxl",
        token=get_required_env("HUGGINGFACE_HUB_KEY"),
    )


@pytest.fixture
def huggingface_hub_minilml6v2_embedding():
    # There's a bug in Llama-Index HuggingFace Hub embedding
    # so we use LangChain's wrapped one for now
    return (
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
        ("openai_ada002_embedding", "openai_gpt35turbo_llm"),
        ("openai_3large_embedding", "openai_gpt35turbo_llm"),
        ("openai_3small_embedding", "openai_gpt4_llm"),
        ("azure_openai_ada002_embedding", "azure_openai_gpt35turbo_llm"),
        ("vertex_gecko_langchain_embedding", "vertex_bison_llm"),
        ("bedrock_titan_embedding", "bedrock_anthropic_claudev2_llm"),
        ("bedrock_cohere_embedding", "bedrock_ai21_jurassic2mid_llm"),
        ("bedrock_cohere_embedding", "bedrock_meta_llama2_llm"),
        # ("huggingface_hub_minilml6v2_embedding", "huggingface_hub_flant5xxl_llm"),
    ],
)
def test_rag(vector_store, embedding, llm, request):
    set_current_test_info(
        "llama_index::rag",
        f"{llm},{embedding},{vector_store}",
    )
    embedding_dimensions, embedding = request.getfixturevalue(embedding)
    vector_store_context: VectorStoreTestContext = request.getfixturevalue(vector_store)
    llm = request.getfixturevalue(llm)
    vector_store = vector_store_context.new_llamaindex_vector_store(
        embedding_dimension=embedding_dimensions
    )
    storage_context = StorageContext.from_defaults(vector_store=vector_store)
    service_context = ServiceContext.from_defaults(llm=llm, embed_model=embedding)

    documents = [
        Document(
            text="MyFakeProductForTesting is a versatile testing tool designed to streamline the testing process for software developers, quality assurance professionals, and product testers. It provides a comprehensive solution for testing various aspects of applications and systems, ensuring robust performance and functionality."
            # noqa: E501
        ),
        Document(
            text="MyFakeProductForTesting comes equipped with an advanced dynamic test scenario generator. This feature allows users to create realistic test scenarios by simulating various user interactions, system inputs, and environmental conditions. The dynamic nature of the generator ensures that tests are not only diverse but also adaptive to changes in the application under test."
            # noqa: E501
        ),
        Document(
            text="The product includes an intelligent bug detection and analysis module. It not only identifies bugs and issues but also provides in-depth analysis and insights into the root causes. The system utilizes machine learning algorithms to categorize and prioritize bugs, making it easier for developers and testers to address critical issues first."
            # noqa: E501
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
    assert "2020" in response, f"Expected 2020 in response, got {response}"


@pytest.fixture
def vertex_gemini_multimodal_embedding():
    # this is from LangChain, LLamaIndex does not have Vertex Embeddings support yet
    return MultiModalEmbeddingModel.from_pretrained("multimodalembedding@001"), 1408


@pytest.fixture
def vertex_gemini_pro_llm():
    return Vertex(model="gemini-pro")


def _complete_multimodal_vertex(llm, prompt, image_path):
    history = [
        ChatMessage(
            role="user",
            content=[
                {
                    "type": "text",
                    "text": prompt,
                },
                {
                    "type": "image_url",
                    "image_url": image_path,
                },
            ],
        ),
    ]
    return llm.chat(history).message.content


@pytest.fixture
def vertex_gemini_pro_vision_llm():
    return Vertex(model="gemini-pro-vision"), _complete_multimodal_vertex


@pytest.fixture
def gemini_pro_llm():
    return Gemini(
        api_key=get_required_env("GOOGLE_API_KEY"), model_name="models/gemini-pro"
    )


@pytest.fixture
def gemini_pro_vision_llm():
    return (
        GeminiMultiModal(
            api_key=get_required_env("GOOGLE_API_KEY"),
            model_name="models/gemini-pro-vision",
        ),
        lambda llm, prompt, image_path: llm.complete(
            prompt=prompt, image_documents=[ImageNode(image_path=image_path)]
        ).text,
    )


@pytest.mark.parametrize(
    "vector_store",
    ["astra_db", "cassandra"],
)
@pytest.mark.parametrize(
    "embedding,llm",
    [
        # disable due to this bug: https://github.com/googleapis/python-aiplatform/issues/3227
        # ("vertex_gemini_multimodal_embedding", "vertex_gemini_pro_vision_llm"),
        ("vertex_gemini_multimodal_embedding", "gemini_pro_vision_llm"),
    ],
)
def test_multimodal(vector_store, embedding, llm, request):
    set_current_test_info(
        "llama_index::multimodal",
        f"{llm},{embedding},{vector_store}",
    )

    resolved_embedding, embedding_size = request.getfixturevalue(embedding)

    vector_store_context = request.getfixturevalue(vector_store)
    enhanced_vector_store = vector_store_context.new_llamaindex_vector_store(
        embedding_dimension=embedding_size
    )
    resolved_llm, llm_complete_fn = request.getfixturevalue(llm)

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

        enhanced_vector_store.put_document(
            p["name"], p["name"], {}, embeddings.image_embedding
        )

    query_image_path = get_local_resource_path("coffee_maker_part.png")
    img = Image.load_from_file(query_image_path)
    embeddings = resolved_embedding.get_embeddings(
        image=img, contextual_text="Coffee Maker Part"
    )

    documents = enhanced_vector_store.search_documents(embeddings.image_embedding, 3)
    docs_str = ", ".join([f"'{p}'" for p in documents])
    prompt = f"Tell me which one of these products it is part of. Only include product from the ones below: {docs_str}."
    logging.info(f"Prompt: {prompt}")
    response = llm_complete_fn(resolved_llm, prompt, query_image_path)
    assert (
        "Coffee Machine Ultra Cool" in response
    ), f"Expected Coffee Machine Ultra Cool in response, got {response}"


@pytest.mark.parametrize(
    "chat",
    ["gemini_pro_llm", "vertex_gemini_pro_llm"],
)
def test_chat(chat, request):
    set_current_test_info("llama_index::chat", chat)
    chat_model = request.getfixturevalue(chat)
    response = chat_model.complete("Hello! Where Archimede was born?")
    assert (
        "Syracuse" in response.text
    ), f"Expected Syracuse in response, got {response.text}"
