import logging

import pytest
from langchain.embeddings import VertexAIEmbeddings, HuggingFaceInferenceAPIEmbeddings

try:
    # llamaindex 0.9.x
    from llama_index import ServiceContext, StorageContext, VectorStoreIndex, Document
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
        Gemini,
    )
    from llama_index.multi_modal_llms import GeminiMultiModal
    from llama_index.schema import ImageNode
except ImportError:
    # llamaindex 0.10.x
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
    from llama_index.llms.huggingface import HuggingFaceInferenceAPI
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
        region_name=get_required_env("BEDROCK_AWS_REGION"),
    )


@pytest.fixture
def bedrock_meta_llm():
    return "bedrock-meta", Bedrock(
        model="meta.llama2-13b-chat-v1",
        aws_access_key_id=get_required_env("AWS_ACCESS_KEY_ID"),
        aws_secret_access_key=get_required_env("AWS_SECRET_ACCESS_KEY"),
        region_name=get_required_env("BEDROCK_AWS_REGION"),
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
    import boto3

    return (
        "bedrock-cohere",
        1024,
        BedrockEmbedding(
            client=boto3.Session(
                region_name=get_required_env("BEDROCK_AWS_REGION")
            ).client("bedrock-runtime"),
            model="cohere.embed-english-v3",
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
        ("bedrock_titan_embedding", "bedrock_anthropic_llm"),
        ("bedrock_cohere_embedding", "bedrock_meta_llm"),
        ("huggingface_hub_embedding", "huggingface_hub_llm"),
    ],
)
def test_rag(vector_store, embedding, llm, request):
    embedding_name, embedding_dimensions, embedding = request.getfixturevalue(embedding)
    vector_store_context: VectorStoreTestContext = request.getfixturevalue(vector_store)
    llm_name, llm = request.getfixturevalue(llm)
    set_current_test_info(
        "llama_index::rag",
        f"{llm_name},{embedding_name},{vector_store}",
    )
    vector_store = vector_store_context.new_llamaindex_vector_store(
        embedding_dimension=embedding_dimensions
    )
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
