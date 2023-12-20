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
)
from llama_index.vector_stores import AstraDBVectorStore, CassandraVectorStore

from e2e_tests.conftest import (
    set_current_test_info,
    get_required_env,
    get_astra_ref,
    delete_all_astra_collections,
    delete_astra_collection,
)


@pytest.fixture
def astra_db():
    astra_ref = get_astra_ref()
    delete_all_astra_collections(astra_ref)

    yield (
        "astradb",
        lambda embedding_dimension: AstraDBVectorStore(
            collection_name=astra_ref.collection,
            embedding_dimension=embedding_dimension,
            token=astra_ref.token,
            api_endpoint=astra_ref.api_endpoint,
        ),
    )
    delete_astra_collection(astra_ref)


@pytest.fixture
def cassandra():
    astra_ref = get_astra_ref()
    delete_all_astra_collections(astra_ref)
    cassio.init(token=astra_ref.token, database_id=astra_ref.id)
    yield "cassandra", lambda embedding_dimension: CassandraVectorStore(
        embedding_dimension=embedding_dimension,
        session=None,
        keyspace="default_keyspace",
        table=astra_ref.collection,
    )
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


@pytest.mark.parametrize("vector_store", ["astra_db"])
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
    _run_test(
        request.getfixturevalue(vector_store),
        request.getfixturevalue(embedding),
        request.getfixturevalue(llm),
    )


def _run_test(vector_store, embedding, llm):
    embedding_name, embedding_dimensions, embedding = embedding
    vector_store_name, vector_store = vector_store
    vector_store = vector_store(embedding_dimensions)
    llm_name, llm = llm
    set_current_test_info(
        "llama_index::rag",
        f"{llm_name},{embedding_name},{vector_store_name}",
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
    assert "2020" in response
