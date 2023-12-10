import cassio
import pytest
from e2e_tests.conftest import (
    AstraRef,
    set_current_test_info,
    get_required_env,
    get_astra_dev_ref,
    get_astra_prod_ref,
    delete_all_astra_collections,
    delete_astra_collection,
)
from e2e_tests.langchain.chat_application import run_application
from langchain.chat_models import ChatOpenAI, AzureChatOpenAI, ChatVertexAI, BedrockChat
from langchain.embeddings import (
    OpenAIEmbeddings,
    VertexAIEmbeddings,
    BedrockEmbeddings,
    HuggingFaceInferenceAPIEmbeddings,
)
from langchain.embeddings.azure_openai import AzureOpenAIEmbeddings
from langchain.llms.huggingface_hub import HuggingFaceHub
from langchain.vectorstores import AstraDB, Cassandra


def astra_db(name, astra_ref: AstraRef):
    delete_all_astra_collections(astra_ref)

    yield (
        name,
        lambda embedding: AstraDB(
            collection_name=astra_ref.collection,
            embedding=embedding,
            token=astra_ref.token,
            api_endpoint=astra_ref.api_endpoint,
        ),
    )
    delete_astra_collection(astra_ref)


@pytest.fixture
def astra_db_prod():
    yield from astra_db("astradb-prod", get_astra_prod_ref())


@pytest.fixture
def astra_db_dev():
    yield from astra_db("astradb-dev", get_astra_dev_ref())


@pytest.fixture
def cassandra():
    astra_ref = get_astra_prod_ref()
    delete_all_astra_collections(astra_ref)
    cassio.init(token=astra_ref.token, database_id=astra_ref.id)
    yield "cassandra", lambda embedding: Cassandra(
        embedding=embedding,
        session=None,
        keyspace="default_keyspace",
        table_name=astra_ref.collection,
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
        model_kwargs={"temperature": 0.5, "max_length": 64},
    )


@pytest.fixture
def huggingface_hub_embedding():
    return "huggingface-hub", HuggingFaceInferenceAPIEmbeddings(
        api_key=get_required_env("HUGGINGFACE_HUB_KEY"),
        model_name="sentence-transformers/all-MiniLM-l6-v2",
    )


def test_openai_azure_astra_dev(astra_db_dev, azure_openai_embedding, azure_openai_llm):
    _run_test(astra_db_dev, azure_openai_embedding, azure_openai_llm)


@pytest.mark.parametrize("vector_store", ["astra_db_prod", "cassandra"])
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
    embedding_name, embedding = embedding
    vector_store_name, vector_store = vector_store
    vector_store = vector_store(embedding)
    llm_name, llm = llm
    set_current_test_info(
        "simple_rag",
        f"{llm_name},{embedding_name},{vector_store_name}",
    )
    response = run_application(
        question="When was released MyFakeProductForTesting for the first time ?",
        vector_store=vector_store,
        llm=llm,
    )
    print(f"Got response ${response}")
    assert "2020" in response
