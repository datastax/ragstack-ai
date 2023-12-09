from abc import ABC, abstractmethod
from typing import Type

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
from langchain.schema.embeddings import Embeddings
from langchain.schema.vectorstore import VectorStore
from langchain.vectorstores import AstraDB, Cassandra


class ContextMixin(ABC):
    @property
    @abstractmethod
    def name(self) -> str:
        ...

    @abstractmethod
    def __enter__(self):
        ...

    def __exit__(self, exc_type, exc_val, exc_tb):
        pass


class VectorStoreContext(ContextMixin, ABC):
    def __init__(self, embedding: Embeddings):
        self.embedding = embedding

    @abstractmethod
    def __enter__(self) -> VectorStore:
        ...


class AstraCloudVectorStoreContext(VectorStoreContext, ABC):
    def __init__(self, astra_ref: AstraRef, embedding: Embeddings):
        super().__init__(embedding)
        self.astra_ref = astra_ref

    def __enter__(self):
        delete_all_astra_collections(self.astra_ref)

    def __exit__(self, exc_type, exc_value, traceback):
        delete_astra_collection(self.astra_ref)


class AstraDBVectorStoreContext(AstraCloudVectorStoreContext, ABC):
    def __enter__(self) -> AstraDB:
        super().__enter__()
        return AstraDB(
            collection_name=self.astra_ref.collection,
            embedding=self.embedding,
            token=self.astra_ref.token,
            api_endpoint=self.astra_ref.api_endpoint,
        )


class DevAstraDBVectorStoreContext(AstraDBVectorStoreContext):
    name = "astradb-dev"

    def __init__(self, embedding: Embeddings):
        super().__init__(get_astra_dev_ref(), embedding)


class ProdAstraDBVectorStoreContext(AstraDBVectorStoreContext):
    name = "astradb-prod"

    def __init__(self, embedding: Embeddings):
        super().__init__(get_astra_prod_ref(), embedding)


class CassandraVectorStoreContext(AstraCloudVectorStoreContext):
    name = "cassandra"

    def __init__(self, embedding: Embeddings):
        super().__init__(get_astra_prod_ref(), embedding)

    def __enter__(self) -> Cassandra:
        super().__enter__()
        cassio.init(token=self.astra_ref.token, database_id=self.astra_ref.id)
        return Cassandra(
            embedding=self.embedding,
            session=None,
            keyspace="default_keyspace",
            table_name=self.astra_ref.collection,
        )


class OpenAILLMContext(ContextMixin):
    name = "openai"

    def __enter__(self) -> ChatOpenAI:
        return ChatOpenAI(
            openai_api_key=get_required_env("OPEN_AI_KEY"),
            model="gpt-3.5-turbo-16k",
            streaming=True,
            temperature=0,
        )


class OpenAIEmbeddingsContext(ContextMixin):
    name = "openai"

    def __enter__(self) -> OpenAIEmbeddings:
        return OpenAIEmbeddings(openai_api_key=get_required_env("OPEN_AI_KEY"))


class AzureOpenAILLMContext(ContextMixin):
    name = "openai-azure"

    def __enter__(self) -> AzureChatOpenAI:
        return AzureChatOpenAI(
            azure_deployment=get_required_env("AZURE_OPEN_AI_CHAT_MODEL_DEPLOYMENT"),
            openai_api_base=get_required_env("AZURE_OPEN_AI_ENDPOINT"),
            openai_api_key=get_required_env("AZURE_OPEN_AI_KEY"),
            openai_api_type="azure",
            openai_api_version="2023-07-01-preview",
        )


class AzureOpenAIEmbeddingsContext(ContextMixin):
    name = "openai-azure"

    def __enter__(self) -> AzureOpenAIEmbeddings:
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


class VertexLLMContext(ContextMixin):
    name = "vertex-ai"

    def __enter__(self) -> ChatVertexAI:
        return ChatVertexAI()


class VertexEmbeddingsContext(ContextMixin):
    name = "vertex-ai"

    def __enter__(self) -> VertexAIEmbeddings:
        return VertexAIEmbeddings(model_name="textembedding-gecko")


class BedrockAnthropicLLMContext(ContextMixin):
    name = "bedrock-anthropic"

    def __enter__(self) -> BedrockChat:
        return BedrockChat(
            model_id="anthropic.claude-v2",
            region_name=get_required_env("BEDROCK_AWS_REGION"),
        )


class BedrockMetaLLMContext(ContextMixin):
    name = "bedrock-meta"

    def __enter__(self) -> BedrockChat:
        return BedrockChat(
            model_id="meta.llama2-13b-chat-v1",
            region_name=get_required_env("BEDROCK_AWS_REGION"),
        )


class BedrockTitanEmbeddingsContext(ContextMixin):
    name = "bedrock-titan"

    def __enter__(self) -> BedrockEmbeddings:
        return BedrockEmbeddings(
            model_id="amazon.titan-embed-text-v1",
            region_name=get_required_env("BEDROCK_AWS_REGION"),
        )


class BedrockCohereEmbeddingsContext(ContextMixin):
    name = "bedrock-cohere"

    def __enter__(self) -> BedrockEmbeddings:
        return BedrockEmbeddings(
            model_id="cohere.embed-english-v3",
            region_name=get_required_env("BEDROCK_AWS_REGION"),
        )


class HuggingFaceHubLLMContext(ContextMixin):
    name = "huggingface-hub"

    def __enter__(self) -> HuggingFaceHub:
        return HuggingFaceHub(
            repo_id="google/flan-t5-xxl",
            huggingfacehub_api_token=get_required_env("HUGGINGFACE_HUB_KEY"),
            model_kwargs={"temperature": 0.5, "max_length": 64},
        )


class HuggingFaceHubEmbeddingsContext(ContextMixin):
    name = "huggingface-hub"

    def __enter__(self) -> HuggingFaceInferenceAPIEmbeddings:
        return HuggingFaceInferenceAPIEmbeddings(
            api_key=get_required_env("HUGGINGFACE_HUB_KEY"),
            model_name="sentence-transformers/all-MiniLM-l6-v2",
        )


def test_openai_azure_astra_dev():
    test_rag(
        DevAstraDBVectorStoreContext,
        AzureOpenAIEmbeddingsContext,
        AzureOpenAILLMContext,
    )


@pytest.mark.parametrize(
    "vector_store_ctx_cls",
    [
        ProdAstraDBVectorStoreContext,
        CassandraVectorStoreContext,
    ],
)
@pytest.mark.parametrize(
    "embedding_ctx_cls,llm_ctx_cls",
    [
        (OpenAIEmbeddingsContext, OpenAILLMContext),
        (AzureOpenAIEmbeddingsContext, AzureOpenAILLMContext),
        (VertexEmbeddingsContext, VertexLLMContext),
        (BedrockTitanEmbeddingsContext, BedrockAnthropicLLMContext),
        (BedrockCohereEmbeddingsContext, BedrockMetaLLMContext),
        (HuggingFaceHubEmbeddingsContext, HuggingFaceHubLLMContext),
    ],
)
def test_rag(
    vector_store_ctx_cls: Type[VectorStoreContext],
    embedding_ctx_cls: Type[ContextMixin],
    llm_ctx_cls: Type[ContextMixin],
):
    embedding_ctx = embedding_ctx_cls()
    with embedding_ctx as embedding:
        llm_ctx = llm_ctx_cls()
        vector_store_ctx = vector_store_ctx_cls(embedding)
        with vector_store_ctx as vector_store, llm_ctx as llm:
            set_current_test_info(
                "simple_rag",
                f"{llm_ctx.name},{embedding_ctx.name},{vector_store_ctx.name}",
            )
            response = run_application(
                question="When was released MyFakeProductForTesting for the first time ?",  # noqa: E501
                vector_store=vector_store,
                llm=llm,
            )
            print(f"Got response ${response}")
            assert "2020" in response
