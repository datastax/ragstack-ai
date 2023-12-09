from abc import ABC, abstractmethod
from typing import Type

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
)
from llama_index.embeddings.utils import EmbedType
from llama_index.llms import OpenAI, AzureOpenAI, Vertex, HuggingFaceInferenceAPI
from llama_index.vector_stores import AstraDBVectorStore
from llama_index.vector_stores.types import VectorStore


class ContextMixin(ABC):
    def __init__(self):
        pass

    @property
    @abstractmethod
    def name(self) -> str:
        ...

    @abstractmethod
    def __enter__(self):
        ...

    def __exit__(self, exc_type, exc_val, exc_tb):
        pass


class EmbeddingsContextMixin(ContextMixin):
    @abstractmethod
    def __enter__(self) -> EmbedType:
        ...

    @property
    @abstractmethod
    def embedding_dimension(self) -> int:
        ...


class VectorStoreContext(ContextMixin, ABC):
    def __init__(self, embedding_dimension: int):
        super().__init__()
        self.embedding_dimension = embedding_dimension

    @abstractmethod
    def __enter__(self) -> VectorStore:
        ...


class AstraDBVectorStoreContext(VectorStoreContext, ABC):
    def __init__(self, astra_ref: AstraRef, embedding_dimension: int):
        super().__init__(embedding_dimension)
        self.astra_ref = astra_ref
        self.vector_store = None

    def __enter__(self) -> AstraDBVectorStore:
        delete_all_astra_collections(self.astra_ref)

        self.vector_store = AstraDBVectorStore(
            collection_name=self.astra_ref.collection,
            embedding_dimension=self.embedding_dimension,
            token=self.astra_ref.token,
            api_endpoint=self.astra_ref.api_endpoint,
        )
        return self.vector_store

    def __exit__(self, exc_type, exc_value, traceback):
        delete_astra_collection(self.astra_ref)


class DevAstraDBVectorStoreContext(AstraDBVectorStoreContext):
    name = "astradb-dev"

    def __init__(self, embedding_dimension: int):
        super().__init__(get_astra_dev_ref(), embedding_dimension)


class ProdAstraDBVectorStoreContext(AstraDBVectorStoreContext):
    name = "astradb-prod"

    def __init__(self, embedding_dimension: int):
        super().__init__(get_astra_prod_ref(), embedding_dimension)


class OpenAILLMContext(ContextMixin):
    name = "openai"

    def __enter__(self) -> OpenAI:
        key = get_required_env("OPEN_AI_KEY")
        return OpenAI(api_key=key)


class OpenAIEmbeddingsContext(EmbeddingsContextMixin):
    name = "openai"
    embedding_dimension = 1536

    def __enter__(self) -> OpenAIEmbedding:
        key = get_required_env("OPEN_AI_KEY")
        return OpenAIEmbedding(api_key=key)


class AzureOpenAILLMContext(ContextMixin):
    name = "openai-azure"

    def __enter__(self) -> AzureOpenAI:
        return AzureOpenAI(
            azure_deployment=get_required_env("AZURE_OPEN_AI_CHAT_MODEL_DEPLOYMENT"),
            azure_endpoint=get_required_env("AZURE_OPEN_AI_ENDPOINT"),
            api_key=get_required_env("AZURE_OPEN_AI_KEY"),
            api_version="2023-07-01-preview",
        )


class AzureOpenAIEmbeddingsContext(EmbeddingsContextMixin):
    name = "openai-azure"
    embedding_dimension = 1536

    def __enter__(self) -> AzureOpenAIEmbedding:
        model_and_deployment = get_required_env(
            "AZURE_OPEN_AI_EMBEDDINGS_MODEL_DEPLOYMENT"
        )
        return AzureOpenAIEmbedding(
            model=model_and_deployment,
            deployment_name=model_and_deployment,
            api_key=get_required_env("AZURE_OPEN_AI_KEY"),
            azure_endpoint=get_required_env("AZURE_OPEN_AI_ENDPOINT"),
            api_version="2023-05-15",
            embed_batch_size=1,
        )


class VertexLLMContext(ContextMixin):
    name = "vertex-ai"

    def __enter__(self):
        return Vertex(model="chat-bison")


class VertexEmbeddingsContext(EmbeddingsContextMixin):
    name = "vertex-ai"
    embedding_dimension = 768

    def __enter__(self) -> VertexAIEmbeddings:
        # Llama-Index doesn't have Vertex AI embedding
        # so we use LangChain's wrapped one
        return VertexAIEmbeddings(model_name="textembedding-gecko")


class HuggingFaceHubLLMContext(ContextMixin):
    name = "huggingface-hub"

    def __enter__(self):
        return HuggingFaceInferenceAPI(
            model_name="google/flan-t5-xxl",
            token=get_required_env("HUGGINGFACE_HUB_KEY"),
        )


class HuggingFaceHubEmbeddingsContext(EmbeddingsContextMixin):
    name = "huggingface-hub"
    embedding_dimension = 384

    def __enter__(self) -> HuggingFaceInferenceAPIEmbeddings:
        # There's a bug in Llama-Index HuggingFace Hub embedding
        # so we use LangChain's wrapped one for now
        return HuggingFaceInferenceAPIEmbeddings(
            api_key=get_required_env("HUGGINGFACE_HUB_KEY"),
            model_name="sentence-transformers/all-MiniLM-l6-v2",
        )


def test_openai_azure_astra_dev():
    _run_test(
        DevAstraDBVectorStoreContext,
        AzureOpenAIEmbeddingsContext,
        AzureOpenAILLMContext,
    )


@pytest.mark.parametrize(
    "embedding,llm",
    [
        (OpenAIEmbeddingsContext, OpenAILLMContext),
        (AzureOpenAIEmbeddingsContext, AzureOpenAILLMContext),
        (VertexEmbeddingsContext, VertexLLMContext),
        (HuggingFaceHubEmbeddingsContext, HuggingFaceHubLLMContext),
    ],
)
def test_rag(embedding, llm):
    _run_test(ProdAstraDBVectorStoreContext, embedding, llm)


def _run_test(
    vector_store_ctx_cls: Type[VectorStoreContext],
    embed_model_ctx_cls: Type[EmbeddingsContextMixin],
    llm_ctx_cls: Type[ContextMixin],
):
    embed_model_ctx = embed_model_ctx_cls()
    llm_ctx = llm_ctx_cls()
    vector_store_ctx = vector_store_ctx_cls(embed_model_ctx.embedding_dimension)
    set_current_test_info(
        "llama_index_retrieve",
        f"{llm_ctx.name},{embed_model_ctx.name},{vector_store_ctx.name}",
    )
    with vector_store_ctx as vector_store, llm_ctx as llm, embed_model_ctx as embed_model:
        storage_context = StorageContext.from_defaults(vector_store=vector_store)
        service_context = ServiceContext.from_defaults(llm=llm, embed_model=embed_model)

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
            Document(
                text="MyFakeProductForTesting first release happened in June 2020."
            ),
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
