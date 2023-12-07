import os
from abc import ABC, abstractmethod

import pytest
from e2e_tests.conftest import (
    AstraRef,
    get_required_env,
    get_astra_dev_ref,
    get_astra_prod_ref,
    delete_all_astra_collections,
    delete_astra_collection,
)
from llama_index import (
    VectorStoreIndex,
    StorageContext,
    ServiceContext,
    Document,
    OpenAIEmbedding,
)
from llama_index.embeddings import AzureOpenAIEmbedding
from llama_index.llms import OpenAI, AzureOpenAI
from llama_index.vector_stores import AstraDBVectorStore


class ContextMixin(ABC):
    @property
    @abstractmethod
    def name(self) -> str:
        ...

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        pass


class AstraDBVectorStoreContext(ContextMixin, ABC):
    def __init__(self, astra_ref: AstraRef):
        self.astra_ref = astra_ref
        self.vector_store = None

    def __enter__(self):
        delete_all_astra_collections(self.astra_ref)

        self.vector_store = AstraDBVectorStore(
            collection_name=self.astra_ref.collection,
            embedding_dimension=1536,
            token=self.astra_ref.token,
            api_endpoint=self.astra_ref.api_endpoint,
        )
        return self.vector_store

    def __exit__(self, exc_type, exc_value, traceback):
        delete_astra_collection(self.astra_ref)


class DevAstraDBVectorStoreContext(AstraDBVectorStoreContext):
    name = "astradb-dev"

    def __init__(self):
        super().__init__(get_astra_dev_ref())


class ProdAstraDBVectorStoreContext(AstraDBVectorStoreContext):
    name = "astradb-prod"

    def __init__(self):
        super().__init__(get_astra_prod_ref())


class OpenAILLMContext(ContextMixin):
    name = "openai"

    def __enter__(self):
        key = get_required_env("OPEN_AI_KEY")
        return OpenAI(api_key=key)


class OpenAIEmbeddingsContext(ContextMixin):
    name = "openai"

    def __enter__(self):
        key = get_required_env("OPEN_AI_KEY")
        return OpenAIEmbedding(api_key=key)


class AzureOpenAILLMContext(ContextMixin):
    name = "openai-azure"

    def __enter__(self):
        return AzureOpenAI(
            azure_deployment=get_required_env("AZURE_OPEN_AI_CHAT_MODEL_DEPLOYMENT"),
            azure_endpoint=get_required_env("AZURE_OPEN_AI_ENDPOINT"),
            api_key=get_required_env("AZURE_OPEN_AI_KEY"),
            api_version="2023-07-01-preview",
        )


class AzureOpenAIEmbeddingsContext(ContextMixin):
    name = "openai-azure"

    def __enter__(self):
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


def test_openai():
    _run_test(ProdAstraDBVectorStoreContext, OpenAILLMContext, OpenAIEmbeddingsContext)


@pytest.mark.parametrize(
    "vector_store", [ProdAstraDBVectorStoreContext, DevAstraDBVectorStoreContext]
)
def test_openai_azure(vector_store):
    _run_test(vector_store, AzureOpenAILLMContext, AzureOpenAIEmbeddingsContext)


def _run_test(vector_store_ctx, llm_ctx, embed_model_ctx):
    os.environ[
        "RAGSTACK_E2E_TESTS_TEST_INFO"
    ] = f"llama_index_retrieve::{llm_ctx.name},{embed_model_ctx.name},{vector_store_ctx.name}"
    with vector_store_ctx() as vector_store, llm_ctx() as llm, embed_model_ctx() as embed_model:
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
