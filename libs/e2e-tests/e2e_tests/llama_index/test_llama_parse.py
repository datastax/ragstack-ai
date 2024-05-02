import pytest

try:
    from llama_parse import LlamaParse
except ImportError:
    pytest.skip("llama_parse is not supported, skipping tests", allow_module_level=True)


from llama_index.core import ServiceContext, StorageContext, VectorStoreIndex
from llama_index.llms.openai import OpenAI
from llama_index.embeddings.openai import OpenAIEmbedding


from e2e_tests.conftest import (
    set_current_test_info,
    get_required_env,
)
from e2e_tests.test_utils import get_local_resource_path
from e2e_tests.test_utils.vector_store_handler import (
    VectorStoreTestContext,
)


@pytest.fixture
def llama_parse_text():
    return "text", LlamaParse(result_type="text")


@pytest.fixture
def llama_parse_markdown():
    return "markdown", LlamaParse(result_type="markdown")


@pytest.mark.parametrize("vector_store", ["cassandra", "astra_db"])
@pytest.mark.parametrize(
    "llama_parse_instance",
    ["llama_parse_text", "llama_parse_markdown"],
)
def test_llama_parse(vector_store, llama_parse_instance, request):
    vector_store_context: VectorStoreTestContext = request.getfixturevalue(vector_store)
    lp_type, lp = request.getfixturevalue(llama_parse_instance)
    llm = OpenAI(api_key=get_required_env("OPENAI_API_KEY"))
    embedding = OpenAIEmbedding(api_key=get_required_env("OPENAI_API_KEY"))

    set_current_test_info(
        "llama_index::llama_parse",
        f"{lp_type},{vector_store}",
    )
    vector_store = vector_store_context.new_llamaindex_vector_store(
        embedding_dimension=1536
    )
    storage_context = StorageContext.from_defaults(vector_store=vector_store)
    service_context = ServiceContext.from_defaults(llm=llm, embed_model=embedding)

    file_path = get_local_resource_path("tree.pdf")
    documents = lp.load_data(file_path)

    index = VectorStoreIndex.from_documents(
        documents, storage_context=storage_context, service_context=service_context
    )

    retriever = index.as_retriever()
    assert len(retriever.retrieve("What was Eldenroot?")) > 0
