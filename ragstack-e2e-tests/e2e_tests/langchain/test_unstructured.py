import pytest

from langchain_community.document_loaders import UnstructuredAPIFileLoader
from langchain.text_splitter import TokenTextSplitter
from langchain.schema.output_parser import StrOutputParser
from langchain.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain.chat_models import ChatOpenAI

from e2e_tests.langchain.rag_application import BASIC_QA_PROMPT
from e2e_tests.test_utils import get_local_resource_path

from e2e_tests.conftest import (
    set_current_test_info,
    get_required_env,
    get_vector_store_handler,
)

from e2e_tests.test_utils.vector_store_handler import (
    VectorStoreImplementation,
    VectorStoreTestContext,
)


@pytest.fixture
def astra_db():
    handler = get_vector_store_handler(VectorStoreImplementation.ASTRADB)
    context = handler.before_test()
    yield context
    handler.after_test()


@pytest.fixture
def cassandra():
    handler = get_vector_store_handler(VectorStoreImplementation.CASSANDRA)
    context = handler.before_test()
    yield context
    handler.after_test()


@pytest.mark.parametrize("vector_store", ["cassandra", "astra_db"])
@pytest.mark.parametrize("unstructured_mode", ["single", "elements"])
def test_unstructured(vector_store, unstructured_mode, request):
    set_current_test_info(
        "lang_chain::unstructured",
        f"{unstructured_mode},{vector_store}",
    )

    vector_store_context: VectorStoreTestContext = request.getfixturevalue(vector_store)
    vector_store = vector_store_context.new_langchain_vector_store(
        embedding_dimension=1536
    )

    loader = UnstructuredAPIFileLoader(
        file_path=get_local_resource_path("tree.pdf"),
        mode=unstructured_mode,
        strategy="auto",
        api_key=get_required_env("UNSTRUCTURED_API_KEY"),
    )

    splitter = TokenTextSplitter(chunk_size=512, chunk_overlap=0)
    vector_store.add_documents(splitter.split_documents(loader.load()))

    prompt = PromptTemplate.from_template(BASIC_QA_PROMPT)
    llm = ChatOpenAI(
        openai_api_key=get_required_env("OPEN_AI_KEY"),
        model="gpt-3.5-turbo-16k",
        streaming=False,
        temperature=0,
    )

    chain = (
        {"context": vector_store.as_retriever(), "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )
    response = chain.invoke("What was Eldenroot?")
    print(response)
    assert len(response) > 0
