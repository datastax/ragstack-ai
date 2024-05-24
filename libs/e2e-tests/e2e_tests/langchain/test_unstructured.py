import pytest

from langchain.text_splitter import TokenTextSplitter
from langchain_community.document_loaders import UnstructuredAPIFileLoader
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough

from langchain_openai import (
    ChatOpenAI,
    OpenAIEmbeddings,
)

from e2e_tests.langchain.rag_application import BASIC_QA_PROMPT
from e2e_tests.test_utils import get_local_resource_path

from e2e_tests.conftest import (
    set_current_test_info,
    get_required_env,
)

from e2e_tests.test_utils.vector_store_handler import (
    VectorStoreTestContext,
)


@pytest.mark.parametrize("vector_store", ["cassandra", "astra_db"])
@pytest.mark.parametrize("unstructured_mode", ["single", "elements"])
def test_unstructured_api(vector_store, unstructured_mode, request):
    set_current_test_info(
        "langchain::unstructured-api",
        f"{unstructured_mode},{vector_store}",
    )

    vector_store_context: VectorStoreTestContext = request.getfixturevalue(vector_store)
    embedding = OpenAIEmbeddings(openai_api_key=get_required_env("OPENAI_API_KEY"))
    vector_store = vector_store_context.new_langchain_vector_store(embedding=embedding)

    loader = UnstructuredAPIFileLoader(
        file_path=get_local_resource_path("tree.pdf"),
        mode=unstructured_mode,
        strategy="auto",
        api_key=get_required_env("UNSTRUCTURED_API_KEY"),
        url=get_required_env("UNSTRUCTURED_API_URL"),
    )

    splitter = TokenTextSplitter(chunk_size=512, chunk_overlap=0)
    vector_store.add_documents(splitter.split_documents(loader.load()))

    prompt = PromptTemplate.from_template(BASIC_QA_PROMPT)
    llm = ChatOpenAI(
        openai_api_key=get_required_env("OPENAI_API_KEY"),
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
    assert len(response) > 0
