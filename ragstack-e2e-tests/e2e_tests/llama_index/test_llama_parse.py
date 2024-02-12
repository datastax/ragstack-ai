from llama_parse import LlamaParse
from llama_index import VectorStoreIndex

from test_astra import Environment
from test_compatibility_rag import get_local_resource_path


def test_llamaparse_as_text_with_vector_search(environment: Environment):
    print("test_llamaparse_with_vector_search")

    file_path = get_local_resource_path("tree.pdf")
    documents = LlamaParse(result_type="test").load_data(file_path)

    index = VectorStoreIndex.from_documents(
        documents,
        storage_context=environment.storage_context,
        service_context=environment.service_context,
    )

    # Verify that the document is in the vector store
    retriever = index.as_retriever()
    assert len(retriever.retrieve("What was Eldenroot?")) > 0


def test_llamaparse_as_markdown_with_vector_search(environment: Environment):
    print("test_llamaparse_with_vector_search")

    file_path = get_local_resource_path("tree.pdf")
    documents = LlamaParse(result_type="markdown").load_data(file_path)

    index = VectorStoreIndex.from_documents(
        documents,
        storage_context=environment.storage_context,
        service_context=environment.service_context,
    )

    # Verify that the document is in the vector store
    retriever = index.as_retriever()
    assert len(retriever.retrieve("What was Eldenroot?")) > 0
