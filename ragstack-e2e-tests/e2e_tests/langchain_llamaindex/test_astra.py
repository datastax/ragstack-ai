from uuid import uuid4

import langchain_core.documents
import pytest
from e2e_tests.conftest import get_required_env, is_astra
from langchain.chains import ConversationalRetrievalChain
from langchain.chat_models import ChatOpenAI
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores.astradb import AstraDB

try:
    # llamaindex 0.9.x
    from llama_index import (
        Document,
        OpenAIEmbedding,
        ServiceContext,
        StorageContext,
        VectorStoreIndex,
    )
    from llama_index.llms import OpenAI
    from llama_index.vector_stores import (
        AstraDBVectorStore,
        MetadataFilters,
        ExactMatchFilter,
    )
except ImportError:
    # llamaindex 0.10.x
    from llama_index.core import ServiceContext, StorageContext, VectorStoreIndex
    from llama_index.core.schema import Document
    from llama_index.core.vector_stores import (
        MetadataFilters,
        ExactMatchFilter,
    )
    from llama_index.vector_stores.astra import AstraDBVectorStore
    from llama_index.embeddings.openai import OpenAIEmbedding
    from llama_index.llms.openai import OpenAI


from e2e_tests.test_utils import skip_test_due_to_implementation_not_supported
from e2e_tests.test_utils.astradb_vector_store_handler import (
    AstraDBVectorStoreHandler,
    AstraRef,
)
from e2e_tests.test_utils.vector_store_handler import VectorStoreImplementation


def test_ingest_llama_retrieve_langchain(astra_ref: AstraRef):
    print("Running test_ingest_llama_retrieve_langchain")
    token = astra_ref.token
    api_endpoint = astra_ref.api_endpoint
    openai_key = get_required_env("OPEN_AI_KEY")
    collection = astra_ref.collection
    llm_model = "gpt-3.5-turbo"

    langchain_embeddings = OpenAIEmbeddings(openai_api_key=openai_key)

    langchain_vector_db = AstraDB(
        collection_name=collection,
        embedding=langchain_embeddings,
        token=token,
        api_endpoint=api_endpoint,
    )
    langchain_vector_db.delete_collection()

    document_id = str(uuid4())
    document = Document(
        text="RAGStack is a framework to run LangChain and LlamaIndex in production",
        id=document_id,
        metadata={"source": "llama-index-ingest"},
    )
    documents = [document]

    astra_db_store = AstraDBVectorStore(
        token=token,
        api_endpoint=api_endpoint,
        collection_name=collection,
        embedding_dimension=1536,
    )

    embed_model = OpenAIEmbedding(api_key=openai_key)
    llm = OpenAI(api_key=openai_key, model=llm_model, streaming=False, temperature=0)

    service_context = ServiceContext.from_defaults(embed_model=embed_model, llm=llm)

    storage_context = StorageContext.from_defaults(vector_store=astra_db_store)

    index = VectorStoreIndex.from_documents(
        documents, storage_context=storage_context, service_context=service_context
    )

    # Verify that the document is in the vector store
    retriever = index.as_retriever()
    documents_from_llamaindex = retriever.retrieve("What is RAGStack ?")
    assert len(documents_from_llamaindex) > 0
    for doc in documents_from_llamaindex:
        print("doc:", doc)
        assert "framework" in doc.text

    # Basic RAG with LlamaIndex
    query_engine = index.as_query_engine()
    response = query_engine.query("What is RAGStack ?")
    print("response:", response)

    assert "framework" in response.response

    # Use LangChain now
    # Verify that the document is in the vector store

    retriever = langchain_vector_db.as_retriever()
    documents_from_langchain = retriever.invoke("What is RAGStack ?")
    assert len(documents_from_langchain) > 0
    for doc in documents_from_langchain:
        print("doc:", doc)
        assert "framework" in doc.page_content

    # Verify compatibility of metadata filtering
    retriever = langchain_vector_db.as_retriever(
        search_kwargs={"filter": {"source": "llama-index-ingest"}}
    )
    documents_from_langchain = retriever.invoke("What is RAGStack ?")
    assert len(documents_from_langchain) > 0
    for doc in documents_from_langchain:
        print("doc:", doc)
        assert "framework" in doc.page_content

    retriever_no_docs = langchain_vector_db.as_retriever(
        search_kwargs={"filter": {"source": "don't-find-anything-please"}}
    )
    documents_from_langchain = retriever_no_docs.invoke("What is RAGStack ?")
    assert len(documents_from_langchain) == 0

    # Basic RAG with LangChain

    llm = ChatOpenAI(
        model_name=llm_model, temperature=0, openai_api_key=openai_key, streaming=False
    )
    chain = ConversationalRetrievalChain.from_llm(
        llm=llm, retriever=retriever, verbose=False
    )
    response = chain.invoke({"question": "What is RAGStack ?", "chat_history": []})
    print("response:", response)


def test_ingest_langchain_retrieve_llama_index(astra_ref: AstraRef):
    print("Running test_ingest_langchain_retrieve_llama_index")
    token = astra_ref.token
    api_endpoint = astra_ref.api_endpoint
    openai_key = get_required_env("OPEN_AI_KEY")
    collection = astra_ref.collection
    llm_model = "gpt-3.5-turbo"

    document = langchain_core.documents.Document(
        page_content="RAGStack is a framework to run LangChain and LlamaIndex in production",
        metadata={"source": "llama-index-ingest"},
    )
    documents = [document]

    langchain_embeddings = OpenAIEmbeddings(openai_api_key=openai_key)

    vector_db = AstraDB(
        collection_name=collection,
        embedding=langchain_embeddings,
        token=token,
        api_endpoint=api_endpoint,
    )

    vector_db.add_documents(documents)

    # Verify that the document is in the vector store

    retriever = vector_db.as_retriever()
    documents_from_langchain = retriever.invoke("What is RAGStack ?")
    assert len(documents_from_langchain) > 0
    for doc in documents_from_langchain:
        print("doc:", doc)
        assert "framework" in doc.page_content

    # Basic RAG with LangChain

    llm = ChatOpenAI(
        model_name=llm_model, temperature=0, openai_api_key=openai_key, streaming=False
    )
    chain = ConversationalRetrievalChain.from_llm(
        llm=llm, retriever=retriever, verbose=False
    )
    response = chain.invoke({"question": "What is RAGStack ?", "chat_history": []})
    print("response:", response)

    # Now retrieve with LlamaIndex

    astra_db_store = AstraDBVectorStore(
        token=token,
        api_endpoint=api_endpoint,
        collection_name=collection,
        embedding_dimension=1536,
    )

    embed_model = OpenAIEmbedding(api_key=openai_key)
    llm = OpenAI(api_key=openai_key, model=llm_model, streaming=False, temperature=0)

    service_context = ServiceContext.from_defaults(embed_model=embed_model, llm=llm)

    index = VectorStoreIndex.from_vector_store(
        vector_store=astra_db_store, service_context=service_context
    )

    # Verify that the document is in the vector store
    retriever = index.as_retriever()
    documents_from_llamaindex = retriever.retrieve("What is RAGStack ?")
    assert len(documents_from_llamaindex) > 0
    for doc in documents_from_llamaindex:
        print("doc:", doc)
        assert "framework" in doc.text

    # Verify compatibility of metadata filtering
    filters = MetadataFilters(
        filters=[ExactMatchFilter(key="source", value="llama-index-ingest")]
    )
    retriever = index.as_retriever(filters=filters)
    documents_from_llamaindex = retriever.retrieve("What is RAGStack ?")
    assert len(documents_from_llamaindex) > 0
    for doc in documents_from_llamaindex:
        print("doc:", doc)
        assert "framework" in doc.text

    filters = MetadataFilters(
        filters=[ExactMatchFilter(key="source", value="don't-find-anything-please")]
    )
    retriever_no_docs = index.as_retriever(filters=filters)
    documents_from_llamaindex = retriever_no_docs.retrieve("What is RAGStack ?")
    assert len(documents_from_llamaindex) == 0

    # Basic RAG with LlamaIndex
    query_engine = index.as_query_engine()
    response = query_engine.query("What is RAGStack ?")
    print("response:", response)

    assert "framework" in response.response


@pytest.fixture
def astra_ref() -> AstraRef:
    if not is_astra:
        skip_test_due_to_implementation_not_supported("astradb")
    handler = AstraDBVectorStoreHandler(VectorStoreImplementation.ASTRADB)
    handler.before_test()
    yield handler.astra_ref
    handler.after_test()
