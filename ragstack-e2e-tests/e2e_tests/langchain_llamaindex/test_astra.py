from uuid import uuid4

import langchain_core.documents
import pytest
from astrapy.db import AstraDB as LibAstraDB
from e2e_tests.conftest import get_required_env, get_astra_ref
from langchain.chains import ConversationalRetrievalChain
from langchain.chat_models import ChatOpenAI
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores.astradb import AstraDB
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
    MetadataFilter,
    ExactMatchFilter,
)


def test_ingest_llama_retrieve_langchain(environment):
    print("Running test_ingest_llama_retrieve_langchain")
    astra_ref = environment.astra_ref
    token = astra_ref.token
    api_endpoint = astra_ref.api_endpoint
    openai_key = get_required_env("OPEN_AI_KEY")
    collection = astra_ref.collection
    llm_model = "gpt-3.5-turbo"

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

    langchain_embeddings = OpenAIEmbeddings(openai_api_key=openai_key)

    vector_db = AstraDB(
        collection_name=collection,
        embedding=langchain_embeddings,
        token=token,
        api_endpoint=api_endpoint,
    )

    # Verify that the document is in the vector store

    retriever = vector_db.as_retriever()
    documents_from_langchain = retriever.invoke("What is RAGStack ?")
    assert len(documents_from_langchain) > 0
    for doc in documents_from_langchain:
        print("doc:", doc)
        assert "framework" in doc.page_content

    # Verify compatibility of metadata filtering
    retriever = vector_db.as_retriever(
        search_kwargs={"filter": {"source": "llama-index-ingest"}}
    )
    documents_from_langchain = retriever.invoke("What is RAGStack ?")
    assert len(documents_from_langchain) > 0
    for doc in documents_from_langchain:
        print("doc:", doc)
        assert "framework" in doc.page_content

    retriever_no_docs = vector_db.as_retriever(
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


def test_ingest_langchain_retrieve_llama_index(environment):
    print("Running test_ingest_langchain_retrieve_llama_index")
    astra_ref = environment.astra_ref
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

    # TODO: uncomment this after https://github.com/run-llama/llama_index/issues/9432 is fixed

    # Verify compatibility of metadata filtering
    # filters = MetadataFilters(filters=[ExactMatchFilter(key="source", value="llama-index-ingest")])
    # retriever = index.as_retriever(filters=filters)
    # documents_from_llamaindex = retriever.retrieve("What is RAGStack ?")
    # assert len(documents_from_llamaindex) > 0
    # for doc in documents_from_llamaindex:
    #         print("doc:", doc)
    #    assert "framework" in doc.text

    #  filters = MetadataFilters(filters=[ExactMatchFilter(key="source", value="don't-find-anything-please")])
    # retriever_no_docs = index.as_retriever(filters=filters)
    # documents_from_llamaindex = retriever_no_docs.retrieve("What is RAGStack ?")
    # assert len(documents_from_llamaindex) == 0

    # Basic RAG with LlamaIndex
    query_engine = index.as_query_engine()
    response = query_engine.query("What is RAGStack ?")
    print("response:", response)

    assert "framework" in response.response


class Environment:
    def __init__(self, astra_ref):
        self.astra_ref = astra_ref


def delete_collection(astra_ref):
    db = LibAstraDB(token=astra_ref.token, api_endpoint=astra_ref.api_endpoint)
    db.delete_collection(astra_ref.collection)


@pytest.fixture
def environment():
    astra_ref = get_astra_ref()
    yield Environment(astra_ref=astra_ref)
    delete_collection(astra_ref)
