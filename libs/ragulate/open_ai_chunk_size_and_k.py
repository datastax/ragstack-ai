import os
from typing import Any

from langchain_astradb import AstraDBVectorStore
from langchain_community.document_loaders import UnstructuredFileLoader
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import Runnable, RunnablePassthrough
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

EMBEDDING_MODEL = "text-embedding-3-small"
LLM_MODEL = "gpt-3.5-turbo"


def get_vector_store(chunk_size: int) -> AstraDBVectorStore:
    return AstraDBVectorStore(
        embedding=OpenAIEmbeddings(model=EMBEDDING_MODEL),
        collection_name=f"chunk_size_{chunk_size}",
        token=os.getenv("ASTRA_DB_TOKEN"),
        api_endpoint=os.getenv("ASTRA_DB_ENDPOINT"),
    )


def ingest(file_path: str, chunk_size: int, **_: Any) -> None:
    vector_store = get_vector_store(chunk_size=chunk_size)

    chunk_overlap = min(chunk_size / 4, 64)

    text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
        model_name=EMBEDDING_MODEL,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )

    docs = UnstructuredFileLoader(
        file_path=file_path, mode="single", strategy="fast"
    ).load()
    split_docs = text_splitter.split_documents(docs)
    vector_store.add_documents(split_docs)


def query_pipeline(k: int, chunk_size: int, **_: Any) -> Runnable[Any, Any]:
    vector_store = get_vector_store(chunk_size=chunk_size)
    llm = ChatOpenAI(model=LLM_MODEL)

    # build a prompt
    prompt_template = """
    Answer the question based only on the supplied context. If you don't know the answer, say: "I don't know".
    Context: {context}
    Question: {question}
    Your answer:
    """  # noqa: E501
    prompt = ChatPromptTemplate.from_template(prompt_template)

    return (
        {
            "context": vector_store.as_retriever(search_kwargs={"k": k}),
            "question": RunnablePassthrough(),
        }
        | prompt
        | llm
        | StrOutputParser()
    )
