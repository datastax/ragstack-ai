from __future__ import annotations

import logging
import os
import time
from pathlib import Path
from typing import Any

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import UnstructuredFileLoader
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import Runnable, RunnablePassthrough
from langchain_openai import ChatOpenAI
from ragstack_colbert import (
    CassandraDatabase,
    Chunk,
    ColbertEmbeddingModel,
    ColbertVectorStore,
)
from ragstack_langchain.colbert import ColbertVectorStore as LangChainColbertVectorStore
from transformers import BertTokenizer

LLM_MODEL = "gpt-3.5-turbo"

batch_size = 640

astra_token = os.environ["ASTRA_DB_TOKEN"]
database_id = os.environ["ASTRA_DB_ID"]
keyspace = "colbert"


logging.basicConfig(level=logging.INFO)
logging.getLogger("unstructured").setLevel(logging.ERROR)
logging.getLogger("cassandra").setLevel(logging.ERROR)
logging.getLogger("http").setLevel(logging.ERROR)
logging.getLogger("httpx").setLevel(logging.ERROR)


def get_embedding_model(chunk_size: int) -> ColbertEmbeddingModel:
    return ColbertEmbeddingModel(doc_maxlen=chunk_size, chunk_batch_size=batch_size)


def get_database(chunk_size: int) -> CassandraDatabase:
    table_name = f"colbert_chunk_size_{chunk_size}"

    return CassandraDatabase.from_astra(
        astra_token=astra_token,
        database_id=database_id,
        keyspace=keyspace,
        table_name=table_name,
        timeout=500,
    )


def get_lc_vector_store(chunk_size: int) -> LangChainColbertVectorStore:
    database = get_database(chunk_size=chunk_size)
    embedding_model = get_embedding_model(chunk_size=chunk_size)

    return LangChainColbertVectorStore(
        database=database,
        embedding_model=embedding_model,
    )


def get_vector_store(chunk_size: int) -> ColbertVectorStore:
    database = get_database(chunk_size=chunk_size)
    return ColbertVectorStore(database=database)


tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")


def len_function(text: str) -> int:
    return len(tokenizer.tokenize(text))


async def ingest(file_path: str, chunk_size: int, **_: Any) -> None:
    doc_id = Path(file_path).name

    chunk_overlap = min(chunk_size / 4, 64)

    start = time.time()
    docs = UnstructuredFileLoader(
        file_path=file_path, mode="single", strategy="fast"
    ).load()
    duration = time.time() - start
    print(f"It took {duration} seconds to load and parse the document")

    # confirm only one document returned per file
    if not len(docs) == 1:
        raise ValueError("Only one document must be returned per file")

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len_function,
    )

    start = time.time()
    chunked_docs = text_splitter.split_documents(docs)
    duration = time.time() - start
    print(
        f"It took {duration} seconds to split the document "
        f"into {len(chunked_docs)} chunks"
    )

    texts = [doc.page_content for doc in chunked_docs]
    start = time.time()
    embeddings = get_embedding_model(chunk_size=chunk_size).embed_texts(texts=texts)
    duration = time.time() - start
    print(f"It took {duration} seconds to embed {len(chunked_docs)} chunks")

    colbert_vector_store = get_vector_store(chunk_size=chunk_size)

    await colbert_vector_store.adelete_chunks(doc_ids=[doc_id])

    chunks: list[Chunk] = []
    for i, doc in enumerate(chunked_docs):
        chunks.append(
            Chunk(
                doc_id=doc_id,
                chunk_id=i,
                text=doc.page_content,
                metadata={} if doc.metadata is None else doc.metadata,
                embedding=embeddings[i],
            )
        )

    start = time.time()
    await colbert_vector_store.aadd_chunks(chunks=chunks, concurrent_inserts=100)
    duration = time.time() - start
    print(
        f"It took {duration} seconds to insert {len(chunked_docs)} chunks into AstraDB"
    )


def query_pipeline(k: int, chunk_size: int, **_: Any) -> Runnable[Any, Any]:
    vector_store = get_lc_vector_store(chunk_size=chunk_size)
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
